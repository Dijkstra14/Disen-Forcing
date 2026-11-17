"""Causal Wan2.1 backbone of Disen-Forcing.

Self-attention reads a rolling KV cache of `local_attn_size` latent frames whose first `sink_size` frames are a
frame sink. Learnable context routing: a per-head logit bias b_h on the sink keys (sink_gate_logit) controls how
much each head reads the sink once the window is full. The regularizer compares the student's cross-frame
attention maps at a probe layer (reg.probe_layer) with those of an EMA copy of the student (the EMA reference).
"""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from wan.modules.attention import attention
from wan.modules.model import (WAN_CROSSATTENTION_CLASSES, MLPProj, WanLayerNorm, WanRMSNorm, rope_params,
                               sinusoidal_embedding_1d)



def enable_sink_gate(model, init_logit=0.0):
    """One gate logit per head in every self-attention layer; the trainer initialises the values."""
    for blk in model.blocks:
        blk.self_attn.sink_gate_logit = nn.Parameter(torch.full((blk.self_attn.num_heads,), float(init_logit)))


def _gated_attention(q, k, v, bias, n_sink):
    """softmax(q.k / sqrt(d) + bias[h] * [key < n_sink]) v through flash attention: one extra dimension with
    q_extra = sqrt(d) and k_extra = bias[h] on the sink keys (0 elsewhere). The extra dimension is padded to 8
    channels so the head dim stays a multiple of 8 for flash attention."""
    B, Lk, n, d = k.shape
    pad = 8
    q_ext = q.new_zeros(*q.shape[:-1], pad)
    q_ext[..., 0] = math.sqrt(d)
    kb = bias.to(k.dtype).view(1, 1, n, 1).expand(B, n_sink, n, 1)
    k_ext = torch.cat([torch.cat([kb, k.new_zeros(B, n_sink, n, pad - 1)], dim=-1),
                       k.new_zeros(B, Lk - n_sink, n, pad)], dim=1)
    out = attention(torch.cat([q, q_ext], dim=-1), torch.cat([k, k_ext], dim=-1),
                    torch.cat([v, v.new_zeros(B, Lk, n, pad)], dim=-1), softmax_scale=d ** -0.5)
    return out[..., :d]


class _CacheGradScale(torch.autograd.Function):
    """Identity forward; gradient multiplied by alpha."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = float(alpha)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return g * ctx.alpha, None


def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    n, c = x.size(2), x.size(3) // 2
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(seq_len, 1, -1)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        output.append(torch.cat([x_i, x[i, seq_len:]]))
    return torch.stack(output).type_as(x)


def cross_frame_attn_map(q_frame, k_prev_frame):
    """softmax(Q K^T / sqrt(d)) between one query frame and one key frame, all heads, in fp32.
    q_frame, k_prev_frame: [B, fsl, n, d] -> (probs, log_probs), each [B, n, fsl, fsl]."""
    B, fsl, n, d = q_frame.shape
    with torch.autocast('cuda', enabled=False):
        q_t = q_frame.transpose(1, 2).flatten(0, 1).float()
        k_t = k_prev_frame.transpose(1, 2).flatten(0, 1).float()
        scores = torch.bmm(q_t, k_t.transpose(1, 2)) * (d ** -0.5)
        if not bool(torch.isfinite(scores).all()):
            raise FloatingPointError('cross-frame attention map: non-finite scores')
        log_attn = F.log_softmax(scores, dim=-1)
        return log_attn.exp().unflatten(0, (B, n)), log_attn.unflatten(0, (B, n))


def _chunk_maps(q, k, first_chunk, fs):
    """Maps of every query frame of the chunk against the frame before it. q: the chunk's roped queries;
    k: the roped keys read for it (current chunk last). Chunk 0 has no earlier frame: in-chunk pairs 1->0, 2->1."""
    nf = q.shape[1] // fs
    if first_chunk:
        if k.shape[1] != nf * fs or nf < 2:
            raise RuntimeError(f'chunk-0 map: {k.shape[1] // fs} key frames, {nf} query frames')
        frames, k_prev = range(1, nf), lambda fi: k[:, (fi - 1) * fs:fi * fs]
    else:
        kp = k[:, -(nf + 1) * fs:-fs]
        frames, k_prev = range(nf), lambda fi: kp[:, fi * fs:(fi + 1) * fs]
    maps = [cross_frame_attn_map(q[:, fi * fs:(fi + 1) * fs], k_prev(fi)) for fi in frames]
    return torch.stack([m[0] for m in maps], dim=1), torch.stack([m[1] for m in maps], dim=1)


class CausalWanSelfAttention(nn.Module):

    def __init__(self, dim, num_heads, local_attn_size=-1, sink_size=0, qk_norm=True, eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.max_attention_size = 32760 if local_attn_size == -1 else local_attn_size * 1560
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs, kv_cache=None, current_start=0, return_maps=False,
                maps_only=False, num_frame_per_block=None, cache_grad_scale=1.0, capture=None, t_zero=False):
        """kv_cache given: rollout step over one chunk. kv_cache None: teacher-forced forward over
        [clean context || noisy chunks]. return_maps: also return the chunk's cross-frame maps (regularized layer).
        maps_only: maps only, from the cache written by the preceding (checkpointed) call. t_zero: the t = 0 pass
        that writes the chunk into the cache."""
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        if kv_cache is None:
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return self._teacher_forced(q, k, v, seq_lens, grid_sizes, freqs, num_frame_per_block, cache_grad_scale, capture)

        fs = math.prod(grid_sizes[0][1:]).item()
        current_end = current_start + q.shape[1]
        sink_tokens = self.sink_size * fs
        if maps_only:
            local_end_index = kv_cache["local_end_index"].item()
            local_start_index = local_end_index - s
        else:
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            kv_cache_size = kv_cache["k"].shape[1]
            num_new_tokens = q.shape[1]
            if self.local_attn_size != -1 and (current_end > kv_cache["global_end_index"].item()) and (
                    num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
                # Evict the oldest frames after the sink.
                num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
                num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
                kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                local_end_index = kv_cache["local_end_index"].item() + current_end - \
                    kv_cache["global_end_index"].item() - num_evicted_tokens
            else:
                local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
            local_start_index = local_end_index - num_new_tokens
            kv_cache["k"][:, local_start_index:local_end_index] = k
            kv_cache["v"][:, local_start_index:local_end_index] = v

        # Read the whole cache while it fits the window, else [sink; most recent frames]; RoPE = slot order 0..n-1.
        L = self.max_attention_size
        in_window = local_end_index < L   # routing starts at the chunk that exactly fills the window
        in_window = self.local_attn_size == -1 or in_window
        sink_outside_window = not (in_window or ((L - sink_tokens) - q.shape[1]) // fs < 0)
        if self.local_attn_size != -1 and local_end_index > L:
            lo = local_end_index - L + sink_tokens
            k_read = torch.cat([kv_cache["k"][:, :sink_tokens], kv_cache["k"][:, lo:local_end_index]], dim=1)
            v_read = None if maps_only else torch.cat([kv_cache["v"][:, :sink_tokens], kv_cache["v"][:, lo:local_end_index]], dim=1)
        else:
            k_read = kv_cache["k"][:, :local_end_index]
            v_read = None if maps_only else kv_cache["v"][:, :local_end_index]
        n_kv = k_read.shape[1] // fs
        grid_kv = copy.deepcopy(grid_sizes)
        grid_kv[:, 0] = n_kv
        q_rope = causal_rope_apply(q, grid_sizes, freqs, start_frame=n_kv - q.shape[1] // fs).type_as(q)
        k_rope = causal_rope_apply(k_read, grid_kv, freqs, start_frame=0).type_as(q)
        if maps_only:
            return _chunk_maps(q_rope, k_rope, local_start_index < fs, fs)

        gated = sink_tokens > 0 and sink_outside_window
        if t_zero and gated:
            # The cache-writing pass reads no sink frames (exp(-1e4) = 0 in fp32), following Commit Forcing
            # (https://arxiv.org/abs/2611.00937).
            assert not torch.is_grad_enabled(), 'the cache-writing pass runs without gradient'
            bias = torch.full((self.num_heads,), -1e4, dtype=torch.float32, device=q_rope.device)
            x = _gated_attention(q_rope, k_rope, v_read, bias, sink_tokens)
        elif gated and hasattr(self, 'sink_gate_logit'):
            x = _gated_attention(q_rope, k_rope, v_read, self.sink_gate_logit.float(), sink_tokens)
        else:
            x = attention(q_rope, k_rope, v_read)
        kv_cache["global_end_index"].fill_(current_end)
        kv_cache["local_end_index"].fill_(local_end_index)
        x = self.o(x.flatten(2))
        if return_maps:
            return x, _chunk_maps(q_rope, k_rope, local_start_index < fs, fs)
        return x

    def _teacher_forced(self, q, k, v, seq_lens, grid_sizes, freqs, nfb, cache_grad_scale, capture):
        """x = [clean || noisy], F frames each. For chunk c, the clean queries (its cache write) and the noisy queries
        (its exit-step denoise) read the same keys as in the rollout: the clean frames before the chunk (whole history
        while it fits the window, else [sink; recent]), then the chunk's own tokens. Clean queries get the cache-write
        sink bias (-1e4), noisy queries the learned gate. cache_grad_scale multiplies the gradient of the clean context read by noisy
        queries. capture ('probs' / 'log'): per chunk, the noisy queries' cross-frame maps."""
        b, s, n, d = q.shape
        S = int(seq_lens[0])
        fs = int(math.prod(grid_sizes[0][1:]).item())
        sk = int(self.sink_size)
        F_, W = S // fs, self.max_attention_size // fs
        assert S * 2 == s and F_ % nfb == 0 and W - sk - nfb >= 0
        rec = W - sk - nfb
        out = q.new_zeros(b, s, n, d)
        maps = [] if capture is not None else None
        for c in range(F_ // nfb):
            c0 = c * nfb
            pre = list(range(c0)) if c0 + nfb <= W else list(range(sk)) + list(range(c0 - rec, c0))
            sink_outside_window = c0 + nfb >= W
            nk = len(pre) + nfb
            gk = grid_sizes.clone()
            gk[:, 0] = nk
            gq = grid_sizes.clone()
            gq[:, 0] = nfb
            for half, off in ((0, 0), (1, S)):
                k_in = torch.cat([k[:, f * fs:(f + 1) * fs] for f in pre] + [k[:, off + c0 * fs:off + (c0 + nfb) * fs]], dim=1)
                v_in = torch.cat([v[:, f * fs:(f + 1) * fs] for f in pre] + [v[:, off + c0 * fs:off + (c0 + nfb) * fs]], dim=1)
                if half == 1 and cache_grad_scale != 1.0 and len(pre) > 0 and torch.is_grad_enabled():
                    nc = len(pre) * fs
                    k_in = torch.cat([_CacheGradScale.apply(k_in[:, :nc], cache_grad_scale), k_in[:, nc:]], dim=1)
                    v_in = torch.cat([_CacheGradScale.apply(v_in[:, :nc], cache_grad_scale), v_in[:, nc:]], dim=1)
                qh = q[:, off + c0 * fs:off + (c0 + nfb) * fs]
                q_r = causal_rope_apply(qh, gq, freqs, start_frame=nk - nfb).type_as(qh)
                k_r = causal_rope_apply(k_in, gk, freqs, start_frame=0).type_as(qh)
                bias = None
                if sink_outside_window and sk > 0:
                    if half == 0:   # clean context (t = 0): no sink frames, as in the cache-writing pass
                        bias = torch.full((n,), -1e4, dtype=torch.float32, device=q.device)
                    elif half == 1 and hasattr(self, 'sink_gate_logit'):
                        bias = self.sink_gate_logit.float()
                o = _gated_attention(q_r, k_r, v_in, bias, sk * fs) if bias is not None else attention(q_r, k_r, v_in)
                out[:, off + c0 * fs:off + (c0 + nfb) * fs] = o
                if half == 1 and maps is not None:
                    pairs = [cross_frame_attn_map(q_r[:, fi * fs:(fi + 1) * fs],
                                                  k_r[:, (len(pre) + fi - 1) * fs:(len(pre) + fi) * fs])
                             for fi in (range(1, nfb) if c == 0 else range(nfb))]
                    maps.append(torch.stack([m[0 if capture == 'probs' else 1] for m in pairs], dim=1))
        out = self.o(out.flatten(2))
        return (out, maps) if maps is not None else out


class CausalWanAttentionBlock(nn.Module):

    def __init__(self, cross_attn_type, dim, ffn_dim, num_heads, local_attn_size=-1, sink_size=0, qk_norm=True,
                 cross_attn_norm=False, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(dim, num_heads, local_attn_size, sink_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens, kv_cache=None, crossattn_cache=None,
                current_start=0, return_maps=False, maps_only=False, num_frame_per_block=None, cache_grad_scale=1.0,
                capture=None, t_zero=False):
        """x: [B, L, C]; e: [B, F, 6, C]. Returns x, or (x, maps) when the self-attention returns maps."""
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        attn_in = (self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2)
        if maps_only:
            return x, self.self_attn(attn_in, seq_lens, grid_sizes, freqs, kv_cache, current_start, maps_only=True)
        y = self.self_attn(attn_in, seq_lens, grid_sizes, freqs, kv_cache, current_start, return_maps=return_maps,
                           num_frame_per_block=num_frame_per_block, cache_grad_scale=cache_grad_scale, capture=capture,
                           t_zero=t_zero)
        y, maps = y if isinstance(y, tuple) else (y, None)
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)
        x = x + self.cross_attn(self.norm3(x), context, context_lens, crossattn_cache=crossattn_cache)
        y = self.ffn((self.norm2(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2))
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[5]).flatten(1, 2)
        return (x, maps) if maps is not None else x


class CausalHead(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        return self.head(self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0])


class CausalWanModel(ModelMixin, ConfigMixin):
    ignore_for_config = ['patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim']
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self, model_type='t2v', patch_size=(1, 2, 2), text_len=512, in_dim=16, dim=2048, ffn_dim=8192,
                 freq_dim=256, text_dim=4096, out_dim=16, num_heads=16, num_layers=32, local_attn_size=-1, sink_size=0,
                 qk_norm=True, cross_attn_norm=True, eps=1e-6):
        super().__init__()
        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'), nn.Linear(dim, dim))
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads, local_attn_size, sink_size, qk_norm,
                                    cross_attn_norm, eps)
            for _ in range(num_layers)])
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # buffers (not registered, so .to() does not change their dtype)
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([rope_params(1024, d - 4 * (d // 6)), rope_params(1024, 2 * (d // 6)),
                                rope_params(1024, 2 * (d // 6))], dim=1)
        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)
        self.init_weights()

        self.gradient_checkpointing = False
        self.num_frame_per_block = 1
        self.reg_cfg = None   # rollout regularizer {probe_layer, wt}; set on the student by RegDMD
        # EMA reference of the regularizer (student only): the live reference model, set by the trainer, and its
        # per-rollout caches. A plain dict, so the reference is not registered as a submodule.
        self.reference = dict(generator=None, kv=None, xa=None, x0=None)

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def _time(self, t, x):
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        return e, self.time_projection(e).unflatten(1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)

    def _forward_inference(self, x, t, context, seq_len, kv_cache, crossattn_cache=None, current_start=0):
        """One rollout step over a chunk with KV caching (CausVid Alg. 2). Returns (x0-flow, reg loss or None)."""
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)
        raw_x, raw_context = x, context
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(x)
        e, e0 = self._time(t, x)
        context = self.text_embedding(torch.stack([
            torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context]))
        kwargs = dict(e=e0, seq_lens=seq_lens, grid_sizes=grid_sizes, freqs=self.freqs, context=context, context_lens=None,
                      t_zero=bool((t == 0).all()))

        # Regularizer: on the student's exit-step (grad-enabled) forward of every chunk.
        reg = self.reg_cfg
        fs = int(math.prod(grid_sizes[0][1:].tolist()))
        do_reg = reg is not None and torch.is_grad_enabled()
        ref_probs = None
        if do_reg:
            t_scalar = t.flatten()[0].item()
            ref_probs = self._reference_maps(raw_x, raw_context, t, current_start)
        if reg is not None and torch.is_grad_enabled() and self.reference['kv'] is not None:
            self._reference_x0(raw_x, raw_context, t, current_start)

        student_logp = None
        for i, block in enumerate(self.blocks):
            at_reg_layer = do_reg and i == reg['probe_layer']
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                # As upstream: no cross-attention cache on the gradient path. The maps are recomputed outside
                # the checkpoint from the block input and the cache it wrote.
                kw = dict(kwargs, kv_cache=kv_cache[i], current_start=current_start)
                x_in = x
                x = torch.utils.checkpoint.checkpoint(block, x, **kw, use_reentrant=False)
                if at_reg_layer:
                    student_logp = block(x_in, **kw, maps_only=True)[1][1]
            else:
                x = block(x, **kwargs, kv_cache=kv_cache[i], current_start=current_start, return_maps=at_reg_layer,
                          crossattn_cache=crossattn_cache[i] if crossattn_cache is not None else None)
                if at_reg_layer:
                    x, (_, student_logp) = x

        reg_loss = None
        if do_reg:
            # KL(P_ref || P_student) per query token, times w(t); chunk 0 (nfb - 1 frame pairs) weighted per pair.
            kl = F.kl_div(student_logp.float().flatten(0, -2), ref_probs.float().flatten(0, -2), reduction='batchmean')
            w = reg_weight(reg['wt'], t_scalar)
            reg_loss = kl * w
            pw = student_logp.shape[1] / self.num_frame_per_block
            if pw != 1.0:
                reg_loss = reg_loss * pw
            self.reg_last = dict(chunk=int(current_start // (fs * self.num_frame_per_block)), t=t_scalar,
                                 kl=float(kl.detach()), w=float(w), pw=pw)

        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        return torch.stack(self.unpatchify(x, grid_sizes)), reg_loss

    @torch.no_grad()
    def _reference_maps(self, raw_x, raw_context, t, current_start):
        """Attention probabilities of the reference at the regularized layer for the student's exit-step input (its
        own embeddings and KV cache, online gate). Writes the chunk's KV up to that layer; the cache write overwrites
        it."""
        m = self.reference['generator'].model
        kv, xa = self.reference['kv'], self.reference['xa']
        if int(kv[0]['global_end_index'].item()) != current_start:
            raise RuntimeError(f"reference cache at {int(kv[0]['global_end_index'].item())}, student at {current_start}")
        reg_layer = int(self.reg_cfg['probe_layer'])
        with torch.autocast('cuda', dtype=torch.bfloat16):
            x = [m.patch_embedding(u.unsqueeze(0)) for u in raw_x]
            grid = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
            x = torch.cat([u.flatten(2).transpose(1, 2) for u in x])
            e = m.time_embedding(sinusoidal_embedding_1d(m.freq_dim, t.flatten()).type_as(x))
            e0 = m.time_projection(e).unflatten(1, (6, m.dim)).unflatten(dim=0, sizes=t.shape)
            ctx = m.text_embedding(torch.stack([torch.cat([u, u.new_zeros(m.text_len - u.size(0), u.size(1))]) for u in raw_context]))
            kw = dict(e=e0, seq_lens=torch.tensor([x.shape[1]] * x.shape[0], dtype=torch.long), grid_sizes=grid,
                      freqs=m.freqs.to(x.device), context=ctx, context_lens=None)
            for i in range(reg_layer + 1):
                x = m.blocks[i](x, **kw, kv_cache=kv[i], crossattn_cache=xa[i], current_start=current_start,
                                return_maps=i == reg_layer)
        return x[1][0]

    @torch.no_grad()
    def _reference_x0(self, raw_x, raw_context, t, current_start):
        """The reference's own x0 for the same input and t through its own cache; the pipeline writes it into the
        cache."""
        if int(self.reference['kv'][-1]['global_end_index'].item()) != current_start:
            raise RuntimeError('reference cache out of step with the student')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            _, x0 = self.reference['generator'](
                noisy_image_or_video=torch.stack(list(raw_x)).permute(0, 2, 1, 3, 4),
                conditional_dict=dict(prompt_embeds=raw_context), timestep=t, kv_cache=self.reference['kv'],
                crossattn_cache=self.reference['xa'], current_start=current_start)
        self.reference['x0'] = x0.detach().to(torch.bfloat16)

    def _forward_train(self, x, t, context, seq_len, clean_x, aug_t=None, cache_grad_scale=1.0, capture=None):
        """Pass 2 of the KV caching gradient: one teacher-forced forward over [clean context || noisy chunks] (no KV cache).
        capture = (layer, 'probs' | 'log') also returns that layer's per-chunk maps."""
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        def patchify(v):
            v = [self.patch_embedding(u.unsqueeze(0)) for u in v]
            grid = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in v])
            v = [u.flatten(2).transpose(1, 2) for u in v]
            lens = torch.tensor([u.size(1) for u in v], dtype=torch.long)
            assert lens.max() <= seq_len
            return torch.cat([torch.cat([u, u.new_zeros(1, lens[0] - u.size(1), u.size(2))], dim=1) for u in v]), grid, lens

        x, grid_sizes, seq_lens = patchify(x)
        e, e0 = self._time(t, x)
        context = self.text_embedding(torch.stack([
            torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context]))
        clean, _, _ = patchify(clean_x)
        x = torch.cat([clean, x], dim=1)
        if aug_t is None:
            aug_t = torch.zeros_like(t)
        _, e0_clean = self._time(aug_t, x)
        kwargs = dict(e=torch.cat([e0_clean, e0], dim=1), seq_lens=seq_lens, grid_sizes=grid_sizes, freqs=self.freqs,
                      context=context, context_lens=None, num_frame_per_block=self.num_frame_per_block,
                      cache_grad_scale=cache_grad_scale)
        maps = None
        for i, block in enumerate(self.blocks):
            if capture is not None and i == capture[0]:   # not checkpointed: the maps stay in the graph once
                x, maps = block(x, **kwargs, capture=capture[1])
            elif torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(block, x, **kwargs, use_reentrant=False)
            else:
                x = block(x, **kwargs)
        x = x[:, x.shape[1] // 2:]
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        x = torch.stack(self.unpatchify(x, grid_sizes))
        return (x, maps) if capture is not None else x

    def forward(self, *args, **kwargs):
        if kwargs.get('kv_cache', None) is not None:
            return self._forward_inference(*args, **kwargs)
        return self._forward_train(*args, **kwargs)

    def unpatchify(self, x, grid_sizes):
        """[L, C_out * prod(patch)] per sample -> [C_out, F, H / 8, W / 8]."""
        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            out.append(u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)]))
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        nn.init.zeros_(self.head.head.weight)


def reg_weight(wt, t):
    """w(t) of the nearest denoising step (training timesteps arrive bf16-rounded, e.g. 833.3 -> 832)."""
    b = min(wt, key=lambda v: abs(v - float(t)))
    if abs(b - float(t)) > 4.0:
        raise ValueError(f'timestep {t} is not on the reg weight grid {sorted(wt)}')
    return wt[b]
