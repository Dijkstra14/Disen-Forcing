"""Measure how much attention every self-attention head puts on the first latent frame(s).

Runs the inference pipeline with full causal attention over the whole rollout (sink_size 0,
KV window = all frames, so no eviction and no routing bias) and records, for every self-attention
call, the softmax attention mass that the query tokens put on each key frame.

Score of head h in layer l, for one prompt and timestep: the attention mass on the first
--sink_frames latent frames (summed over their key tokens, averaged over the query tokens of the
chunk), averaged over the chunks in --score_chunks.

Outputs in --out_dir:
  sink_scores.json       {timestep: {"layer_L/head_H": [score of each prompt]}}  -> find_anchor_heads.py
  sink_scores.meta.json  measurement settings
  attn_maps.npz          (--maps) per-head chunk-by-chunk attention (query chunk x key chunk),
                         averaged over prompts and timesteps                    -> visualize_heads.py grid

Run from the repository root, e.g.
  python tools/measure_sink_attention.py --config configs/inference.yaml --checkpoint ode_init.pt \
      --prompts prompts.txt --out_dir anchor_scores/my_model --maps
"""
import argparse
import json
import os
import sys
import time
from functools import partial

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--config', required=True, help='inference config (model and sampler settings)')
    p.add_argument('--checkpoint', required=True, help='training checkpoint or exported generator state dict')
    p.add_argument('--weights', default='generator', help="state-dict key in a training checkpoint")
    p.add_argument('--prompts', required=True, help='one prompt per line')
    p.add_argument('--indices', default=None, help='comma-separated prompt line indices (default: all lines)')
    p.add_argument('--out_dir', required=True)
    p.add_argument('--num_frames', type=int, default=None, help='latent frames per rollout (default: config)')
    p.add_argument('--sink_frames', type=int, default=1, help='leading latent frames that count as the sink')
    p.add_argument('--score_chunks', default='last', help="'last' or comma-separated chunk indices")
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--per_prompt_seed', action='store_true',
                   help='seed each prompt with seed + line index (default: seed once, draw noise in prompt order)')
    p.add_argument('--maps', action='store_true', help='also record chunk-by-chunk maps at every chunk')
    p.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE', help='extra config overrides')
    return p.parse_args()


def frame_mass(q, k, fs, scale):
    """q [B, Lq, n, d], k [B, Lk, n, d] -> [n, Lk / fs]: softmax attention mass on each key frame,
    averaged over query tokens (and batch). Computed per head with per-frame log-sum-exp."""
    B, Lq, n, _ = q.shape
    Fk = k.shape[1] // fs
    out = torch.zeros(n, Fk, device=q.device)
    for b in range(B):
        for h in range(n):
            logits = (q[b, :, h] @ k[b, :, h].transpose(0, 1)).float().mul_(scale).view(Lq, Fk, fs)
            lse = torch.logsumexp(logits, dim=-1)
            out[h] += (lse - torch.logsumexp(lse, dim=-1, keepdim=True)).exp().mean(0)
    return out / B


class AttentionRecorder:
    """Wraps the model module's `attention` function; block hooks give the layer, generator hooks the
    timestep and chunk."""

    def __init__(self, pipeline, num_frames, sink_frames, score_chunks, maps):
        gen = pipeline.generator
        blocks = gen.model.blocks
        attn0 = blocks[0].self_attn
        self.fs, self.nfb = pipeline.frame_seq_length, pipeline.num_frame_per_block
        self.num_chunks = num_frames // self.nfb
        self.n, self.d, self.L = attn0.num_heads, attn0.head_dim, len(blocks)
        self.sink_frames, self.maps = sink_frames, maps
        self.score_chunks = {self.num_chunks - 1} if score_chunks == 'last' else {int(c) for c in score_chunks.split(',')}
        self.layer = self.t_key = self.start = None
        self.cur = {}        # (timestep, layer) -> [sum of scores over score chunks, count] for the current prompt
        self.scores = {}     # timestep -> head key -> per-prompt scores
        C = self.num_chunks
        self.map_sum = torch.zeros(self.L, self.n, C, C, dtype=torch.float64)
        self.map_cnt = torch.zeros(self.L, C, dtype=torch.float64)
        self.module = sys.modules[type(attn0).__module__]
        self._attention = self.module.attention
        self.module.attention = self._wrapped
        gen.register_forward_pre_hook(self._on_generator, with_kwargs=True)
        for i, blk in enumerate(blocks):
            blk.self_attn.register_forward_pre_hook(partial(self._enter, i))
            blk.self_attn.register_forward_hook(self._exit)

    def _on_generator(self, module, args, kwargs):
        self.t_key = str(float(kwargs['timestep'].flatten()[0]))
        self.start = kwargs['current_start'] // self.fs

    def _enter(self, i, module, args):
        self.layer = i

    def _exit(self, module, args, out):
        self.layer = None

    def _wrapped(self, q, k, v, *args, **kwargs):
        if self.layer is not None:
            self._record(q, k, kwargs.get('softmax_scale'))
        return self._attention(q, k, v, *args, **kwargs)

    def _record(self, q, k, scale):
        if q.shape[-1] != self.d:
            raise RuntimeError('routing bias is active; measure with sink_size 0')
        fq, fk = q.shape[1] // self.fs, k.shape[1] // self.fs
        if fk != self.start + fq or self.start % self.nfb or fq != self.nfb:
            raise RuntimeError(f'expected keys = frames 0..{self.start + fq - 1}, got {fk} key frames '
                               f'(KV eviction or sink active?)')
        chunk = self.start // self.nfb
        score = chunk in self.score_chunks
        if not (score or self.maps):
            return
        mass = frame_mass(q, k, self.fs, scale or self.d ** -0.5)
        if score:
            s = self.cur.setdefault((self.t_key, self.layer), [0.0, 0])
            s[0] = s[0] + mass[:, :self.sink_frames].sum(1)
            s[1] += 1
        if self.maps:
            self.map_sum[self.layer, :, chunk, :chunk + 1] += mass.view(self.n, chunk + 1, self.nfb).sum(-1).double().cpu()
            self.map_cnt[self.layer, chunk] += 1

    def end_prompt(self):
        for (t, l), (s, c) in self.cur.items():
            for h, v in enumerate((s / c).tolist()):
                self.scores.setdefault(t, {}).setdefault(f'layer_{l}/head_{h}', []).append(v)
        self.cur = {}

    def save(self, out_dir):
        with open(os.path.join(out_dir, 'sink_scores.json'), 'w') as f:
            json.dump(self.scores, f)
        if self.maps:
            m = (self.map_sum / self.map_cnt[:, None, :, None]).numpy()
            m[:, :, np.triu_indices(self.num_chunks, 1)[0], np.triu_indices(self.num_chunks, 1)[1]] = np.nan
            np.savez(os.path.join(out_dir, 'attn_maps.npz'), num_chunks=self.num_chunks,
                     **{f'L{l}H{h}': m[l, h].astype(np.float32) for l in range(self.L) for h in range(self.n)})


def load_weights(path, key):
    from inference import strip_fsdp
    if path.endswith('.safetensors'):
        from safetensors.torch import load_file
        sd = load_file(path)
    else:
        sd = torch.load(path, map_location='cpu', weights_only=False)
    if 'generator' in sd:   # training checkpoint
        if key not in sd:
            raise KeyError(f'{key!r} not in checkpoint; keys: {list(sd)}')
        sd = sd[key]
    return strip_fsdp(sd)


def main():
    args = parse_args()
    config = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.set))
    num_frames = args.num_frames or config.num_output_frames
    # Measurement protocol: keys are always all frames 0..current (full causal attention, no sink).
    protocol = {'model_kwargs': {'local_attn_size': num_frames, 'sink_size': 0, 'sink_gate': False}}
    config = OmegaConf.merge(config, protocol)
    print('protocol overrides:', protocol, 'extra:', args.set, flush=True)

    import utils.wan_wrapper as wan_wrapper
    from pipeline import CausalInferencePipeline
    from utils.misc import set_seed
    wan_wrapper.WAN_MODEL_DIR = config.wan_model_dir
    torch.set_grad_enabled(False)
    device = torch.device('cuda')

    pipeline = CausalInferencePipeline(config, device=device)
    missing, unexpected = pipeline.generator.load_state_dict(load_weights(args.checkpoint, args.weights), strict=False)
    bad = list(missing) + [k for k in unexpected if 'sink_gate_logit' not in k]  # trained checkpoints carry routing-bias tensors
    if bad:
        raise RuntimeError(f'state dict mismatch: {bad[:10]}')
    if unexpected:
        print(f'ignoring {len(unexpected)} routing-bias tensors (unused with sink_size 0)')
    pipeline = pipeline.to(dtype=torch.bfloat16)
    for m in (pipeline.text_encoder, pipeline.generator, pipeline.vae):
        m.to(device=device)

    rec = AttentionRecorder(pipeline, num_frames, args.sink_frames, args.score_chunks, args.maps)
    with open(args.prompts, encoding='utf-8') as f:
        prompts = [line.rstrip() for line in f]
    indices = [int(i) for i in args.indices.split(',')] if args.indices else list(range(len(prompts)))
    os.makedirs(args.out_dir, exist_ok=True)
    meta = dict(vars(args), protocol=protocol, num_frames=num_frames,
                num_frame_per_block=rec.nfb, score_chunks_resolved=sorted(rec.score_chunks), prompt_indices=indices)
    stride = config.vae_spatial_stride
    shape = [1, num_frames, config.latent_channels, config.height // stride, config.width // stride]

    set_seed(args.seed)
    for n_done, idx in enumerate(indices, 1):
        t0 = time.time()
        if args.per_prompt_seed:
            set_seed(args.seed + idx)
        noise = torch.randn(shape, device=device, dtype=torch.bfloat16)
        pipeline.inference(noise=noise, text_prompts=[prompts[idx]], return_latents=True)
        pipeline.vae.model.clear_cache()
        rec.end_prompt()
        rec.save(args.out_dir)
        print(f'[{n_done}/{len(indices)}] prompt {idx}: {time.time() - t0:.1f}s', flush=True)
    meta['timesteps'] = list(rec.scores)
    with open(os.path.join(args.out_dir, 'sink_scores.meta.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    print('wrote', args.out_dir)


if __name__ == '__main__':
    main()
