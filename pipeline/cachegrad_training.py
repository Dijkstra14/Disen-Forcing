"""Training rollout with the KV caching gradient, adapted from Self Gradient Forcing
(https://github.com/zhuang2002/Self_Gradient_Forcing, Apache-2.0).

Pass 1 (no grad): the Self-Forcing rollout over the KV cache; records every chunk's exit-step input x_t and the x0
it writes to the cache.
Pass 2 (grad): one teacher-forced forward over [cached x0 || exit-step inputs] in which each chunk reads exactly
the keys it read in the rollout (CausalWanSelfAttention._teacher_forced). The DMD and regularizer gradients reach
the K/V of the cached context tokens; cache_grad_scale scales the gradient of that context read.
Regularizer: the EMA reference first runs its own rollout on the same exit-step inputs (predicts and caches its
own x0), then a teacher-forced forward on its own context; per chunk, KL(reference maps || student maps).
"""
import torch
import torch.nn.functional as F

import wan.modules.causal_model_anchor_reg as causal_model
from pipeline.self_forcing_training import SelfForcingTrainingPipeline


class CacheGradTrainingPipeline(SelfForcingTrainingPipeline):
    def __init__(self, *args, reg_probe_layer=None, reg_wt=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.reg_probe_layer = reg_probe_layer
        self.reg_wt = reg_wt
        self.cache_grad_scale = 1.0   # set by the trainer every effective step

    @torch.no_grad()
    def _pass1_rollout(self, noise, conditional_dict, exit_idx, num_blocks):
        """Returns the exit-step inputs, the cached x0 (context) and the exit timesteps."""
        batch_size = noise.shape[0]
        nfb, fs = self.num_frame_per_block, self.frame_seq_length
        noisy_at_t = torch.zeros_like(noise)
        x_ctx = torch.zeros_like(noise)
        # float32: the dtype of the Pass-1 timestep (int64 ones * float step)
        tf_timestep = torch.zeros(noise.shape[:2], device=noise.device, dtype=torch.float32)
        for block_index in range(num_blocks):
            cs, ce = block_index * nfb, (block_index + 1) * nfb
            noisy_input = noise[:, cs:ce]
            for index, current_timestep in enumerate(self.denoising_step_list):
                timestep = torch.ones([batch_size, nfb], device=noise.device, dtype=torch.int64) * current_timestep
                if index == exit_idx:
                    noisy_at_t[:, cs:ce] = noisy_input
                    tf_timestep[:, cs:ce] = current_timestep
                _, denoised_pred = self.generator(
                    noisy_image_or_video=noisy_input, conditional_dict=conditional_dict, timestep=timestep,
                    kv_cache=self.kv_cache1, crossattn_cache=self.crossattn_cache, current_start=cs * fs)
                if index == exit_idx:
                    break
                next_timestep = self.denoising_step_list[index + 1]
                noisy_input = self.scheduler.add_noise(
                    denoised_pred.flatten(0, 1),
                    torch.randn_like(denoised_pred.flatten(0, 1)),
                    next_timestep * torch.ones([batch_size * nfb], device=noise.device, dtype=torch.long)
                ).unflatten(0, denoised_pred.shape[:2])

            context_timestep = torch.ones_like(timestep) * self.context_noise
            ctx_eps = torch.randn_like(denoised_pred.flatten(0, 1))   # drawn even at context_noise 0 (random stream)
            x = denoised_pred if self.context_noise == 0 else self.scheduler.add_noise(
                denoised_pred.flatten(0, 1), ctx_eps, context_timestep.reshape(-1).long()
            ).unflatten(0, denoised_pred.shape[:2])
            x_ctx[:, cs:ce] = x
            self._write_cache(self.generator, x, context_timestep, self.kv_cache1, self.crossattn_cache, cs * fs,
                         conditional_dict)
        return noisy_at_t, x_ctx, tf_timestep

    @torch.no_grad()
    def _reference_rollout(self, ref, noisy_at_t, tf_timestep, conditional_dict, num_blocks):
        """EMA reference on the student's exit-step inputs: per chunk it predicts its own x0 through its own
        cache, then writes that x0 into the cache. Returns the reference's x0 (its Pass-2 context)."""
        nfb, fs = self.num_frame_per_block, self.frame_seq_length
        # The Pass-1 caches are no longer needed: fresh caches for the reference.
        self._initialize_kv_cache(batch_size=noisy_at_t.shape[0], dtype=noisy_at_t.dtype, device=noisy_at_t.device)
        self._initialize_crossattn_cache(batch_size=noisy_at_t.shape[0], dtype=noisy_at_t.dtype, device=noisy_at_t.device)
        x0_ref = torch.zeros_like(noisy_at_t)
        for block_index in range(num_blocks):
            cs, ce = block_index * nfb, (block_index + 1) * nfb
            t = tf_timestep[:, cs:ce]
            _, x0 = ref(noisy_image_or_video=noisy_at_t[:, cs:ce], conditional_dict=conditional_dict, timestep=t,
                        kv_cache=self.kv_cache1, crossattn_cache=self.crossattn_cache, current_start=cs * fs)
            x0_ref[:, cs:ce] = x0
            self._write_cache(ref, x0, torch.zeros_like(t), self.kv_cache1, self.crossattn_cache, cs * fs, conditional_dict)
        return x0_ref

    def _reg_loss(self, student_logp, ref_probs, t):
        """Mean over frame pairs (chunk 0 has nfb - 1) of KL(P_ref || P_student) per query token, times w(t).
        Returned as a sum over chunks; RegDMD divides by the chunk count."""
        w = causal_model.reg_weight(self.reg_wt, t)
        total, pair_weights, kls = 0., 0., []
        for logp, probs in zip(student_logp, ref_probs):
            pw = logp.shape[1] / float(self.num_frame_per_block)
            kl = F.kl_div(logp.flatten(0, -2), probs.flatten(0, -2), reduction='batchmean')
            kls.append(float(kl.detach()))
            total = total + kl * pw
            pair_weights += pw
        reg = w * total / pair_weights
        self.last_reg = dict(scale=1. / len(kls), t=float(t), chunk_kls=[[c, kl, w] for c, kl in enumerate(kls)])
        return reg * float(len(kls))

    def inference_with_trajectory(self, noise: torch.Tensor, collect_reg_loss: bool = False, **conditional_dict):
        assert self.same_step_across_blocks, 'Pass 2 needs one exit step shared by all chunks'
        batch_size, num_frames = noise.shape[:2]
        assert num_frames % self.num_frame_per_block == 0
        num_blocks = num_frames // self.num_frame_per_block
        exit_idx = self._exit_flags(num_blocks, noise.device)[0]

        self._initialize_kv_cache(batch_size=batch_size, dtype=noise.dtype, device=noise.device)
        self._initialize_crossattn_cache(batch_size=batch_size, dtype=noise.dtype, device=noise.device)
        noisy_at_t, x_ctx, tf_timestep = self._pass1_rollout(noise, conditional_dict, exit_idx, num_blocks)

        aug_t = torch.ones([batch_size, num_frames], device=noise.device, dtype=torch.int64) * self.context_noise
        pass2 = dict(noisy_image_or_video=noisy_at_t, conditional_dict=conditional_dict, timestep=tf_timestep,
                     clean_x=x_ctx.detach(), aug_t=aug_t, cache_grad_scale=self.cache_grad_scale)
        reg_loss = None
        if self.use_reg and collect_reg_loss and torch.is_grad_enabled():
            ref = getattr(self.generator.model, 'module', self.generator.model).reference['generator']
            x0_ref = self._reference_rollout(ref, noisy_at_t, tf_timestep, conditional_dict, num_blocks)
            with torch.no_grad():
                _, _, ref_probs = ref(noisy_image_or_video=noisy_at_t, conditional_dict=conditional_dict,
                                      timestep=tf_timestep, clean_x=x0_ref, aug_t=aug_t,
                                      reg_capture=(self.reg_probe_layer, 'probs'))
            _, pred_x0, student_logp = self.generator(**pass2, reg_capture=(self.reg_probe_layer, 'log'))
            reg_loss = self._reg_loss(student_logp, ref_probs, float(tf_timestep[0, 0].item()))
            del ref_probs, student_logp, x0_ref
        else:
            _, pred_x0 = self.generator(**pass2)
        if collect_reg_loss:
            return pred_x0, reg_loss
        return pred_x0
