from typing import List

import torch
import torch.distributed as dist

from utils.scheduler import SchedulerInterface
from utils.wan_wrapper import WanDiffusionWrapper


class SelfForcingTrainingPipeline:
    """Self-Forcing rollout: chunk by chunk, few-step denoising over the KV cache, then a t = 0 pass that writes
    the chunk into the cache. Gradient flows through each chunk's exit step only. With the regularizer, the EMA
    reference keeps its own caches and writes its own x0 into them next to the student (see
    CausalWanModel._forward_inference)."""

    def __init__(self,
                 denoising_step_list: List[int],
                 scheduler: SchedulerInterface,
                 generator: WanDiffusionWrapper,
                 num_frame_per_block=3,
                 same_step_across_blocks: bool = False,
                 context_noise: int = 0,
                 use_reg: bool = False,
                 num_transformer_blocks: int = 30,
                 frame_seq_length: int = 1560,
                 num_training_frames: int = 21):
        self.scheduler = scheduler
        self.generator = generator
        self.denoising_step_list = denoising_step_list
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]  # remove the zero timestep for inference

        self.num_transformer_blocks = num_transformer_blocks
        self.frame_seq_length = frame_seq_length
        self.num_frame_per_block = num_frame_per_block
        self.context_noise = context_noise
        self.same_step_across_blocks = same_step_across_blocks
        self.use_reg = use_reg   # regularizer on every chunk
        self.kv_cache_size = num_training_frames * frame_seq_length
        assert int(context_noise) == 0, 'the cache write runs at t = 0'
        self._shared_exit_flags = None   # set by the trainer: one exit step per optimizer step, shared by its micro-steps
        self.last_reg = None

    def generate_and_sync_list(self, num_blocks, num_denoising_steps, device):
        if dist.get_rank() == 0:
            indices = torch.randint(low=0, high=num_denoising_steps, size=(num_blocks,), device=device)
        else:
            indices = torch.empty(num_blocks, dtype=torch.long, device=device)
        dist.broadcast(indices, src=0)  # the same exit steps on every rank
        return indices.tolist()

    def _exit_flags(self, num_blocks, device):
        if self._shared_exit_flags is not None:
            return self._shared_exit_flags
        return self.generate_and_sync_list(num_blocks, len(self.denoising_step_list), device=device)

    def _write_cache(self, model, latents, timestep, kv_cache, crossattn_cache, current_start, conditional_dict):
        """t = 0 re-forward that writes the chunk into the cache (at t = 0 the model reads no sink frames)."""
        with torch.no_grad():
            model(noisy_image_or_video=latents, conditional_dict=conditional_dict, timestep=timestep,
                  kv_cache=kv_cache, crossattn_cache=crossattn_cache, current_start=current_start)

    def inference_with_trajectory(self, noise: torch.Tensor, collect_reg_loss: bool = False, **conditional_dict):
        """Returns the rollout; with collect_reg_loss also the regularizer sum over chunks (None without
        gradient or regularizer)."""
        batch_size, num_frames = noise.shape[:2]
        nfb, fs = self.num_frame_per_block, self.frame_seq_length
        assert num_frames % nfb == 0
        num_blocks = num_frames // nfb
        output = torch.zeros_like(noise)
        self._initialize_kv_cache(batch_size=batch_size, dtype=noise.dtype, device=noise.device)
        self._initialize_crossattn_cache(batch_size=batch_size, dtype=noise.dtype, device=noise.device)
        exit_flags = self._exit_flags(num_blocks, noise.device)

        # EMA reference caches live for one generator rollout (critic rollouts run without gradient or reg).
        use_reg = self.use_reg and collect_reg_loss and torch.is_grad_enabled()
        ref = getattr(self.generator.model, 'module', self.generator.model).reference
        ref.update(kv=None, xa=None, x0=None)
        if use_reg:
            ref.update(
                kv=[dict(k=torch.zeros_like(c['k'], dtype=torch.bfloat16), v=torch.zeros_like(c['v'], dtype=torch.bfloat16),
                         global_end_index=torch.zeros_like(c['global_end_index']),
                         local_end_index=torch.zeros_like(c['local_end_index'])) for c in self.kv_cache1],
                xa=[dict(k=torch.zeros_like(c['k'], dtype=torch.bfloat16), v=torch.zeros_like(c['v'], dtype=torch.bfloat16),
                         is_init=False) for c in self.crossattn_cache])
            ref_cond = {k: (v.to(torch.bfloat16) if torch.is_tensor(v) and v.is_floating_point() else v)
                        for k, v in conditional_dict.items()}
        reg_total, chunks = None, []

        for block_index in range(num_blocks):
            current_start = block_index * nfb * fs
            noisy_input = noise[:, block_index * nfb:(block_index + 1) * nfb]
            exit_idx = exit_flags[0] if self.same_step_across_blocks else exit_flags[block_index]
            for index, current_timestep in enumerate(self.denoising_step_list):
                timestep = torch.ones([batch_size, nfb], device=noise.device, dtype=torch.int64) * current_timestep
                if index < exit_idx:
                    with torch.no_grad():
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input, conditional_dict=conditional_dict, timestep=timestep,
                            kv_cache=self.kv_cache1, crossattn_cache=self.crossattn_cache, current_start=current_start)
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep * torch.ones([batch_size * nfb], device=noise.device, dtype=torch.long)
                    ).unflatten(0, denoised_pred.shape[:2])
                    continue
                # exit step: the only forward that keeps gradient
                if collect_reg_loss:
                    _, denoised_pred, step_reg = self.generator(
                        noisy_image_or_video=noisy_input, conditional_dict=conditional_dict, timestep=timestep,
                        kv_cache=self.kv_cache1, crossattn_cache=self.crossattn_cache, current_start=current_start,
                        return_reg_loss=True)
                    if use_reg:
                        if step_reg is None:
                            raise RuntimeError(f'regularizer not computed at chunk {block_index}')
                        inner = getattr(self.generator.model, 'module', self.generator.model)
                        chunks.append(inner.reg_last)
                        reg_total = step_reg if reg_total is None else reg_total + step_reg
                else:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input, conditional_dict=conditional_dict, timestep=timestep,
                        kv_cache=self.kv_cache1, crossattn_cache=self.crossattn_cache, current_start=current_start)
                break

            output[:, block_index * nfb:(block_index + 1) * nfb] = denoised_pred

            context_timestep = torch.ones_like(timestep) * self.context_noise
            # Drawn even at context_noise 0, so the random stream does not depend on context_noise.
            ctx_eps = torch.randn_like(denoised_pred.flatten(0, 1))
            if self.context_noise != 0:
                denoised_pred = self.scheduler.add_noise(
                    denoised_pred.flatten(0, 1), ctx_eps, context_timestep.reshape(-1).long()
                ).unflatten(0, denoised_pred.shape[:2])
            self._write_cache(self.generator, denoised_pred, context_timestep, self.kv_cache1, self.crossattn_cache,
                         current_start, conditional_dict)
            if use_reg:   # the reference writes its own x0 of this chunk into its cache (same t = 0 rules)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    self._write_cache(ref['generator'], ref['x0'], context_timestep, ref['kv'], ref['xa'],
                                 current_start, ref_cond)
                ref['x0'] = None

        ref.update(kv=None, xa=None, x0=None)   # freed before backward
        if not collect_reg_loss:
            return output
        if use_reg:
            if len(chunks) != num_blocks:
                raise ValueError(f'regularizer covered {len(chunks)} chunks, expected {num_blocks}')
            # Mean over frame pairs: chunk terms carry pairs / nfb, so normalise by their sum.
            self.last_reg = dict(scale=1. / sum(c['pw'] for c in chunks), t=chunks[0]['t'] if chunks else None,
                                 chunk_kls=[[c['chunk'], c['kl'], c['w']] for c in chunks])
        return output, reg_total

    def _initialize_kv_cache(self, batch_size, dtype, device):
        self.kv_cache1 = [{
            "k": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
            "v": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
            "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
            "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
        } for _ in range(self.num_transformer_blocks)]

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        self.crossattn_cache = [{
            "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
            "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
            "is_init": False
        } for _ in range(self.num_transformer_blocks)]
