from typing import Tuple

import torch
import torch.nn.functional as F

from model.base import SelfForcingModel
from utils.wan_wrapper import WanDiffusionWrapper


class DMD(SelfForcingModel):
    """Distribution Matching Distillation (https://arxiv.org/abs/2311.18828): generator loss from the real (teacher)
    and fake (critic) scores of backward-simulated samples, and the critic's denoising loss. No dataset needed."""

    def __init__(self, args, device):
        super().__init__(args, device)
        assert args.denoising_loss_type == 'flow', 'the critic is trained with the flow-matching loss'
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.same_step_across_blocks = getattr(args, "same_step_across_blocks", True)
        self.num_training_frames = getattr(args, "num_training_frames", 21)

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block
        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()
            self.fake_score.enable_gradient_checkpointing()

        self.inference_pipeline = None   # built later on the FSDP-wrapped modules
        self.use_reg, self.reg_probe_layer, self.reg_wt = False, None, None   # regularizer: see RegDMD

        self.num_train_timestep = args.num_train_timestep
        self.min_step = int(0.02 * self.num_train_timestep)
        self.max_step = int(0.98 * self.num_train_timestep)
        self.real_guidance_scale = args.guidance_scale
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)
        self.min_score_timestep = getattr(args, "min_score_timestep", 0)

        if getattr(self.scheduler, "alphas_cumprod", None) is not None:
            self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        else:
            self.scheduler.alphas_cumprod = None

    def _score_timestep(self, batch_size, num_frame):
        timestep = self._get_timestep(self.min_score_timestep, self.num_train_timestep, batch_size, num_frame)
        if self.timestep_shift > 1:
            timestep = self.timestep_shift * \
                (timestep / 1000) / (1 + (self.timestep_shift - 1) * (timestep / 1000)) * 1000
        return timestep.clamp(self.min_step, self.max_step)

    def _compute_kl_grad(self, noisy_image_or_video, estimated_clean_image_or_video, timestep,
                         conditional_dict, unconditional_dict) -> Tuple[torch.Tensor, dict]:
        """DMD gradient (eq. 7) with the eq. 8 normalisation."""
        _, pred_fake_image = self.fake_score(
            noisy_image_or_video=noisy_image_or_video, conditional_dict=conditional_dict, timestep=timestep)
        _, pred_real_image_cond = self.real_score(
            noisy_image_or_video=noisy_image_or_video, conditional_dict=conditional_dict, timestep=timestep)
        _, pred_real_image_uncond = self.real_score(
            noisy_image_or_video=noisy_image_or_video, conditional_dict=unconditional_dict, timestep=timestep)
        pred_real_image = pred_real_image_cond + (
            pred_real_image_cond - pred_real_image_uncond) * self.real_guidance_scale

        grad = (pred_fake_image - pred_real_image)
        p_real = (estimated_clean_image_or_video - pred_real_image)
        normalizer = torch.abs(p_real).mean(dim=[1, 2, 3, 4], keepdim=True)
        grad = grad / normalizer
        grad = torch.nan_to_num(grad)
        return grad, {"dmdtrain_gradient_norm": torch.mean(torch.abs(grad)).detach()}

    def compute_distribution_matching_loss(self, image_or_video, conditional_dict, unconditional_dict):
        original_latent = image_or_video
        batch_size, num_frame = image_or_video.shape[:2]
        with torch.no_grad():
            timestep = self._score_timestep(batch_size, num_frame)
            noise = torch.randn_like(image_or_video)
            noisy_latent = self.scheduler.add_noise(
                image_or_video.flatten(0, 1), noise.flatten(0, 1), timestep.flatten(0, 1)
            ).detach().unflatten(0, (batch_size, num_frame))
            grad, dmd_log_dict = self._compute_kl_grad(
                noisy_image_or_video=noisy_latent, estimated_clean_image_or_video=original_latent, timestep=timestep,
                conditional_dict=conditional_dict, unconditional_dict=unconditional_dict)
        dmd_loss = 0.5 * F.mse_loss(original_latent.double(), (original_latent.double() - grad.double()).detach(),
                                    reduction="mean")
        return dmd_loss, dmd_log_dict

    def critic_loss(self, image_or_video_shape, conditional_dict, unconditional_dict) -> Tuple[torch.Tensor, dict]:
        """Flow-matching denoising loss of the critic on generated samples (generator run without gradient)."""
        with torch.no_grad():
            generated_image, _ = self._run_generator(image_or_video_shape, conditional_dict)
        generated_shape = list(generated_image.shape[:2])
        critic_timestep = self._score_timestep(*generated_shape)
        critic_noise = torch.randn_like(generated_image)
        noisy_generated_image = self.scheduler.add_noise(
            generated_image.flatten(0, 1), critic_noise.flatten(0, 1), critic_timestep.flatten(0, 1)
        ).unflatten(0, generated_shape)
        _, pred_fake_image = self.fake_score(
            noisy_image_or_video=noisy_generated_image, conditional_dict=conditional_dict, timestep=critic_timestep)
        flow_pred = WanDiffusionWrapper._convert_x0_to_flow_pred(
            scheduler=self.scheduler,
            x0_pred=pred_fake_image.flatten(0, 1),
            xt=noisy_generated_image.flatten(0, 1),
            timestep=critic_timestep.flatten(0, 1))
        denoising_loss = self.denoising_loss_func(
            x=generated_image.flatten(0, 1),
            x_pred=pred_fake_image.flatten(0, 1),
            noise=critic_noise.flatten(0, 1),
            noise_pred=None,
            alphas_cumprod=self.scheduler.alphas_cumprod,
            timestep=critic_timestep.flatten(0, 1),
            flow_pred=flow_pred)
        return denoising_loss, {"critic_timestep": critic_timestep.detach()}
