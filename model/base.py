from torch import nn
import torch

from pipeline import SelfForcingTrainingPipeline
from utils.loss import get_denoising_loss
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder


class BaseModel(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self._initialize_models(args, device)

        self.device = device
        self.args = args
        self.dtype = torch.bfloat16 if args.mixed_precision else torch.float32
        self.denoising_step_list = torch.tensor(args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

    def _initialize_models(self, args, device):
        self.generator = WanDiffusionWrapper(**args.model_kwargs, is_causal=True)
        self.generator.model.requires_grad_(True)

        self.real_score = WanDiffusionWrapper(model_name=args.real_name, is_causal=False)
        self.real_score.model.requires_grad_(False)

        self.fake_score = WanDiffusionWrapper(model_name=getattr(args, "fake_name", "Wan2.1-T2V-1.3B"), is_causal=False)
        self.fake_score.model.requires_grad_(True)

        self.text_encoder = WanTextEncoder()
        self.text_encoder.requires_grad_(False)

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

    def _get_timestep(self, min_timestep, max_timestep, batch_size, num_frame):
        """One timestep per sample, uniform in [min_timestep, max_timestep), shared by all frames."""
        return torch.randint(min_timestep, max_timestep, [batch_size, 1], device=self.device,
                             dtype=torch.long).repeat(1, num_frame)


class SelfForcingModel(BaseModel):
    def __init__(self, args, device):
        super().__init__(args, device)
        self.denoising_loss_func = get_denoising_loss(args.denoising_loss_type)()

    def _initialize_inference_pipeline(self):
        """Training rollout pipeline over the FSDP-wrapped generator: Self-Forcing, or with the KV caching gradient
        (kv_cache_gradient: true)."""
        kwargs = dict(
            denoising_step_list=self.denoising_step_list,
            scheduler=self.scheduler,
            generator=self.generator,
            num_frame_per_block=self.num_frame_per_block,
            same_step_across_blocks=self.args.same_step_across_blocks,
            context_noise=self.args.context_noise,
            use_reg=self.use_reg,
            num_training_frames=self.num_training_frames,
        )
        if getattr(self.args, 'kv_cache_gradient', False):
            from pipeline.cachegrad_training import CacheGradTrainingPipeline
            self.inference_pipeline = CacheGradTrainingPipeline(
                **kwargs, reg_probe_layer=self.reg_probe_layer, reg_wt=self.reg_wt)
        else:
            self.inference_pipeline = SelfForcingTrainingPipeline(**kwargs)
