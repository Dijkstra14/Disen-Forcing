from typing import Tuple

import torch

from model.dmd import DMD
from utils.wan_wrapper import WanDiffusionWrapper


class RegDMD(DMD):
    """DMD plus the attention-map regularizer: at layer reg.probe_layer, KL(P_ref || P_student) between the
    cross-frame attention maps of an EMA reference and the student, on the student's own rollout. The live
    reference model is created here; the trainer keeps it equal to the reference EMA weights plus the online sink
    gate."""

    def __init__(self, args, device):
        super().__init__(args, device)
        self.reg_weight = float(args.reg_weight)   # lambda_reg
        self.use_reg = True   # every chunk is regularized
        self.reg_probe_layer = int(args.reg.probe_layer)
        self.reg_wt = {float(k): float(v) for k, v in dict(args.reg.wt_fixed).items()}
        self.generator.model.reg_cfg = dict(probe_layer=self.reg_probe_layer, wt=self.reg_wt)
        self.generator_ref = WanDiffusionWrapper(**args.model_kwargs, is_causal=True)
        self.generator_ref.model.num_frame_per_block = self.num_frame_per_block
        self.generator_ref.model.requires_grad_(False)

    def _run_generator(self, image_or_video_shape, conditional_dict):
        if self.inference_pipeline is None:
            self._initialize_inference_pipeline()
        pred, reg_sum = self.inference_pipeline.inference_with_trajectory(
            noise=torch.randn(list(image_or_video_shape), device=self.device, dtype=self.dtype),
            collect_reg_loss=True, **conditional_dict)
        return pred.to(self.dtype), reg_sum

    def generator_loss(self, image_or_video_shape, conditional_dict, unconditional_dict) -> Tuple[torch.Tensor, dict]:
        pred_image, reg_sum = self._run_generator(image_or_video_shape, conditional_dict)
        torch.cuda.empty_cache()
        dmd_loss, log_dict = self.compute_distribution_matching_loss(pred_image, conditional_dict, unconditional_dict)
        log_dict['dmd_loss'] = dmd_loss.detach()
        if reg_sum is None:
            return dmd_loss, log_dict
        last = self.inference_pipeline.last_reg
        reg_loss = reg_sum * last['scale']
        log_dict.update(reg_loss=reg_loss.detach(), reg_t=last['t'], reg_chunk_kls=last['chunk_kls'])
        if self.reg_weight == 0.0:   # logged only, no graph
            reg_loss = reg_loss.detach()
        return dmd_loss + self.reg_weight * reg_loss, log_dict
