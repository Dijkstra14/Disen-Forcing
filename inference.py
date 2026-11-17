"""Text-to-video inference for Disen-Forcing.

Example:
    python inference.py --config configs/inference.yaml --checkpoint model.pt \
        --prompts prompts/MovieGenVideoBench_extended.txt --output_dir videos

--checkpoint is either a training checkpoint (dict with 'generator' and
'generator_ema') or a generator state dict written by scripts/export_release.py.
For a training checkpoint, --weights picks the weights:
    ema_online_gate  EMA weights, with the per-head routing logits from the non-EMA model (default)
    ema              EMA weights only
    generator        non-EMA weights
The routing logits (learnable context routing) are stored under keys containing 'sink_gate_logit'.
Multi-GPU: torchrun --nproc_per_node=N inference.py ... (prompts are split across ranks).
"""
import argparse
import os

import torch
import torch.distributed as dist
from einops import rearrange
from omegaconf import OmegaConf
from torchvision.io import write_video
from tqdm import tqdm

import utils.wan_wrapper as wan_wrapper
from pipeline import CausalInferencePipeline
from utils.misc import set_seed
from demo_utils.memory import get_cuda_free_memory_gb, DynamicSwapInstaller

GATE_KEY = 'sink_gate_logit'
FSDP_PREFIX = '._fsdp_wrapped_module'


def strip_fsdp(sd):
    return {k.replace(FSDP_PREFIX, ''): v for k, v in sd.items()}


def ema_with_online_gate(online, ema):
    """EMA weights, with every routing-bias tensor taken from the online weights."""
    assert set(online) == set(ema), sorted(set(online) ^ set(ema))[:10]
    gate_keys = [k for k in ema if GATE_KEY in k]
    assert gate_keys, f'no {GATE_KEY} tensors in the checkpoint'
    merged = dict(ema)
    for k in gate_keys:
        merged[k] = online[k].clone()
    return merged, gate_keys


def load_generator_weights(path, weights):
    """Return a generator state dict (FSDP prefixes removed)."""
    if path.endswith('.safetensors'):
        from safetensors.torch import load_file
        ckpt = load_file(path)
    else:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)

    if 'generator' not in ckpt:   # exported state dict: already merged
        assert weights is None, '--weights applies to training checkpoints only'
        print(f'Loaded exported generator weights: {len(ckpt)} tensors')
        return strip_fsdp(ckpt)

    weights = weights or 'ema_online_gate'
    if weights == 'generator':
        return strip_fsdp(ckpt['generator'])
    assert 'generator_ema' in ckpt, 'checkpoint has no EMA weights'
    ema = strip_fsdp(ckpt['generator_ema'])
    if weights == 'ema':
        return ema
    sd, gate_keys = ema_with_online_gate(strip_fsdp(ckpt['generator']), ema)
    print(f'EMA weights + non-EMA routing logits: {len(gate_keys)} {GATE_KEY} tensors replaced')
    return sd


def read_lines(path):
    with open(path, encoding='utf-8') as f:
        return [line.rstrip() for line in f]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--prompts', required=True, help='one prompt per line')
    parser.add_argument('--extended_prompts', default=None,
                        help='optional file aligned with --prompts; its lines are used for generation')
    parser.add_argument('--indices', default=None,
                        help='comma-separated prompt line indices (default: all); seeds and names keep the line index')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--weights', choices=['ema_online_gate', 'ema', 'generator'], default=None,
                        help='default: ema_online_gate')
    parser.add_argument('--num_output_frames', type=int, default=None, help='latent frames (default: config)')
    parser.add_argument('--num_samples', type=int, default=None, help='videos per prompt (default: config)')
    parser.add_argument('--seed', type=int, default=None, help='default: config')
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    num_frames = args.num_output_frames or config.num_output_frames
    num_samples = args.num_samples or config.num_samples
    seed = config.seed if args.seed is None else args.seed

    if 'LOCAL_RANK' in os.environ:
        dist.init_process_group(backend='nccl')
        rank, world_size = dist.get_rank(), dist.get_world_size()
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        rank, world_size, local_rank = 0, 1, 0
        device = torch.device('cuda')
    set_seed(seed + local_rank)
    torch.set_grad_enabled(False)

    free_gb = get_cuda_free_memory_gb(device)
    low_memory = free_gb < config.low_memory_threshold_gb
    print(f'Free VRAM {free_gb:.1f} GB, low_memory={low_memory}')

    wan_wrapper.WAN_MODEL_DIR = config.wan_model_dir

    pipeline = CausalInferencePipeline(config, device=device)
    state_dict = load_generator_weights(args.checkpoint, args.weights)
    pipeline.generator.load_state_dict(state_dict, strict=True)
    pipeline = pipeline.to(dtype=torch.bfloat16)
    if low_memory:
        DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
    else:
        pipeline.text_encoder.to(device=device)
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)

    prompts = read_lines(args.prompts)
    if args.extended_prompts:
        extended = read_lines(args.extended_prompts)
        assert len(extended) == len(prompts), (len(extended), len(prompts))
        prompts = extended
    indices = [int(i) for i in args.indices.split(',')] if args.indices else list(range(len(prompts)))
    indices = indices[rank::world_size]

    os.makedirs(args.output_dir, exist_ok=True)
    stride = config.vae_spatial_stride
    latent_shape = [num_samples, num_frames, config.latent_channels,
                    config.height // stride, config.width // stride]

    for idx in tqdm(indices, disable=rank != 0):
        set_seed(seed + idx)   # per-prompt seed: output does not depend on the GPU split
        noise = torch.randn(latent_shape, device=device, dtype=torch.bfloat16)
        video, _ = pipeline.inference(
            noise=noise,
            text_prompts=[prompts[idx]] * num_samples,
            return_latents=True,
            initial_latent=None,
            low_memory=low_memory,
        )
        video = 255.0 * rearrange(video, 'b t c h w -> b t h w c').cpu()
        pipeline.vae.model.clear_cache()
        for s in range(num_samples):
            write_video(os.path.join(args.output_dir, f'{idx}-{s}.mp4'), video[s], fps=config.fps)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
