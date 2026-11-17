"""Disen-Forcing training. Launch with torchrun, e.g.

    torchrun --nnodes=2 --nproc_per_node=4 ... train.py --config configs/disen.yaml --logdir outputs/disen

Extra `key=value` arguments override config entries (OmegaConf dot-list, e.g. max_train_steps=4).
"""
import argparse
import os

import torch.distributed as dist
from omegaconf import OmegaConf

from trainer import RegTrainer


def load_yaml(path):
    """A config may name a `base` config that it overrides."""
    config = OmegaConf.load(path)
    if "base" in config:
        base = config.pop("base")
        config = OmegaConf.merge(load_yaml(base), config)
    return config


def load_config(args, overrides):
    config = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), load_yaml(args.config),
                             OmegaConf.from_dotlist(overrides))
    if args.resume:
        config.resume_from = args.resume
    config.logdir = args.logdir
    config.log_first_losses = args.log_first_losses
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None, help="checkpoint_model_XXXXXX/model.pt to resume from")
    parser.add_argument("--logdir", required=True, help="checkpoints and logs are written here")
    parser.add_argument("--log_first_losses", type=int, default=0,
                        help="print the losses of the first N optimizer steps at full precision "
                             "(reproducibility checks)")
    args, overrides = parser.parse_known_args()
    config = load_config(args, overrides)
    os.makedirs(config.logdir, exist_ok=True)
    if int(os.environ.get("RANK", 0)) == 0:
        OmegaConf.save(config, os.path.join(config.logdir, "config.yaml"))
    trainer = RegTrainer(config)
    trainer.train()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
