import contextlib
from datetime import timedelta
from functools import partial
import os
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy, StateDictType
from torch.distributed.fsdp.api import CPUOffload
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy

GATE_KEY = 'sink_gate_logit'


def clean_name(name):
    return name.replace("_fsdp_wrapped_module.", "").replace("_checkpoint_wrapped_module.", "").replace("_orig_mod.", "")


def fsdp_state_dict(model):
    fsdp_fullstate_save_policy = FullStateDictConfig(
        offload_to_cpu=True, rank0_only=True
    )
    with FSDP.state_dict_type(
        model, StateDictType.FULL_STATE_DICT, fsdp_fullstate_save_policy
    ):
        checkpoint = model.state_dict()

    return checkpoint


def fsdp_wrap(module, sharding_strategy="full", mixed_precision=False, wrap_strategy="size", min_num_params=int(5e7), transformer_module=None, cpu_offload=False):
    if mixed_precision:
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
            cast_forward_inputs=False
        )
    else:
        mixed_precision_policy = None

    if wrap_strategy == "transformer":
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_module
        )
    elif wrap_strategy == "size":
        auto_wrap_policy = partial(
            size_based_auto_wrap_policy,
            min_num_params=min_num_params
        )
    else:
        raise ValueError(f"Invalid wrap strategy: {wrap_strategy}")

    os.environ["NCCL_CROSS_NIC"] = "1"

    sharding_strategy = {
        "full": ShardingStrategy.FULL_SHARD,
        "hybrid_full": ShardingStrategy.HYBRID_SHARD,
        "hybrid_zero2": ShardingStrategy._HYBRID_SHARD_ZERO2,
        "no_shard": ShardingStrategy.NO_SHARD,
    }[sharding_strategy]

    module = FSDP(
        module,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=sharding_strategy,
        mixed_precision=mixed_precision_policy,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        use_orig_params=True,
        cpu_offload=CPUOffload(offload_params=cpu_offload),
        sync_module_states=False  # every rank loads the weights itself
    )
    return module


def launch_distributed_job(backend: str = "nccl", timeout_minutes: int = 60):
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    host = os.environ["MASTER_ADDR"]
    port = int(os.environ["MASTER_PORT"])

    if ":" in host:  # IPv6
        init_method = f"tcp://[{host}]:{port}"
    else:  # IPv4
        init_method = f"tcp://{host}:{port}"
    # Checkpoint saves are long and rank-asymmetric: raise the NCCL timeout of the default group and of the
    # sub-groups HYBRID_SHARD creates (they inherit c10d's module default, 10 min).
    c10d.default_pg_nccl_timeout = timedelta(minutes=timeout_minutes)
    dist.init_process_group(rank=rank, world_size=world_size, backend=backend,
                            init_method=init_method, timeout=timedelta(minutes=timeout_minutes))
    torch.cuda.set_device(local_rank)


class ShardEMA:
    """EMA with a shard-local fp32 CPU shadow (FSDP use_orig_params: named_parameters() are the local fp32 shards).
    Full weights exist only transiently (full_rank0, sync_live, load_full): the shadow is written into the local
    shards, FSDP's unshard is read, and the online values are restored exactly."""

    def __init__(self, fsdp_module, decay=0.999):
        self.decay = decay
        self.shadow = {}
        with torch.no_grad():
            for n, p in fsdp_module.module.named_parameters():
                if p.dtype != torch.float32:
                    raise RuntimeError(f'ShardEMA needs fp32 local shards ({n}: {p.dtype})')
                self.shadow[n] = p.detach().clone().float().cpu()

    @torch.no_grad()
    def update(self, fsdp_module):
        d = self.decay
        for n, p in fsdp_module.module.named_parameters():
            self.shadow[n].mul_(d).add_(p.detach().float().cpu(), alpha=1. - d)

    @contextlib.contextmanager
    def _swapped(self, fsdp_module, keep):
        named = list(fsdp_module.module.named_parameters())
        saved = [p.detach().clone() for _, p in named]
        try:
            with torch.no_grad():
                for n, p in named:
                    if keep(n):
                        p.data.copy_(self.shadow[n].to(device=p.device, dtype=p.dtype))
            yield
        finally:
            with torch.no_grad():
                for (_, p), v in zip(named, saved):
                    p.data.copy_(v)
            del saved

    @torch.no_grad()
    def full_rank0(self, fsdp_module, keep_key=None):
        """Full fp32 CPU dict of the EMA on rank 0 (None on other ranks). Collective."""
        out = {} if dist.get_rank() == 0 else None
        with self._swapped(fsdp_module, lambda n: True):
            with FSDP.summon_full_params(fsdp_module, writeback=False):
                for n, p in fsdp_module.module.named_parameters():
                    if out is not None and (keep_key is None or keep_key(n)):
                        out[n] = p.detach().cpu()
        return out

    @torch.no_grad()
    def sync_live(self, fsdp_module, live):
        """live (plain module) = EMA weights + the current online sink gate. Collective."""
        target = dict(live.named_parameters())
        with self._swapped(fsdp_module, lambda n: GATE_KEY not in n):
            with FSDP.summon_full_params(fsdp_module, writeback=False):
                named = list(fsdp_module.module.named_parameters())
                if len(target) != len(named):
                    raise RuntimeError(f'live model has {len(target)} parameters, online {len(named)}')
                for n, p in named:
                    q = target[clean_name(n)]
                    q.data.copy_(p.detach().to(device=q.device, dtype=q.dtype))

    @torch.no_grad()
    def load_full(self, fsdp_module, full):
        """Shadow = the shards of a full fp32 EMA dict (checkpoint), sliced by FSDP; online weights restored. Collective."""
        named = list(fsdp_module.module.named_parameters())
        if set(full) != {n for n, _ in named}:
            raise RuntimeError('EMA checkpoint keys differ from the online model')
        saved = [p.detach().clone() for _, p in named]
        try:
            with FSDP.summon_full_params(fsdp_module, writeback=True):
                for n, p in fsdp_module.module.named_parameters():
                    p.data.copy_(full[n].to(device=p.device))
            for n, p in named:
                self.shadow[n] = p.detach().float().cpu()
        finally:
            for (_, p), v in zip(named, saved):
                p.data.copy_(v)
            del saved
