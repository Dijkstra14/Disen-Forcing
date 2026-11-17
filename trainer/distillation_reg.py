import contextlib
import gc
import json
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType
from torch.distributed.fsdp import FullStateDictConfig, FullOptimStateDictConfig

import utils.wan_wrapper as wan_wrapper
from model import RegDMD
from utils.dataset import TextDataset, cycle
from utils.distributed import GATE_KEY, ShardEMA, fsdp_state_dict, fsdp_wrap, launch_distributed_job
from utils.misc import set_seed
from utils.optimizer_init import initialize_adamw_state


def _layer(name):
    return int(name.split('blocks.')[1].split('.')[0])


class RegTrainer:
    """Disen-Forcing distillation: DMD with a critic, learnable context routing (sink gate logits), and the
    regularizer against an EMA reference. Two EMAs of the generator: the reference EMA (from step 0,
    drives the regularizer) and the release EMA (generator_ema, from ema_start_step). Steps: `self.step` counts
    micro-steps; one effective step is gradient_accumulation_steps micro-steps (step-based config values are given
    in effective steps)."""

    def __init__(self, config):
        self.config = config
        self.step = 0
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        launch_distributed_job()
        self.world_size = dist.get_world_size()
        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = dist.get_rank() == 0
        self.output_path = config.logdir

        self.accum = int(config.gradient_accumulation_steps)
        # effective steps -> micro-steps
        config.log_iters = config.log_iters * self.accum
        config.ema_start_step = config.ema_start_step * self.accum
        config.max_train_steps = config.max_train_steps * self.accum
        set_seed(config.seed + dist.get_rank())
        wan_wrapper.WAN_MODEL_DIR = config.wan_model_dir

        self.model = RegDMD(config, device=self.device)
        self.model.generator = fsdp_wrap(self.model.generator, sharding_strategy=config.sharding_strategy,
                                         mixed_precision=config.mixed_precision,
                                         wrap_strategy=config.generator_fsdp_wrap_strategy)
        self.model.real_score = fsdp_wrap(self.model.real_score, sharding_strategy=config.sharding_strategy,
                                          mixed_precision=config.mixed_precision,
                                          wrap_strategy=config.real_score_fsdp_wrap_strategy,
                                          cpu_offload=getattr(config, "real_score_cpu_offload", False))
        self.model.fake_score = fsdp_wrap(self.model.fake_score, sharding_strategy=config.sharding_strategy,
                                          mixed_precision=config.mixed_precision,
                                          wrap_strategy=config.fake_score_fsdp_wrap_strategy)
        self.model.text_encoder = fsdp_wrap(self.model.text_encoder, sharding_strategy=config.sharding_strategy,
                                            mixed_precision=config.mixed_precision,
                                            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
                                            cpu_offload=getattr(config, "text_encoder_cpu_offload", False))

        # Generator optimizer: the sink gate logits get their own group (lr, betas, eps, weight decay).
        named = [(n, p) for n, p in self.model.generator.named_parameters() if p.requires_grad]
        self.gate_params = [p for n, p in named if GATE_KEY in n]
        if not self.gate_params:
            raise RuntimeError('model_kwargs.sink_gate must be on')
        self.generator_optimizer = torch.optim.AdamW(
            [{'params': [p for n, p in named if GATE_KEY not in n]}, {'params': self.gate_params}],
            lr=config.lr, betas=(config.beta1, config.beta2), weight_decay=config.weight_decay)
        self._set_group_hparams()
        self.critic_optimizer = torch.optim.AdamW(
            [p for p in self.model.fake_score.parameters() if p.requires_grad],
            lr=config.lr_critic, betas=(config.beta1_critic, config.beta2_critic), weight_decay=config.weight_decay)

        dataset = TextDataset(config.data_path)
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True, drop_last=True)
        self.dataloader = cycle(torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, sampler=sampler,
                                                            num_workers=8))
        self._batches_drawn = 0   # per rank; restored on resume (data position)

        resume = self._load_weights()

        # Release EMA: created after the cycle that reaches ema_start_step; a checkpoint saved at exactly that step
        # has none yet, and a fresh copy of the restored weights is exact.
        self.generator_ema = None
        if resume and self.step >= config.ema_start_step:
            self.generator_ema = ShardEMA(self.model.generator, decay=config.ema_weight)
            if resume.get('generator_ema') is not None:
                self.generator_ema.load_full(self.model.generator, resume['generator_ema'])
        # Reference EMA: the online weights at step 0, restored afterwards.
        self.ref_ema = ShardEMA(self.model.generator, decay=config.ema_weight)
        self._ref_ema_updates = 0
        if self.step > 0:
            if resume.get('generator_ref_ema_own') is None:
                raise RuntimeError('checkpoint has no generator_ref_ema_own')
            self.ref_ema.load_full(self.model.generator, resume['generator_ref_ema_own'])
            self._ref_ema_updates = int(resume['generator_ref_ema_own_meta']['updates'])

        # Live reference: plain bf16 copy = reference EMA weights + online gate, re-synced after every update.
        self.model.generator_ref = self.model.generator_ref.to(device=self.device, dtype=torch.bfloat16).eval()
        self.ref_ema.sync_live(self.model.generator, self.model.generator_ref)
        student = getattr(self.model.generator.model, 'module', self.model.generator.model)
        student.reference['generator'] = self.model.generator_ref

        initialize_adamw_state(self.generator_optimizer)
        self._g_updates = self._c_updates = self._ema_updates = 0
        if resume:
            self._restore_resume_state(resume)
        else:
            # Gate moments start at the gate's first update.
            for p in self.gate_params:
                self.generator_optimizer.state.pop(p, None)
        self._set_group_hparams()   # a restored optimizer state carries the saved group hyper-parameters
        del resume

        self.model._initialize_inference_pipeline()
        self.max_grad_norm_generator = config.max_grad_norm_generator
        self.max_grad_norm_critic = config.max_grad_norm_critic
        self.log_first_losses = int(getattr(config, 'log_first_losses', 0) or 0)
        torch.cuda.empty_cache()

    def _set_group_hparams(self):
        g = self.config.sink_gate
        rest, gate = self.generator_optimizer.param_groups
        rest['lr'] = float(self.config.lr)
        gate.update(lr=float(g.lr), weight_decay=float(g.weight_decay), eps=float(g.eps),
                    betas=tuple(float(b) for b in g.betas))

    def _load_weights(self):
        """Fresh run: generator from generator_ckpt and sink gates from the anchor heads. Resume: generator, critic and
        step from a checkpoint; returns its dict (EMAs, meta) or None."""
        cfg = self.config
        if not getattr(cfg, 'resume_from', None):
            sd = torch.load(cfg.generator_ckpt, map_location="cpu")
            for key in ("generator", "model", "generator_ema"):
                if key in sd:
                    sd = sd[key]
                    break
            own = self.model.generator.state_dict()
            missing = [k for k in own if k not in sd]
            assert all(GATE_KEY in k for k in missing), missing[:5]   # init checkpoints have no gate logits
            self.model.generator.load_state_dict(dict(sd, **{k: own[k] for k in missing}), strict=True)
            self._init_sink_gates()
            return None
        ck = torch.load(cfg.resume_from, map_location='cpu', mmap=True)
        meta = ck['resume']
        if int(meta['gradient_accumulation_steps']) != self.accum:
            raise RuntimeError('resume: gradient_accumulation_steps differs from the checkpoint')
        self.model.generator.load_state_dict(ck['generator'], strict=True)
        self.model.fake_score.load_state_dict(ck['critic'], strict=True)
        self.step = int(meta['raw_step'])
        if self.step % self.accum:
            raise RuntimeError('resume step is not at an accumulation boundary')
        if self.step > cfg.ema_start_step and 'generator_ema' not in ck:
            raise RuntimeError('checkpoint past ema_start_step has no generator_ema')
        if self.is_main_process:
            print(f"[Resume] {cfg.resume_from}: raw_step={self.step} meta={json.dumps(meta)}", flush=True)
        return dict(ck, meta=meta, path=cfg.resume_from)

    @torch.no_grad()
    def _init_sink_gates(self):
        """Anchor heads (sink_gate.anchor_json) start with b = 0 (gate fully open), every other head at b0."""
        with open(self.config.sink_gate.anchor_json) as f:
            anchors = json.load(f)
        b0 = float(self.config.sink_gate.b0)
        with FSDP.summon_full_params(self.model.generator, writeback=True):
            for n, p in self.model.generator.named_parameters():
                if GATE_KEY in n:
                    heads = set(anchors.get(f'layer_{_layer(n)}', []))
                    v = torch.tensor([0.0 if h in heads else b0 for h in range(p.numel())], dtype=p.dtype)
                    p.data.copy_(v.to(p.device))

    def _restore_resume_state(self, resume):
        meta = resume['meta']
        self._g_updates = int(meta['generator_updates'])
        self._c_updates = int(meta['critic_updates'])
        self._ema_updates = int(meta['ema_updates'])
        ckdir = os.path.dirname(resume['path'])
        rs = torch.load(os.path.join(ckdir, 'resume_state.pt'), map_location='cpu', weights_only=False)
        if int(rs['raw_step']) != self.step:
            raise RuntimeError('resume_state.pt step differs from model.pt')
        # Full optimizer state dicts are world-size agnostic; FSDP re-shards them.
        for module, opt, key in ((self.model.generator, self.generator_optimizer, 'generator'),
                                 (self.model.fake_score, self.critic_optimizer, 'critic')):
            osd = torch.load(os.path.join(ckdir, rs['optimizer_files'][key]), map_location='cpu', mmap=True,
                             weights_only=False)
            with FSDP.state_dict_type(module, StateDictType.FULL_STATE_DICT,
                                      FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
                                      FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False)):
                opt.load_state_dict(FSDP.optim_state_dict_to_load(module, opt, osd))
            del osd
        # Data position and RNG: exact with the same world size, else a deterministic reseed.
        for _ in range(int(rs['batches_drawn'])):
            next(self.dataloader)
        self._batches_drawn = int(rs['batches_drawn'])
        self.resume_exact = int(rs['world_size']) == dist.get_world_size()
        if self.resume_exact:
            g = rs['rng'][dist.get_rank()]
            torch.set_rng_state(g['cpu'])
            torch.cuda.set_rng_state(g['cuda'])
            random.setstate(g['py'])
            np.random.set_state(g['np'])
        else:
            set_seed(self.config.seed + dist.get_rank() + 1000003 * self.step)
        if self.is_main_process:
            print(f'[Resume] restored step={self.step} g={self._g_updates} c={self._c_updates} ema={self._ema_updates} '
                  f'ref_ema={self._ref_ema_updates} batches={self._batches_drawn} exact={self.resume_exact}', flush=True)

    def save(self):
        """checkpoint_model_<step>/: model.pt (generator, critic, EMAs, counters) and the full optimizer states with the
        per-rank RNG and data position for an exact resume."""
        ckdir = os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}")
        state = {"generator": fsdp_state_dict(self.model.generator), "critic": fsdp_state_dict(self.model.fake_score)}
        if self.config.ema_start_step < self.step:
            state["generator_ema"] = self.generator_ema.full_rank0(self.model.generator)
        state['generator_ref_ema_own'] = self.ref_ema.full_rank0(self.model.generator)
        state['generator_ref_ema_own_meta'] = dict(
            decay=self.ref_ema.decay, updates=self._ref_ema_updates, raw_step=self.step,
            start_raw=0)
        state['resume'] = dict(
            raw_step=self.step, effective_step=self.step // self.accum, gradient_accumulation_steps=self.accum,
            generator_updates=self._g_updates, critic_updates=self._c_updates, ema_updates=self._ema_updates,
            world_size=dist.get_world_size(), ema_saved='generator_ema' in state, ref_ema_updates=self._ref_ema_updates)
        if self.is_main_process:
            os.makedirs(ckdir, exist_ok=True)
            torch.save(state, os.path.join(ckdir, "model.pt"))
            print("Model saved to", os.path.join(ckdir, "model.pt"), flush=True)
        del state
        files = {}
        for module, opt, key in ((self.model.generator, self.generator_optimizer, 'generator'),
                                 (self.model.fake_score, self.critic_optimizer, 'critic')):
            with FSDP.state_dict_type(module, StateDictType.FULL_STATE_DICT,
                                      FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
                                      FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)):
                osd = FSDP.optim_state_dict(module, opt)
            files[key] = f'resume_optim_{key}.pt'
            if self.is_main_process:
                torch.save(osd, os.path.join(ckdir, files[key]))
            del osd
            gc.collect()
        rng = [None] * dist.get_world_size()
        dist.all_gather_object(rng, dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(),
                                         py=random.getstate(), np=np.random.get_state(), batches=self._batches_drawn))
        if len({r['batches'] for r in rng}) != 1:
            raise RuntimeError('ranks drew different batch counts')
        if self.is_main_process:
            torch.save(dict(raw_step=self.step, world_size=dist.get_world_size(), batches_drawn=self._batches_drawn,
                            rng=rng, optimizer_files=files), os.path.join(ckdir, 'resume_state.pt'))
        dist.barrier()

    @torch.no_grad()
    def log_sink_gates(self):
        """Append the online (and release-EMA) gates g_h = exp(b_h) per layer to sink_gates.jsonl. Collective."""
        rows = {}
        with FSDP.summon_full_params(self.model.generator, writeback=False, rank0_only=True, offload_to_cpu=True):
            if self.is_main_process:
                for n, p in self.model.generator.named_parameters():
                    if GATE_KEY in n:
                        rows[_layer(n)] = p.detach().float().cpu().exp().tolist()
        ema = None
        if self.generator_ema is not None:
            ema = self.generator_ema.full_rank0(self.model.generator, keep_key=lambda n: GATE_KEY in n)
        if self.is_main_process:
            row = dict(effective_step=self.step // self.accum, gates=[rows[i] for i in sorted(rows)])
            if ema is not None:
                ema = {_layer(k): v.float().exp().tolist() for k, v in ema.items()}
                row['gates_ema'] = [ema[i] for i in sorted(ema)]
            with open(os.path.join(self.output_path, 'sink_gates.jsonl'), 'a') as f:
                f.write(json.dumps(row) + '\n')
        return rows

    def fwdbwd_one_step(self, batch, train_generator):
        self.model.eval()  # no dropout
        if self.step % 20 == 0:
            torch.cuda.empty_cache()
        text_prompts = batch["prompts"]
        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(text_prompts=text_prompts)
            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(text_prompts=[self.config.negative_prompt] * batch_size)
                self.unconditional_dict = {k: v.detach() for k, v in unconditional_dict.items()}
            unconditional_dict = self.unconditional_dict

        if train_generator:
            torch.cuda.empty_cache()
            loss, log = self.model.generator_loss(image_or_video_shape, conditional_dict, unconditional_dict)
            loss = loss / self.accum
            loss.backward()
            log["generator_loss"] = loss.detach()
            torch.cuda.empty_cache()
            return log
        loss, log = self.model.critic_loss(image_or_video_shape, conditional_dict, unconditional_dict)
        loss = loss / self.accum
        loss.backward()
        log["critic_loss"] = loss.detach()
        return log

    def _cache_grad_scale(self, effective_step):
        """KV caching gradient factor: 0 at start_eff, linear to alpha over warmup_eff (1 when unset)."""
        c = getattr(self.config, 'cache_grad_scale', None)
        if c is None:
            return 1.0
        wu = float(c['warmup_eff'])
        return float(c['alpha']) * (min(1.0, max(0.0, (effective_step - float(c['start_eff'])) / wu)) if wu > 0 else 1.0)

    def _log_first_losses(self, kind, micro, log, effective_step):
        """Full-precision losses of the first cycles (rank 0), for reproducibility checks."""
        if self.log_first_losses > 0 and self.is_main_process:
            vals = {k: repr(float(torch.as_tensor(log[k]).double().mean())) for k in
                    ('dmd_loss', 'reg_loss', 'generator_loss', 'critic_loss') if k in log}
            print(f'[first-losses] eff={effective_step} {kind} micro={micro} {json.dumps(vals)} '
                  f'reg_t={log.get("reg_t")}', flush=True)

    def train(self):
        cfg, accum, pipe = self.config, self.accum, self.model.inference_pipeline
        start_step = self.step
        if self.step % accum or cfg.max_train_steps % accum:
            raise ValueError('training must start and end at an accumulation boundary')
        self.log_sink_gates()
        self.generator_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        while self.step < cfg.max_train_steps:
            effective_step = self.step // accum
            train_generator = effective_step % cfg.dfake_gen_update_ratio == 0
            if getattr(cfg, 'kv_cache_gradient', False):
                # as in Self Gradient Forcing: no generator step at 0
                train_generator = train_generator and effective_step > 0

            pipe.cache_grad_scale = self._cache_grad_scale(effective_step)
            cycle_start = self.step

            # One exit step per optimizer step, shared by its micro-steps (separate draws for generator and critic).
            n_blocks = cfg.num_training_frames // pipe.num_frame_per_block
            gen_flags = pipe.generate_and_sync_list(n_blocks, len(pipe.denoising_step_list), device=self.device)
            crit_flags = pipe.generate_and_sync_list(n_blocks, len(pipe.denoising_step_list), device=self.device)
            generator_batches, critic_batches = [], []
            for _ in range(accum):   # prompt order: generator, critic, generator, ...
                if train_generator:
                    generator_batches.append(next(self.dataloader))
                critic_batches.append(next(self.dataloader))
            self._batches_drawn += accum * (2 if train_generator else 1)

            gen_log = None
            try:
                if train_generator:
                    pipe._shared_exit_flags = gen_flags
                    for micro, batch in enumerate(generator_batches):
                        self.step = cycle_start + micro
                        with self.model.generator.no_sync() if micro < accum - 1 else contextlib.nullcontext():
                            gen_log = self.fwdbwd_one_step(batch, True)
                        self._log_first_losses('generator', micro, gen_log, effective_step)
                    self.last_generator_grad_norm = self.model.generator.clip_grad_norm_(self.max_grad_norm_generator)
                    self.generator_optimizer.step()
                    self._g_updates += 1
                    self.generator_optimizer.zero_grad(set_to_none=True)
                    if self.generator_ema is not None:
                        self.generator_ema.update(self.model.generator)
                        self._ema_updates += 1
                    self.ref_ema.update(self.model.generator)
                    self._ref_ema_updates += 1
                    self.ref_ema.sync_live(self.model.generator, self.model.generator_ref)

                torch.cuda.empty_cache()
                pipe._shared_exit_flags = crit_flags
                for micro, batch in enumerate(critic_batches):
                    self.step = cycle_start + micro
                    with self.model.fake_score.no_sync() if micro < accum - 1 else contextlib.nullcontext():
                        crit_log = self.fwdbwd_one_step(batch, False)
                    self._log_first_losses('critic', micro, crit_log, effective_step)
                self.last_critic_grad_norm = self.model.fake_score.clip_grad_norm_(self.max_grad_norm_critic)
                self.critic_optimizer.step()
                self._c_updates += 1
                self.critic_optimizer.zero_grad(set_to_none=True)
            finally:
                pipe._shared_exit_flags = None
            self.log_first_losses -= 1

            self.step = cycle_start + accum
            norms = [('critic', self.last_critic_grad_norm)] + ([('generator', self.last_generator_grad_norm)]
                                                                  if train_generator else [])
            for name, norm in norms:
                if not bool(torch.isfinite(norm)):
                    raise FloatingPointError(f'non-finite {name} gradient norm')
            if self.is_main_process:
                self._write_log(train_generator, gen_log, crit_log, pipe.cache_grad_scale)
            if cfg.sink_gate.log_every and (self.step // accum) % cfg.sink_gate.log_every == 0:
                self.log_sink_gates()
            if self.step >= cfg.ema_start_step and self.generator_ema is None:
                self.generator_ema = ShardEMA(self.model.generator, decay=cfg.ema_weight)
            if not cfg.no_save and self.step > start_step and self.step % cfg.log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()
            if self.step % cfg.gc_interval == 0:
                gc.collect()
                torch.cuda.empty_cache()

    def _write_log(self, train_generator, gen_log, crit_log, cache_grad_scale):
        row = dict(effective_step=self.step // self.accum, generator_step=bool(train_generator),
                   generator_updates=self._g_updates, critic_updates=self._c_updates, ema_updates=self._ema_updates,
                   ref_ema_updates=self._ref_ema_updates, critic_loss=float(crit_log['critic_loss']),
                   critic_grad_norm=float(self.last_critic_grad_norm), reg_weight=float(self.model.reg_weight),
                   wall_time=time.time())
        if train_generator:
            row.update(dmd_loss=float(gen_log['dmd_loss']), generator_loss=float(gen_log['generator_loss']),
                       generator_grad_norm=float(self.last_generator_grad_norm), cache_grad_scale=cache_grad_scale)
            if 'reg_loss' in gen_log:
                row.update(reg_loss=float(gen_log['reg_loss']), reg_t=gen_log['reg_t'],
                           reg_chunk_kls=gen_log['reg_chunk_kls'])
        with open(os.path.join(self.output_path, 'training_log.jsonl'), 'a') as f:
            f.write(json.dumps(row) + '\n')
        print('[train] ' + json.dumps({k: v for k, v in row.items() if k not in ('reg_chunk_kls', 'wall_time')}),
              flush=True)
