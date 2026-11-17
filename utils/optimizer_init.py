"""Allocate AdamW moments without changing parameters or bias-correction time."""
import torch


@torch.no_grad()
def initialize_adamw_state(optimizer):
    if not isinstance(optimizer, torch.optim.AdamW):
        raise TypeError('This initializer supports AdamW only')
    for group in optimizer.param_groups:
        for p in group['params']:
            if optimizer.state[p]:
                raise RuntimeError('Refusing to reset existing AdamW state')
            state = optimizer.state[p]
            step_device = p.device if group.get('capturable') or group.get('fused') else 'cpu'
            state['step'] = torch.zeros((), dtype=torch.float32, device=step_device)
            state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
            state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
            if group.get('amsgrad', False):
                state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
    if any(float(s['step']) != 0 for s in optimizer.state.values()):
        raise RuntimeError('AdamW must start at step zero')
