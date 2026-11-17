"""Export the release generator weights from a training checkpoint.

Writes the EMA weights, with the per-head routing logits (keys containing 'sink_gate_logit') from the
non-EMA model; generator only, FSDP prefixes removed. The output loads with
`inference.py --checkpoint <out>` and gives the same videos as
`--weights ema_online_gate` on the training checkpoint (bf16 included: inference
casts the generator to bf16).

    python scripts/export_release.py --checkpoint model.pt --output disen_forcing.safetensors [--bf16]
"""
import argparse

import torch

GATE_KEY = 'sink_gate_logit'
FSDP_PREFIX = '._fsdp_wrapped_module'


def strip_fsdp(sd):
    return {k.replace(FSDP_PREFIX, ''): v for k, v in sd.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='training checkpoint (model.pt)')
    parser.add_argument('--output', required=True, help='.safetensors or .pt')
    parser.add_argument('--bf16', action='store_true', help='store floating-point tensors in bf16')
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    online = strip_fsdp(ckpt['generator'])
    ema = strip_fsdp(ckpt['generator_ema'])
    assert set(online) == set(ema), sorted(set(online) ^ set(ema))[:10]
    gate_keys = [k for k in ema if GATE_KEY in k]
    assert gate_keys, f'no {GATE_KEY} tensors in the checkpoint'

    out = {}
    for k, v in ema.items():
        v = online[k] if k in gate_keys else v
        if args.bf16 and v.is_floating_point():
            v = v.to(torch.bfloat16)
        out[k] = v.detach().clone().contiguous()

    if args.output.endswith('.safetensors'):
        from safetensors.torch import save_file
        save_file(out, args.output)
    else:
        torch.save(out, args.output)

    dtypes = sorted({str(v.dtype) for v in out.values()})
    n_params = sum(v.numel() for v in out.values())
    print(f'keys: online={len(online)} ema={len(ema)} exported={len(out)} '
          f'({len(gate_keys)} {GATE_KEY} from online)')
    print(f'parameters: {n_params:,}  dtypes: {dtypes}  -> {args.output}')
    assert len(out) == len(ema)


if __name__ == '__main__':
    main()
