"""Select anchor heads from sink attention scores (output of tools/measure_sink_attention.py).

A head is an anchor head when its score, averaged over prompts and then over timesteps, is above
the p-th percentile of all heads. Writes {"layer_L": [head, ...]}, the format of configs/anchors/*.json.

  python tools/find_anchor_heads.py --scores anchor_scores/my_model/sink_scores.json --percentile 80 \
      --out configs/anchors/my_model_p80.json [--compare configs/anchors/anchor_cf_ode_p80.json]
"""
import argparse
import json

import numpy as np


def head_scores(scores):
    """{timestep: {"layer_L/head_H": score or [score per prompt]}} -> {head key: mean over prompts, then timesteps}."""
    ts = list(scores)
    return {k: float(np.mean([np.mean(scores[t][k]) for t in ts])) for k in scores[ts[0]]}


def select(avg, percentile):
    threshold = float(np.percentile(list(avg.values()), percentile))
    return {k for k, v in avg.items() if v > threshold}, threshold


def parse_key(key):
    layer, head = key.split('/')
    return int(layer.split('_')[1]), int(head.split('_')[1])


def to_layer_dict(pairs):
    out = {}
    for layer, head in sorted(pairs):
        out.setdefault(f'layer_{layer}', []).append(head)
    return out


def load_anchor_set(path):
    with open(path) as f:
        return {(int(k.split('_')[1]), h) for k, heads in json.load(f).items() for h in heads}


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--scores', required=True)
    p.add_argument('--percentile', type=float, default=80.0)
    p.add_argument('--out', required=True)
    p.add_argument('--compare', default=None, help='anchor json to compare the selection with')
    args = p.parse_args()

    with open(args.scores) as f:
        avg = head_scores(json.load(f))
    keys, threshold = select(avg, args.percentile)
    pairs = {parse_key(k) for k in keys}
    with open(args.out, 'w') as f:
        json.dump(to_layer_dict(pairs), f, indent=1)
    per_layer = {k: len(v) for k, v in to_layer_dict(pairs).items()}
    print(f'p{args.percentile:g} threshold {threshold:.4f}: {len(pairs)}/{len(avg)} anchor heads in {len(per_layer)} layers')
    print('per layer:', per_layer)
    print('wrote', args.out)

    if args.compare:
        ref = load_anchor_set(args.compare)
        both = pairs & ref
        print(f'vs {args.compare}: {len(pairs)} vs {len(ref)} heads, {len(both)} shared, '
              f'Jaccard {len(both) / max(1, len(pairs | ref)):.3f}, exact match {pairs == ref}')
        print('  only here:', to_layer_dict(pairs - ref))
        print('  only in reference:', to_layer_dict(ref - pairs))


if __name__ == '__main__':
    main()
