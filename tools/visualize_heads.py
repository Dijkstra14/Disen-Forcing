"""Plot per-head sink attention and the paper's attention-map figures.

  layer-head  layer x head heatmap of the mean sink score, anchor heads outlined
  combined    sink score of every head per prompt and per timestep (paper: "anchor-head positions are
              input invariant")
  grid        chunk-by-chunk attention map of every head, anchor heads with red borders (paper
              appendix: per-(layer, head) attention map grid); needs measure_sink_attention.py --maps

  python tools/visualize_heads.py layer-head --scores S/sink_scores.json --anchors A.json --out layer_head.png
  python tools/visualize_heads.py combined --scores S/sink_scores.json --out combined.png
  python tools/visualize_heads.py grid --maps S/attn_maps.npz --anchors A.json --out grid.png
Without --anchors, anchor heads are selected from --scores at --percentile.
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from find_anchor_heads import head_scores, load_anchor_set, parse_key, select  # noqa: E402

ANCHOR_RED, WORKING_BLUE = '#C0392B', '#4472C4'
REDS = LinearSegmentedColormap.from_list(
    'reds', [(0.0, '#FFF7F3'), (0.25, '#FCBBA1'), (0.55, '#FB6A4A'), (0.8, '#C0392B'), (1.0, '#7B1E1E')])
COOLWARM = plt.cm.coolwarm.copy()
COOLWARM.set_bad('gray', alpha=0.6)


def load_scores(path):
    """-> scores dict and array A [timesteps, layers*heads (layer-major), prompts], timestep keys, L, H."""
    with open(path) as f:
        scores = json.load(f)
    ts = list(scores)
    keys = sorted(scores[ts[0]], key=parse_key)
    L, H = (max(v) + 1 for v in zip(*map(parse_key, keys)))
    A = np.array([[np.atleast_1d(scores[t][k]) for k in keys] for t in ts], dtype=np.float64)
    return scores, A, ts, L, H


def anchor_set(args, scores=None):
    if args.anchors:
        return load_anchor_set(args.anchors)
    if scores is None:
        return set()
    return {parse_key(k) for k in select(head_scores(scores), args.percentile)[0]}


def plot_layer_head(args):
    scores, A, _, L, H = load_scores(args.scores)
    M = A.mean(2).mean(0).reshape(L, H)
    anchors = anchor_set(args, scores)
    fig, ax = plt.subplots(figsize=(0.3 * H + 1.4, 0.15 * L + 0.9))
    im = ax.imshow(M, origin='lower', aspect='auto', cmap=COOLWARM, vmin=0.0, vmax=args.vmax, interpolation='nearest')
    for layer, head in anchors:
        ax.add_patch(Rectangle((head - 0.5, layer - 0.5), 1, 1, fill=False, ec='black', lw=1.2))
    ax.set_xticks(range(H))
    ax.set_yticks(range(0, L, 2))
    ax.set_xlabel('Head')
    ax.set_ylabel('Layer')
    ax.set_title(f'Attention to the first frame ({len(anchors)} anchor heads outlined)', fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.06, pad=0.03).set_label('Mean attn score')
    return fig


def plot_combined(args):
    """Layout and colours of the paper's combined heatmap (by prompt | by timestep)."""
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
                         'mathtext.fontset': 'stix', 'font.size': 10, 'axes.titlesize': 10,
                         'axes.titleweight': 'bold', 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
                         'axes.linewidth': 0.5, 'pdf.fonttype': 42})
    _, A, ts, L, H = load_scores(args.scores)
    X_p, X_t, n_hl = A.mean(0).T, A.mean(2), L * H
    vmax = args.vmax if args.vmax is not None else float(np.nanpercentile(X_p, 99))
    fig, (ax_p, ax_t) = plt.subplots(1, 2, figsize=(7.0, 2.15))
    fig.subplots_adjust(left=0.07, right=0.93, top=0.88, bottom=0.22, wspace=0.26)
    ticks = list(range(0, n_hl + 1, 5 * H))
    for ax, X, title in ((ax_p, X_p, '(a) Attention Score by Prompt'), (ax_t, X_t, '(b) Attention Score by Timestep')):
        im = ax.imshow(X, aspect='auto', origin='upper', cmap=REDS, vmin=0.0, vmax=vmax, interpolation='nearest',
                       extent=(-0.5, n_hl - 0.5, len(X) - 0.5, -0.5), rasterized=True)
        ax.set_title(title, pad=4)
        ax.set_xticks(ticks)
        ax.set_xlim(-0.5, n_hl - 0.5)
        ax.set_xlabel('Attention Head Index')
    p_ticks = sorted(set(range(0, len(X_p), 10)) | {len(X_p) - 1})
    ax_p.set_yticks(p_ticks)
    ax_p.set_ylabel('Prompt index')
    ax_t.set_yticks(range(len(ts)))
    ax_t.set_yticklabels([f'{float(t):.4g}' for t in ts])
    ax_t.set_ylabel('Timestep')
    pos = ax_t.get_position()
    cbar = fig.colorbar(im, cax=fig.add_axes([pos.x1 + 0.006, pos.y0, 0.010, pos.height]))
    cbar.ax.tick_params(labelsize=7.5, width=0.5, length=2.0)
    cbar.outline.set_linewidth(0.4)
    return fig


def plot_grid(args):
    """Layout and colours of the paper's per-(layer, head) attention map grid."""
    npz = np.load(args.maps)
    heads = [parse_key(k.replace('L', 'layer_').replace('H', '/head_')) for k in npz.files if k.startswith('L')]
    L, H = (max(v) + 1 for v in zip(*heads))
    anchors = anchor_set(args, json.load(open(args.scores)) if args.scores else None)
    fig = plt.figure(figsize=(0.5 * H + 1.2, 0.5 * L + 0.6))
    gs = GridSpec(L, H, figure=fig, hspace=0.1, wspace=0.1, top=0.97, bottom=0.03, left=0.06, right=0.92)
    for layer in range(L):
        for head in range(H):
            ax = fig.add_subplot(gs[layer, head])
            ax.imshow(np.ma.masked_invalid(npz[f'L{layer}H{head}']), aspect='auto', origin='upper',
                      cmap=COOLWARM, vmin=0.0, vmax=args.vmax, interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(ANCHOR_RED if (layer, head) in anchors else WORKING_BLUE)
                spine.set_linewidth(1.0)
            if layer == 0:
                ax.set_title(f'H{head}', fontsize=11, pad=3)
            if head == 0:
                ax.set_ylabel(f'L{layer}', fontsize=11, rotation=0, ha='right', va='center', labelpad=5)
    sm = ScalarMappable(norm=Normalize(vmin=0.0, vmax=args.vmax), cmap=COOLWARM)
    cbar = fig.colorbar(sm, cax=fig.add_axes([0.94, 0.2, 0.012, 0.6]))
    cbar.set_label('Mean attn score', fontsize=12)
    cbar.ax.tick_params(labelsize=10, length=2.0, width=0.6)
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = p.add_subparsers(dest='plot', required=True)
    for name, needs in (('layer-head', 'scores'), ('combined', 'scores'), ('grid', 'maps')):
        s = sub.add_parser(name)
        s.add_argument('--scores', required=needs == 'scores', help='sink_scores.json')
        if needs == 'maps':
            s.add_argument('--maps', required=True, help='attn_maps.npz')
        s.add_argument('--anchors', default=None, help='anchor json (default: select from --scores)')
        s.add_argument('--percentile', type=float, default=80.0)
        s.add_argument('--vmax', type=float, default=None if name == 'combined' else 0.4)
        s.add_argument('--dpi', type=int, default=200)
        s.add_argument('--out', required=True, help='.png or .pdf')
    args = p.parse_args()
    fig = {'layer-head': plot_layer_head, 'combined': plot_combined, 'grid': plot_grid}[args.plot](args)
    fig.savefig(args.out, bbox_inches='tight', dpi=args.dpi)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
