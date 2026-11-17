#!/usr/bin/env bash
# Training with KV caching gradient: the same 700-step training, with the KV caching gradient added from step 200.
# Usage: bash scripts/train_cachegrad.sh [key=value overrides passed to both runs]
# Env: NPROC (GPUs per node, default 8), OUT (output root, default outputs).
set -euo pipefail
NPROC=${NPROC:-8}
OUT=${OUT:-outputs}
LAUNCH=(torchrun --nnodes=1 --nproc_per_node="$NPROC")

"${LAUNCH[@]}" train.py --config configs/disen.yaml --logdir "$OUT/disen_200" max_train_steps=200 "$@"

CKPT=$(ls -d "$OUT"/disen_200/checkpoint_model_* | sort | tail -n 1)/model.pt
"${LAUNCH[@]}" train.py --config configs/disen_cachegrad.yaml --logdir "$OUT/disen_cachegrad" --resume "$CKPT" "$@"
