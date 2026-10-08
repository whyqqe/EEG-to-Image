#!/bin/bash
# Pre-registered reading #3 for pillar A2 (v5-ep2): does the trained-vs-held-out
# subject gap shrink? Run on the CLIP shared filesystem (NOT /tmp -- that is node-local).
set -euo pipefail
ROOT=/project/peilab/why/CLIP
export PYTHONNOUSERSITE=1 HF_HOME=/project/peilab/why/cache/huggingface
export OPENCLIP_CACHE_DIR=/project/peilab/why/cache/open_clip TORCH_HOME=/project/peilab/why/cache/torch
export XDG_CONFIG_HOME=$ROOT/scratch/config MPLCONFIGDIR=$ROOT/scratch/mpl
export TMPDIR="/tmp/clip-gap-${SLURM_JOB_ID:-local}"; mkdir -p "$TMPDIR" "$XDG_CONFIG_HOME" "$MPLCONFIGDIR"
export PYTHONPATH=$ROOT/src
cd "$ROOT"
for ck in outputs/stage1/v5-a1-k20/last.pt outputs/stage1/v5-ep2/last.pt; do
  echo "===== $ck ====="
  "$ROOT/.venv/bin/python" scripts/probe_per_subject_accuracy.py --ckpt "$ck" \
      --held-out 8 --device cuda 2>&1 | tail -16
done
