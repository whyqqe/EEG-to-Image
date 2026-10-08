#!/bin/bash
# 2x2 attribution for the Stage-1 regression (see scripts/probe_regression.py).
#
# Arm (a) `routed_sr` + `z_norm: none` reproduces job 636661, which trained. Arm (d)
# `routed` + `z_norm: unit` reproduces job 637180, which was dead. The other two split
# the difference. One epoch per arm: 636661 moved test top-1 from 0.50 to 6.00 across
# its first epoch, so one epoch separates a live run from a dead one.
#
#SBATCH --job-name=probe-regr
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --output=/project/peilab/why/CLIP/outputs/slurm/probe-regr-%j.out
#SBATCH --error=/project/peilab/why/CLIP/outputs/slurm/probe-regr-%j.err

set -euo pipefail

ROOT="/project/peilab/why/CLIP"
PY="${ROOT}/.venv/bin/python"
mkdir -p "${ROOT}/outputs/slurm"

# --- caches (AGENTS.md §2.4): MUST come before any torch/HF import ---
export PYTHONNOUSERSITE=1
export HF_HOME=/project/peilab/why/cache/huggingface
export HF_HUB_CACHE=/project/peilab/why/cache/huggingface/hub
export OPENCLIP_CACHE_DIR=/project/peilab/why/cache/open_clip
export TORCH_HOME=/project/peilab/why/cache/torch
export XDG_CACHE_HOME=/project/peilab/why/cache/xdg
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${ROOT}"
echo "[probe] job=${SLURM_JOB_ID} host=$(hostname) date=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# --- hard CUDA gate: never silently run this on CPU ---
"${PY}" -c "import sys,torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo "[FATAL] CUDA unavailable on $(hostname)"; exit 1; }

"${PY}" scripts/probe_regression.py \
  --config configs/loso_sub08.yaml \
  --epochs 1 --log-every 400 --seeds 2025 2026 \
  --out-dir outputs/probe_regression

echo "[probe] finished rc=$? at $(date -Is)"
