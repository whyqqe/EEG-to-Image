#!/bin/bash
#SBATCH --job-name=nw4_diag
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/nw4_diag_%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/nw4_diag_%j.err
#
# Trace where the nw4 conditioning bank collapses (S1 head vs EEG->bank ridge vs
# manifold_project).  Cheap: ~2 min of forward passes.

set -uo pipefail
cd /project/peilab/why/NeuroBridge

PYTHON=/project/peilab/why/eeg-brainit/.venv/bin/python
export OMP_NUM_THREADS=8
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/project/peilab/why/cache/xdg}"
export HOME="${XDG_CACHE_HOME}"

O=outputs/nw4/sub-08
mkdir -p outputs/slurm "${O}/logs"

"${PYTHON}" scripts/nda/nw4_diag_reanchor.py \
  --s1 "${O}/s1" \
  --out "${O}/diag_reanchor.json" \
  --device cuda:0
rc=$?
echo "[nw4-diag] exit ${rc}"
exit "${rc}"
