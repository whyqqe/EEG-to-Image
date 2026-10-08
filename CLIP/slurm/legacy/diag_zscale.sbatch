#!/bin/bash
# Separates the two candidate mechanisms behind the dead support-conditioned Stage 1:
# H1 (conditioning input scale too large) vs H2 (fresh support set = noisy modulation).
# See scripts/diag_zscale.py.
#
#SBATCH --job-name=diag-zscale
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --output=/project/peilab/why/CLIP/outputs/slurm/diag-zscale-%j.out
#SBATCH --error=/project/peilab/why/CLIP/outputs/slurm/diag-zscale-%j.err

set -euo pipefail
ROOT="/project/peilab/why/CLIP"
PY="${ROOT}/.venv/bin/python"
mkdir -p "${ROOT}/outputs/slurm"

export PYTHONNOUSERSITE=1
export HF_HOME=/project/peilab/why/cache/huggingface
export HF_HUB_CACHE=/project/peilab/why/cache/huggingface/hub
export OPENCLIP_CACHE_DIR=/project/peilab/why/cache/open_clip
export TORCH_HOME=/project/peilab/why/cache/torch
export XDG_CACHE_HOME=/project/peilab/why/cache/xdg
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${ROOT}"
echo "[diag] job=${SLURM_JOB_ID} host=$(hostname) date=$(date -Is)"
"${PY}" -c "import sys,torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo "[FATAL] CUDA unavailable on $(hostname)"; exit 1; }

"${PY}" scripts/diag_zscale.py --config configs/loso_sub08.yaml --steps 400
echo "[diag] finished rc=$? at $(date -Is)"
