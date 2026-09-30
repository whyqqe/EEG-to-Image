#!/usr/bin/env bash
#SBATCH --job-name=erdc-cn-dl
#SBATCH --account=peilab
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

# CPU-only download of SDXL ControlNet-canny for next structure-routing experiments.

set -euo pipefail
ROOT=/project/peilab/why/eeg-brainit
cd "${ROOT}"
source scripts/activate.sh
export PYTHONUNBUFFERED=1
export HF_HOME=/project/peilab/why/cache/eeg-brainit/hf
export HF_HUB_CACHE=/project/peilab/why/cache/eeg-brainit/hf/hub
# allow network for download
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
python scripts/download_controlnet_sdxl.py
echo "[OK] controlnet download job done"
