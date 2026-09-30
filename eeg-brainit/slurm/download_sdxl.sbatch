#!/usr/bin/env bash
#SBATCH --job-name=eeg-brainit-sdxl-dl
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

# Download complete SDXL fp16 + IP-Adapter into project HF cache (not /home).
# Note: peilab QOS requires GRES/GPU even for download-only jobs.

set -euo pipefail
ROOT=/project/peilab/why/eeg-brainit
module load slurm nvhpc-hpcx-cuda12/23.11
cd "${ROOT}"
source scripts/activate.sh
export PYTHONUNBUFFERED=1
export HF_HOME=/project/peilab/why/cache/eeg-brainit/hf
export HF_HUB_CACHE=/project/peilab/why/cache/eeg-brainit/hf/hub
export TRANSFORMERS_CACHE=/project/peilab/why/cache/eeg-brainit/hf/hub
export HUGGINGFACE_HUB_CACHE=/project/peilab/why/cache/eeg-brainit/hf/hub
export TMPDIR=/project/peilab/why/cache/eeg-brainit/tmp
mkdir -p "$TMPDIR" "$HF_HUB_CACHE/.locks"

python scripts/download_sdxl.py
test -f "$HF_HUB_CACHE/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/"*/model_index.json
echo "[OK] SDXL download finished"
