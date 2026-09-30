#!/usr/bin/env bash
#SBATCH --job-name=eeg-brainit-atmgen
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

# SDXL generation + image metrics for ATM prior (sub-08).
# Uses project HF cache only (home disk is full).

set -euo pipefail
ROOT=/project/peilab/why/eeg-brainit
module load slurm nvhpc-hpcx-cuda12/23.11
cd "${ROOT}"
source scripts/activate.sh
export PYTHONUNBUFFERED=1
export HF_HOME=/project/peilab/why/cache/eeg-brainit/hf
export HF_HUB_CACHE=/project/peilab/why/cache/eeg-brainit/hf/hub
export TRANSFORMERS_CACHE=/project/peilab/why/cache/eeg-brainit/hf/hub
export DIFFUSERS_CACHE=/project/peilab/why/cache/eeg-brainit/hf/hub
export OPENCLIP_CACHE_DIR=/project/peilab/why/cache/eeg-brainit/open_clip
export TORCH_HOME=/project/peilab/why/cache/eeg-brainit/torch
export TMPDIR=/project/peilab/why/cache/eeg-brainit/tmp
mkdir -p "$TMPDIR" "$HF_HUB_CACHE/.locks"

python -c "import torch; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0))"
test -d "$HF_HUB_CACHE/models--stabilityai--stable-diffusion-xl-base-1.0"
test -d "$HF_HUB_CACHE/models--h94--IP-Adapter"
test -f outputs/eval/atm_pipeline_sub08/sub-08_prior_clip_1024.npy || true

# Reuse prior embeds if present; otherwise recompute full pipeline.
python scripts/eval_atm_pipeline.py \
  --subject sub-08 \
  --output-dir outputs/eval/atm_pipeline_sub08 \
  --prior-steps 50 \
  --prior-guidance 5.0 \
  --gen-steps 30 \
  --gen-guidance 5.0 \
  --gen-size 512
echo "[OK] ATM generation eval finished"
