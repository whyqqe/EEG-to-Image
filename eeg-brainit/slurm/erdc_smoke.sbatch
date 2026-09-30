#!/usr/bin/env bash
#SBATCH --job-name=erdc-smoke
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --exclude=dgx-10,dgx-11
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

# Fast smoke: 4 images, B0 fixed only — prove SDXL+IP-Adapter runs on allocated GPU.

set -euo pipefail
ROOT=/project/peilab/why/eeg-brainit
module load slurm nvhpc-hpcx-cuda12/23.11
cd "${ROOT}"
source scripts/activate.sh
export PYTHONUNBUFFERED=1
export HF_HOME=/project/peilab/why/cache/eeg-brainit/hf
export HF_HUB_CACHE=/project/peilab/why/cache/eeg-brainit/hf/hub
export HUGGINGFACE_HUB_CACHE=/project/peilab/why/cache/eeg-brainit/hf/hub
export TRANSFORMERS_CACHE=/project/peilab/why/cache/eeg-brainit/hf/hub
export DIFFUSERS_CACHE=/project/peilab/why/cache/eeg-brainit/hf/hub
export OPENCLIP_CACHE_DIR=/project/peilab/why/cache/eeg-brainit/open_clip
export TORCH_HOME=/project/peilab/why/cache/eeg-brainit/torch
export TMPDIR=/project/peilab/why/cache/eeg-brainit/tmp
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# Broken xformers (built for torch 2.2) crashes Triton on import — force SDPA path.
export XFORMERS_DISABLED=1
mkdir -p "$TMPDIR" outputs/erdc outputs/slurm

python - <<'PY'
import torch, torchvision
assert torch.cuda.is_available(), "CUDA unavailable — exclude/requeue"
from diffusers import StableDiffusionXLPipeline
print("GPU", torch.cuda.get_device_name(0), "tv", torchvision.__version__, "SDXL OK", flush=True)
PY

python scripts/erdc_b0_generate.py \
  --subject sub-08 \
  --embed-source atm \
  --mode fixed \
  --output-dir outputs/erdc/smoke_b0_atm_sub08 \
  --max-images 4 \
  --gen-steps 20 \
  --gen-guidance 5.0 \
  --gen-size 512

echo "[OK] ERDC smoke finished"
