#!/usr/bin/env bash
#SBATCH --job-name=eeg-s4pix
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --exclude=dgx-10,dgx-11
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

# ATM distill Stage-4: Pix-oriented BIT fine-tune from S3 (lower distill pressure).

set -euo pipefail
ROOT=/project/peilab/why/eeg-brainit
module load slurm nvhpc-hpcx-cuda12/23.11
cd "${ROOT}"
source scripts/activate.sh
export PYTHONUNBUFFERED=1
export HF_HOME=/project/peilab/why/cache/eeg-brainit/hf
export OPENCLIP_CACHE_DIR=/project/peilab/why/cache/eeg-brainit/open_clip
export TORCH_HOME=/project/peilab/why/cache/eeg-brainit/torch

python -c "import torch; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0))"
test -f outputs/atm_distill_s3_sub08/checkpoints/atm_stage3_best.pt

python scripts/train_atm_bridge.py \
  --config configs/base.yaml \
  --override configs/atm_distill_s4_pix_sub08.yaml

echo "[OK] ATM distill stage-4 (Pix-oriented) finished"
