#!/usr/bin/env bash
#SBATCH --job-name=eeg-brainit-atmds1
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

# Dual-channel S1: ATM distill + bridge only (BIT frozen).

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
test -f outputs/atm_bridge/sub-08_train_eeg_avg_1024.npy
test -f outputs/eval/atm_baseline/test_ViT-H-14_laion2b_s32b_b79k_features.npy

python scripts/train_atm_bridge.py \
  --config configs/base.yaml \
  --override configs/atm_distill_s1_sub08.yaml
echo "[OK] ATM distill stage-1 finished"
