#!/usr/bin/env bash
#SBATCH --job-name=eeg-brainit-nb-sa3
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

# Standalone +BIT (init from standalone S1).

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
test -f outputs/nb_standalone_s1_sub08/checkpoints/nb_stage1_best.pt

python scripts/train_neurobolt_bridge.py \
  --config configs/base.yaml \
  --override configs/nb_standalone_s3_sub08.yaml
echo "[OK] NeuroBOLT standalone S3 finished"
