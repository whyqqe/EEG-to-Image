#!/usr/bin/env bash
#SBATCH --job-name=eeg-brainit-nb-s0
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

# Ablation A: NeuroBOLT → direct MLP → CLIP (sub-08).

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
test -f /project/peilab/why/cache/things_eeg2_b2/neurobolt/sub-08/train_tokens.npy
test -f outputs/atm_bridge/sub-08_train_eeg_avg_1024.npy

python scripts/train_neurobolt_bridge.py \
  --config configs/base.yaml \
  --override configs/nb_direct_s0_sub08.yaml
echo "[OK] NeuroBOLT direct S0 finished"
