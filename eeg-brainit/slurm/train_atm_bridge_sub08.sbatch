#!/usr/bin/env bash
#SBATCH --job-name=eeg-brainit-atmbrdg
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=28
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail
ROOT=/project/peilab/why/eeg-brainit
module load slurm nvhpc-hpcx-cuda12/23.11
cd "${ROOT}"
source scripts/activate.sh
export PYTHONUNBUFFERED=1
python -c "import torch; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0))"
test -f outputs/atm_bridge/sub-08_train_eeg_avg_1024.npy
python scripts/train_atm_bridge.py \
  --config configs/base.yaml \
  --override configs/atm_bridge_sub08.yaml
echo "[OK] ATM bridge training finished"
