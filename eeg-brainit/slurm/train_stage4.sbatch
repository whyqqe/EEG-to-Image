#!/usr/bin/env bash
#SBATCH --job-name=eeg-brainit-s4
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=28
#SBATCH --mem=160G
#SBATCH --time=48:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail
ROOT=/project/peilab/why/eeg-brainit
RESUME="${RESUME:-${ROOT}/outputs/stage3_brainit/checkpoints/stage3_best_e18.pt}"
module load slurm nvhpc-hpcx-cuda12/23.11
cd "${ROOT}"
source scripts/activate.sh
export PYTHONUNBUFFERED=1
python -c "import torch; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0))"
test -f "${RESUME}"
echo "[INFO] resume=${RESUME}"
python scripts/train.py \
  --config configs/base.yaml \
  --override configs/stage4_e2e.yaml \
  --resume "${RESUME}"
echo "[OK] stage4 finished"
