#!/usr/bin/env bash
#SBATCH --job-name=eeg-brainit-sigdbg
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=28
#SBATCH --mem=160G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail
ROOT=/project/peilab/why/eeg-brainit
RESUME="${ROOT}/outputs/clip_align_e2e/checkpoints/stage4_best_e10.pt"
module load slurm nvhpc-hpcx-cuda12/23.11
cd "${ROOT}"
source scripts/activate.sh
export PYTHONUNBUFFERED=1

python -c "import torch; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0))"
test -f "${RESUME}"

python scripts/train.py \
  --config configs/base.yaml \
  --override configs/debug_siglip_hard.yaml \
  --resume "${RESUME}"
echo "[OK] debug siglip-hard finished"
