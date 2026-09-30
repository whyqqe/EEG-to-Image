#!/usr/bin/env bash
#SBATCH --job-name=eeg-brainit-clipe2e
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=28
#SBATCH --mem=160G
#SBATCH --time=12:00:00
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

# Prefer newest stage2 best from clip_align; fall back to last.
RESUME="$(ls -1t "${ROOT}"/outputs/clip_align/checkpoints/stage2_best_*.pt 2>/dev/null | head -1 || true)"
if [[ -z "${RESUME}" ]]; then
  RESUME="$(ls -1t "${ROOT}"/outputs/clip_align/checkpoints/stage2_last_*.pt 2>/dev/null | head -1 || true)"
fi
test -n "${RESUME}" && test -f "${RESUME}"
echo "[INFO] resume=${RESUME}"

python scripts/train.py \
  --config configs/base.yaml \
  --override configs/clip_align_e2e.yaml \
  --resume "${RESUME}"
echo "[OK] clip-align e2e finished"
