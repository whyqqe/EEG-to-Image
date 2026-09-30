#!/usr/bin/env bash
#SBATCH --job-name=eeg-brainit-clipdbg
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=28
#SBATCH --mem=160G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail
ROOT=/project/peilab/why/eeg-brainit
RESUME="${ROOT}/outputs/stage4_e2e/checkpoints/stage4_best_e30.pt"
TEACHER="${ROOT}/data/processed/teachers/clip_ViT-L-14_openai/embeddings.npy"
module load slurm nvhpc-hpcx-cuda12/23.11
cd "${ROOT}"
source scripts/activate.sh
export PYTHONUNBUFFERED=1

python -c "import torch; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0))"
test -f "${RESUME}"
test -f "${TEACHER}"

python scripts/train.py \
  --config configs/base.yaml \
  --override configs/debug_clip_align.yaml \
  --resume "${RESUME}"
echo "[OK] debug clip-align finished"
