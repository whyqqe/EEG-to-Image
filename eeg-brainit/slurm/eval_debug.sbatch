#!/usr/bin/env bash
#SBATCH --job-name=eeg-brainit-evaldbg
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=28
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail
ROOT=/project/peilab/why/eeg-brainit
CKPT="${ROOT}/outputs/stage4_e2e/checkpoints/stage4_best_e30.pt"
module load slurm nvhpc-hpcx-cuda12/23.11
cd "${ROOT}"
source scripts/activate.sh
export PYTHONUNBUFFERED=1
export HF_HOME=/project/peilab/why/cache/eeg-brainit/hf
export OPENCLIP_CACHE_DIR=/project/peilab/why/cache/eeg-brainit/open_clip
export TORCH_HOME=/project/peilab/why/cache/eeg-brainit/torch
python -c "import torch; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0))"
test -f "${CKPT}"
python scripts/evaluate_retrieval.py \
  --config configs/base.yaml \
  --checkpoint "${CKPT}" \
  --smoke \
  --output-dir outputs/eval/stage4_smoke
echo "[OK] eval smoke finished"
