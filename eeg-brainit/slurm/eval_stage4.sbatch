#!/usr/bin/env bash
#SBATCH --job-name=eeg-brainit-eval
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=28
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail
ROOT=/project/peilab/why/eeg-brainit
CKPT="${CKPT:-${ROOT}/outputs/stage4_e2e/checkpoints/stage4_best_e30.pt}"
module load slurm nvhpc-hpcx-cuda12/23.11
cd "${ROOT}"
source scripts/activate.sh
export PYTHONUNBUFFERED=1
export HF_HOME=/project/peilab/why/cache/eeg-brainit/hf
export OPENCLIP_CACHE_DIR=/project/peilab/why/cache/eeg-brainit/open_clip
export TORCH_HOME=/project/peilab/why/cache/eeg-brainit/torch
python -c "import torch; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0))"
test -f "${CKPT}"
echo "[INFO] checkpoint=${CKPT}"
# Full test (2000 trials) + val probe (capped for speed)
python scripts/evaluate_retrieval.py \
  --config configs/base.yaml \
  --checkpoint "${CKPT}" \
  --batch-size 8 \
  --probe-train-split val \
  --max-probe-batches 400 \
  --max-test-batches 0 \
  --clip-model ViT-L-14 \
  --clip-pretrained openai \
  --output-dir outputs/eval/stage4
echo "[OK] eval finished"
