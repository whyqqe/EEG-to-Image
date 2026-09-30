#!/usr/bin/env bash
#SBATCH --job-name=eeg-brainit-direval
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=28
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail
ROOT=/project/peilab/why/eeg-brainit
SUBJECT="${SUBJECT:-sub-08}"
FAMILY="${FAMILY:-direct_sub08}"
OUT_DIR="${OUT_DIR:-outputs/eval/${FAMILY}}"

module load slurm nvhpc-hpcx-cuda12/23.11
cd "${ROOT}"
source scripts/activate.sh
export PYTHONUNBUFFERED=1
export HF_HOME=/project/peilab/why/cache/eeg-brainit/hf
export OPENCLIP_CACHE_DIR=/project/peilab/why/cache/eeg-brainit/open_clip
export TORCH_HOME=/project/peilab/why/cache/eeg-brainit/torch

python -c "import torch; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0))"
CKPT="$(ls -1t "${ROOT}/outputs/${FAMILY}/checkpoints/"stage0_best_*.pt 2>/dev/null | head -1 || true)"
if [[ -z "${CKPT}" ]]; then
  CKPT="$(ls -1t "${ROOT}/outputs/${FAMILY}/checkpoints/"*_best_*.pt 2>/dev/null | head -1 || true)"
fi
test -n "${CKPT}" && test -f "${CKPT}"
echo "[INFO] subject=${SUBJECT} ckpt=${CKPT} out=${OUT_DIR}"

python scripts/evaluate_retrieval.py \
  --config configs/base.yaml \
  --override configs/direct_sub08.yaml \
  --checkpoint "${CKPT}" \
  --direct-clip \
  --subjects "${SUBJECT}" \
  --batch-size 64 \
  --probe-train-split val \
  --max-probe-batches 0 \
  --max-test-batches 0 \
  --clip-model ViT-L-14 \
  --clip-pretrained openai \
  --output-dir "${OUT_DIR}"
echo "[OK] direct eval finished -> ${OUT_DIR}"
