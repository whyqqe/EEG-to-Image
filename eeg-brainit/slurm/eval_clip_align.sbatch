#!/usr/bin/env bash
#SBATCH --job-name=eeg-brainit-clipeval
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=28
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail
ROOT=/project/peilab/why/eeg-brainit
# Which checkpoint family to evaluate: clip_align | clip_align_e2e
FAMILY="${FAMILY:-clip_align}"
OUT_DIR="${OUT_DIR:-outputs/eval/${FAMILY}}"
STAGE_GLOB="${STAGE_GLOB:-stage2_best_*.pt}"

module load slurm nvhpc-hpcx-cuda12/23.11
cd "${ROOT}"
source scripts/activate.sh
export PYTHONUNBUFFERED=1
export HF_HOME=/project/peilab/why/cache/eeg-brainit/hf
export OPENCLIP_CACHE_DIR=/project/peilab/why/cache/eeg-brainit/open_clip
export TORCH_HOME=/project/peilab/why/cache/eeg-brainit/torch

python -c "import torch; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0))"

CKPT="$(ls -1t "${ROOT}/outputs/${FAMILY}/checkpoints/"${STAGE_GLOB} 2>/dev/null | head -1 || true)"
if [[ -z "${CKPT}" ]]; then
  CKPT="$(ls -1t "${ROOT}/outputs/${FAMILY}/checkpoints/"*_best_*.pt 2>/dev/null | head -1 || true)"
fi
test -n "${CKPT}" && test -f "${CKPT}"
echo "[INFO] family=${FAMILY} checkpoint=${CKPT} out=${OUT_DIR}"

python scripts/evaluate_retrieval.py \
  --config configs/base.yaml \
  --override configs/clip_align.yaml \
  --checkpoint "${CKPT}" \
  --direct-clip \
  --batch-size 8 \
  --probe-train-split val \
  --max-probe-batches 400 \
  --max-test-batches 0 \
  --clip-model ViT-L-14 \
  --clip-pretrained openai \
  --output-dir "${OUT_DIR}"
echo "[OK] clip-align eval finished -> ${OUT_DIR}"
