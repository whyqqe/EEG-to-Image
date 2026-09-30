#!/usr/bin/env bash
#SBATCH --job-name=loso-diff
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=160G
#SBATCH --time=1-12:00:00
#SBATCH --output=/project/peilab/why/third_party/loso_pipeline/logs/%x-%j.out
#SBATCH --error=/project/peilab/why/third_party/loso_pipeline/logs/%x-%j.err

# Stage 3 -- freeze SDXL, train EEG->IP-Adapter projector + UNet LoRA.
set -uo pipefail
module load slurm nvhpc-hpcx-cuda12/23.11

HERE=/project/peilab/why/third_party/loso_pipeline
# shellcheck disable=SC1091
source "${HERE}/env.sh"
PY="${LOSO_VENV}/bin/python"

ALIGN_CKPT="${LOSO_ALIGN_CKPT:-${LOSO_OUT}/align/sub-08/best.pt}"
if [[ ! -f "${ALIGN_CKPT}" ]]; then
  echo "[FATAL] missing Stage-2 checkpoint: ${ALIGN_CKPT}" >&2
  exit 10
fi

echo "host=$(hostname) job=${SLURM_JOB_ID:-local}"
echo "align_ckpt=${ALIGN_CKPT}"
echo "=== Stage 3 diffusion adapter sub-08 ==="

"${PY}" "${HERE}/scripts/train_diffusion.py" \
    --test-subject sub-08 \
    --align-ckpt "${ALIGN_CKPT}" \
    --epochs "${LOSO_DIFF_EPOCHS:-20}" \
    --batch-size "${LOSO_DIFF_BS:-8}" \
    --lr 1e-4 \
    --warmup-steps 200 \
    --amp-dtype bf16 \
    --lora-rank 16 --lora-alpha 16 \
    --num-workers 8 \
    || exit 20

echo "=== [OK] Stage 3 complete $(date -Iseconds) ==="
