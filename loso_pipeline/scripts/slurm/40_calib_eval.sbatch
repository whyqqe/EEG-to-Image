#!/usr/bin/env bash
#SBATCH --job-name=loso-eval
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=160G
#SBATCH --time=0-12:00:00
#SBATCH --output=/project/peilab/why/third_party/loso_pipeline/logs/%x-%j.out
#SBATCH --error=/project/peilab/why/third_party/loso_pipeline/logs/%x-%j.err

# Stage 4 -- few-shot calibration (k=0/5/10/25/50) + retrieval + reconstruction.
set -uo pipefail
module load slurm nvhpc-hpcx-cuda12/23.11

HERE=/project/peilab/why/third_party/loso_pipeline
# shellcheck disable=SC1091
source "${HERE}/env.sh"
PY="${LOSO_VENV}/bin/python"

ALIGN_CKPT="${LOSO_ALIGN_CKPT:-${LOSO_OUT}/align/sub-08/best.pt}"
DIFF_CKPT="${LOSO_DIFF_CKPT:-${LOSO_OUT}/diffusion/sub-08/best.pt}"

if [[ ! -f "${ALIGN_CKPT}" ]]; then
  echo "[FATAL] missing Stage-2 checkpoint: ${ALIGN_CKPT}" >&2
  exit 10
fi
EXTRA=()
if [[ -f "${DIFF_CKPT}" ]]; then
  EXTRA+=(--diffusion-ckpt "${DIFF_CKPT}")
  echo "diffusion_ckpt=${DIFF_CKPT}"
else
  echo "[warn] no diffusion ckpt at ${DIFF_CKPT}; retrieval-only evaluation"
fi

echo "host=$(hostname) job=${SLURM_JOB_ID:-local}"
echo "align_ckpt=${ALIGN_CKPT}"
echo "=== Stage 4 calib+eval sub-08 ==="

"${PY}" "${HERE}/scripts/calibrate_eval.py" \
    --test-subject sub-08 \
    --align-ckpt "${ALIGN_CKPT}" \
    "${EXTRA[@]}" \
    --k-shots 0 5 10 25 50 \
    --calib-steps 200 \
    --gen-steps 20 \
    --max-gen "${LOSO_MAX_GEN:-50}" \
    --seed 0 \
    || exit 20

echo "=== [OK] Stage 4 complete $(date -Iseconds) ==="
