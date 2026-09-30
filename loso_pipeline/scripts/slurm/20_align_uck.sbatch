#!/usr/bin/env bash
#SBATCH --job-name=loso-align-uck
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=160G
#SBATCH --time=0-16:00:00
#SBATCH --output=/project/peilab/why/third_party/loso_pipeline/logs/%x-%j.out
#SBATCH --error=/project/peilab/why/third_party/loso_pipeline/logs/%x-%j.err

# Stage 2 -- UCK-optimised multi-modal alignment, LOSO on sub-08, then few-shot
# SubjectAdapter calibration on the held-out subject (NeuroWeave Stage 5 / NEED IAM).
#
# Speed vs the previous run:
#   * avg_trials          4x fewer steps/epoch (SAMGA/UCK recipe)
#   * --w-time 0          skips set_alignment (~4 GB similarity at bs=512)
#   * --proj-layers 1     linear projector (UCK/SAMGA)
#   * --n-layers 2        thinner trunk (Transformer was 85% of encoder params)
#
# Cross-subject:
#   * SubjectAdapter already in the trunk during the 9-subject train
#   * after best.pt lands, calibrate a fresh 1-subject adapter on k={10,25,50}
#     held-out *train* images of sub-08 (never the 200 test concepts)
set -uo pipefail
module load slurm nvhpc-hpcx-cuda12/23.11

HERE=/project/peilab/why/third_party/loso_pipeline
# shellcheck disable=SC1091
source "${HERE}/env.sh"
PY="${LOSO_VENV}/bin/python"

echo "host=$(hostname) job=${SLURM_JOB_ID:-local} python=${PY}"
echo "=== Stage 2 UCK-align sub-08 ==="

"${PY}" "${HERE}/scripts/validate_targets.py" --splits train test || exit 10

"${PY}" "${HERE}/scripts/train_align.py" \
    --test-subject sub-08 \
    --run-tag uck \
    --epochs "${LOSO_ALIGN_EPOCHS:-40}" \
    --batch-size "${LOSO_ALIGN_BS:-512}" \
    --lr 3e-4 \
    --warmup-epochs 2 \
    --adv-warmup-epochs 3 \
    --eval-every 2 \
    --eval-avg-reps 1 80 \
    --avg-trials \
    --amp-dtype bf16 \
    --num-workers 8 \
    --gradient-budget-steps 2 \
    --log-every 50 \
    --d-model 256 --n-heads 4 --n-layers 2 --branch-width 64 \
    --d-inv 512 --d-sub 128 --adapter-blocks 2 \
    --proj-layers 1 --proj-hidden 512 \
    --w-time 0 --w-trial 0 \
    --w-gallery 1.0 --w-mem 0.3 \
    --w-img 1.0 --w-text 0.2 --w-dino 0.3 --w-vae 0.1 \
    --w-var 1.0 --w-cov 0.1 --w-adv 0.3 --w-dist 0.3 --w-orth 0.1 \
    || exit 20

echo "=== Stage 2 done; few-shot SubjectAdapter calibration ==="
CKPT="${LOSO_OUT_ROOT:-${HERE}/outputs}/align/sub-08/uck/best.pt"
if [[ ! -f "${CKPT}" ]]; then
  CKPT="${LOSO_OUT_ROOT:-${HERE}/outputs}/align/sub-08/best.pt"
fi
[[ -f "${CKPT}" ]] || { echo "missing checkpoint ${CKPT}"; exit 30; }

"${PY}" "${HERE}/scripts/calibrate_eval.py" \
    --align-ckpt "${CKPT}" \
    --test-subject sub-08 \
    --k-shots 0 10 25 50 \
    --calib-steps 200 \
    || exit 40

echo "=== [OK] UCK align + calib complete $(date -Iseconds) ==="
