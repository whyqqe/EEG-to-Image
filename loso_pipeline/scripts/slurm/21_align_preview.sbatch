#!/usr/bin/env bash
#SBATCH --job-name=loso-align-preview
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=120G
#SBATCH --time=0-03:00:00
#SBATCH --output=/project/peilab/why/third_party/loso_pipeline/logs/%x-%j.out
#SBATCH --error=/project/peilab/why/third_party/loso_pipeline/logs/%x-%j.err

# Stage 2 short real-data preview, on the FULL architecture and FULL data path.
#
# Purpose: the CPU smoke test in scripts/smoke_align.py uses synthetic tensors and
# checks shapes, gradients and invariants.  It cannot check the thing that actually
# failed last time -- whether the loss moves and whether the representation stays
# alive over real epochs.  This runs the real datasets, the real target store and the
# real optimiser for a few hundred steps and writes to a separate `--run-tag`
# directory so it cannot overwrite the canonical checkpoint.
#
# What to read in the log:
#   * `gradient budget` -- which term owns the encoder gradient.  Last time L_time
#     took ~58%; if that is still happening the reweighting did not work.
#   * `init:` vs the first eval -- `top1_sv_ratio` must not grow by more than ~3x and
#     `mean_cosine` must not climb; `top1` must rise above the constant baseline.
#   * `constant_encoder_baseline` vs `top1` -- if the loss and the baseline move
#     together, the encoder is not contributing.
set -uo pipefail
module load slurm nvhpc-hpcx-cuda12/23.11

HERE=/project/peilab/why/third_party/loso_pipeline
# shellcheck disable=SC1091
source "${HERE}/env.sh"
PY="${LOSO_VENV}/bin/python"

echo "host=$(hostname) job=${SLURM_JOB_ID:-local} python=${PY}"
echo "=== Stage 2 align preview (sub-08, short) ==="

"${PY}" "${HERE}/scripts/validate_targets.py" --splits train test || exit 10

# Gradient-budget probe on the first two steps, then 1 epoch capped at 120 steps.
"${PY}" "${HERE}/scripts/train_align.py" \
    --test-subject sub-08 \
    --run-tag preview \
    --epochs 2 \
    --max-steps-per-epoch 120 \
    --batch-size 512 \
    --lr 3e-4 \
    --warmup-epochs 1 \
    --adv-warmup-epochs 1 \
    --eval-every 1 \
    --eval-triv \
    --eval-avg-reps 1 80 \
    --gradient-budget-steps 2 \
    --log-every 20 \
    --amp-dtype bf16 \
    --num-workers 8 \
    --d-model 256 --n-heads 4 --n-layers 4 --branch-width 64 \
    --d-inv 512 --d-sub 128 --adapter-blocks 2 \
    || exit 20

echo "=== [OK] preview complete $(date -Iseconds) ==="
echo "inspect: ${LOSO_OUT_ROOT:-/project/peilab/why/loso_out}/align/sub-08/preview"
