#!/usr/bin/env bash
#SBATCH --job-name=loso-align
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=160G
#SBATCH --time=1-12:00:00
#SBATCH --output=/project/peilab/why/third_party/loso_pipeline/logs/%x-%j.out
#SBATCH --error=/project/peilab/why/third_party/loso_pipeline/logs/%x-%j.err

# Stage 2 -- multi-modal alignment, LOSO on sub-08.
set -uo pipefail
module load slurm nvhpc-hpcx-cuda12/23.11

HERE=/project/peilab/why/third_party/loso_pipeline
# shellcheck disable=SC1091
source "${HERE}/env.sh"
PY="${LOSO_VENV}/bin/python"

echo "host=$(hostname) job=${SLURM_JOB_ID:-local} python=${PY}"
echo "=== Stage 2 align sub-08 ==="

# Gate: refuse to train if Stage-0 targets are incomplete / mis-paired.
"${PY}" "${HERE}/scripts/validate_targets.py" --splits train test || exit 10

"${PY}" "${HERE}/scripts/train_align.py" \
    --test-subject sub-08 \
    --epochs "${LOSO_ALIGN_EPOCHS:-60}" \
    --batch-size "${LOSO_ALIGN_BS:-512}" \
    --lr 3e-4 \
    --warmup-epochs 3 \
    --adv-warmup-epochs 5 \
    --eval-every 5 \
    --eval-triv \
    --amp-dtype bf16 \
    --num-workers 8 \
    --d-model 256 --n-heads 4 --n-layers 4 --branch-width 64 \
    --d-inv 512 --d-sub 128 --adapter-blocks 2 \
    || exit 20

echo "=== [OK] Stage 2 complete $(date -Iseconds) ==="
