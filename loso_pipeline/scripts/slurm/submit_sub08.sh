#!/usr/bin/env bash
# Submit the full sub-08 LOSO chain with SLURM dependencies.
#
#   Stage 0  data prep        (already done; re-submit only if targets missing)
#   Stage 2  align            depends on nothing (targets validated at job start)
#   Stage 3  diffusion        after align
#   Stage 4  calib + eval     after diffusion
#
# Usage:
#   bash scripts/slurm/submit_sub08.sh
#   SKIP_PREP=1 bash scripts/slurm/submit_sub08.sh   # default: skip prep
set -euo pipefail

HERE=/project/peilab/why/third_party/loso_pipeline
cd "${HERE}"
mkdir -p logs

SKIP_PREP="${SKIP_PREP:-1}"
ALIGN_DEP=""

if [[ "${SKIP_PREP}" != "1" ]]; then
  PREP_ID=$(sbatch --parsable scripts/slurm/10_prep_data.sbatch)
  echo "submitted prep  job ${PREP_ID}"
  ALIGN_DEP="--dependency=afterok:${PREP_ID}"
else
  echo "skipping prep (SKIP_PREP=1); validating targets are on disk"
  # shellcheck disable=SC1091
  source "${HERE}/env.sh"
  "${LOSO_VENV}/bin/python" scripts/validate_targets.py --splits train test
fi

ALIGN_ID=$(sbatch --parsable ${ALIGN_DEP} scripts/slurm/20_align.sbatch)
echo "submitted align job ${ALIGN_ID}"

DIFF_ID=$(sbatch --parsable --dependency=afterok:${ALIGN_ID} scripts/slurm/30_diffusion.sbatch)
echo "submitted diff  job ${DIFF_ID}"

EVAL_ID=$(sbatch --parsable --dependency=afterok:${DIFF_ID} scripts/slurm/40_calib_eval.sbatch)
echo "submitted eval  job ${EVAL_ID}"

echo
echo "chain: align=${ALIGN_ID} -> diff=${DIFF_ID} -> eval=${EVAL_ID}"
squeue -u "${USER}" -o "%.10i %.20j %.8T %.10M %R" | head -20
