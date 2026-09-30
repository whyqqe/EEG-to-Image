#!/usr/bin/env bash
set -euo pipefail
NB_ROOT=/project/peilab/why/NeuroBridge
OUT_ROOT="${NB_ROOT}/outputs/hcma_loso_fid129"
# Default: remaining folds after sub-01 pilot. Override: HOLDOUTS=1,2,3,...
HOLDOUTS="${HOLDOUTS:-2,3,4,5,6,7,8,9,10}"
mkdir -p "${OUT_ROOT}" "${NB_ROOT}/outputs/slurm"
# Avoid sbatch --export commas (Slurm splits on ','). Use a file instead.
HOLD_FILE="${OUT_ROOT}/holdouts.txt"
printf '%s\n' "${HOLDOUTS}" > "${HOLD_FILE}"
JOB=$(sbatch --parsable --export=ALL,HOLDOUTS_FILE="${HOLD_FILE}",OUT_ROOT="${OUT_ROOT}",DEVICE=cuda:0 \
  "${NB_ROOT}/slurm/hcma_loso_fid129.sbatch")
echo "{\"pipeline\":\"hcma_loso_fid129\",\"job\":\"${JOB}\",\"output\":\"${OUT_ROOT}\",\"holdouts\":\"${HOLDOUTS}\",\"holdouts_file\":\"${HOLD_FILE}\",\"submitted\":\"$(date -Iseconds)\",\"baseline\":\"hcma_10subj hcma_full_a40 pooled_FID=129.47\",\"note\":\"Slurm --export cannot pass comma-lists; use HOLDOUTS_FILE\"}" \
  | tee "${OUT_ROOT}/pipeline_submit.json"
echo "Submitted job ${JOB} holdouts=${HOLDOUTS}"
