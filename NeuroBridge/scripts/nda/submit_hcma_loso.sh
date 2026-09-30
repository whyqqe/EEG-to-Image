#!/usr/bin/env bash
set -euo pipefail
NB_ROOT=/project/peilab/why/NeuroBridge
OUT_ROOT="${NB_ROOT}/outputs/hcma_loso"
# Full 10-fold LOSO by default. Override: HOLDOUTS=8 bash submit_hcma_loso.sh
HOLDOUTS="${HOLDOUTS:-1,2,3,4,5,6,7,8,9,10}"
mkdir -p "${OUT_ROOT}" "${NB_ROOT}/outputs/slurm"
JOB=$(sbatch --parsable --export=ALL,HOLDOUTS="${HOLDOUTS}",OUT_ROOT="${OUT_ROOT}",DEVICE=cuda:0 \
  "${NB_ROOT}/slurm/hcma_loso.sbatch")
echo "{\"pipeline\":\"hcma_loso\",\"job\":\"${JOB}\",\"output\":\"${OUT_ROOT}\",\"holdouts\":\"${HOLDOUTS}\",\"submitted\":\"$(date -Iseconds)\",\"protocol\":\"9-subject pretrain → held-out FT → eval\",\"purpose\":\"cross-subject generalization (response to LOSO critique)\"}" \
  | tee "${OUT_ROOT}/pipeline_submit.json"
echo "Submitted job ${JOB}"
