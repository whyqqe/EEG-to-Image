#!/usr/bin/env bash
set -euo pipefail
NB_ROOT=/project/peilab/why/NeuroBridge
OUT="${NB_ROOT}/outputs/rgt_cfm/sub-08"
mkdir -p "${OUT}" "${NB_ROOT}/outputs/slurm"
JOB=$(sbatch --parsable "${NB_ROOT}/slurm/rgt_cfm_sub08.sbatch")
echo "{\"pipeline\":\"rgt_cfm\",\"job\":\"${JOB}\",\"output\":\"${OUT}\",\"submitted\":\"$(date -Iseconds)\",\"claim\":\"Retrieval-Generation Transport CFM\",\"subjects\":\"1,2,4,5,6,7,8,9,10\"}" | tee "${OUT}/pipeline_submit.json"
echo "Submitted job ${JOB}"
