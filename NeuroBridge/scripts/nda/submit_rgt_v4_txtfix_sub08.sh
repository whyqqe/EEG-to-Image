#!/usr/bin/env bash
set -euo pipefail
NB_ROOT=/project/peilab/why/NeuroBridge
OUT="${NB_ROOT}/outputs/rgt_v4_txtfix/sub-08"
mkdir -p "${OUT}" "${NB_ROOT}/outputs/slurm"
# cancel any stale duplicate rgt-v4 if re-run
JOB=$(sbatch --parsable "${NB_ROOT}/slurm/rgt_v4_txtfix_sub08.sbatch")
echo "{\"pipeline\":\"rgt_v4_txtfix\",\"job\":\"${JOB}\",\"output\":\"${OUT}\",\"submitted\":\"$(date -Iseconds)\",\"fix\":[\"200-way test gallery prompts\",\"post-norm fusion constraints\"]}" | tee "${OUT}/pipeline_submit.json"
echo "Submitted job ${JOB}"
