#!/usr/bin/env bash
set -euo pipefail
NB_ROOT=/project/peilab/why/NeuroBridge
OUT="${NB_ROOT}/outputs/nda_v2/sub-08"
mkdir -p "${OUT}" "${NB_ROOT}/outputs/slurm"
JOB=$(sbatch --parsable "${NB_ROOT}/slurm/nda_v2_sub08.sbatch")
echo "{\"pipeline\":\"nda_v2\",\"job\":\"${JOB}\",\"output\":\"${OUT}\",\"submitted\":\"$(date -Iseconds)\"}" | tee "${OUT}/pipeline_submit.json"
echo "Submitted job ${JOB}"
