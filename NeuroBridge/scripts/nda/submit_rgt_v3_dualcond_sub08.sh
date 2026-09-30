#!/usr/bin/env bash
set -euo pipefail
NB_ROOT=/project/peilab/why/NeuroBridge
OUT="${NB_ROOT}/outputs/rgt_v3_dualcond/sub-08"
mkdir -p "${OUT}" "${NB_ROOT}/outputs/slurm"
JOB=$(sbatch --parsable "${NB_ROOT}/slurm/rgt_v3_dualcond_sub08.sbatch")
echo "{\"pipeline\":\"rgt_v3_dualcond\",\"job\":\"${JOB}\",\"output\":\"${OUT}\",\"submitted\":\"$(date -Iseconds)\",\"focus\":[\"dual image+text cond\",\"clean fusion\",\"EEG-text multi-granularity\"]}" | tee "${OUT}/pipeline_submit.json"
echo "Submitted job ${JOB}"
