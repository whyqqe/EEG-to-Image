#!/usr/bin/env bash
set -euo pipefail
NB_ROOT=/project/peilab/why/NeuroBridge
OUT="${NB_ROOT}/outputs/nda_v2_semtxt/sub-08"
mkdir -p "${OUT}" "${NB_ROOT}/outputs/slurm"
JOB=$(sbatch --parsable "${NB_ROOT}/slurm/nda_v2_semtxt_sub08.sbatch")
echo "{\"pipeline\":\"nda_v2_semtxt\",\"job\":\"${JOB}\",\"output\":\"${OUT}\",\"submitted\":\"$(date -Iseconds)\",\"semantic\":\"CLIP-Image+CLIP-Text\"}" | tee "${OUT}/pipeline_submit.json"
echo "Submitted job ${JOB}"
