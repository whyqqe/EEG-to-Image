#!/usr/bin/env bash
set -euo pipefail
NB_ROOT=/project/peilab/why/NeuroBridge
OUT="${NB_ROOT}/outputs/nda_ss/sub-08"
mkdir -p "${OUT}" "${NB_ROOT}/outputs/slurm"
JOB=$(sbatch --parsable "${NB_ROOT}/slurm/nda_ss_sub08.sbatch")
echo "{\"pipeline\":\"nda_ss\",\"job\":\"${JOB}\",\"output\":\"${OUT}\",\"submitted\":\"$(date -Iseconds)\",\"refs\":[\"MindCross\",\"MindBridge\",\"ShaSpec\"],\"note\":\"shared+specific pretrain+calib then NDA dual\"}" | tee "${OUT}/pipeline_submit.json"
echo "Submitted job ${JOB}"
