#!/usr/bin/env bash
set -euo pipefail
NB_ROOT=/project/peilab/why/NeuroBridge
OUT="${NB_ROOT}/outputs/scr_decode/sub-08"
mkdir -p "${OUT}" "${NB_ROOT}/outputs/slurm"
JOB=$(sbatch --parsable "${NB_ROOT}/slurm/scr_decode_sub08.sbatch")
echo "{\"pipeline\":\"scr_decode\",\"job\":\"${JOB}\",\"output\":\"${OUT}\",\"submitted\":\"$(date -Iseconds)\",\"fix\":[\"SCR cn/ip routing\",\"paper-grade skimage SSIM\"]}" | tee "${OUT}/pipeline_submit.json"
echo "Submitted job ${JOB}"
