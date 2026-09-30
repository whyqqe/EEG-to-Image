#!/usr/bin/env bash
set -euo pipefail
NB_ROOT=/project/peilab/why/NeuroBridge
OUT="${NB_ROOT}/outputs/cn_ip_decode/sub-08"
mkdir -p "${OUT}" "${NB_ROOT}/outputs/slurm"
JOB=$(sbatch --parsable "${NB_ROOT}/slurm/cn_ip_decode_sub08.sbatch")
echo "{\"pipeline\":\"cn_ip_decode\",\"job\":\"${JOB}\",\"output\":\"${OUT}\",\"submitted\":\"$(date -Iseconds)\",\"fix\":\"decoder=SDXL ControlNet-Canny + NDA-SS IP + CPA\",\"note\":\"do not wait; check summary.json when done\"}" | tee "${OUT}/pipeline_submit.json"
echo "Submitted job ${JOB}"
