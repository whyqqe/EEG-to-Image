#!/usr/bin/env bash
set -euo pipefail
NB_ROOT=/project/peilab/why/NeuroBridge
OUT="${NB_ROOT}/outputs/hcma_10subj"
mkdir -p "${OUT}" "${NB_ROOT}/outputs/slurm"
JOB=$(sbatch --parsable "${NB_ROOT}/slurm/hcma_standard_seven.sbatch")
echo "{\"pipeline\":\"HCMA-standard-seven\",\"job\":\"${JOB}\",\"output\":\"${OUT}\",\"submitted\":\"$(date -Iseconds)\",\"metrics\":[\"PixCorr\",\"SSIM\",\"AlexNet2\",\"AlexNet5\",\"Inception\",\"CLIP\",\"SwAV\"],\"protocol\":\"MindEye/ATM/CogCap seven\"}" | tee "${OUT}/standard_seven_submit.json"
echo "Submitted job ${JOB}"
