#!/usr/bin/env bash
set -euo pipefail
NB_ROOT=/project/peilab/why/NeuroBridge
OUT="${NB_ROOT}/outputs/struct_inject_v1/sub-08"
mkdir -p "${OUT}" "${NB_ROOT}/outputs/slurm"
JOB=$(sbatch --parsable "${NB_ROOT}/slurm/struct_inject_v1_sub08.sbatch")
echo "{\"pipeline\":\"struct_inject_v1\",\"job\":\"${JOB}\",\"output\":\"${OUT}\",\"submitted\":\"$(date -Iseconds)\",\"goal\":\"SSIM≥0.28 Pix≥0.15 under semantic gate\",\"method\":\"timed Depth/Canny-CN early + HCMA IP late; ATM mild Pc/LL latent init; frozen SDXL\",\"inspired_by\":[\"ATM img2img low-level\",\"CogCap depth spatial inject\"]}" \
  | tee "${OUT}/pipeline_submit.json"
echo "Submitted job ${JOB}"
