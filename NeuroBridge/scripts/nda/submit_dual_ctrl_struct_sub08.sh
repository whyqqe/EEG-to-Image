#!/usr/bin/env bash
set -euo pipefail
NB_ROOT=/project/peilab/why/NeuroBridge
OUT="${NB_ROOT}/outputs/dual_ctrl_struct/sub-08"
mkdir -p "${OUT}" "${NB_ROOT}/outputs/slurm"
JOB=$(sbatch --parsable "${NB_ROOT}/slurm/dual_ctrl_struct_sub08.sbatch")
echo "{\"pipeline\":\"dual_ctrl_struct\",\"job\":\"${JOB}\",\"output\":\"${OUT}\",\"submitted\":\"$(date -Iseconds)\",\"goal\":\"lift SSIM/PixCorr under semantic gate\",\"method\":\"frozen SDXL + mild/gated Pc/LL img2img + semantic-recovery luma\",\"no_unet_ft\":true}" \
  | tee "${OUT}/pipeline_submit.json"
echo "Submitted job ${JOB}"
