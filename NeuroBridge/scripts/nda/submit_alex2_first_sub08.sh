#!/usr/bin/env bash
set -euo pipefail
NB_ROOT=/project/peilab/why/NeuroBridge
OUT="${NB_ROOT}/outputs/alex2_first/sub-08"
mkdir -p "${OUT}" "${NB_ROOT}/outputs/slurm"
JOB=$(sbatch --parsable "${NB_ROOT}/slurm/alex2_first_sub08.sbatch")
echo "{\"pipeline\":\"alex2_first\",\"job\":\"${JOB}\",\"output\":\"${OUT}\",\"submitted\":\"$(date -Iseconds)\",\"goal\":\"Alex2↑ under CLIP/A5/Inc/SwAV/FID gate; ignore PixCorr/SSIM\",\"ref\":\"hcma_full_a40\",\"alex2_target\":0.776}" | tee "${OUT}/pipeline_submit.json"
echo "Submitted job ${JOB}"
