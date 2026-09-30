#!/usr/bin/env bash
set -euo pipefail
NB_ROOT=/project/peilab/why/NeuroBridge
OUT="${NB_ROOT}/outputs/hcma_lite/sub-08"
mkdir -p "${OUT}" "${NB_ROOT}/outputs/slurm"
JOB=$(sbatch --parsable "${NB_ROOT}/slurm/hcma_lite_sub08.sbatch")
echo "{\"pipeline\":\"HCMA-lite\",\"job\":\"${JOB}\",\"output\":\"${OUT}\",\"submitted\":\"$(date -Iseconds)\",\"focus\":\"semantic\",\"methods\":[\"hier_prompts_subj_det_bg\",\"luma_matched_pc_fuse\",\"frozen_a40\"],\"excluded\":[\"saliency_R\",\"pc_img2img\"]}" | tee "${OUT}/pipeline_submit.json"
echo "Submitted job ${JOB}"
