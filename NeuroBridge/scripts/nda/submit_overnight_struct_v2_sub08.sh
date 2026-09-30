#!/usr/bin/env bash
set -euo pipefail
NB_ROOT=/project/peilab/why/NeuroBridge
OUT="${NB_ROOT}/outputs/overnight_struct_v2/sub-08"
mkdir -p "${OUT}" "${NB_ROOT}/outputs/slurm"
JOB=$(sbatch --parsable "${NB_ROOT}/slurm/overnight_struct_v2_sub08.sbatch")
echo "{\"pipeline\":\"overnight_struct_v2\",\"job\":\"${JOB}\",\"output\":\"${OUT}\",\"submitted\":\"$(date -Iseconds)\",\"goal\":\"lift PixCorr/SSIM while gating CLIP 2-way\",\"mechanisms\":[\"eeg_depth_head\",\"depth_cn\",\"luma_pc_ll\",\"freq_fuse\",\"gated_luma\"],\"ref\":\"hcma_10subj/sub-08/hcma_full_a40\"}" | tee "${OUT}/pipeline_submit.json"
echo "Submitted job ${JOB}"
