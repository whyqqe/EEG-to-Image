#!/usr/bin/env bash
set -euo pipefail
NB_ROOT=/project/peilab/why/NeuroBridge
OUT="${NB_ROOT}/outputs/alex2_sota_official/sub-08"
mkdir -p "${OUT}" "${NB_ROOT}/outputs/slurm"
JOB=$(sbatch --parsable "${NB_ROOT}/slurm/alex2_sota_official_sub08.sbatch")
echo "{\"pipeline\":\"alex2_sota_official\",\"job\":\"${JOB}\",\"output\":\"${OUT}\",\"submitted\":\"$(date -Iseconds)\",\"eval\":\"ATM/MindEye/CogCap official seven\",\"goal\":\"Alex2→ATM under CLIP/A5/Inc/SwAV/FID gate\",\"methods\":[\"eeg_alex_mid\",\"atm_img2img_pc_ll\",\"alex_gated_luma\",\"combo_baseline\"]}" | tee "${OUT}/pipeline_submit.json"
echo "Submitted job ${JOB}"
