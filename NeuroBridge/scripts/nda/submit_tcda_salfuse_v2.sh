#!/usr/bin/env bash
set -euo pipefail
NB_ROOT=/project/peilab/why/NeuroBridge
OUT="${NB_ROOT}/outputs/tcda_salfuse_v2/sub-08"
mkdir -p "${OUT}" "${NB_ROOT}/outputs/slurm"
source "${NB_ROOT}/scripts/nmb/nmb_sota_v2_env.sh" 2>/dev/null || true
JOB=$(sbatch --parsable "${NB_ROOT}/slurm/tcda_salfuse_v2.sbatch")
echo "{\"pipeline\":\"TCDA-salfuse-v2\",\"job\":\"${JOB}\",\"output\":\"${OUT}\",\"submitted\":\"$(date -Iseconds)\",\"focus\":\"sal_fuse\",\"changes\":[\"spectral_saliency_no_center\",\"stronger_w_r\",\"sem_floor_grid\",\"drop_i2i\"]}" | tee "${OUT}/pipeline_submit.json"
echo "Submitted job ${JOB}"
