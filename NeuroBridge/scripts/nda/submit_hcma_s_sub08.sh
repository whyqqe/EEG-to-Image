#!/usr/bin/env bash
set -euo pipefail
NB_ROOT=/project/peilab/why/NeuroBridge
OUT="${NB_ROOT}/outputs/hcma_s/sub-08"
mkdir -p "${OUT}" "${NB_ROOT}/outputs/slurm"
source "${NB_ROOT}/scripts/nmb/nmb_sota_v2_env.sh" 2>/dev/null || true
JOB=$(sbatch --parsable "${NB_ROOT}/slurm/hcma_s_sub08.sbatch")
echo "{\"pipeline\":\"HCMA-S\",\"job\":\"${JOB}\",\"output\":\"${OUT}\",\"submitted\":\"$(date -Iseconds)\",\"recipe\":\"retrain_Depth+LL_init+DepthCN_Img2Img\",\"frozen_semantic\":\"hcma_full_a40\",\"grid\":\"cn={0.25,0.32,0.40}×s={0.82,0.86,0.88}\",\"gate\":\"strict_hcma\",\"target_ssim\":0.28}" \
  | tee "${OUT}/pipeline_submit.json"
echo "Submitted job ${JOB}"
