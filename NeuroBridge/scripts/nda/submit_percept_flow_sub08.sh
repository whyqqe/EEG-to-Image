#!/usr/bin/env bash
set -euo pipefail
NB_ROOT=/project/peilab/why/NeuroBridge
OUT="${NB_ROOT}/outputs/percept_flow/sub-08"
mkdir -p "${OUT}" "${NB_ROOT}/outputs/slurm"
source "${NB_ROOT}/scripts/nmb/nmb_sota_v2_env.sh" 2>/dev/null || true
export PYTHON="${PYTHON:-python}"
if ! "${PYTHON}" -c "import torch,lpips,numpy,diffusers" 2>/dev/null; then
  echo "[WARN] env import check failed; submitting anyway (sbatch env will load)"
fi
JOB=$(sbatch --parsable "${NB_ROOT}/slurm/percept_flow_sub08.sbatch")
echo "{\"pipeline\":\"PerceptFlow\",\"job\":\"${JOB}\",\"output\":\"${OUT}\",\"submitted\":\"$(date -Iseconds)\",\"methods\":[\"vae_head\",\"depth_head\",\"spatial_cond_cfm\",\"lpips_ssim\",\"i2i_low_s\",\"freq_fuse\",\"weak_depth_cn\"],\"frozen_semantic\":\"mg_blend_a40_dual\"}" | tee "${OUT}/pipeline_submit.json"
echo "Submitted job ${JOB}"
