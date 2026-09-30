#!/usr/bin/env bash
set -euo pipefail
NB_ROOT=/project/peilab/why/NeuroBridge
OUT="${NB_ROOT}/outputs/tcda/sub-08"
mkdir -p "${OUT}" "${NB_ROOT}/outputs/slurm"
source "${NB_ROOT}/scripts/nmb/nmb_sota_v2_env.sh" 2>/dev/null || true
export PYTHON="${PYTHON:-python}"
if ! "${PYTHON}" -c "import torch,numpy,PIL" 2>/dev/null; then
  echo "[WARN] env import check failed; submitting anyway"
fi
JOB=$(sbatch --parsable "${NB_ROOT}/slurm/tcda_sub08.sbatch")
echo "{\"pipeline\":\"TCDA\",\"job\":\"${JOB}\",\"output\":\"${OUT}\",\"submitted\":\"$(date -Iseconds)\",\"channels\":[\"S_frozen_a40\",\"P_multi_Pc_Pf\",\"R_saliency\"],\"injection\":[\"i2i\",\"freq\",\"sal_fuse\",\"cn_pf\"],\"gate\":\"2way_fid_vs_a40\"}" | tee "${OUT}/pipeline_submit.json"
echo "Submitted job ${JOB}"
