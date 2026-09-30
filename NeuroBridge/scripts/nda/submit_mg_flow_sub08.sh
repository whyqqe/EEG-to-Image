#!/usr/bin/env bash
set -euo pipefail
NB_ROOT=/project/peilab/why/NeuroBridge
OUT="${NB_ROOT}/outputs/mg_flow/sub-08"
mkdir -p "${OUT}" "${NB_ROOT}/outputs/slurm"
# quick local sanity: targets build does not need GPU
source "${NB_ROOT}/scripts/nmb/nmb_sota_v2_env.sh" 2>/dev/null || true
export PYTHON="${PYTHON:-python}"
if ! "${PYTHON}" -c "import torch,open_clip,numpy" 2>/dev/null; then
  echo "[WARN] env import check failed; submitting anyway (sbatch env will load)"
fi
JOB=$(sbatch --parsable "${NB_ROOT}/slurm/mg_flow_sub08.sbatch")
echo "{\"pipeline\":\"MG-Flow\",\"job\":\"${JOB}\",\"output\":\"${OUT}\",\"submitted\":\"$(date -Iseconds)\",\"methods\":[\"dual_granularity\",\"hier_cfm\",\"gated_residual\"],\"no_forced_rgt\":true}" | tee "${OUT}/pipeline_submit.json"
echo "Submitted job ${JOB}"
