#!/usr/bin/env bash
# Submit the LG-SELECT validation (compose existing grid images per-sample).
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
LOG_DIR="${NB_ROOT}/outputs/slurm"
mkdir -p "${LOG_DIR}"

JOBID=$(sbatch "${NB_ROOT}/slurm/lg_select_sub08.sbatch" | awk '{print $4}')
echo "${JOBID}" > "${NB_ROOT}/outputs/lg_select_sub08.jobid" 2>/dev/null || true

cat <<EOF > "${LOG_DIR}/lg_select_sub08.submit.json"
{
  "pipeline": "lg_select_sub08",
  "job": "${JOBID}",
  "submitted": "$(date -Iseconds)",
  "script": "${NB_ROOT}/slurm/lg_select_sub08.sbatch",
  "goal": "Decisive cheap test of per-sample CN gating: compose EXISTING pure-intra grid images by router/oracle u (no re-gen). Two-level (CN 0<->.40) and four-level at s082, two-level at s086.",
  "rows": ["lgsel_r2_s082","lgsel_o2_s082","lgsel_r4_s082","lgsel_o4_s082","lgsel_r2_s086","lgsel_o2_s086"],
  "interpret": "lgsel_o* is the u_true upper bound (diagnostic only); lgsel_r* is the deployable router result.",
  "reuses": "lg_gate/sub-08/router/{u_hat,u_true}_test.npy + intra_hcma_s/sub-08/generation grid images"
}
EOF
echo "[SUBMITTED] job ${JOBID}"
squeue -u "${USER}" | grep "${JOBID}" || true
