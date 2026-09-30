#!/usr/bin/env bash
# Submit the LG-Gate (learnable structural gate) sub-08 overnight experiment.
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
LOG_DIR="${NB_ROOT}/outputs/slurm"
mkdir -p "${LOG_DIR}"

JOBID=$(sbatch "${NB_ROOT}/slurm/lg_gate_sub08.sbatch" | awk '{print $4}')
echo "${JOBID}" > "${NB_ROOT}/outputs/lg_gate_sub08.jobid" 2>/dev/null || true

cat <<EOF > "${LOG_DIR}/lg_gate_sub08.submit.json"
{
  "pipeline": "lg_gate_sub08",
  "job": "${JOBID}",
  "submitted": "$(date -Iseconds)",
  "script": "${NB_ROOT}/slurm/lg_gate_sub08.sbatch",
  "goal": "Learnable structural gate on HCMA-S sub-08: EEG->u router (K-fold OOF depth-quality labels, sub-08 only) sets per-sample CN scale; high-u => HCMA-S dual path, low-u => sdedit-LL semantic-safe fallback. Targets: beat sdedit_ll_intra AND best intra grid cell on all standard-7 + FID.",
  "rows": ["lg_router_s082", "lg_router_s086", "lg_oracle_s082(diag upper bound)"],
  "reuses": "outputs/intra_hcma_s/sub-08 (pure-intra semantic embed, VAE-LL, Depth-CN, prompts)",
  "note": "New trained weights (fold depth heads + router) are sub-08 ONLY."
}
EOF
echo "[SUBMITTED] job ${JOBID}"
squeue -u "${USER}" | grep "${JOBID}" || true
