#!/usr/bin/env bash
set -euo pipefail
NB_ROOT=/project/peilab/why/NeuroBridge
OUT="${NB_ROOT}/outputs/nb_atm_vith/sub-08"
mkdir -p "${OUT}" "${NB_ROOT}/outputs/slurm"
export RETRAIN="${RETRAIN:-0}"
JOB=$(sbatch --parsable "${NB_ROOT}/slurm/nb_atm_vith_sub08.sbatch")
echo "[submit] NB-ATM job=${JOB}"
cat > "${OUT}/pipeline_submit.json" <<EOF
{
  "submitted_at": "$(date -Iseconds)",
  "pipeline": "NB-ATM-vith-sub08",
  "job": "${JOB}",
  "output": "${OUT}",
  "retrain": "${RETRAIN}"
}
EOF
squeue -u "$USER" -o "%.10i %.12j %.8T %.10M %R" | head -8
