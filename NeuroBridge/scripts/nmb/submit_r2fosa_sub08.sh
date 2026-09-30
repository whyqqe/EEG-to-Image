#!/usr/bin/env bash
set -euo pipefail
NB_ROOT=/project/peilab/why/NeuroBridge
OUT="${NB_ROOT}/outputs/nb_r2fosa/sub-08"
mkdir -p "${OUT}" "${NB_ROOT}/outputs/slurm"
df -h /project/peilab/why | tail -1
export RETRAIN=1
JOB=$(sbatch --parsable "${NB_ROOT}/slurm/nb_r2fosa_sub08.sbatch")
echo "[submit] R²-FOSA job=${JOB}"
cat > "${OUT}/pipeline_submit.json" <<EOF
{
  "submitted_at": "$(date -Iseconds)",
  "pipeline": "R2-FOSA",
  "no_erdc": true,
  "job": "${JOB}",
  "output": "${OUT}"
}
EOF
squeue -u "$USER" -o "%.10i %.12j %.8T %.10M %R" | head -8
