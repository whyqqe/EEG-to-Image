#!/usr/bin/env bash
set -euo pipefail
NB_ROOT=/project/peilab/why/NeuroBridge
OUT="${NB_ROOT}/outputs/nb_scc/sub-08"
mkdir -p "${OUT}" "${NB_ROOT}/outputs/slurm"
df -h /project/peilab/why | tail -1
JOB=$(sbatch --parsable "${NB_ROOT}/slurm/nb_scc_sub08.sbatch")
echo "[submit] DA-Calibrator SCC job=${JOB}"
cat > "${OUT}/pipeline_submit.json" <<EOF
{
  "submitted_at": "$(date -Iseconds)",
  "pipeline": "DA-Calibrator-SCC",
  "job": "${JOB}",
  "output": "${OUT}"
}
EOF
squeue -u "$USER" -o "%.10i %.12j %.8T %.10M %R" | head -8
