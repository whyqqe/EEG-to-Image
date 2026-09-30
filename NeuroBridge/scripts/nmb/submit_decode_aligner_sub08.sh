#!/usr/bin/env bash
set -euo pipefail
NB_ROOT=/project/peilab/why/NeuroBridge
OUT="${NB_ROOT}/outputs/nb_decode_aligner/sub-08"
mkdir -p "${OUT}" "${NB_ROOT}/outputs/slurm"
df -h /project/peilab/why | tail -1
JOB=$(sbatch --parsable "${NB_ROOT}/slurm/nb_decode_aligner_sub08.sbatch")
echo "[submit] DecodeAligner job=${JOB}"
cat > "${OUT}/pipeline_submit.json" <<EOF
{
  "submitted_at": "$(date -Iseconds)",
  "pipeline": "DecodeAligner",
  "job": "${JOB}",
  "output": "${OUT}"
}
EOF
squeue -u "$USER" -o "%.10i %.12j %.8T %.10M %R" | head -8
