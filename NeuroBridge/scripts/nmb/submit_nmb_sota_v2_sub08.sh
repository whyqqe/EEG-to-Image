#!/usr/bin/env bash
# Submit DA-HLM v2 SOTA冲击流水线（ERDC + S2 + S2 eval，带依赖链）
set -euo pipefail

NB_ROOT=/project/peilab/why/NeuroBridge
SLURM_DIR="${NB_ROOT}/slurm"
OUT="${NB_ROOT}/outputs/nb_nmb_sota_v2/sub-08"
mkdir -p "${OUT}" "${NB_ROOT}/outputs/slurm"

echo "===== Disk before submit ====="
df -h /project/peilab/why | tail -1

JOB_ERDC=$(sbatch --parsable "${SLURM_DIR}/nb_nmb_sota_v2_erdc.sbatch")
echo "[submit] ERDC track  job=${JOB_ERDC}"

JOB_S2=$(sbatch --parsable "${SLURM_DIR}/nb_nmb_sota_v2_s2.sbatch")
echo "[submit] S2 train    job=${JOB_S2}"

JOB_S2E=$(sbatch --parsable --dependency=afterok:"${JOB_S2}" "${SLURM_DIR}/nb_nmb_sota_v2_s2_eval.sbatch")
echo "[submit] S2 eval     job=${JOB_S2E} (after S2)"

MANIFEST="${OUT}/pipeline_submit.json"
cat > "${MANIFEST}" <<EOF
{
  "submitted_at": "$(date -Iseconds)",
  "pipeline": "NMB-SOTA-v2",
  "subject": "sub-08",
  "output_root": "${OUT}",
  "jobs": {
    "erdc": "${JOB_ERDC}",
    "s2_train": "${JOB_S2}",
    "s2_eval": "${JOB_S2E}"
  },
  "tracks": {
    "E_erdc": "pair-fuse existing overnight + RAS bank (pruned)",
    "S2": "DINOv2 targets + dual-teacher decode align finetune",
    "S2_eval": "blend + focused gen + ERDC vs v1"
  },
  "disk_policy": "no fusion_prior finetune; max 4 gen paths; prune RAS candidate PNGs"
}
EOF

echo "[OK] manifest -> ${MANIFEST}"
squeue -u "$USER" -o "%.10i %.12j %.8T %.10M %.6D %R" | head -20
