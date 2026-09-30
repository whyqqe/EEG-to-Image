#!/usr/bin/env bash
# Submit the strict pure-intra HCMA-S (sub-08) experiment.
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
LOG_DIR="${NB_ROOT}/outputs/slurm"
mkdir -p "${LOG_DIR}"

JOBID=$(sbatch "${NB_ROOT}/slurm/intra_hcma_s_sub08.sbatch" | awk '{print $4}')
echo "${JOBID}" > "${NB_ROOT}/outputs/intra_hcma_s_sub08.jobid" 2>/dev/null || true

cat <<EOF > "${LOG_DIR}/intra_hcma_s_sub08.submit.json"
{
  "pipeline": "intra_hcma_s_sub08",
  "job": "${JOBID}",
  "submitted": "$(date -Iseconds)",
  "script": "${NB_ROOT}/slurm/intra_hcma_s_sub08.sbatch",
  "goal": "STRICT pure-intra HCMA-S sub-08: all EEG-side project weights trained on sub-08 only (EEGProject->NDA-v2 dual->VAE/Depth heads); image-side pretrained (CLIP/DINO/SDXL/Depth/RN50 gallery) reused. Compare vs cross-subject SOTA rows.",
  "note": "NO SharedSpecificEncoder / RGT bank / hcma_10subj MG-Flow / nda_ss SS EEG features."
}
EOF
echo "[SUBMITTED] job ${JOBID}"
squeue -u "${USER}" | grep "${JOBID}" || true
