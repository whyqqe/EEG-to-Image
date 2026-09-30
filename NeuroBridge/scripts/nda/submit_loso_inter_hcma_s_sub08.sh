#!/usr/bin/env bash
# Submit the LOSO-inter sub-08 companion job (depends on intra job 564261).
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
LOG_DIR="${NB_ROOT}/outputs/slurm"
mkdir -p "${LOG_DIR}"

JOBID=$(sbatch "${NB_ROOT}/slurm/loso_inter_hcma_s_sub08.sbatch" | awk '{print $4}')
echo "${JOBID}" > "${NB_ROOT}/outputs/loso_inter_hcma_s_sub08.jobid" 2>/dev/null || true

cat <<EOF > "${LOG_DIR}/loso_inter_hcma_s_sub08.submit.json"
{
  "pipeline": "loso_inter_hcma_s_sub08",
  "job": "${JOBID}",
  "depends_on": "564261 (intra structure heads, afterany)",
  "submitted": "$(date -Iseconds)",
  "goal": "Controlled LOSO-inter: semantic = LOSO fold-08 MG-Flow pure-forward (9-subj pretrain, sub-08 unseen); structure = intra sub-08 heads from 564261. Quantify inter-semantic vs intra-semantic on identical structure towers.",
  "semantic_embed": "${NB_ROOT}/outputs/inter_ll_full10/sub-08/inter_embeds/embeds/blend_nda_cfm_f_a40_test.npy"
}
EOF
echo "[SUBMITTED] job ${JOBID} (after 564261)"
squeue -u "${USER}" | grep -E "${JOBID}|564261" || true
