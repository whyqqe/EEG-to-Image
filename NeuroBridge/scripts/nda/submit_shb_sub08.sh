#!/usr/bin/env bash
# Submit the SHB third-branch experiment (sub-08, overnight).
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
LOG_DIR="${NB_ROOT}/outputs/slurm"
mkdir -p "${LOG_DIR}"

JOBID=$(sbatch "${NB_ROOT}/slurm/shb_sub08.sbatch" | awk '{print $4}')
echo "${JOBID}" > "${NB_ROOT}/outputs/shb_sub08.jobid" 2>/dev/null || true

cat <<EOF > "${LOG_DIR}/shb_sub08.submit.json"
{
  "pipeline": "shb_sub08",
  "job": "${JOBID}",
  "submitted": "$(date -Iseconds)",
  "script": "${NB_ROOT}/slurm/shb_sub08.sbatch",
  "goal": "Third branch (Structural Hypothesis Branch) for HCMA-S sub-08: structure-specialised head off the frozen backbone (objective decoupling), heteroscedastic geometry field at 128^2 with a SPATIAL uncertainty gate on the ControlNet control image, trial-level posterior hypotheses, and image-side verification/selection. Aims to move the structure<->semantics Pareto frontier rather than slide along it (the scalar gate was shown unable to move it, even with oracle u).",
  "rows": ["shb_pt_c040_s086", "shb_sp_c040_s086", "shb_mu_c040_s086", "shb_h1_c040_s086", "shb_h2_c040_s086", "shb_sel_sem_c040_s086", "shb_sel_geo_c040_s086", "shb_sel_cons_c040_s086"],
  "controls": "same semantic IP embed, same VAE-LL SDEdit init, same prompts/seed; only the STRUCTURAL channel changes",
  "gate0": "trial_posterior_report.json tests whether trial-to-trial dispersion is reproducible signal and correlates with structural quality"
}
EOF
echo "[SUBMITTED] job ${JOBID}"
squeue -u "${USER}" | grep "${JOBID}" || true
