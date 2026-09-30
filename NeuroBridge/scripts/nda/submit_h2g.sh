#!/usr/bin/env bash
# Submit the HCMA-2G (granularity axis) pipeline for sub-08.
set -euo pipefail
NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
mkdir -p "${NB_ROOT}/outputs/slurm"
JID=$(sbatch --parsable "${NB_ROOT}/slurm/h2g_sub08.sbatch")
echo "submitted h2g-sub08 job: ${JID}"
echo "${JID}" > "${NB_ROOT}/outputs/h2g_sub08.jobid"
echo "monitor:  squeue -j ${JID}"
echo "log:      tail -f ${NB_ROOT}/outputs/slurm/h2g-sub08-${JID}.out"
