#!/usr/bin/env bash
# Submit the BRDT (band-routed dual tower) pipeline for sub-08.
set -euo pipefail
NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
mkdir -p "${NB_ROOT}/outputs/slurm"
JID=$(sbatch --parsable "${NB_ROOT}/slurm/brdt_sub08.sbatch")
echo "submitted brdt-sub08 job: ${JID}"
echo "${JID}" > "${NB_ROOT}/outputs/brdt_sub08.jobid"
echo "monitor:  squeue -j ${JID}"
echo "log:      tail -f ${NB_ROOT}/outputs/slurm/brdt-sub08-${JID}.out"
