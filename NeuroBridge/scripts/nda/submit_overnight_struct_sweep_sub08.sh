#!/usr/bin/env bash
set -euo pipefail
NB_ROOT=/project/peilab/why/NeuroBridge
OUT="${NB_ROOT}/outputs/overnight_struct_sweep/sub-08"
mkdir -p "${OUT}" "${NB_ROOT}/outputs/slurm"
JOB=$(sbatch --parsable "${NB_ROOT}/slurm/overnight_struct_sweep_sub08.sbatch")
echo "{\"pipeline\":\"overnight_struct_sweep\",\"job\":\"${JOB}\",\"output\":\"${OUT}\",\"submitted\":\"$(date -Iseconds)\",\"no_train\":true,\"families\":[\"alpha_fuse\",\"sal_fuse\",\"sal_floor\",\"freq\",\"cn_pf\"],\"assets\":[\"a40\",\"tcda_pc\",\"tcda_r\",\"r_post\",\"ll\",\"ts_ll\"]}" | tee "${OUT}/pipeline_submit.json"
echo "Submitted job ${JOB}"
