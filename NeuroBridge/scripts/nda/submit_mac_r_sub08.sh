#!/usr/bin/env bash
set -euo pipefail
NB_ROOT=/project/peilab/why/NeuroBridge
OUT="${NB_ROOT}/outputs/mac_r/sub-08"
mkdir -p "${OUT}" "${NB_ROOT}/outputs/slurm"
JOB=$(sbatch --parsable "${NB_ROOT}/slurm/mac_r_sub08.sbatch")
echo "{\"pipeline\":\"mac_r_p1\",\"job\":\"${JOB}\",\"output\":\"${OUT}\",\"submitted\":\"$(date -Iseconds)\",\"modules\":[\"NDA-SS mem⊕decode\",\"CPA prompts\",\"confidence router\",\"stage-wise assembly\",\"optional LL fuse\"]}" | tee "${OUT}/pipeline_submit.json"
echo "Submitted job ${JOB}"
