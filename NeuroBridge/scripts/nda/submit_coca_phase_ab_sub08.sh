#!/usr/bin/env bash
set -euo pipefail
NB_ROOT=/project/peilab/why/NeuroBridge
OUT="${NB_ROOT}/outputs/coca_depth/sub-08"
mkdir -p "${OUT}" "${NB_ROOT}/outputs/slurm"
JOB=$(sbatch --parsable "${NB_ROOT}/slurm/coca_phase_ab_sub08.sbatch")
echo "{\"pipeline\":\"coca_phase_ab\",\"job\":\"${JOB}\",\"output\":\"${OUT}\",\"submitted\":\"$(date -Iseconds)\",\"phases\":[\"A clean vs CPA Top-1\",\"A paper SSIM\",\"B1 Depth-ControlNet\"],\"cleaned\":[\"mac_r\",\"oracle_chase_v1\",\"tmp smokes\"]}" | tee "${OUT}/pipeline_submit.json"
echo "Submitted job ${JOB}"
