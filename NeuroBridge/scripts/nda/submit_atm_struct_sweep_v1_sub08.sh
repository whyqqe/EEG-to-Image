#!/usr/bin/env bash
set -euo pipefail
NB_ROOT=/project/peilab/why/NeuroBridge
OUT="${NB_ROOT}/outputs/atm_struct_sweep_v1/sub-08"
mkdir -p "${OUT}" "${NB_ROOT}/outputs/slurm"
JOB=$(sbatch --parsable "${NB_ROOT}/slurm/atm_struct_sweep_v1_sub08.sbatch")
echo "{\"pipeline\":\"atm_struct_sweep_v1\",\"job\":\"${JOB}\",\"output\":\"${OUT}\",\"submitted\":\"$(date -Iseconds)\",\"goal\":\"maximize SSIM/Pix under strict HCMA semantic gate\",\"focus\":[\"Pc strength 0.78-0.86\",\"fewer steps 16/20\",\"lower CFG 3.5/4.0\",\"LL/blend transfer\"],\"prior\":\"sdedit_ll_s082 gated; sdedit_pc_s082 SSIM0.282 near-miss\"}" \
  | tee "${OUT}/pipeline_submit.json"
echo "Submitted job ${JOB}"
