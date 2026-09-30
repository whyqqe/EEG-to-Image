#!/usr/bin/env bash
set -euo pipefail
NB_ROOT=/project/peilab/why/NeuroBridge
OUT="${NB_ROOT}/outputs/rcfm_ll/sub-08"
mkdir -p "${OUT}" "${NB_ROOT}/outputs/slurm"
source "${NB_ROOT}/scripts/nmb/nmb_sota_v2_env.sh" 2>/dev/null || true
export PYTHON="${PYTHON:-python}"
JOB=$(sbatch --parsable "${NB_ROOT}/slurm/rcfm_ll_sub08.sbatch")
echo "{\"pipeline\":\"R-CFM-LL\",\"job\":\"${JOB}\",\"output\":\"${OUT}\",\"submitted\":\"$(date -Iseconds)\",\"recipe\":\"L1_mu+residual_CondCFM→blur→HCMA_SDEdit\",\"gate\":\"strict_hcma\",\"frozen_semantic\":\"hcma_full_a40\"}" \
  | tee "${OUT}/pipeline_submit.json"
echo "Submitted job ${JOB}"
