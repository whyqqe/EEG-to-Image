#!/usr/bin/env bash
set -euo pipefail
NB_ROOT=/project/peilab/why/NeuroBridge
OUT="${NB_ROOT}/outputs/mg_flow_a40_all"
mkdir -p "${OUT}" "${NB_ROOT}/outputs/slurm"
JOB=$(sbatch --parsable "${NB_ROOT}/slurm/mg_flow_a40_all.sbatch")
echo "{\"pipeline\":\"MG-Flow-a40-all\",\"job\":\"${JOB}\",\"output\":\"${OUT}\",\"subjects\":\"1-10\",\"tag\":\"mg_blend_a40_dual\",\"compare\":\"semantic_top12\",\"submitted\":\"$(date -Iseconds)\"}" | tee "${OUT}/pipeline_submit.json"
echo "Submitted job ${JOB}"
