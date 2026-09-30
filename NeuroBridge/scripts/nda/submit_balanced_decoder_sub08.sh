#!/usr/bin/env bash
set -euo pipefail
NB_ROOT=/project/peilab/why/NeuroBridge
OUT="${NB_ROOT}/outputs/balanced_decoder/sub-08"
mkdir -p "${OUT}" "${NB_ROOT}/outputs/slurm"
JOB=$(sbatch --parsable "${NB_ROOT}/slurm/balanced_decoder_sub08.sbatch")
echo "{\"pipeline\":\"balanced_decoder\",\"job\":\"${JOB}\",\"output\":\"${OUT}\",\"submitted\":\"$(date -Iseconds)\",\"clip_metric\":\"2-way\",\"methods\":[\"freq_fuse\",\"twostage_refine\"],\"compare_grid\":true}" | tee "${OUT}/pipeline_submit.json"
echo "Submitted job ${JOB}"
