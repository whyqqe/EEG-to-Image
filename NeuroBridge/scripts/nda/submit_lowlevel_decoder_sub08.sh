#!/usr/bin/env bash
set -euo pipefail
NB_ROOT=/project/peilab/why/NeuroBridge
OUT="${NB_ROOT}/outputs/lowlevel_decoder/sub-08"
mkdir -p "${OUT}" "${NB_ROOT}/outputs/slurm"
JOB=$(sbatch --parsable "${NB_ROOT}/slurm/lowlevel_decoder_sub08.sbatch")
echo "{\"pipeline\":\"lowlevel_decoder\",\"job\":\"${JOB}\",\"output\":\"${OUT}\",\"submitted\":\"$(date -Iseconds)\",\"design\":\"EEG→SDXL-VAE + MindEye fuse / img2img\",\"disk\":\"float16 VAE cache; drop train latents after train; prune gen top-2\"}" | tee "${OUT}/pipeline_submit.json"
echo "Submitted job ${JOB}"
