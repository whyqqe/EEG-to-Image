#!/usr/bin/env bash
set -euo pipefail
NB_ROOT=/project/peilab/why/NeuroBridge
OUT="${NB_ROOT}/outputs/atm_aligned_decode/sub-08"
mkdir -p "${OUT}" "${NB_ROOT}/outputs/slurm"
JOB=$(sbatch --parsable "${NB_ROOT}/slurm/atm_aligned_decode_sub08.sbatch")
echo "{\"pipeline\":\"atm_aligned_decode\",\"job\":\"${JOB}\",\"output\":\"${OUT}\",\"submitted\":\"$(date -Iseconds)\",\"goal\":\"ATM dual-stream decode aligned; protect HCMA semantics\",\"method\":\"HCMA IP/prompts + VAE/LL/Pc SDEdit high-strength (+ optional ATM-exact)\",\"innovation\":\"unchanged — decode interface only\",\"strict_gate\":\"CLIP/A5/Inc≥ref-0.010 FID≤ref+15\"}" \
  | tee "${OUT}/pipeline_submit.json"
echo "Submitted job ${JOB}"
