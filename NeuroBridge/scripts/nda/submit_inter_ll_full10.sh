#!/usr/bin/env bash
# Submit inter_ll_full10 (pure zero-shot inter-subject semantic decode, NO per-subject FT).
# Goal: quantify FT-vs-zero-shot delta on the standard-7 semantic metrics + pooled FID.
set -euo pipefail
NB_ROOT=/project/peilab/why/NeuroBridge
OUT="${NB_ROOT}/outputs/inter_ll_full10"
mkdir -p "${OUT}" "${NB_ROOT}/outputs/slurm"
JOB=$(sbatch --parsable "${NB_ROOT}/slurm/inter_ll_full10.sbatch")
echo "{\"pipeline\":\"inter_ll_full10\",\"job\":\"${JOB}\",\"output\":\"${OUT}\",\"submitted\":\"$(date -Iseconds)\",\"design\":\"LOSO fold MG-Flow ckpt pure forward (9-subject pretrain, NO subject FT) + sdedit_ll decode; compare vs sdedit_ll_full10 FT\",\"question\":\"does dropping per-subject FT keep SOTA semantics?\"}" \
  | tee "${OUT}/pipeline_submit.json"
echo "Submitted job ${JOB}"
