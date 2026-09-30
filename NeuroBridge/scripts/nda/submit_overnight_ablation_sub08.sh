#!/usr/bin/env bash
# Submit overnight ablation AFTER top1_structure job (default 546654) succeeds.
set -euo pipefail
NB_ROOT=/project/peilab/why/NeuroBridge
OUT="${NB_ROOT}/outputs/overnight_ablation/sub-08"
DEP_JOB="${DEP_JOB:-546654}"
mkdir -p "${OUT}" "${NB_ROOT}/outputs/slurm"

SBATCH_ARGS=(--parsable)
if [[ -n "${DEP_JOB}" ]]; then
  # afterok: only start if dependency completed successfully
  SBATCH_ARGS+=(--dependency="afterok:${DEP_JOB}")
fi

JOB=$(sbatch "${SBATCH_ARGS[@]}" "${NB_ROOT}/slurm/overnight_ablation_sub08.sbatch")
echo "{\"pipeline\":\"overnight_ablation\",\"job\":\"${JOB}\",\"depends_on\":\"${DEP_JOB}\",\"output\":\"${OUT}\",\"submitted\":\"$(date -Iseconds)\",\"disk_policy\":\"prune PNGs keep top-2; symlink hybrid; no GT rebuild; drop ablate embeds\",\"mechanisms\":[\"fuse\",\"gated_clean\",\"hybrid_depth\",\"vith_depth_head\",\"fullstack\",\"clean_heavy_lambda\"]}" | tee "${OUT}/pipeline_submit.json"
echo "Submitted job ${JOB} (dependency afterok:${DEP_JOB})"
