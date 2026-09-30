#!/usr/bin/env bash
set -euo pipefail
NB_ROOT=/project/peilab/why/NeuroBridge
OUT="${NB_ROOT}/outputs/top1_structure/sub-08"
mkdir -p "${OUT}" "${NB_ROOT}/outputs/slurm"
JOB=$(sbatch --parsable "${NB_ROOT}/slurm/top1_structure_sub08.sbatch")
echo "{\"pipeline\":\"top1_structure\",\"job\":\"${JOB}\",\"output\":\"${OUT}\",\"submitted\":\"$(date -Iseconds)\",\"tracks\":[\"T dual clean+CPA finetune\",\"S EEG→Depth + COCA\",\"merge gen ablations\"],\"plan\":\"docs/TOP1_STRUCTURE_PLAN.md\"}" | tee "${OUT}/pipeline_submit.json"
echo "Submitted job ${JOB}"
