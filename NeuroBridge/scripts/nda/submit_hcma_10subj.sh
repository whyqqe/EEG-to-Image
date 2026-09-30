#!/usr/bin/env bash
set -euo pipefail
NB_ROOT=/project/peilab/why/NeuroBridge
OUT="${NB_ROOT}/outputs/hcma_10subj"
mkdir -p "${OUT}" "${NB_ROOT}/outputs/slurm"
JOB=$(sbatch --parsable "${NB_ROOT}/slurm/hcma_10subj.sbatch")
echo "{\"pipeline\":\"HCMA-10subj\",\"job\":\"${JOB}\",\"output\":\"${OUT}\",\"submitted\":\"$(date -Iseconds)\",\"protocol\":\"cross_subj_from_scratch+per_subj_FT_delta\",\"init_ckpt\":null,\"metrics\":[\"clip_2way_vitl\",\"erdc_2wc\",\"erdc_full\",\"fid\",\"ssim\",\"class\"],\"compare\":\"auto_select_semantic\"}" | tee "${OUT}/pipeline_submit.json"
echo "Submitted job ${JOB}"
