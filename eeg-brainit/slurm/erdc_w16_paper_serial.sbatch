#!/usr/bin/env bash
#SBATCH --job-name=erdc-w16-all
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=96G
#SBATCH --time=72:00:00
#SBATCH --requeue
#SBATCH --exclude=dgx-10,dgx-11
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

# Single-job paper pipeline (QOS 1 concurrent job):
#   A) sub-08 finalize (2WC / bootstrap / panels / main table)
#   B) 10-subject per-subject S1→S3 + official + ours fuse
#   C) ten-subject table + w16_paper_freeze.md

set -euo pipefail
ROOT=/project/peilab/why/eeg-brainit
module load slurm nvhpc-hpcx-cuda12/23.11
cd "${ROOT}"
source scripts/activate.sh
export PYTHONUNBUFFERED=1
export HF_HOME=/project/peilab/why/cache/eeg-brainit/hf
export HF_HUB_CACHE=/project/peilab/why/cache/eeg-brainit/hf/hub
export OPENCLIP_CACHE_DIR=/project/peilab/why/cache/eeg-brainit/open_clip
export TORCH_HOME=/project/peilab/why/cache/eeg-brainit/torch
export TMPDIR=/project/peilab/why/cache/eeg-brainit/tmp
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export XFORMERS_DISABLED=1
mkdir -p "$TMPDIR" outputs/erdc/w15_metrics outputs/erdc/w16_loso_metrics outputs/slurm
pip install -q torch-fidelity 2>/dev/null || true

echo "################ [A] sub-08 finalize ################"
bash scripts/erdc_paper_finalize.sh

echo "################ [B] 10-subject serial LOSO ################"
bash scripts/erdc_loso_serial.sh

echo "################ [C] final freeze ################"
bash scripts/erdc_paper_finalize.sh

echo "[OK] W16 paper serial finished"
date
