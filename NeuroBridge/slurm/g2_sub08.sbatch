#!/usr/bin/env bash
#SBATCH --job-name=g2-sub08
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/g2-sub08-%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/g2-sub08-%j.err

set -euo pipefail
export NB_ROOT=/project/peilab/why/NeuroBridge
export DEVICE=cuda:0
export SUBJECT=8
export EPOCHS="${EPOCHS:-60}"
export STAGES="${STAGES:-012345}"

# The pipeline script handles its own sub-environments: the caption stage runs in an
# isolated venv (transformers 4.57 + the user's default HF cache, where the
# Qwen2.5-VL weights actually live), the rest uses the shared pipeline env.
bash "${NB_ROOT}/scripts/nda/run_g2_pipeline.sh"
