#!/usr/bin/env bash
#SBATCH --job-name=g2-final
#SBATCH --account=peilab
# `normal` rather than `preempt`: this is ONE job that must run to the end, and the
# normal partition has an unlimited time limit and ~21 usable nodes. The earlier
# QOSMaxJobsPerUserLimit problem was caused by submitting an array plus a dependent
# job at once; a single submission cannot hit it.
#SBATCH --partition=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/g2-final-%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/g2-final-%j.err

# Single job, full chain: train (10 LOSO folds + sub-08 intra) -> generate the
# controlled condition rows -> one evaluation pass producing standard-7 and pooled
# FID for our rows AND for the already-generated sdedit_ll reference folds.
set -euo pipefail
export NB_ROOT=/project/peilab/why/NeuroBridge
export DEVICE=cuda:0
export SUBJECTS="${SUBJECTS:-1 2 3 4 5 6 7 8 9 10}"
export EPOCHS="${EPOCHS:-30}"
export OUT="${OUT:-${NB_ROOT}/outputs/g2}"

bash "${NB_ROOT}/scripts/nda/run_g2_final.sh"
