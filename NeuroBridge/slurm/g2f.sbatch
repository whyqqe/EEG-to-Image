#!/usr/bin/env bash
#SBATCH --job-name=g2f
#SBATCH --account=peilab
# `normal`, not `preempt`: this is ONE job that must run to the end, and the
# preempt partition would interrupt it repeatedly. The earlier
# QOSMaxSubmitJobPerUserLimit was caused by submitting an array plus a dependent
# job simultaneously; a single submission cannot hit it.
#SBATCH --partition=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/g2f-%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/g2f-%j.err

# Single job, full chain:
#   train 10 LOSO folds (+ sub-08 intra)
#   -> generate the controlled prompt/anchor rows
#   -> one evaluation pass producing standard-7 and pooled FID for our rows AND
#      for the already-generated sdedit_ll folds, so the reference number comes
#      from the same code on the same GT cache.
set -euo pipefail
export NB_ROOT=/project/peilab/why/NeuroBridge
export DEVICE=cuda:0
export EPOCHS="${EPOCHS:-26}"
export BATCH="${BATCH:-512}"
export OUT="${OUT:-${NB_ROOT}/outputs/g2f}"

bash "${NB_ROOT}/scripts/nda/run_g2f.sh"
