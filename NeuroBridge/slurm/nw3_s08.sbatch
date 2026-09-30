#!/bin/bash
#SBATCH --job-name=nw3_s08
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/nw3_s08_%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/nw3_s08_%j.err
#
# TIME BUDGET (sub-08, single GPU)
#   V0 seven on init                 ~5 min
#   S1 factorized 30 ep              ~25-40 min
#   S2 prior 40 ep × 4 mods          ~40-60 min
#   S3 fusion 30 ep                  ~15-25 min
#   V1 gen 6 arms × 200 @ 28 steps   ~40-60 min
#   seven + spatial cycle × 6        ~40-50 min
#   => ~3-4.5 h typical; 12 h wall for slow nodes / resume.

set -euo pipefail
cd /project/peilab/why/NeuroBridge
export NB_ROOT=/project/peilab/why/NeuroBridge
export DEVICE=cuda:0
export NW3_OUT="${NW3_OUT:-${NB_ROOT}/outputs/nw3/sub-08}"
export NW3_BOOTSTRAP="${NW3_BOOTSTRAP:-0}"
export S1_EPOCHS="${S1_EPOCHS:-30}"
export S2_EPOCHS="${S2_EPOCHS:-40}"
export S3_EPOCHS="${S3_EPOCHS:-30}"

mkdir -p "${NB_ROOT}/outputs/slurm" "${NW3_OUT}/logs"
bash scripts/nda/run_nw3_sub08.sh 2>&1 | tee "${NW3_OUT}/logs/orchestrator_${SLURM_JOB_ID:-local}.log"
