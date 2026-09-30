#!/bin/bash
#SBATCH --job-name=nw4_pre
#SBATCH --partition=cpu
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/nw4_pre_%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/nw4_pre_%j.err
#
# Runs the full NW4 preflight + CPU smoke OFF the login node.
# `--check-only` so it validates and never submits the GPU job itself.

set -uo pipefail
cd /project/peilab/why/NeuroBridge
export NB_ROOT=/project/peilab/why/NeuroBridge
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
mkdir -p outputs/slurm
bash scripts/nda/submit_nw4_sub08.sh --check-only
