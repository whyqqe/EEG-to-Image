#!/bin/bash
#SBATCH --job-name=nw4_facts
#SBATCH --partition=cpu
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/nw4_facts_%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/nw4_facts_%j.err
#
# Facts NW4's design depends on (trial structure, bank ordering).  Pure numpy.

set -euo pipefail
cd /project/peilab/why/NeuroBridge
export NB_ROOT=/project/peilab/why/NeuroBridge
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

mkdir -p "${NB_ROOT}/outputs/slurm"
python3 scripts/nda/nw4_preflight_facts.py
