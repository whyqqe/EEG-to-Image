#!/bin/bash
#SBATCH --job-name=gvm_geo_test
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/gvm_geo_test_%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/gvm_geo_test_%j.err

set -euo pipefail
cd /project/peilab/why/NeuroBridge
export NB_ROOT=/project/peilab/why/NeuroBridge
export DEVICE=cuda:0
bash scripts/nda/gvm_geo_selftest.sh
