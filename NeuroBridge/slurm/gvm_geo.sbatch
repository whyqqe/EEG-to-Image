#!/bin/bash
#SBATCH --job-name=gvm_geo
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/gvm_geo_%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/gvm_geo_%j.err

set -euo pipefail
cd /project/peilab/why/NeuroBridge
export NB_ROOT=/project/peilab/why/NeuroBridge
export DEVICE=cuda:0

echo "[slurm] gem_geo ${SLURM_JOB_ID:-local} on $(hostname) gpu ${CUDA_VISIBLE_DEVICES:-?}"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
bash scripts/nda/run_gem_geo.sh
echo "[slurm] gem_geo finished @ $(date -Iseconds)"
