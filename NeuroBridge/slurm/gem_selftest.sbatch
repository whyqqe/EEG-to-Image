#!/bin/bash
#SBATCH --job-name=gem_test
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/gem_test_%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/gem_test_%j.err

# GPU shape/logic validation of the whole GEM path on REAL data at batch size 8.
# Short by design: 40 minutes is enough for the full path plus the theta grid and
# every export, and short enough that a defective run does not hold a node.
set -euo pipefail
cd /project/peilab/why/NeuroBridge
export NB_ROOT=/project/peilab/why/NeuroBridge
export DEVICE=cuda:0

echo "[slurm] selftest job ${SLURM_JOB_ID:-local} on $(hostname) gpu ${CUDA_VISIBLE_DEVICES:-?}"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
bash scripts/nda/gem_selftest.sh
echo "[slurm] selftest finished @ $(date -Iseconds)"
