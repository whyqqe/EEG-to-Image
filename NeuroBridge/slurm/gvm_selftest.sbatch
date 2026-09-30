#!/bin/bash
#SBATCH --job-name=gvm_test
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/gvm_test_%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/gvm_test_%j.err

# GPU shape/logic validation of the GVM path on REAL data, 1 024 rows, 3 epochs, with
# the information-weighted schedule armed at epoch 1.  Short by design: a defective
# run releases the node in under an hour.
set -euo pipefail
cd /project/peilab/why/NeuroBridge
export NB_ROOT=/project/peilab/why/NeuroBridge
export DEVICE=cuda:0

echo "[slurm] GVM selftest job ${SLURM_JOB_ID:-local} on $(hostname) gpu ${CUDA_VISIBLE_DEVICES:-?}"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
bash scripts/nda/gvm_selftest.sh
echo "[slurm] GVM selftest finished @ $(date -Iseconds)"
