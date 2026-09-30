#!/bin/bash
#SBATCH --job-name=ack_s08
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/ack_s08_%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/ack_s08_%j.err

set -euo pipefail
cd /project/peilab/why/NeuroBridge
export NB_ROOT=/project/peilab/why/NeuroBridge
export DEVICE=cuda:0
export KEEP_IMAGES=0
export ACK_OUT=/project/peilab/why/NeuroBridge/outputs/ack_s08
export UCK_OUT=/project/peilab/why/NeuroBridge/outputs/uck
export EPOCHS="${EPOCHS:-20}"

echo "[slurm] ACK-DT sub-08 ${SLURM_JOB_ID:-local} on $(hostname) gpu ${CUDA_VISIBLE_DEVICES:-?} out=${ACK_OUT}"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
bash scripts/nda/run_ack_sub08.sh
echo "[slurm] finished rc=$? @ $(date -Iseconds)"
