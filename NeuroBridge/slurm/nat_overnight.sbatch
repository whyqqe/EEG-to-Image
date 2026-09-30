#!/bin/bash
#SBATCH --job-name=nat_night
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/nat_night_%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/nat_night_%j.err

set -euo pipefail
cd /project/peilab/why/NeuroBridge
export NB_ROOT=/project/peilab/why/NeuroBridge
export DEVICE=cuda:0
export KEEP_IMAGES="${KEEP_IMAGES:-0}"
export NAT_OUT=/project/peilab/why/NeuroBridge/outputs/nat
export UCK_OUT=/project/peilab/why/NeuroBridge/outputs/uck
export SUBJECTS="${SUBJECTS:-8,1,2,3,4,5,6,7,9,10}"
export EPOCHS="${EPOCHS:-30}"

echo "[slurm] NAT overnight job ${SLURM_JOB_ID:-local} on $(hostname) gpu ${CUDA_VISIBLE_DEVICES:-?} out=${NAT_OUT}"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
bash scripts/nda/run_nat_overnight.sh
echo "[slurm] finished rc=$? @ $(date -Iseconds)"
