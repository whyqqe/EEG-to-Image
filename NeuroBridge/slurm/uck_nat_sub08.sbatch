#!/bin/bash
#SBATCH --job-name=uck_nat_s08
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/uck_nat_s08_%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/uck_nat_s08_%j.err

set -euo pipefail
cd /project/peilab/why/NeuroBridge
export NB_ROOT=/project/peilab/why/NeuroBridge
export DEVICE=cuda:0
export KEEP_IMAGES=1
export UCK_NAT_OUT=/project/peilab/why/NeuroBridge/outputs/uck_nat_s08
export UCK_OUT=/project/peilab/why/NeuroBridge/outputs/uck
export K="${K:-8}"
export LAM="${LAM:-0.5}"
export DO_RANDOM="${DO_RANDOM:-1}"

echo "[slurm] UCK-NAT sub-08 ${SLURM_JOB_ID:-local} on $(hostname) K=${K} LAM=${LAM}"
echo "[slurm] shared node co-tenants:"
squeue -w "$(hostname)" -h -o "%i %u %j %M" || true
nvidia-smi --query-gpu=name,memory.total,utilization.gpu --format=csv || true
bash scripts/nda/run_uck_nat_sub08.sh
echo "[slurm] finished rc=$? @ $(date -Iseconds)"
