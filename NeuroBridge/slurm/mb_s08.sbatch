#!/bin/bash
#SBATCH --job-name=mb_s08
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/mb_s08_%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/mb_s08_%j.err

set -euo pipefail
cd /project/peilab/why/NeuroBridge
export NB_ROOT=/project/peilab/why/NeuroBridge
export DEVICE=cuda:0
export KEEP_IMAGES=1
export MB_OUT=/project/peilab/why/NeuroBridge/outputs/mb_s08
export UCK_OUT=/project/peilab/why/NeuroBridge/outputs/uck
export K="${K:-8}"
export LAM="${LAM:-0.5}"
export STRUCT_W="${STRUCT_W:-0.5}"
export BRANCH_SCALES="${BRANCH_SCALES:-1.0,1.0,1.0}"
export TURBO_STEPS="${TURBO_STEPS:-5}"
export BASE_STEPS="${BASE_STEPS:-28}"
export SCOPE="${SCOPE:-all}"

echo "[slurm] multibranch sub-08 ${SLURM_JOB_ID:-local} on $(hostname)"
echo "[slurm] K=${K} STRUCT_W=${STRUCT_W} BRANCH_SCALES=${BRANCH_SCALES} SCOPE=${SCOPE}"
echo "[slurm] shared node co-tenants:"
squeue -w "$(hostname)" -h -o "%i %u %j %M" || true
nvidia-smi --query-gpu=name,memory.total,utilization.gpu --format=csv || true
bash scripts/nda/run_multibranch_sub08.sh
echo "[slurm] finished rc=$? @ $(date -Iseconds)"
