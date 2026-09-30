#!/bin/bash
#SBATCH --job-name=cfmsf_s08
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/cfmsf_s08_%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/cfmsf_s08_%j.err

set -euo pipefail
cd /project/peilab/why/NeuroBridge
export NB_ROOT=/project/peilab/why/NeuroBridge
export DEVICE=cuda:0
export CFMSF_OUT=/project/peilab/why/NeuroBridge/outputs/cfmsf_s08
export EPOCHS="${EPOCHS:-80}"
export INST_W="${INST_W:-0.2}"
export ROUTES="${ROUTES:-img,text,depth,edge}"

echo "[slurm] CF-MSF sub-08 ${SLURM_JOB_ID:-local} on $(hostname)"
echo "[slurm] EPOCHS=${EPOCHS} INST_W=${INST_W} ROUTES=${ROUTES}"
squeue -w "$(hostname)" -h -o "%i %u %j %M" || true
nvidia-smi --query-gpu=name,memory.total,utilization.gpu --format=csv || true
bash scripts/nda/run_cfmsf_sub08.sh
echo "[slurm] finished rc=$? @ $(date -Iseconds)"
