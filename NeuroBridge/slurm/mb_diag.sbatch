#!/bin/bash
#SBATCH --job-name=mb_diag
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/mb_diag_%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/mb_diag_%j.err

set -euo pipefail
cd /project/peilab/why/NeuroBridge
export NB_ROOT=/project/peilab/why/NeuroBridge
export DEVICE=cuda:0
export DIAG_OUT=/project/peilab/why/NeuroBridge/outputs/mb_diag
echo "[slurm] IP-mass diagnostic ${SLURM_JOB_ID:-local} on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
bash scripts/nda/run_diag_ipmass.sh
echo "[slurm] finished rc=$? @ $(date -Iseconds)"
