#!/bin/bash
#SBATCH --job-name=cfmsf_probe
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/cfmsf_probe_%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/cfmsf_probe_%j.err

set -euo pipefail
cd /project/peilab/why/NeuroBridge
export NB_ROOT=/project/peilab/why/NeuroBridge
export DEVICE=cuda:0
export CFMSF_PROBE_OUT=/project/peilab/why/NeuroBridge/outputs/cfmsf_probe/sub-08
export EPOCHS="${EPOCHS:-80}"

echo "[slurm] CF-MSF route-quality probe ${SLURM_JOB_ID:-local} on $(hostname)"
echo "[slurm] EPOCHS=${EPOCHS}"
nvidia-smi --query-gpu=name,memory.total,utilization.gpu --format=csv || true
bash scripts/nda/run_cfmsf_probe_sub08.sh
echo "[slurm] finished rc=$? @ $(date -Iseconds)"
