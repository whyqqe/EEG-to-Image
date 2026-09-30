#!/bin/bash
#SBATCH --job-name=cfmsf_joint
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/cfmsf_joint_%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/cfmsf_joint_%j.err

set -euo pipefail
cd /project/peilab/why/NeuroBridge
export NB_ROOT=/project/peilab/why/NeuroBridge
export DEVICE=cuda:0
export JOINT_OUT=/project/peilab/why/NeuroBridge/outputs/cfmsf_joint/sub-08
export TARGET="${TARGET:-levels_mean}"
export ARMS="${ARMS:-joint,frozen}"
export EPOCHS="${EPOCHS:-40}"
export PROBE_EPOCHS="${PROBE_EPOCHS:-80}"

echo "[slurm] CF-MSF joint encoder training ${SLURM_JOB_ID:-local} on $(hostname)"
echo "[slurm] TARGET=${TARGET} ARMS=${ARMS} EPOCHS=${EPOCHS} PROBE_EPOCHS=${PROBE_EPOCHS}"
nvidia-smi --query-gpu=name,memory.total,utilization.gpu --format=csv || true
bash scripts/nda/run_cfmsf_joint_sub08.sh
echo "[slurm] finished rc=$? @ $(date -Iseconds)"
