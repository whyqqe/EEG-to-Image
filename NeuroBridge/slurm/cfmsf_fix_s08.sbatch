#!/bin/bash
#SBATCH --job-name=cfmsf_fix
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/cfmsf_fix_%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/cfmsf_fix_%j.err

set -euo pipefail
cd /project/peilab/why/NeuroBridge
export NB_ROOT=/project/peilab/why/NeuroBridge
export DEVICE=cuda:0
export FIX_OUT=/project/peilab/why/NeuroBridge/outputs/cfmsf_fix/sub-08
export TARGET="${TARGET:-levels_mean}"
export ARMS="${ARMS:-frozen,joint,direct,disc,lora}"
export EPOCHS="${EPOCHS:-40}"
export PROBE_EPOCHS="${PROBE_EPOCHS:-80}"

echo "[slurm] CF-MSF fix round ${SLURM_JOB_ID:-local} on $(hostname)"
echo "[slurm] ARMS=${ARMS} TARGET=${TARGET} EPOCHS=${EPOCHS}"
nvidia-smi --query-gpu=name,memory.total,utilization.gpu --format=csv || true

set +e
bash scripts/nda/run_cfmsf_fix_sub08.sh
rc=$?
set -e
echo "[slurm] fix round rc=${rc} @ $(date -Iseconds)"
echo "[slurm] Re-submitting resumes: finished arms and probes are skipped."
exit "${rc}"
