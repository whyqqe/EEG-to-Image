#!/bin/bash
#SBATCH --job-name=uck_re7
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/uck_remeasure_%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/uck_remeasure_%j.err

set -euo pipefail
cd /project/peilab/why/NeuroBridge
export NB_ROOT=/project/peilab/why/NeuroBridge
export DEVICE=cuda:0
export KEEP_IMAGES=0
export UCK_OUT=/project/peilab/why/NeuroBridge/outputs/uck

echo "[slurm] UCK remasure ${SLURM_JOB_ID:-local} on $(hostname) out=${UCK_OUT}"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
bash scripts/nda/run_uck_remeasure.sh
echo "[slurm] finished rc=$? @ $(date -Iseconds)"
