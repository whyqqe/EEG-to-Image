#!/bin/bash
#SBATCH --job-name=uge_s08
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/uge_s08_%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/uge_s08_%j.err

set -euo pipefail
cd /project/peilab/why/NeuroBridge
export NB_ROOT=/project/peilab/why/NeuroBridge
export DEVICE=cuda:0
export SID=8
export EPOCHS="${EPOCHS:-26}"
export KEEP_IMAGES="${KEEP_IMAGES:-0}"
# Fresh root: skip logic would otherwise reuse GVM fused checkpoints.
export OUT=/project/peilab/why/NeuroBridge/outputs/uge/sub-08

echo "[slurm] UGE job ${SLURM_JOB_ID:-local} on $(hostname) gpu ${CUDA_VISIBLE_DEVICES:-?} out=${OUT}"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
bash scripts/nda/run_gem_intra.sh
echo "[slurm] finished rc=$? @ $(date -Iseconds)"
