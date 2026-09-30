#!/bin/bash
#SBATCH --job-name=gvm_s08
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/gvm_s08_%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/gvm_s08_%j.err

set -euo pipefail
cd /project/peilab/why/NeuroBridge
export NB_ROOT=/project/peilab/why/NeuroBridge
export DEVICE=cuda:0
export SID=8
export EPOCHS="${EPOCHS:-26}"
export KEEP_IMAGES="${KEEP_IMAGES:-0}"
# A FRESH OUTPUT ROOT, DELIBERATELY.  `run_gem_intra.sh` skips an arm when its
# exported condition already exists, so pointing GVM at `outputs/gem/sub-08` would
# silently reuse the pre-GVM checkpoints and report GVM numbers for a model that
# never had the anchor head, the NVOL head, the arbitration head or the schedule.
# The shared condition cache (`outputs/gem/cond_cache`) IS reused: it is the audited
# CLIP image extraction, identical for both architectures by construction.
export OUT=/project/peilab/why/NeuroBridge/outputs/gvm/sub-08

echo "[slurm] GVM job ${SLURM_JOB_ID:-local} on $(hostname) gpu ${CUDA_VISIBLE_DEVICES:-?} out=${OUT}"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
bash scripts/nda/run_gem_intra.sh
echo "[slurm] finished rc=$? @ $(date -Iseconds)"
