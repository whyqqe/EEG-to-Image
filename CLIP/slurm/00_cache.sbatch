#!/bin/bash
# Pipeline stage 0 -- prebuild EEG + image-target caches for one LOSO fold.
# CPU only (no GPU needed for loading/normalisation), so this runs on `cpu`.
#SBATCH --job-name=samclip-cache
#SBATCH --partition=cpu
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/project/peilab/why/CLIP/outputs/slurm/samclip-cache-%j.out
#SBATCH --error=/project/peilab/why/CLIP/outputs/slurm/samclip-cache-%j.err

set -euo pipefail

ROOT="/project/peilab/why/CLIP"
PY="${ROOT}/.venv/bin/python"
mkdir -p "${ROOT}/outputs/slurm"

# caches MUST leave $HOME before any torch import (AGENTS.md §2.4)
export PYTHONNOUSERSITE=1
export HF_HOME=/project/peilab/why/cache/huggingface
export HF_HUB_CACHE=/project/peilab/why/cache/huggingface/hub
export TORCH_HOME=/project/peilab/why/cache/torch
export XDG_CACHE_HOME=/project/peilab/why/cache/xdg
# TMPDIR is NODE-LOCAL on purpose (see samclip.config): NFS + multiprocessing cannot
# rmtree its own per-worker temp dirs (unlink-while-open is silly-renamed to `.nfs*`,
# which stays visible -> EBUSY), which buried every job in false tracebacks. The other
# two are small persistent state, so they stay in the project.
export TMPDIR="/tmp/clip-${SLURM_JOB_ID:-local}"
export XDG_CONFIG_HOME="${ROOT}/scratch/config"
export MPLCONFIGDIR="${ROOT}/scratch/mpl"
mkdir -p "${TMPDIR}" "${XDG_CONFIG_HOME}" "${MPLCONFIGDIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${ROOT}"
echo "[slurm] cache job=${SLURM_JOB_ID} host=$(hostname) date=$(date -Is)"

  "${PY}" scripts/build_cache.py \
    --target-subject "${TARGET_SUBJECT:?TARGET_SUBJECT is required}" \
    --mvnn "${MVNN:-train}" \
    --channel-set "${CHANNEL_SET:-all63}" \
    --feature-set "${FEATURE_SET:-internvit_multilevel}"

echo "[slurm] cache finished rc=$? at $(date -Is)"
