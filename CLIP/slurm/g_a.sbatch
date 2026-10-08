#!/bin/bash
#SBATCH --job-name=clip-g-a
#SBATCH --partition=cpu
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/project/peilab/why/CLIP/outputs/slurm/clip-g-a-%j.out
#SBATCH --error=/project/peilab/why/CLIP/outputs/slurm/clip-g-a-%j.err

# G-a: the pre-registered gate for the noise-corrected low-rank concept frame.
#
# CPU partition on purpose. Every arm is linear algebra on cached features -- the only heavy
# step is embedding the R=80 repetition cloud, and the `.npz` caches for all 10 folds x 3
# seeds are written on first use. Nothing here needs a GPU, so taking one would only make a
# GPU node unavailable to someone else.

set -euo pipefail

ROOT="/project/peilab/why/CLIP"
PY="${ROOT}/.venv/bin/python"
mkdir -p "${ROOT}/outputs/slurm"

# --- caches (AGENTS.md §2.4): MUST come before any torch/HF import ---
export PYTHONNOUSERSITE=1
export HF_HOME=/project/peilab/why/cache/huggingface
export HF_HUB_CACHE=/project/peilab/why/cache/huggingface/hub
export OPENCLIP_CACHE_DIR=/project/peilab/why/cache/open_clip
export TORCH_HOME=/project/peilab/why/cache/torch
export XDG_CACHE_HOME=/project/peilab/why/cache/xdg
# --- TMPDIR must be NODE-LOCAL, not NFS (AGENTS.md §2.4) ---
export TMPDIR="/tmp/clip-${SLURM_JOB_ID:-local}"
export XDG_CONFIG_HOME="${ROOT}/scratch/config"
export MPLCONFIGDIR="${ROOT}/scratch/mpl"
mkdir -p "${TMPDIR}" "${XDG_CONFIG_HOME}" "${MPLCONFIGDIR}"
export PYTHONPATH="${ROOT}/src:${ROOT}/scripts${PYTHONPATH:+:${PYTHONPATH}}"

cd "${ROOT}"

echo "[slurm] job=${SLURM_JOB_ID} host=$(hostname) date=$(date -Is)"
nproc
co_tenants=$(squeue -w "$(hostname)" -h -o '%i %u %j' | grep -v "${SLURM_JOB_ID}" || true)
[[ -n "${co_tenants}" ]] && echo "[slurm] shared-node co-tenants:" && echo "${co_tenants}" || true

# No GPU gate: this job is deliberately CPU-only. Assert the caches are readable instead,
# because a missing shared cache would otherwise surface as a download into a full $HOME.
[[ -d "${HF_HOME}" ]] || { echo "[FATAL] HF_HOME missing: ${HF_HOME}"; exit 1; }

"${PY}" scripts/probe_g_a.py \
  --seeds 2025 2026 2027 \
  --subjects 1 2 3 4 5 6 7 8 9 10 \
  --out "${ROOT}/outputs/probe/g_a.json"

echo "[slurm] finished rc=$? at $(date -Is)"
