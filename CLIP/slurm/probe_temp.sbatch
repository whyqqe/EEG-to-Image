#!/bin/bash
# A/B probe for the InfoNCE temperature collapse (see scripts/probe_temperature.py).
#
# Arms: (A) the fix in place, (B) the floor disabled, (C) `routed_sr` as a reference.
# A and B differ in exactly one module global, so if A trains and B is dead at ln(72)
# the floor is the load-bearing change and not a coincidence of the other edits.
#
#SBATCH --job-name=probe-temp
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --output=/project/peilab/why/CLIP/outputs/slurm/probe-temp-%j.out
#SBATCH --error=/project/peilab/why/CLIP/outputs/slurm/probe-temp-%j.err

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
# TMPDIR is NODE-LOCAL on purpose (see samclip.config): NFS + multiprocessing cannot
# rmtree its own per-worker temp dirs (unlink-while-open is silly-renamed to `.nfs*`,
# which stays visible -> EBUSY), which buried every job in false tracebacks. The other
# two are small persistent state, so they stay in the project.
export TMPDIR="/tmp/clip-${SLURM_JOB_ID:-local}"
export XDG_CONFIG_HOME="${ROOT}/scratch/config"
export MPLCONFIGDIR="${ROOT}/scratch/mpl"
mkdir -p "${TMPDIR}" "${XDG_CONFIG_HOME}" "${MPLCONFIGDIR}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${ROOT}"
echo "[probe] job=${SLURM_JOB_ID} host=$(hostname) date=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# --- hard CUDA gate: never silently run this on CPU ---
"${PY}" -c "import sys,torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo "[FATAL] CUDA unavailable on $(hostname)"; exit 1; }

"${PY}" scripts/probe_temperature.py \
  --config configs/loso_sub08.yaml \
  --steps 300 --log-every 50 --seeds 2025 2026 \
  --out-dir outputs/probe

echo "[probe] finished rc=$? at $(date -Is)"
