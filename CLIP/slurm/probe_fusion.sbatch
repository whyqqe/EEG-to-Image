#!/bin/bash
#SBATCH --job-name=probe-fusion
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --output=/project/peilab/why/CLIP/outputs/slurm/probe-fusion-%j.out
#SBATCH --error=/project/peilab/why/CLIP/outputs/slurm/probe-fusion-%j.err
#
# v11 PREDICTION-1 GATE (docs/eeg2image_v11_unified_architecture.md §3, §7).
#
# Runs the fusion probe on a GPU node rather than the login node, per this repo's rule that
# nothing heavy runs on the shared login node (AGENTS.md §3.1) and the user's instruction to use
# Slurm for probes. The algebra is small (200x200 metrics, 9 views, 10 folds) so a GPU is not the
# bottleneck; the point of the job is (a) to keep it off the login node and (b) to certify the run
# in the same environment (CUDA, NFS, wheel ABI) the real experiments use, so the go/no-go cannot
# be overturned by an environment difference later.
set -euo pipefail

ROOT="/project/peilab/why/CLIP"
PY="${ROOT}/.venv/bin/python"
mkdir -p "${ROOT}/outputs/slurm" "${ROOT}/outputs/probe/fusion"

export PYTHONNOUSERSITE=1
export HF_HOME=/project/peilab/why/cache/huggingface
export HF_HUB_CACHE=/project/peilab/why/cache/huggingface/hub
export OPENCLIP_CACHE_DIR=/project/peilab/why/cache/open_clip
export TORCH_HOME=/project/peilab/why/cache/torch
export XDG_CACHE_HOME=/project/peilab/why/cache/xdg
export TMPDIR="/tmp/clip-${SLURM_JOB_ID:-local}"      # node-local, NOT NFS (AGENTS.md 2.4)
export XDG_CONFIG_HOME="${ROOT}/scratch/config"
export MPLCONFIGDIR="${ROOT}/scratch/mpl"
mkdir -p "${TMPDIR}" "${XDG_CONFIG_HOME}" "${MPLCONFIGDIR}"
export PYTHONPATH="${ROOT}/src"

cd "${ROOT}"
echo "[slurm] job=${SLURM_JOB_ID} host=$(hostname) date=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

"${PY}" scripts/probe_fusion.py "$@"

echo "[fusion] DONE $(date -Is)"
