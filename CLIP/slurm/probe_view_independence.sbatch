#!/bin/bash
#SBATCH --job-name=probe-viewindep
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --output=/project/peilab/why/CLIP/outputs/slurm/probe-viewindep-%j.out
#SBATCH --error=/project/peilab/why/CLIP/outputs/slurm/probe-viewindep-%j.err
#
# v13/CROMA GATE -- the zero-training precondition for the order-2 architecture.
#
# The architecture's whole argument is that the deployed object (the fused concept metric D*) has
# exactly two movable error components, and that a training term can only pay if one of them has
# headroom. This job measures both on the ten banked v8 checkpoints, with no labels and no
# retrieval, and each carries a concept-shuffle control:
#
#   Q1 bias  -- corr(D_{R'=1}, D_gallery) vs corr(D_{R'=80}, D_gallery)
#   Q2 rho   -- split-half reliability of two DISJOINT repetition halves vs how many reps each
#               averages; rho = slope/intercept of the linearised curve; K_eff = R/(1+(R-1)rho)
#
# GO on the bias channel iff the R80-R1 gap > 0.03 on average AND the shuffled control collapses
# (<0.10). If it does not, the design doc's own kill criterion fires: record the falsification and
# keep the shipped 54.83. Submitted through Slurm rather than the login node (AGENTS.md 3.1).
#
# Submit:  sbatch slurm/probe_view_independence.sbatch
# Smoke:   FOLDS="8" sbatch slurm/probe_view_independence.sbatch
set -euo pipefail

ROOT="/project/peilab/why/CLIP"
PY="${ROOT}/.venv/bin/python"
mkdir -p "${ROOT}/outputs/slurm" "${ROOT}/outputs/probe/view_independence"

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

# hard CUDA gate: the embedding pass needs the real device, fail fast rather than on CPU
"${PY}" -c "import sys,torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { echo "[FATAL] CUDA unavailable on $(hostname)"; exit 1; }

FOLDS="${FOLDS:-1 2 3 4 5 6 7 8 9 10}"
SEED="${SEED:-2025}"

"${PY}" scripts/probe_view_independence.py \
  --stage1-root "${ROOT}/outputs/stage1/v8" \
  --folds ${FOLDS} --seed "${SEED}" \
  --fuse 16 --rep-shrink 0.1 --n-split 20 \
  --outdir "${ROOT}/outputs/probe/view_independence"

echo "[vi] DONE $(date -Is)"
