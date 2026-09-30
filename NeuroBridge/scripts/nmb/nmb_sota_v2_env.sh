#!/usr/bin/env bash
# Shared env for DA-HLM v2 / SOTA v2 Slurm jobs (do not pip-install heavy deps here).
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
CACHE="${CACHE:-/project/peilab/why/cache/eeg-brainit}"
BRAINIT="${BRAINIT:-/project/peilab/why/eeg-brainit}"

module load slurm nvhpc-hpcx-cuda12/23.11 2>/dev/null || true
cd "${NB_ROOT}"
# shellcheck disable=SC1091
source "${BRAINIT}/scripts/activate.sh"

export BRAIN_HIVE=/project/peilab/why/Brain-HIVE
export PYTHONPATH="${BRAIN_HIVE}:${BRAINIT}/scripts:${BRAINIT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HOME="${CACHE}/xdg-home"
export HF_HOME="${CACHE}/hf"
export HF_HUB_CACHE="${CACHE}/hf/hub"
export HUGGINGFACE_HUB_CACHE="${CACHE}/hf/hub"
export TRANSFORMERS_CACHE="${CACHE}/hf/hub"
export DIFFUSERS_CACHE="${CACHE}/hf/hub"
export OPENCLIP_CACHE_DIR="${CACHE}/open_clip"
export TORCH_HOME="${CACHE}/torch"
export TMPDIR="${SLURM_TMPDIR:-${CACHE}/tmp}"
mkdir -p "${TMPDIR}" outputs/slurm
export HF_DATASETS_CACHE="${TMPDIR}/hf_datasets"
export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export XFORMERS_DISABLED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export DEVICE="${DEVICE:-cuda:0}"
