#!/usr/bin/env bash
# Activate the project venv and point caches at /project/peilab/why/cache/eeg-brainit.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_ROOT="${EEG_BRAINIT_CACHE:-/project/peilab/why/cache/eeg-brainit}"
# shellcheck disable=SC1091
source "${ROOT}/.venv/bin/activate"
export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR="${CACHE_ROOT}/pip"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export XDG_CONFIG_HOME="${CACHE_ROOT}/xdg-config"
export TMPDIR="${CACHE_ROOT}/tmp"
# If HOME was incorrectly pointed at a full HF cache, restore the real home.
if [[ "${HOME:-}" == *'/.cache/huggingface'* ]]; then
  export HOME="$(getent passwd "$(id -un)" | cut -d: -f6)"
  echo "[WARN] Reset HOME to ${HOME} (was under huggingface cache)"
fi
mkdir -p "${XDG_CACHE_HOME}" "${XDG_CONFIG_HOME}" "${TMPDIR}"
export HF_HOME="${CACHE_ROOT}/hf"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TORCH_HOME="${CACHE_ROOT}/torch"
export OPENCLIP_CACHE_DIR="${CACHE_ROOT}/open_clip"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
# Avoid forcing HF offline by default (breaks new downloads).
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE 2>/dev/null || true
echo "[INFO] Activated eeg-brainit venv ($(python --version))"
echo "[INFO] CACHE_ROOT=${CACHE_ROOT}"
