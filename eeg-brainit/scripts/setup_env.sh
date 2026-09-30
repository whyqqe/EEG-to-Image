#!/usr/bin/env bash
# Create a pip virtualenv for eeg-brainit on peilab (no conda).
#
# Cluster-tested stack (aligned with eeg-vilex-neurolm):
#   Python 3.11, PyTorch 2.2.2 + cu121, module nvhpc-hpcx-cuda12/23.11
#   SLURM partitions: normal / preempt, 8 GPUs per DGX node, DefCpuPerGPU=28
#
# Prefer running inside a GPU allocation so CUDA availability can be verified:
#   module load slurm nvhpc-hpcx-cuda12/23.11
#   srun --account=peilab --partition=normal --gpus=1 --cpus-per-gpu=28 \
#        --mem=64G --time=01:00:00 --pty bash
#   bash scripts/setup_env.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_ROOT="${EEG_BRAINIT_CACHE:-/project/peilab/why/cache/eeg-brainit}"
VENV_DIR="${VENV_DIR:-${ROOT}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"

mkdir -p "${ROOT}/outputs/slurm" "${CACHE_ROOT}/pip" "${CACHE_ROOT}/hf" \
         "${CACHE_ROOT}/torch" "${CACHE_ROOT}/xdg" "${CACHE_ROOT}/open_clip"

export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR="${CACHE_ROOT}/pip"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export HF_HOME="${CACHE_ROOT}/hf"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TORCH_HOME="${CACHE_ROOT}/torch"
export OPENCLIP_CACHE_DIR="${CACHE_ROOT}/open_clip"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  if [[ -x /cm/shared/apps/Anaconda3/2023.09-0/bin/python3.11 ]]; then
    PYTHON_BIN=/cm/shared/apps/Anaconda3/2023.09-0/bin/python3.11
  else
    PYTHON_BIN=python3
  fi
fi

echo "[INFO] Bootstrap Python: ${PYTHON_BIN} ($("${PYTHON_BIN}" --version))"
echo "[INFO] CACHE_ROOT=${CACHE_ROOT}"
echo "[INFO] Creating venv at ${VENV_DIR}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
PYTHON="${VENV_DIR}/bin/python"

"${PYTHON}" -m pip install --upgrade pip wheel setuptools

echo "[INFO] Installing PyTorch 2.2.2 + cu121 (peilab CUDA 12.x driver / nvhpc module)..."
"${PYTHON}" -m pip install \
  --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.2.2 torchvision==0.17.2

# Torch 2.2.2 extensions were compiled against NumPy 1.x.
"${PYTHON}" -m pip install "numpy==1.26.4"
"${PYTHON}" -m pip install -e "${ROOT}"

"${PYTHON}" - <<'PY'
import torch
print(f"PyTorch={torch.__version__}, CUDA_built={torch.version.cuda}, cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU={torch.cuda.get_device_name(0)}")
else:
    print("[WARN] CUDA not visible. Re-run inside a Slurm GPU allocation to verify.")
PY

cat <<EOF
[OK] Environment ready.
Activate with:
  source ${ROOT}/.venv/bin/activate
  # or: source ${ROOT}/scripts/activate.sh

Next:
  bash scripts/download_third_party.sh
  bash scripts/download_checkpoints.sh
  bash scripts/download_things_eeg2.sh   # optional if shared data already exists
  python scripts/smoke_test.py
EOF
