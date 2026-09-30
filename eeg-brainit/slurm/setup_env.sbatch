#!/usr/bin/env bash
#SBATCH --job-name=eeg-brainit-setup
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=28
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

set -euo pipefail
ROOT=/project/peilab/why/eeg-brainit
CACHE_ROOT=/project/peilab/why/cache/eeg-brainit
mkdir -p "${ROOT}/outputs/slurm" "${CACHE_ROOT}/pip"
export PIP_CACHE_DIR="${CACHE_ROOT}/pip"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export HF_HOME="${CACHE_ROOT}/hf"

module load slurm nvhpc-hpcx-cuda12/23.11
cd "${ROOT}"
bash scripts/setup_env.sh
source scripts/activate.sh
python - <<'PY'
import torch
assert torch.cuda.is_available(), "GPU allocation / CUDA library unavailable"
print(torch.cuda.get_device_name(0), torch.__version__, torch.version.cuda)
PY
