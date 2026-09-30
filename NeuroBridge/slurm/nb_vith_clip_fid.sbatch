#!/usr/bin/env bash
#SBATCH --job-name=nb-vith-metrics
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/%x-%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/%x-%j.err

set -euo pipefail
NB_ROOT=/project/peilab/why/NeuroBridge
CACHE=/project/peilab/why/cache/eeg-brainit
GEN_ROOT="${NB_ROOT}/outputs/nb_adapter/sub-08-vit-h/generation"
OUT="${NB_ROOT}/outputs/nb_adapter/sub-08-vit-h/clip_fid_metrics.json"

module load slurm nvhpc-hpcx-cuda12/23.11
cd "${NB_ROOT}"
# shellcheck disable=SC1091
source /project/peilab/why/eeg-brainit/scripts/activate.sh
cd "${NB_ROOT}"

export PYTHONUNBUFFERED=1
export HOME="${CACHE}/xdg-home"
export HF_HOME="${CACHE}/hf"
export HF_HUB_CACHE="${CACHE}/hf/hub"
export OPENCLIP_CACHE_DIR="${CACHE}/open_clip"
export TORCH_HOME="${CACHE}/torch"
export TMPDIR="${SLURM_TMPDIR:-${CACHE}/tmp}"

python -c "import torch; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0))"

python scripts/nb_adapter/eval_clip_fid.py \
  --gen-root "${GEN_ROOT}" \
  --output-json "${OUT}" \
  --max-images 50

echo "[OK] wrote ${OUT}"
