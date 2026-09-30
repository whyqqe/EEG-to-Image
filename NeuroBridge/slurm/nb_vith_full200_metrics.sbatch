#!/usr/bin/env bash
#SBATCH --job-name=nb-vith200
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/%x-%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/%x-%j.err

# Full 200 test: SDXL gen (ViT-H NB+MLP / teacher / ATM prior) + CLIP + FID.

set -euo pipefail
NB_ROOT=/project/peilab/why/NeuroBridge
CACHE=/project/peilab/why/cache/eeg-brainit

module load slurm nvhpc-hpcx-cuda12/23.11
cd "${NB_ROOT}"
# shellcheck disable=SC1091
source /project/peilab/why/eeg-brainit/scripts/activate.sh
cd "${NB_ROOT}"

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
export OMP_NUM_THREADS=8
export DEVICE=cuda:0
export NB_ROOT
export OUT="${NB_ROOT}/outputs/nb_adapter/sub-08-vit-h"

mkdir -p outputs/slurm "${OUT}"
echo "{\"job_id\":\"${SLURM_JOB_ID:-local}\",\"phase\":\"full200\",\"started_at\":\"$(date -Iseconds)\"}" \
  > "${OUT}/job_full200_running.json"

python -c "import torch; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0))"

bash "${NB_ROOT}/scripts/nb_adapter/run_full200_metrics.sh"

echo "{\"job_id\":\"${SLURM_JOB_ID:-local}\",\"finished_at\":\"$(date -Iseconds)\"}" \
  > "${OUT}/job_full200_done.json"
