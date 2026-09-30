#!/usr/bin/env bash
#SBATCH --job-name=nb-adapt08
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/%x-%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/%x-%j.err

# NeuroBridge sub-08 adapter experiment: Linear vs MLP vs ATM-prior vs teacher → SDXL.

set -euo pipefail
NB_ROOT=/project/peilab/why/NeuroBridge
CACHE=/project/peilab/why/cache/eeg-brainit

module load slurm nvhpc-hpcx-cuda12/23.11
cd "${NB_ROOT}"
# Use brainit venv (has diffusers/open_clip); do NOT source thor activate (overwrites ROOT).
# shellcheck disable=SC1091
source /project/peilab/why/eeg-brainit/scripts/activate.sh

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
# First pass: 50 images. Set MAX_IMAGES=0 for full 200.
export MAX_IMAGES="${MAX_IMAGES:-50}"
export NB_ROOT
export OUT="${NB_ROOT}/outputs/nb_adapter/sub-08"

mkdir -p outputs/slurm "${OUT}"
echo "{\"job_id\":\"${SLURM_JOB_ID:-local}\",\"started_at\":\"$(date -Iseconds)\",\"max_images\":${MAX_IMAGES}}" \
  > "${OUT}/job_running.json"

python -c "import torch; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0))"
bash "${NB_ROOT}/scripts/nb_adapter/run_pipeline_sub08.sh"

echo "{\"job_id\":\"${SLURM_JOB_ID:-local}\",\"finished_at\":\"$(date -Iseconds)\"}" \
  > "${OUT}/job_done.json"
