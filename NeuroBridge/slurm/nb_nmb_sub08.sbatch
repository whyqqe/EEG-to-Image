#!/usr/bin/env bash
#SBATCH --job-name=nb-nmb08
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/%x-%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/%x-%j.err

# NeuroMem-Bridge (NMB) full pipeline for sub-08.

set -euo pipefail
NB_ROOT=/project/peilab/why/NeuroBridge
CACHE=/project/peilab/why/cache/eeg-brainit

module load slurm nvhpc-hpcx-cuda12/23.11
cd "${NB_ROOT}"
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
export BRAIN_HIVE=/project/peilab/why/Brain-HIVE
export FUSION_PRIOR=/project/peilab/why/cache/fusion_prior/H14_B32_VAE
export MAX_IMAGES="${MAX_IMAGES:-0}"
export NB_ROOT
export OUT="${NB_ROOT}/outputs/nb_nmb/sub-08"

mkdir -p outputs/slurm "${OUT}"
echo "{\"job_id\":\"${SLURM_JOB_ID:-local}\",\"started_at\":\"$(date -Iseconds)\"}" > "${OUT}/job_running.json"

# Ensure Fusion Prior (skip if already downloaded)
if [[ ! -f "${FUSION_PRIOR}/fusion_encoder/config.json" ]]; then
  python -c "
from huggingface_hub import snapshot_download
snapshot_download('fakekungfu/Fusion-Prior-H14_B32_VAE', local_dir='${FUSION_PRIOR}')
"
fi

python -c "import torch; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0))"
bash "${NB_ROOT}/scripts/nmb/run_nmb_sub08.sh"

echo "{\"job_id\":\"${SLURM_JOB_ID:-local}\",\"finished_at\":\"$(date -Iseconds)\"}" > "${OUT}/job_done.json"
