#!/usr/bin/env bash
#SBATCH --job-name=mg-a40-all
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/%x-%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/%x-%j.err

set -euo pipefail
NB_ROOT=/project/peilab/why/NeuroBridge
OUT="${NB_ROOT}/outputs/mg_flow_a40_all"
source "${NB_ROOT}/scripts/nmb/nmb_sota_v2_env.sh"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
export NB_ROOT ROOT_OUT="${OUT}" DEVICE=cuda:0
export SUBJECTS="${SUBJECTS:-1,2,3,4,5,6,7,8,9,10}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/project/peilab/why/cache/eeg-brainit/hf/hub}"
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-/project/peilab/why/cache/eeg-brainit/open_clip}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"

mkdir -p "${OUT}" outputs/slurm
echo "{\"pipeline\":\"MG-Flow-a40-all\",\"job_id\":\"${SLURM_JOB_ID:-local}\",\"subjects\":\"${SUBJECTS}\",\"started\":\"$(date -Iseconds)\"}" > "${OUT}/job_running.json"
python -c "import torch; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0))"
bash "${NB_ROOT}/scripts/nda/run_mg_flow_a40_all.sh"
echo "{\"pipeline\":\"MG-Flow-a40-all\",\"job_id\":\"${SLURM_JOB_ID:-local}\",\"finished\":\"$(date -Iseconds)\"}" > "${OUT}/job_done.json"
