#!/usr/bin/env bash
#SBATCH --job-name=hcma-10subj
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/%x-%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/%x-%j.err

set -euo pipefail
NB_ROOT=/project/peilab/why/NeuroBridge
OUT_ROOT="${NB_ROOT}/outputs/hcma_10subj"
source "${NB_ROOT}/scripts/nmb/nmb_sota_v2_env.sh"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
export NB_ROOT OUT_ROOT DEVICE=cuda:0
export HF_HUB_CACHE="${HF_HUB_CACHE:-/project/peilab/why/cache/eeg-brainit/hf/hub}"
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-/project/peilab/why/cache/eeg-brainit/open_clip}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
mkdir -p "${OUT_ROOT}" outputs/slurm
echo "{\"pipeline\":\"HCMA-10subj\",\"job_id\":\"${SLURM_JOB_ID:-local}\",\"started\":\"$(date -Iseconds)\"}" > "${OUT_ROOT}/job_running.json"
python -c "import torch; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0))"
bash "${NB_ROOT}/scripts/nda/run_hcma_10subj.sh"
echo "{\"pipeline\":\"HCMA-10subj\",\"job_id\":\"${SLURM_JOB_ID:-local}\",\"finished\":\"$(date -Iseconds)\"}" > "${OUT_ROOT}/job_done.json"
