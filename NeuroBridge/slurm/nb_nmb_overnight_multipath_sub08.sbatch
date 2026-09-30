#!/usr/bin/env bash
#SBATCH --job-name=nb-nmb-overnight
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/%x-%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/%x-%j.err

# Overnight multi-path SOTA sweep: Fusion->ViT-H bridge + IP-Adapter img2img + Fusion Prior rerank.

set -euo pipefail
NB_ROOT=/project/peilab/why/NeuroBridge
CACHE=/project/peilab/why/cache/eeg-brainit

module load slurm nvhpc-hpcx-cuda12/23.11
cd "${NB_ROOT}"
# shellcheck disable=SC1091
source /project/peilab/why/eeg-brainit/scripts/activate.sh

export BRAIN_HIVE=/project/peilab/why/Brain-HIVE
pip install -q 'diffusers>=0.27,<0.32' 'accelerate>=0.27,<1.2' 'huggingface-hub>=0.25,<1.0' datasets einops 2>/dev/null || true
export PYTHONPATH="${BRAIN_HIVE}:${PYTHONPATH:-}"

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
mkdir -p "${TMPDIR}"
export HF_DATASETS_CACHE="${TMPDIR}/hf_datasets"
export HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=8
export DEVICE=cuda:0
export NB_ROOT
export OUT="${NB_ROOT}/outputs/nb_nmb_sota/sub-08"
export MAX_IMAGES="${MAX_IMAGES:-0}"
export IMG2IMG_STRENGTH=0.5
export RERANK_SAMPLES=8

mkdir -p outputs/slurm "${OUT}"
echo "{\"job_id\":\"${SLURM_JOB_ID:-local}\",\"started_at\":\"$(date -Iseconds)\",\"config\":\"overnight-multipath\"}" > "${OUT}/job_overnight_running.json"

python -c "import torch; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0))"
bash "${NB_ROOT}/scripts/nmb/run_nmb_overnight_multipath_sub08.sh"

echo "{\"job_id\":\"${SLURM_JOB_ID:-local}\",\"finished_at\":\"$(date -Iseconds)\",\"config\":\"overnight-multipath\"}" > "${OUT}/job_overnight_done.json"
