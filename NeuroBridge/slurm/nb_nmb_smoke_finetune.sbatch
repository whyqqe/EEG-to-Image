#!/usr/bin/env bash
#SBATCH --job-name=nb-nmb-smoke
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/%x-%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/%x-%j.err

# Quick GPU smoke test: Fusion Prior finetune 15 steps only.

set -euo pipefail
NB_ROOT=/project/peilab/why/NeuroBridge
CACHE=/project/peilab/why/cache/eeg-brainit

module load slurm nvhpc-hpcx-cuda12/23.11
cd "${NB_ROOT}"
source /project/peilab/why/eeg-brainit/scripts/activate.sh

pip install -q 'diffusers>=0.27,<0.32' 'accelerate>=0.27,<1.2' 'huggingface-hub>=0.25,<1.0' datasets einops 2>/dev/null || true

export PYTHONUNBUFFERED=1
export HOME="${CACHE}/xdg-home"
export HF_HOME="${CACHE}/hf"
export HF_HUB_CACHE="${CACHE}/hf/hub"
export OPENCLIP_CACHE_DIR="${CACHE}/open_clip"
export TORCH_HOME="${CACHE}/torch"
export TMPDIR="${SLURM_TMPDIR:-${CACHE}/tmp}"
mkdir -p "${TMPDIR}"
export HF_DATASETS_CACHE="${TMPDIR}/hf_datasets"
export HF_HUB_DISABLE_XET=1
export BRAIN_HIVE=/project/peilab/why/Brain-HIVE
export PYTHONPATH="${BRAIN_HIVE}:${PYTHONPATH:-}"
export FUSION_PRIOR=/project/peilab/why/cache/fusion_prior/H14_B32_VAE
export OUT_EMB="${NB_ROOT}/outputs/nb_nmb_sota/sub-08/things_embeddings"
export OUT_PRIOR="${NB_ROOT}/outputs/nb_nmb_sota/sub-08/fusion_prior_finetuned_smoke"
export FT_EPOCHS=1
export FT_MAX_STEPS=15
export FT_BATCH=4
export FT_LR=5e-5

python -c "import torch; print('GPU', torch.cuda.get_device_name(0))"
bash "${NB_ROOT}/scripts/nmb/nmb_finetune_fusion_prior.sh"
echo "[SMOKE OK] finetune started and completed ${FT_MAX_STEPS} steps"
