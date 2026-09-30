#!/usr/bin/env bash
#SBATCH --job-name=nb-vith
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/%x-%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/%x-%j.err

# NeuroBridge Intra retrain with OpenCLIP ViT-H-14 teacher (10 subjects),
# then sub-08 adapter→SDXL generation in matched ViT-H space.

set -euo pipefail
NB_ROOT=/project/peilab/why/NeuroBridge
CACHE=/project/peilab/why/cache/eeg-brainit

module load slurm nvhpc-hpcx-cuda12/23.11
cd "${NB_ROOT}"
# Prefer brainit venv (has open_clip/diffusers); avoid thor activate overwriting paths.
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
export MAX_IMAGES="${MAX_IMAGES:-50}"
export NB_ROOT
export OUTPUT_DIR="${NB_ROOT}/results/things_eeg/intra-subjects-vit-h"

mkdir -p outputs/slurm "${OUTPUT_DIR}"

IMG="${NB_ROOT}/data/things_eeg/image_feature/ViT-H-14"
AUG="${IMG}/GaussianBlur-GaussianNoise-LowResolution-Mosaic"
EEG_DIR="${NB_ROOT}/data/things_eeg/preprocessed_eeg"
test -f "${IMG}/image_train.npy"
test -f "${IMG}/image_test.npy"
test -f "${AUG}/train.npy"
test -f "${EEG_DIR}/info.json"

echo "{\"job_id\":\"${SLURM_JOB_ID:-local}\",\"phase\":\"vit_h_intra\",\"started_at\":\"$(date -Iseconds)\"}" \
  > "${OUTPUT_DIR}/job_running.json"

python -c "import torch; assert torch.cuda.is_available(); print('GPU', torch.cuda.get_device_name(0))"

echo "===== Phase A: ViT-H Intra train (10 subjects) ====="
bash "${NB_ROOT}/scripts/run_intra_things_eeg_vit_h.sh"

echo "===== Phase B: sub-08 matched-space adapter + SDXL ====="
bash "${NB_ROOT}/scripts/nb_adapter/run_vith_sub08_gen.sh"

echo "{\"job_id\":\"${SLURM_JOB_ID:-local}\",\"finished_at\":\"$(date -Iseconds)\"}" \
  > "${OUTPUT_DIR}/job_done.json"
echo "===== ALL DONE ====="
