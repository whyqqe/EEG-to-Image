#!/usr/bin/env bash
#SBATCH --job-name=nb-intra
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/%x-%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/%x-%j.err

# NeuroBridge official Intra-subject reproduction (Things-EEG, RN50, EEGProject).
# Resume-safe via train.py result.csv guard + --requeue.

set -euo pipefail
# NOTE: eeg-thor-a activate.sh overwrites ROOT → keep NB_ROOT immutable.
NB_ROOT=/project/peilab/why/NeuroBridge
CACHE=/project/peilab/why/cache/eeg-brainit

module load slurm nvhpc-hpcx-cuda12/23.11
cd "${NB_ROOT}"
source /project/peilab/why/eeg-thor-a/scripts/activate.sh
cd "${NB_ROOT}"

export PYTHONUNBUFFERED=1
export HOME="${CACHE}/xdg-home"
export HF_HOME=/project/peilab/why/cache/huggingface
export HF_HUB_CACHE=/project/peilab/why/cache/huggingface/hub
export TORCH_HOME="${CACHE}/torch"
export TMPDIR="${SLURM_TMPDIR:-${CACHE}/tmp}"
export TRITON_CACHE_DIR="${CACHE}/tmp/triton"
export XFORMERS_DISABLED=1
export OMP_NUM_THREADS=8
export PYTHONWARNINGS=ignore::UserWarning
export DEVICE=cuda:0

mkdir -p outputs/slurm results/things_eeg/intra-subjects

EEG_DIR="${NB_ROOT}/data/things_eeg/preprocessed_eeg"
IMG="${NB_ROOT}/data/things_eeg/image_feature/RN50"
AUG="${IMG}/GaussianBlur-GaussianNoise-LowResolution-Mosaic"
test -f "${EEG_DIR}/info.json"
test -f "${IMG}/image_train.npy"
test -f "${AUG}/train.npy"
for s in 01 02 03 04 05 06 07 08 09 10; do
  test -f "${EEG_DIR}/sub-${s}/train.npy"
  test -f "${EEG_DIR}/sub-${s}/test.npy"
done
echo "[INFO] data ready under ${NB_ROOT}"

echo "{\"job_id\":\"${SLURM_JOB_ID:-local}\",\"phase\":\"neurobridge_intra\",\"started_at\":\"$(date -Iseconds)\"}" \
  > "${NB_ROOT}/results/things_eeg/intra-subjects/job_running.json"

bash "${NB_ROOT}/scripts/run_intra_things_eeg.sh"

echo "{\"job_id\":\"${SLURM_JOB_ID:-local}\",\"phase\":\"neurobridge_intra\",\"finished_at\":\"$(date -Iseconds)\"}" \
  > "${NB_ROOT}/results/things_eeg/intra-subjects/job_done.json"
