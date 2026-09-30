#!/usr/bin/env bash
#SBATCH --job-name=nb-adapt08x
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/%x-%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/%x-%j.err

# Extended NB adapters on sub-08: ridge / linear_nobias / mlp_deep / mlp_res / cfm / diffprior.

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
export MAX_IMAGES="${MAX_IMAGES:-50}"
export NB_ROOT
export OUT="${NB_ROOT}/outputs/nb_adapter/sub-08"
export ARCHS="${ARCHS:-ridge,linear_nobias,mlp_deep,mlp_res,cfm,diffprior}"

mkdir -p outputs/slurm "${OUT}"

# Wait for base pipeline embeds (job 541083 may still be generating).
for i in $(seq 1 120); do
  if [[ -f "${OUT}/embeds/z_eeg_proj_train.npy" && -f "${OUT}/embeds/z_eeg_proj_test.npy" ]]; then
    echo "[INFO] embeds ready"
    break
  fi
  if [[ "${i}" -eq 120 ]]; then
    echo "[ERROR] embeds not found" >&2
    exit 1
  fi
  echo "[INFO] waiting embeds (${i}/120)..."
  sleep 30
done

echo "{\"job_id\":\"${SLURM_JOB_ID:-local}\",\"phase\":\"adapter_ext\",\"started_at\":\"$(date -Iseconds)\",\"archs\":\"${ARCHS}\"}" \
  > "${OUT}/job_ext_running.json"

bash "${NB_ROOT}/scripts/nb_adapter/run_pipeline_sub08_ext.sh"

echo "{\"job_id\":\"${SLURM_JOB_ID:-local}\",\"finished_at\":\"$(date -Iseconds)\"}" \
  > "${OUT}/job_ext_done.json"
