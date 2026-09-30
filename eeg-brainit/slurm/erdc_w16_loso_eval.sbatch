#!/usr/bin/env bash
#SBATCH --job-name=erdc-w16-ev
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --array=1-10%2
#SBATCH --exclude=dgx-10,dgx-11
#SBATCH --output=outputs/slurm/%x-%A_%a.out
#SBATCH --error=outputs/slurm/%x-%A_%a.err

# W16 — per-subject eval: official prior_atm + ours bit_clip fuse (Turbo stack)

set -euo pipefail
ROOT=/project/peilab/why/eeg-brainit
module load slurm nvhpc-hpcx-cuda12/23.11
cd "${ROOT}"
source scripts/activate.sh
export PYTHONUNBUFFERED=1
export HF_HOME=/project/peilab/why/cache/eeg-brainit/hf
export HF_HUB_CACHE=/project/peilab/why/cache/eeg-brainit/hf/hub
export OPENCLIP_CACHE_DIR=/project/peilab/why/cache/eeg-brainit/open_clip
export TORCH_HOME=/project/peilab/why/cache/eeg-brainit/torch
export TMPDIR=/project/peilab/why/cache/eeg-brainit/tmp
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export XFORMERS_DISABLED=1
mkdir -p "$TMPDIR" outputs/erdc/w16_loso_metrics outputs/eval/atm_pipeline_loso outputs/slurm

pip install -q torch-fidelity 2>/dev/null || true

MAX_IMAGES="${ERDC_MAX_IMAGES:-0}"
EXTRA=()
[[ "${MAX_IMAGES}" != "0" ]] && EXTRA+=(--max-images "${MAX_IMAGES}")

IDX=${SLURM_ARRAY_TASK_ID:-1}
SUB=$(printf "sub-%02d" "${IDX}")
FUSE_LAM="0.15"
MET_DIR="outputs/erdc/w16_loso_metrics"

S1_CKPT="outputs/atm_distill_s1_${SUB}/checkpoints/atm_stage1_best.pt"
S3_CKPT="outputs/atm_distill_s3_${SUB}/checkpoints/atm_stage3_best.pt"
if [[ ! -f "${S3_CKPT}" ]]; then
  echo "[ERROR] missing ${S3_CKPT}; run erdc_w16_loso_train first"
  exit 1
fi

full_metrics () {
  local gen="$1"; local tag="$2"
  local out="${MET_DIR}/${tag}.json"
  [[ -f "${out}" ]] && echo "[SKIP] ${tag}" && return 0
  [[ -d "${gen}" ]] || { echo "[WARN] miss ${gen}"; return 0; }
  python scripts/erdc_full_metrics.py --gen-dir "${gen}" --output-json "${out}" --tag "${tag}" "${EXTRA[@]}"
}

fid_metrics () {
  local gen="$1"; local tag="$2"
  local out="${MET_DIR}/${tag}_fid.json"
  [[ -f "${out}" ]] && echo "[SKIP] fid ${tag}" && return 0
  [[ -d "${gen}" ]] || return 0
  python scripts/erdc_fid_metrics.py --gen-dir "${gen}" --output-json "${out}" --tag "${tag}" "${EXTRA[@]}"
}

echo "----- Export bit_clip ${SUB} -----"
BIT_NPY="outputs/eval/atm_pipeline_loso/${SUB}_bit_clip_1024.npy"
if [[ ! -f "${BIT_NPY}" ]]; then
  python scripts/eval_atm_pipeline.py \
    --subject "${SUB}" \
    --prior-ckpt "checkpoints/atm_diffusion_prior/${SUB}/diffusion_prior.pt" \
    --s1-ckpt "${S1_CKPT}" \
    --s3-ckpt "${S3_CKPT}" \
    --output-dir outputs/eval/atm_pipeline_loso \
    --skip-generate \
    --gen-sources bit_clip \
    "${EXTRA[@]}"
fi

echo "----- Official Turbo prior_atm ${SUB} -----"
OFF_OUT="outputs/erdc/w16_official_prior_${SUB}"
OFF_TAG="w16_official_prior_${SUB}"
if [[ ! -f "${OFF_OUT}/metrics.json" ]]; then
  python scripts/erdc_official_atm_pipeline.py \
    --subject "${SUB}" \
    --embed-source prior_atm \
    --prior-ckpt "checkpoints/atm_diffusion_prior/${SUB}/diffusion_prior.pt" \
    --low-level-mode neighbor_image \
    --top-m 2 \
    --strengths 0.50,0.65,0.85 \
    --use-turbo --gen-steps 4 --gen-guidance 0.0 \
    --output-dir "${OFF_OUT}" \
    "${EXTRA[@]}"
fi
full_metrics "${OFF_OUT}/selected_brain" "${OFF_TAG}"
fid_metrics "${OFF_OUT}/selected_brain" "${OFF_TAG}"

echo "----- Ours Turbo bit + fuse ${SUB} -----"
TURBO_OUT="outputs/erdc/w16_loso_turbo_bit_${SUB}"
if [[ ! -f "${TURBO_OUT}/metrics.json" ]]; then
  python scripts/erdc_official_atm_pipeline.py \
    --subject "${SUB}" \
    --embed-source bit_clip \
    --bit-npy "${BIT_NPY}" \
    --low-level-mode neighbor_image \
    --top-m 2 \
    --strengths 0.35,0.50,0.65,0.85 \
    --use-turbo --gen-steps 4 --gen-guidance 0.0 \
    --output-dir "${TURBO_OUT}" \
    "${EXTRA[@]}"
fi

FUSE_OUT="outputs/erdc/w16_loso_fuse_${SUB}_l0p15"
FUSE_TAG="w16_loso_fuse_${SUB}"
if [[ ! -f "${FUSE_OUT}/metrics.json" ]]; then
  python scripts/erdc_fuse_reselect.py \
    --cand-dir "${TURBO_OUT}/candidates" \
    --eeg-npy "${BIT_NPY}" \
    --lambda-struct "${FUSE_LAM}" \
    --output-dir "${FUSE_OUT}" \
    "${EXTRA[@]}"
fi
full_metrics "${FUSE_OUT}/selected_fused" "${FUSE_TAG}"
fid_metrics "${FUSE_OUT}/selected_fused" "${FUSE_TAG}"

echo "[OK] LOSO eval ${SUB} done"
date
