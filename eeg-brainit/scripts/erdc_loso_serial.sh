#!/usr/bin/env bash
# Per-subject within-subject train + eval (serial)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
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
mkdir -p "$TMPDIR"

MAX_IMAGES="${ERDC_MAX_IMAGES:-0}"
EXTRA=()
[[ "${MAX_IMAGES}" != "0" ]] && EXTRA+=(--max-images "${MAX_IMAGES}")

MET_DIR="outputs/erdc/w16_loso_metrics"
FUSE_LAM="0.15"

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

resolve_ckpts () {
  local sub="$1"
  python - <<PY
from pathlib import Path
import sys
sys.path.insert(0, "scripts")
from erdc_loso_paths import ensure_stage_best, resolve_stage_ckpt

root = Path("${ROOT}")
sub = "${sub}"
s1 = ensure_stage_best(root, sub, 1) or resolve_stage_ckpt(root, sub, 1)
s3 = ensure_stage_best(root, sub, 3) or resolve_stage_ckpt(root, sub, 3)
if s1:
    print(f"S1_CKPT={s1.relative_to(root)}")
if s3:
    print(f"S3_CKPT={s3.relative_to(root)}")
PY
}

for IDX in $(seq 1 10); do
  SUB=$(printf "sub-%02d" "${IDX}")
  echo "===== ${SUB} ====="

  python scripts/erdc_make_loso_config.py --subject "${SUB}"

  eval "$(resolve_ckpts "${SUB}")"
  S1_CKPT="${S1_CKPT:-}"
  S3_CKPT="${S3_CKPT:-}"

  if [[ -z "${S1_CKPT}" ]]; then
    echo "----- Train S1 ${SUB} -----"
    python scripts/train_atm_bridge.py \
      --config configs/base.yaml \
      --override "configs/_loso_s1_${SUB}.yaml"
    eval "$(resolve_ckpts "${SUB}")"
    S1_CKPT="${S1_CKPT:-}"
  fi
  [[ -n "${S1_CKPT}" ]] || { echo "[ERROR] no S1 ckpt for ${SUB}"; exit 1; }

  python scripts/erdc_make_loso_config.py --subject "${SUB}"
  eval "$(resolve_ckpts "${SUB}")"
  S3_CKPT="${S3_CKPT:-}"

  if [[ -z "${S3_CKPT}" ]]; then
    echo "----- Train S3 ${SUB} -----"
    python scripts/train_atm_bridge.py \
      --config configs/base.yaml \
      --override "configs/_loso_s3_${SUB}.yaml"
    eval "$(resolve_ckpts "${SUB}")"
    S3_CKPT="${S3_CKPT:-}"
  fi
  [[ -n "${S3_CKPT}" ]] || { echo "[ERROR] no S3 ckpt for ${SUB}"; exit 1; }

  BIT_NPY="outputs/eval/atm_pipeline_loso/${SUB}_bit_clip_1024.npy"
  if [[ ! -f "${BIT_NPY}" ]]; then
    echo "----- Export bit_clip ${SUB} -----"
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
  [[ -f "${BIT_NPY}" ]] || { echo "[ERROR] missing ${BIT_NPY}"; exit 1; }

  OFF_OUT="outputs/erdc/w16_official_prior_${SUB}"
  OFF_TAG="w16_official_prior_${SUB}"
  if [[ ! -f "${OFF_OUT}/metrics.json" ]]; then
    echo "----- Official prior_atm ${SUB} -----"
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
  else
    echo "[SKIP] official ${SUB}"
  fi
  full_metrics "${OFF_OUT}/selected_brain" "${OFF_TAG}"
  fid_metrics "${OFF_OUT}/selected_brain" "${OFF_TAG}"

  TURBO_OUT="outputs/erdc/w16_loso_turbo_bit_${SUB}"
  if [[ ! -f "${TURBO_OUT}/metrics.json" ]]; then
    echo "----- Turbo bit ${SUB} -----"
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
  else
    echo "[SKIP] turbo ${SUB}"
  fi

  FUSE_OUT="outputs/erdc/w16_loso_fuse_${SUB}_l0p15"
  FUSE_TAG="w16_loso_fuse_${SUB}"
  if [[ ! -f "${FUSE_OUT}/metrics.json" ]]; then
    echo "----- Fuse ${SUB} -----"
    python scripts/erdc_fuse_reselect.py \
      --cand-dir "${TURBO_OUT}/candidates" \
      --eeg-npy "${BIT_NPY}" \
      --lambda-struct "${FUSE_LAM}" \
      --output-dir "${FUSE_OUT}" \
      "${EXTRA[@]}"
  fi
  full_metrics "${FUSE_OUT}/selected_fused" "${FUSE_TAG}"
  fid_metrics "${FUSE_OUT}/selected_fused" "${FUSE_TAG}"
done

python scripts/erdc_ten_subject_table.py
python scripts/erdc_write_paper_freeze.py
echo "[OK] LOSO serial eval done"
