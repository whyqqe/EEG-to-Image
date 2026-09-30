#!/usr/bin/env bash
# Generate 200 test images + CLIP/FID for ViT-H NB pipeline.
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
BRAINIT="${BRAINIT:-/project/peilab/why/eeg-brainit}"
OUT="${OUT:-${NB_ROOT}/outputs/nb_adapter/sub-08-vit-h}"
GEN_ROOT="${OUT}/generation_full200"
METRICS_JSON="${OUT}/clip_fid_metrics_full200.json"
MAX_IMAGES=0  # all 200
DEVICE="${DEVICE:-cuda:0}"
PYTHON="${PYTHON:-python}"

CLIP_TEST="${BRAINIT}/outputs/atm_bridge/clip_img_test_1024.npy"
PRIOR_NPY="${BRAINIT}/outputs/eval/atm_pipeline_sub08/sub-08_prior_clip_1024.npy"
ADAPT_DIR="${OUT}/adapters"

mkdir -p "${GEN_ROOT}"
cd "${NB_ROOT}"

echo "===== Generate 200 images @ $(date -Iseconds) ====="
declare -A EMBEDS=(
  [mlp]="${ADAPT_DIR}/mlp_test_clip_1024.npy"
  [teacher]="${CLIP_TEST}"
  [atm_prior]="${PRIOR_NPY}"
)

for tag in mlp teacher atm_prior; do
  echo "--- ${tag} (n=200) ---"
  "${PYTHON}" scripts/nb_adapter/generate_from_embeds.py \
    --embed-npy "${EMBEDS[${tag}]}" \
    --output-dir "${GEN_ROOT}/${tag}" \
    --tag "vith_full200_${tag}" \
    --max-images "${MAX_IMAGES}" \
    --seed 42 \
    --skip-metrics
done

echo "===== CLIP + FID (200) @ $(date -Iseconds) ====="
"${PYTHON}" scripts/nb_adapter/eval_clip_fid.py \
  --gen-root "${GEN_ROOT}" \
  --tags "mlp,teacher,atm_prior" \
  --output-json "${METRICS_JSON}" \
  --max-images 0

echo "===== DONE full200 @ $(date -Iseconds) ====="
