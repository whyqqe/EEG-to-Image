#!/usr/bin/env bash
# NeuroMem-Bridge (NMB) full pipeline for sub-08.
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
BRAINIT="${BRAINIT:-/project/peilab/why/eeg-brainit}"
BRAIN_HIVE="${BRAIN_HIVE:-/project/peilab/why/Brain-HIVE}"
FUSION_PRIOR="${FUSION_PRIOR:-/project/peilab/why/cache/fusion_prior/H14_B32_VAE}"
OUT="${OUT:-${NB_ROOT}/outputs/nb_nmb/sub-08}"
SUBJECT="${SUBJECT:-8}"
MAX_IMAGES="${MAX_IMAGES:-0}"
DEVICE="${DEVICE:-cuda:0}"
PYTHON="${PYTHON:-python}"

CKPT_RN50="${CKPT_RN50:-${NB_ROOT}/results/things_eeg/intra-subjects/20260901-074319-sub-08/checkpoint_test_best.pth}"
CLIP_TRAIN="${CLIP_TRAIN:-${BRAINIT}/outputs/atm_bridge/clip_img_train_1024.npy}"
CLIP_TEST="${CLIP_TEST:-${BRAINIT}/outputs/atm_bridge/clip_img_test_1024.npy}"
GALLERY_FUSION="${GALLERY_FUSION:-${OUT}/fusion/fusion_train.npy}"

EMBED_DIR="${OUT}/embeds"
MEMORY_DIR="${OUT}/memory"
FUSION_DIR="${OUT}/fusion"
CFT_DIR="${OUT}/cft"
GEN_ROOT="${OUT}/generation_full200"
METRICS_JSON="${OUT}/clip_fid_metrics_full200.json"
SUMMARY_JSON="${OUT}/summary.json"

export BRAIN_HIVE FUSION_PRIOR

mkdir -p "${OUT}" "${EMBED_DIR}" "${MEMORY_DIR}" "${FUSION_DIR}" "${CFT_DIR}" "${GEN_ROOT}"
cd "${NB_ROOT}"

echo "===== [0] Check Fusion Prior @ $(date -Iseconds) ====="
test -f "${FUSION_PRIOR}/fusion_encoder/config.json" || {
  echo "[ERROR] Fusion Prior missing at ${FUSION_PRIOR}" >&2
  exit 1
}
echo "[OK] Fusion Prior ready"

echo "===== [1] NB RN50 embeds (Head-R) @ $(date -Iseconds) ====="
FULL_PHASE_EMBED="${NB_ROOT}/outputs/nb_full_phases/sub-08/embeds"
if [[ ! -f "${EMBED_DIR}/z_eeg_proj_test.npy" ]] && [[ -f "${FULL_PHASE_EMBED}/z_eeg_proj_test.npy" ]]; then
  echo "[INFO] reuse embeds from nb_full_phases"
  cp -a "${FULL_PHASE_EMBED}/." "${EMBED_DIR}/"
fi
if [[ ! -f "${EMBED_DIR}/z_eeg_proj_test.npy" ]]; then
  "${PYTHON}" scripts/nb_adapter/extract_nb_embeds.py \
    --nb-root "${NB_ROOT}" \
    --checkpoint "${CKPT_RN50}" \
    --subject "${SUBJECT}" \
    --output-dir "${EMBED_DIR}" \
    --device "${DEVICE}"
else
  echo "[INFO] embeds exist"
fi

echo "===== [2] Memory Router (M_R -> gallery ViT-H) @ $(date -Iseconds) ====="
if [[ ! -f "${MEMORY_DIR}/rag_soft5_test_clip_1024.npy" ]]; then
  "${PYTHON}" scripts/nmb/nmb_memory_router.py \
    --embed-dir "${EMBED_DIR}" \
    --clip-train "${CLIP_TRAIN}" \
    --clip-test "${CLIP_TEST}" \
    --output-dir "${MEMORY_DIR}"
else
  echo "[INFO] memory router done"
fi

echo "===== [3] Fusion GT targets (M_G teacher coords) @ $(date -Iseconds) ====="
if [[ ! -f "${FUSION_DIR}/fusion_test.npy" ]]; then
  "${PYTHON}" scripts/nmb/nmb_build_fusion_targets.py \
    --embed-dir "${EMBED_DIR}" \
    --images-root "/project/peilab/why/data/images_set" \
    --clip-h-train "${CLIP_TRAIN}" \
    --clip-h-test "${CLIP_TEST}" \
    --prior-path "${FUSION_PRIOR}" \
    --output-dir "${FUSION_DIR}" \
    --batch-size 32 \
    --device "${DEVICE}"
else
  echo "[INFO] fusion targets exist"
fi

echo "===== [4] CFT transport (NB+mem -> Fusion) @ $(date -Iseconds) ====="
if [[ ! -f "${CFT_DIR}/cft_mlp_test_fusion.npy" ]]; then
  "${PYTHON}" scripts/nmb/nmb_train_cft.py \
    --embed-dir "${EMBED_DIR}" \
    --mem-train "${MEMORY_DIR}/rag_soft5_train_clip_1024.npy" \
    --mem-test "${MEMORY_DIR}/rag_soft5_test_clip_1024.npy" \
    --fusion-train "${FUSION_DIR}/fusion_train.npy" \
    --fusion-test "${FUSION_DIR}/fusion_test.npy" \
    --gallery "${FUSION_DIR}/fusion_train.npy" \
    --output-dir "${CFT_DIR}" \
    --epochs 80 \
    --device "${DEVICE}"
else
  echo "[INFO] CFT done"
fi

echo "===== [5] Tri-condition generation @ $(date -Iseconds) ====="
declare -A FUSION_NPYS=(
  [fusion_teacher]="${FUSION_DIR}/fusion_test.npy"
  [nmb_cft_mlp]="${CFT_DIR}/cft_mlp_test_fusion.npy"
)

for tag in fusion_teacher nmb_cft_mlp; do
  echo "--- gen ${tag} ---"
  "${PYTHON}" scripts/nmb/nmb_generate.py \
    --fusion-npy "${FUSION_NPYS[${tag}]}" \
    --prior-path "${FUSION_PRIOR}" \
    --output-dir "${GEN_ROOT}/${tag}" \
    --max-images "${MAX_IMAGES}" \
    --tag "${tag}" \
    --seed 42 \
    --device "${DEVICE}"
done

echo "--- gen nmb_cft_lowlevel ---"
"${PYTHON}" scripts/nmb/nmb_generate.py \
  --fusion-npy "${CFT_DIR}/cft_mlp_test_fusion.npy" \
  --prior-path "${FUSION_PRIOR}" \
  --neighbor-idx-npy "${MEMORY_DIR}/rag_soft5_neighbor_idx_test.npy" \
  --output-dir "${GEN_ROOT}/nmb_cft_lowlevel" \
  --max-images "${MAX_IMAGES}" \
  --img2img-strength 0.5 \
  --tag "nmb_cft_lowlevel" \
  --seed 42 \
  --device "${DEVICE}"

echo "--- gen nmb_cft_rerank ---"
"${PYTHON}" scripts/nmb/nmb_generate.py \
  --fusion-npy "${CFT_DIR}/cft_mlp_test_fusion.npy" \
  --prior-path "${FUSION_PRIOR}" \
  --nb-proj-npy "${EMBED_DIR}/z_eeg_proj_test.npy" \
  --output-dir "${GEN_ROOT}/nmb_cft_rerank" \
  --max-images "${MAX_IMAGES}" \
  --num-samples 8 \
  --rerank \
  --tag "nmb_cft_rerank" \
  --seed 42 \
  --device "${DEVICE}"

echo "===== [6] CLIP + FID @ $(date -Iseconds) ====="
"${PYTHON}" scripts/nb_adapter/eval_clip_fid.py \
  --gen-root "${GEN_ROOT}" \
  --tags "fusion_teacher,nmb_cft_mlp,nmb_cft_lowlevel,nmb_cft_rerank" \
  --output-json "${METRICS_JSON}" \
  --max-images "${MAX_IMAGES}"

echo "===== [7] Summary @ $(date -Iseconds) ====="
"${PYTHON}" - <<PY
import json
from pathlib import Path
out = Path("${OUT}")
summary = {
  "pipeline": "NeuroMem-Bridge",
  "subject": "sub-08",
  "fusion_prior": "${FUSION_PRIOR}",
  "memory": json.loads((out/"memory/memory_report.json").read_text()) if (out/"memory/memory_report.json").is_file() else None,
  "cft": json.loads((out/"cft/cft_report.json").read_text()) if (out/"cft/cft_report.json").is_file() else None,
  "clip_fid": json.loads((out/"clip_fid_metrics_full200.json").read_text()) if (out/"clip_fid_metrics_full200.json").is_file() else None,
}
(out/"summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY

echo "===== DONE NMB @ $(date -Iseconds) ====="
