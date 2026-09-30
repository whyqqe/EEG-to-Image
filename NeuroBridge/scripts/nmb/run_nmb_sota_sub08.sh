#!/usr/bin/env bash
# NeuroMem-Bridge SOTA-optimized pipeline for sub-08.
# Prioritizes mechanism validation + highest-expected CLIP paths.
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
BRAINIT="${BRAINIT:-/project/peilab/why/eeg-brainit}"
BRAIN_HIVE="${BRAIN_HIVE:-/project/peilab/why/Brain-HIVE}"
OUT="${OUT:-${NB_ROOT}/outputs/nb_nmb_sota/sub-08}"
FUSION_PRIOR_BASE="${FUSION_PRIOR_BASE:-/project/peilab/why/cache/fusion_prior/H14_B32_VAE}"
FUSION_PRIOR_FT="${FUSION_PRIOR_FT:-${OUT}/fusion_prior_finetuned}"
THINGS_EMB="${OUT}/things_embeddings"
FUSION_PRIOR="${FUSION_PRIOR:-${FUSION_PRIOR_BASE}}"
SUBJECT="${SUBJECT:-8}"
MAX_IMAGES="${MAX_IMAGES:-0}"
DEVICE="${DEVICE:-cuda:0}"
PYTHON="${PYTHON:-python}"

CKPT_RN50="${CKPT_RN50:-${NB_ROOT}/results/things_eeg/intra-subjects/20260901-074319-sub-08/checkpoint_test_best.pth}"
CLIP_TRAIN="${CLIP_TRAIN:-${BRAINIT}/outputs/atm_bridge/clip_img_train_1024.npy}"
CLIP_TEST="${CLIP_TEST:-${BRAINIT}/outputs/atm_bridge/clip_img_test_1024.npy}"

EMBED_DIR="${OUT}/embeds"
MEMORY_DIR="${OUT}/memory"
FUSION_DIR="${OUT}/fusion"
CFT_DIR="${OUT}/cft"
ENSEMBLE_DIR="${OUT}/ensemble"
GEN_ROOT="${OUT}/generation_full200"
METRICS_JSON="${OUT}/clip_fid_metrics_full200.json"
SUMMARY_JSON="${OUT}/summary.json"

# SOTA hyperparameters (tuned from nb_full_phases ablations)
CFT_EPOCHS="${CFT_EPOCHS:-120}"
CFT_PATIENCE="${CFT_PATIENCE:-18}"
ENSEMBLE_ALPHA="${ENSEMBLE_ALPHA:-0.45}"
IMG2IMG_STRENGTH="${IMG2IMG_STRENGTH:-0.5}"
RERANK_SAMPLES="${RERANK_SAMPLES:-8}"

export BRAIN_HIVE FUSION_PRIOR

mkdir -p "${OUT}" "${EMBED_DIR}" "${MEMORY_DIR}" "${FUSION_DIR}" "${CFT_DIR}" "${ENSEMBLE_DIR}" "${GEN_ROOT}"
cd "${NB_ROOT}"

echo "===== [0] Check Fusion Prior @ $(date -Iseconds) ====="
test -f "${FUSION_PRIOR_BASE}/fusion_encoder/config.json" || {
  echo "[ERROR] Fusion Prior missing at ${FUSION_PRIOR_BASE}" >&2
  exit 1
}
echo "[OK] Base Fusion Prior ready"

echo "===== [0a] THINGS parquet embeddings (for finetune) @ $(date -Iseconds) ====="
if [[ ! -f "${THINGS_EMB}/things_train_vae-part-00000-of-00000.parquet" ]]; then
  bash "${NB_ROOT}/scripts/nmb/nmb_build_things_embeddings.sh"
else
  echo "[INFO] THINGS embeddings exist"
fi

echo "===== [0b] Fusion Prior finetune on THINGS @ $(date -Iseconds) ====="
if [[ ! -f "${FUSION_PRIOR_FT}/fusion_encoder/config.json" ]]; then
  export OUT_EMB="${THINGS_EMB}"
  export OUT_PRIOR="${FUSION_PRIOR_FT}"
  export FUSION_PRIOR="${FUSION_PRIOR_BASE}"
  bash "${NB_ROOT}/scripts/nmb/nmb_finetune_fusion_prior.sh"
else
  echo "[INFO] finetuned Fusion Prior exists"
fi
export FUSION_PRIOR="${FUSION_PRIOR_FT}"
echo "[OK] Using finetuned Fusion Prior: ${FUSION_PRIOR}"

echo "===== [1] NB RN50 embeds (Head-R + raw) @ $(date -Iseconds) ====="
FULL_PHASE_EMBED="${NB_ROOT}/outputs/nb_full_phases/sub-08/embeds"
NMB_EMBED="${NB_ROOT}/outputs/nb_nmb/sub-08/embeds"
for src in "${FULL_PHASE_EMBED}" "${NMB_EMBED}"; do
  if [[ ! -f "${EMBED_DIR}/z_eeg_proj_test.npy" ]] && [[ -f "${src}/z_eeg_proj_test.npy" ]]; then
    echo "[INFO] reuse embeds from ${src}"
    cp -a "${src}/." "${EMBED_DIR}/"
    break
  fi
done
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

echo "===== [2] ViT-H Memory Router (M_R) @ $(date -Iseconds) ====="
if [[ ! -f "${MEMORY_DIR}/rag_soft5_test_clip_1024.npy" ]]; then
  "${PYTHON}" scripts/nmb/nmb_memory_router.py \
    --embed-dir "${EMBED_DIR}" \
    --clip-train "${CLIP_TRAIN}" \
    --clip-test "${CLIP_TEST}" \
    --output-dir "${MEMORY_DIR}" \
    --soft-k 5 \
    --soft-tau 0.07
else
  echo "[INFO] ViT-H memory router done"
fi

echo "===== [3] Fusion GT targets (M_G teacher) @ $(date -Iseconds) ====="
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

echo "===== [3b] Fusion-space Memory Router (M_G gallery) @ $(date -Iseconds) ====="
if [[ ! -f "${MEMORY_DIR}/fusion_mem_test.npy" ]]; then
  "${PYTHON}" scripts/nmb/nmb_build_fusion_mem.py \
    --embed-dir "${EMBED_DIR}" \
    --fusion-train "${FUSION_DIR}/fusion_train.npy" \
    --fusion-test "${FUSION_DIR}/fusion_test.npy" \
    --output-dir "${MEMORY_DIR}" \
    --soft-k 5 \
    --soft-tau 0.07
else
  echo "[INFO] fusion memory done"
fi

echo "===== [4] CFT (proj+mem_vith+mem_fusion+raw -> Fusion) @ $(date -Iseconds) ====="
if [[ ! -f "${CFT_DIR}/cft_mlp_test_fusion.npy" ]]; then
  "${PYTHON}" scripts/nmb/nmb_train_cft.py \
    --embed-dir "${EMBED_DIR}" \
    --mem-train "${MEMORY_DIR}/rag_soft5_train_clip_1024.npy" \
    --mem-test "${MEMORY_DIR}/rag_soft5_test_clip_1024.npy" \
    --fusion-mem-train "${MEMORY_DIR}/fusion_mem_train.npy" \
    --fusion-mem-test "${MEMORY_DIR}/fusion_mem_test.npy" \
    --fusion-train "${FUSION_DIR}/fusion_train.npy" \
    --fusion-test "${FUSION_DIR}/fusion_test.npy" \
    --gallery "${FUSION_DIR}/fusion_train.npy" \
    --output-dir "${CFT_DIR}" \
    --use-raw \
    --epochs "${CFT_EPOCHS}" \
    --patience "${CFT_PATIENCE}" \
    --device "${DEVICE}"
else
  echo "[INFO] CFT done"
fi

echo "===== [4b] Ensemble CFT + Fusion-mem @ $(date -Iseconds) ====="
if [[ ! -f "${ENSEMBLE_DIR}/ensemble_a${ENSEMBLE_ALPHA}_test_fusion.npy" ]]; then
  "${PYTHON}" scripts/nmb/nmb_ensemble_fusion.py \
    --cft-npy "${CFT_DIR}/cft_mlp_test_fusion.npy" \
    --mem-npy "${MEMORY_DIR}/fusion_mem_test.npy" \
    --gt-npy "${FUSION_DIR}/fusion_test.npy" \
    --output-npy "${ENSEMBLE_DIR}/ensemble_a${ENSEMBLE_ALPHA}_test_fusion.npy" \
    --alpha "${ENSEMBLE_ALPHA}" \
    --report-json "${ENSEMBLE_DIR}/ensemble_report.json"
else
  echo "[INFO] ensemble embed done"
fi

echo "===== [5] Generation (validation + SOTA candidates) @ $(date -Iseconds) ====="

# Mechanism validation: Fusion teacher upper bound (~0.63 CLIP)
echo "--- [5a] fusion_teacher (decode chain validation) ---"
"${PYTHON}" scripts/nmb/nmb_generate.py \
  --fusion-npy "${FUSION_DIR}/fusion_test.npy" \
  --prior-path "${FUSION_PRIOR}" \
  --output-dir "${GEN_ROOT}/fusion_teacher" \
  --max-images "${MAX_IMAGES}" \
  --tag "fusion_teacher" \
  --seed 42 \
  --device "${DEVICE}"

# Primary SOTA candidate: ensemble + lowlevel img2img (best prior combo family)
echo "--- [5b] nmb_sota_lowlevel (ensemble + img2img) ---"
"${PYTHON}" scripts/nmb/nmb_generate.py \
  --fusion-npy "${ENSEMBLE_DIR}/ensemble_a${ENSEMBLE_ALPHA}_test_fusion.npy" \
  --prior-path "${FUSION_PRIOR}" \
  --neighbor-idx-npy "${MEMORY_DIR}/rag_soft5_neighbor_idx_test.npy" \
  --output-dir "${GEN_ROOT}/nmb_sota_lowlevel" \
  --max-images "${MAX_IMAGES}" \
  --img2img-strength "${IMG2IMG_STRENGTH}" \
  --tag "nmb_sota_lowlevel" \
  --seed 42 \
  --device "${DEVICE}"

# Highest expected CLIP: ensemble + lowlevel + fusion-space rerank
echo "--- [5c] nmb_sota_lowlevel_rerank (ensemble + img2img + fusion rerank) ---"
"${PYTHON}" scripts/nmb/nmb_generate.py \
  --fusion-npy "${ENSEMBLE_DIR}/ensemble_a${ENSEMBLE_ALPHA}_test_fusion.npy" \
  --prior-path "${FUSION_PRIOR}" \
  --neighbor-idx-npy "${MEMORY_DIR}/rag_soft5_neighbor_idx_test.npy" \
  --output-dir "${GEN_ROOT}/nmb_sota_lowlevel_rerank" \
  --max-images "${MAX_IMAGES}" \
  --img2img-strength "${IMG2IMG_STRENGTH}" \
  --num-samples "${RERANK_SAMPLES}" \
  --rerank \
  --rerank-mode fusion \
  --tag "nmb_sota_lowlevel_rerank" \
  --seed 42 \
  --device "${DEVICE}"

# Ablation: CFT-only lowlevel (mechanism: CFT without ensemble)
echo "--- [5d] nmb_cft_lowlevel (CFT ablation) ---"
"${PYTHON}" scripts/nmb/nmb_generate.py \
  --fusion-npy "${CFT_DIR}/cft_mlp_test_fusion.npy" \
  --prior-path "${FUSION_PRIOR}" \
  --neighbor-idx-npy "${MEMORY_DIR}/rag_soft5_neighbor_idx_test.npy" \
  --output-dir "${GEN_ROOT}/nmb_cft_lowlevel" \
  --max-images "${MAX_IMAGES}" \
  --img2img-strength "${IMG2IMG_STRENGTH}" \
  --tag "nmb_cft_lowlevel" \
  --seed 42 \
  --device "${DEVICE}"

echo "===== [6] CLIP + FID @ $(date -Iseconds) ====="
"${PYTHON}" scripts/nb_adapter/eval_clip_fid.py \
  --gen-root "${GEN_ROOT}" \
  --tags "fusion_teacher,nmb_sota_lowlevel,nmb_sota_lowlevel_rerank,nmb_cft_lowlevel" \
  --output-json "${METRICS_JSON}" \
  --max-images "${MAX_IMAGES}"

echo "===== [7] Summary @ $(date -Iseconds) ====="
"${PYTHON}" - <<PY
import json
from pathlib import Path
out = Path("${OUT}")
summary = {
  "pipeline": "NeuroMem-Bridge-SOTA",
  "subject": "sub-08",
  "fusion_prior_base": "${FUSION_PRIOR_BASE}",
  "fusion_prior": "${FUSION_PRIOR}",
  "hyperparams": {
    "cft_epochs": int("${CFT_EPOCHS}"),
    "ensemble_alpha": float("${ENSEMBLE_ALPHA}"),
    "img2img_strength": float("${IMG2IMG_STRENGTH}"),
    "rerank_samples": int("${RERANK_SAMPLES}"),
  },
  "memory": json.loads((out/"memory/memory_report.json").read_text()) if (out/"memory/memory_report.json").is_file() else None,
  "fusion_mem": json.loads((out/"memory/fusion_mem_report.json").read_text()) if (out/"memory/fusion_mem_report.json").is_file() else None,
  "cft": json.loads((out/"cft/cft_report.json").read_text()) if (out/"cft/cft_report.json").is_file() else None,
  "ensemble": json.loads((out/"ensemble/ensemble_report.json").read_text()) if (out/"ensemble/ensemble_report.json").is_file() else None,
  "clip_fid": json.loads((out/"clip_fid_metrics_full200.json").read_text()) if (out/"clip_fid_metrics_full200.json").is_file() else None,
}
(out/"summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY

echo "===== DONE NMB-SOTA @ $(date -Iseconds) ====="
