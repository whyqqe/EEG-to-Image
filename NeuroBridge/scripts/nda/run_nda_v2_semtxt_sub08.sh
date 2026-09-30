#!/usr/bin/env bash
# NDA-v2 + CLIP-Text semantic alignment (sub-08)
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
BRAINIT="${BRAINIT:-/project/peilab/why/eeg-brainit}"
OUT="${OUT:-${NB_ROOT}/outputs/nda_v2_semtxt/sub-08}"
OLD_OUT="${OLD_OUT:-${NB_ROOT}/outputs/nda_v2/sub-08}"
CKPT_RN50="${CKPT_RN50:-${NB_ROOT}/results/things_eeg/intra-subjects/20260901-074319-sub-08/checkpoint_test_best.pth}"
V2_TARGETS="${NB_ROOT}/outputs/nb_nmb_sota_v2/sub-08/targets"
DINO_TRAIN="${DINO_TRAIN:-${V2_TARGETS}/dinov2_train.npy}"
DINO_TEST="${DINO_TEST:-${V2_TARGETS}/dinov2_test.npy}"
CLIP_TRAIN="${BRAINIT}/outputs/atm_bridge/clip_img_train_1024.npy"
CLIP_TEST="${BRAINIT}/outputs/atm_bridge/clip_img_test_1024.npy"
IMAGES_ROOT="${IMAGES_ROOT:-/project/peilab/why/data/images_set}"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"
LAYERS="${LAYERS:-8,10,12,14,16,18,20,22,24,28}"

mkdir -p "${OUT}/clip_layers" "${OUT}/clip_text" "${OUT}/embeds" "${OUT}/train" "${OUT}/memory" "${OUT}/generation" "${OUT}/blend"
cd "${NB_ROOT}"

echo "===== [0] Prefetch pretrained models @ $(date -Iseconds) ====="
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
"${PYTHON}" - <<'PY'
import os
os.environ.setdefault("HF_HUB_CACHE", "/project/peilab/why/cache/eeg-brainit/hf/hub")
os.environ.setdefault("OPENCLIP_CACHE_DIR", "/project/peilab/why/cache/eeg-brainit/open_clip")
os.environ.setdefault("TORCH_HOME", "/project/peilab/why/cache/eeg-brainit/torch")
import open_clip, timm
print("[INFO] OpenCLIP ViT-H-14 ...")
m, _, _ = open_clip.create_model_and_transforms("ViT-H-14", pretrained="laion2b_s32b_b79k", device="cpu")
print("[OK] blocks=", len(m.visual.transformer.resblocks))
print("[INFO] DINOv2 ...")
_ = timm.create_model("vit_large_patch14_reg4_dinov2.lvd142m", pretrained=True, num_classes=0)
print("[OK] pretrained ready")
PY

echo "===== [1] DINOv2 targets ====="
if [[ ! -f "${DINO_TRAIN}" ]]; then
  "${PYTHON}" scripts/nmb/nmb_build_offline_targets.py --output-dir "${V2_TARGETS}" --device "${DEVICE}"
else
  echo "[SKIP] DINOv2"
fi

echo "===== [2] NB SSP embeds ====="
if [[ ! -f "${OUT}/embeds/z_eeg_proj_test.npy" ]]; then
  if [[ -f "${OLD_OUT}/embeds/z_eeg_proj_test.npy" ]]; then
    cp -a "${OLD_OUT}/embeds/." "${OUT}/embeds/"
    echo "[OK] reused embeds from ${OLD_OUT}"
  else
    "${PYTHON}" scripts/nmb/nmb_encode_aligner_embeds.py \
      --checkpoint "${CKPT_RN50}" --output-dir "${OUT}/embeds" --device "${DEVICE}"
  fi
else
  echo "[SKIP] embeds"
fi

echo "===== [3] CLIP intermediate layers (Perception) ====="
if [[ ! -f "${OUT}/clip_layers/clip_layers_report.json" ]]; then
  if [[ -f "${OLD_OUT}/clip_layers/clip_layers_report.json" ]]; then
    cp -a "${OLD_OUT}/clip_layers/." "${OUT}/clip_layers/"
    echo "[OK] reused clip layers"
  else
    "${PYTHON}" scripts/nda/extract_clip_layers.py \
      --images-root "${IMAGES_ROOT}" \
      --output-dir "${OUT}/clip_layers" \
      --layers "${LAYERS}" \
      --batch-size 16 \
      --device "${DEVICE}"
  fi
else
  echo "[SKIP] clip layers"
fi

echo "===== [3b] CLIP-Text features (Semantic) ====="
if [[ ! -f "${OUT}/clip_text/clip_text_report.json" ]]; then
  "${PYTHON}" scripts/nda/extract_clip_text.py \
    --images-root "${IMAGES_ROOT}" \
    --output-dir "${OUT}/clip_text" \
    --device "${DEVICE}"
else
  echo "[SKIP] clip text"
fi

echo "===== [4] NVOL scan ====="
NVOL_JSON="${OUT}/nvol_scan.json"
if [[ ! -f "${NVOL_JSON}" ]]; then
  "${PYTHON}" scripts/nda/nda_nvol_scan.py \
    --eeg-train "${OUT}/embeds/z_eeg_proj_train.npy" \
    --eeg-test "${OUT}/embeds/z_eeg_proj_test.npy" \
    --clip-layers-dir "${OUT}/clip_layers" \
    --output-json "${NVOL_JSON}" \
    --top-k-layers 3
else
  echo "[SKIP] NVOL"
fi

echo "===== [5] Probe supervision ====="
PROBE="${OUT}/probe/probe_supervision.npz"
DA_PROBE="${NB_ROOT}/outputs/nb_decode_aligner/sub-08/probe/probe_supervision.npz"
mkdir -p "${OUT}/probe"
if [[ ! -f "${PROBE}" && -f "${DA_PROBE}" ]]; then
  cp -f "${DA_PROBE}" "${PROBE}"
  echo "[OK] reused DA probe"
fi

echo "===== [6] Dual-stream train (CLIP-Image + CLIP-Text) ====="
if [[ ! -f "${OUT}/train/nda_train_report.json" ]]; then
  PROBE_ARG=()
  [[ -f "${PROBE}" ]] && PROBE_ARG=(--probe-supervision "${PROBE}")
  "${PYTHON}" scripts/nda/nda_dual_train.py \
    --checkpoint "${CKPT_RN50}" \
    --output-dir "${OUT}/train" \
    --clip-layers-dir "${OUT}/clip_layers" \
    --nvol-json "${NVOL_JSON}" \
    --dino-train-npy "${DINO_TRAIN}" \
    --dino-test-npy "${DINO_TEST}" \
    --clip-train-npy "${CLIP_TRAIN}" \
    --clip-test-npy "${CLIP_TEST}" \
    --text-train-npy "${OUT}/clip_text/train/text_flat_clip.npy" \
    --text-test-npy "${OUT}/clip_text/test/text_flat_clip.npy" \
    --lambda-rn50 0.8 \
    --lambda-txt 0.2 \
    --num-epochs 40 \
    --phase1-epochs 10 \
    --phase2-epochs 25 \
    --batch-size 512 \
    --device "${DEVICE}" \
    --freeze-backbone \
    "${PROBE_ARG[@]}"
  cp -f "${OUT}/train/z_decode_vith_train.npy" "${OUT}/train/decode_vith1024_train_clip_1024.npy"
  cp -f "${OUT}/train/z_decode_vith_test.npy" "${OUT}/train/decode_vith1024_test_clip_1024.npy"
else
  echo "[SKIP] train"
fi

echo "===== [7] Memory ====="
if [[ ! -f "${OUT}/memory/rag_soft5_test_clip_1024.npy" ]]; then
  "${PYTHON}" scripts/nmb/nmb_memory_router.py \
    --embed-dir "${OUT}/train" \
    --clip-train "${CLIP_TRAIN}" \
    --clip-test "${CLIP_TEST}" \
    --output-dir "${OUT}/memory" \
    --input-key proj --soft-k 5 --soft-tau 0.07
else
  echo "[SKIP] memory"
fi

echo "===== [8] Blends ====="
mkdir -p "${OUT}/blend"
BLEND_DEC="${OUT}/blend/mem_decode_a50.npy"
BLEND_FUSE="${OUT}/blend/mem_fuse_a50.npy"
[[ -f "${BLEND_DEC}" ]] || "${PYTHON}" scripts/nb_adapter/ensemble_embeds.py \
  --rag-npy "${OUT}/memory/rag_soft5_test_clip_1024.npy" \
  --prior-npy "${OUT}/train/z_decode_vith_test.npy" \
  --output-npy "${BLEND_DEC}" --alpha 0.5
[[ -f "${BLEND_FUSE}" ]] || "${PYTHON}" scripts/nb_adapter/ensemble_embeds.py \
  --rag-npy "${OUT}/memory/rag_soft5_test_clip_1024.npy" \
  --prior-npy "${OUT}/train/z_fuse_test.npy" \
  --output-npy "${BLEND_FUSE}" --alpha 0.5

echo "===== [9] Generation ====="
NEIGH="${OUT}/memory/rag_soft5_neighbor_idx_test.npy"
run_gen() {
  local tag="$1" emb="$2"
  local gdir="${OUT}/generation/${tag}"
  [[ -f "${gdir}/generated/199.png" ]] && echo "[SKIP] gen ${tag}" && return 0
  "${PYTHON}" scripts/nb_adapter/generate_rag_lowlevel.py \
    --embed-npy "${emb}" --neighbor-idx-npy "${NEIGH}" \
    --output-dir "${gdir}" --strength 0.4 --seed 42 --tag "${tag}" --skip-metrics
}
run_gen "nda_decode_s40" "${OUT}/train/z_decode_vith_test.npy"
run_gen "nda_fuse_s40" "${OUT}/train/z_fuse_test.npy"
run_gen "nda_mem_decode_s40" "${BLEND_DEC}"
run_gen "nda_mem_fuse_s40" "${BLEND_FUSE}"

echo "===== [10] Metrics ====="
METRICS="${OUT}/clip_fid_metrics.json"
"${PYTHON}" scripts/nb_adapter/eval_clip_fid.py \
  --gen-root "${OUT}/generation" \
  --tags "nda_decode_s40,nda_fuse_s40,nda_mem_decode_s40,nda_mem_fuse_s40" \
  --output-json "${METRICS}"

"${PYTHON}" - <<PY
import json
from pathlib import Path
out = Path("${OUT}")
metrics = json.loads((out/"clip_fid_metrics.json").read_text()) if (out/"clip_fid_metrics.json").is_file() else {}
results = metrics.get("results", [])
best = max(results, key=lambda r: r.get("clip_cosine", 0)) if results else None
nvol = json.loads((out/"nvol_scan.json").read_text()) if (out/"nvol_scan.json").is_file() else {}
train = json.loads((out/"train/nda_train_report.json").read_text()) if (out/"train/nda_train_report.json").is_file() else {}
summary = {
  "pipeline": "NDA-v2-semtxt-sub08",
  "semantic": "CLIP-Image(RN50/SSP) + CLIP-Text (λ=0.8/0.2)",
  "perception": "HCF(NVOL)+DINOv2",
  "nvol_best_layer": nvol.get("best_layer"),
  "nvol_top_k": nvol.get("top_k_layers"),
  "train": {k: train.get(k) for k in ("best_epoch","best_score","layer_ids","semantic_primary","lambda_img","lambda_txt") if train},
  "baseline_v1": 0.412,
  "baseline_erdc": 0.422,
  "best_gen": best,
  "all_gen": results,
}
(out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

echo "===== DONE NDA-v2-semtxt @ $(date -Iseconds) ====="
