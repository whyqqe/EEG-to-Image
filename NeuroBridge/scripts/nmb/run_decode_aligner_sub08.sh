#!/usr/bin/env bash
# DecodeAligner pipeline: probe targets -> train -> memory -> generation eval.
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
BRAINIT="${BRAINIT:-/project/peilab/why/eeg-brainit}"
OUT="${OUT:-${NB_ROOT}/outputs/nb_decode_aligner/sub-08}"
SOTA_V1="${NB_ROOT}/outputs/nb_nmb_sota/sub-08"
V2="${NB_ROOT}/outputs/nb_nmb_sota_v2/sub-08"
CKPT_RN50="${CKPT_RN50:-${NB_ROOT}/results/things_eeg/intra-subjects/20260901-074319-sub-08/checkpoint_test_best.pth}"
WARM_S2="${WARM_S2:-${V2}/s2_dual/checkpoint_dual_best.pth}"
CLIP_TRAIN="${BRAINIT}/outputs/atm_bridge/clip_img_train_1024.npy"
CLIP_TEST="${BRAINIT}/outputs/atm_bridge/clip_img_test_1024.npy"
DINO_TRAIN="${DINO_TRAIN:-${V2}/targets/dinov2_train.npy}"
DINO_TEST="${DINO_TEST:-${V2}/targets/dinov2_test.npy}"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"

mkdir -p "${OUT}/probe" "${OUT}/memory" "${OUT}/generation"
cd "${NB_ROOT}"

echo "===== [0] Check DINOv2 targets ====="
if [[ ! -f "${DINO_TRAIN}" ]]; then
  echo "[RUN] build DINOv2 targets"
  export HF_HUB_CACHE="/project/peilab/why/cache/huggingface/hub"
  export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
  "${PYTHON}" scripts/nmb/nmb_build_offline_targets.py --output-dir "${V2}/targets" --device "${DEVICE}"
else
  echo "[SKIP] DINOv2 targets"
fi

echo "===== [1] Encode init embeds (RN50 ckpt + S2 warm head) ====="
if [[ ! -f "${OUT}/decode_vith1024_train_clip_1024.npy" ]]; then
  WS_ARG=()
  if [[ -f "${WARM_S2}" ]]; then
    WS_ARG=(--warm-start "${WARM_S2}")
  fi
  "${PYTHON}" scripts/nmb/nmb_encode_aligner_embeds.py \
    --checkpoint "${CKPT_RN50}" \
    --output-dir "${OUT}" \
    --device "${DEVICE}" \
    "${WS_ARG[@]}"
else
  echo "[SKIP] init embeds"
fi

echo "===== [2] Train neighbor indices for probe ====="
if [[ ! -f "${OUT}/train_neighbor_idx.npy" ]]; then
  "${PYTHON}" - <<PY
import numpy as np
from pathlib import Path
out = Path("${OUT}")
# Use ViT-H 1024-d query keys (same space as CLIP gallery)
q = np.load(out / "decode_vith1024_train_clip_1024.npy").astype(np.float32)
g = np.load("${CLIP_TRAIN}").astype(np.float32)
q = q / np.linalg.norm(q, axis=1, keepdims=True).clip(1e-8)
g = g / np.linalg.norm(g, axis=1, keepdims=True).clip(1e-8)
sim = q @ g.T
idx = np.argsort(-sim, axis=1)[:, :5]
np.save(out / "train_neighbor_idx.npy", idx)
print("saved train_neighbor_idx", idx.shape)
PY
else
  echo "[SKIP] train neighbors"
fi

echo "===== [3] Build Probe-Decoder supervision (512 samples) ====="
if [[ ! -f "${OUT}/probe/probe_supervision.npz" ]]; then
  "${PYTHON}" scripts/nmb/nmb_build_probe_targets.py \
    --embed-npy "${OUT}/decode_vith1024_train_clip_1024.npy" \
    --neighbor-idx-npy "${OUT}/train_neighbor_idx.npy" \
    --clip-train-npy "${CLIP_TRAIN}" \
    --output-dir "${OUT}/probe" \
    --max-samples 512 \
    --strength 0.4 \
    --device "${DEVICE}"
else
  echo "[SKIP] probe supervision"
fi

echo "===== [4] DecodeAligner training ====="
if [[ ! -f "${OUT}/decode_aligner_report.json" ]]; then
  WS_ARG=()
  if [[ -f "${WARM_S2}" ]]; then
    WS_ARG=(--warm-start "${WARM_S2}")
  fi
  "${PYTHON}" scripts/nmb/nmb_decode_aligner_train.py \
    --checkpoint "${CKPT_RN50}" \
    --dino-train-npy "${DINO_TRAIN}" \
    --dino-test-npy "${DINO_TEST}" \
    --clip-train-npy "${CLIP_TRAIN}" \
    --clip-test-npy "${CLIP_TEST}" \
    --probe-supervision "${OUT}/probe/probe_supervision.npz" \
    --output-dir "${OUT}" \
    --num-epochs 40 \
    --batch-size 512 \
    --device "${DEVICE}" \
    "${WS_ARG[@]}"
else
  echo "[SKIP] aligner training"
fi

echo "===== [5] Memory router (new proj keys) ====="
if [[ ! -f "${OUT}/memory/rag_soft5_test_clip_1024.npy" ]]; then
  mkdir -p "${OUT}/memory"
  "${PYTHON}" scripts/nmb/nmb_memory_router.py \
    --embed-dir "${OUT}" \
    --clip-train "${CLIP_TRAIN}" \
    --clip-test "${CLIP_TEST}" \
    --output-dir "${OUT}/memory" \
    --soft-k 5 --soft-tau 0.07
else
  echo "[SKIP] memory router"
fi

echo "===== [6] Blend mem + decode embed ====="
BLEND="${OUT}/blend/mem_decode_a50.npy"
if [[ ! -f "${BLEND}" ]]; then
  mkdir -p "${OUT}/blend"
  "${PYTHON}" scripts/nb_adapter/ensemble_embeds.py \
    --rag-npy "${OUT}/memory/rag_soft5_test_clip_1024.npy" \
    --prior-npy "${OUT}/decode_vith1024_test_clip_1024.npy" \
    --output-npy "${BLEND}" \
    --alpha 0.5
else
  echo "[SKIP] blend"
fi

echo "===== [7] Generation (focused paths) ====="
NEIGH="${OUT}/memory/rag_soft5_neighbor_idx_test.npy"
run_gen() {
  local tag="$1" emb="$2" s="$3"
  local gdir="${OUT}/generation/${tag}"
  if [[ -f "${gdir}/generated/199.png" ]]; then
    echo "[SKIP] gen ${tag}"
    return 0
  fi
  "${PYTHON}" scripts/nb_adapter/generate_rag_lowlevel.py \
    --embed-npy "${emb}" \
    --neighbor-idx-npy "${NEIGH}" \
    --output-dir "${gdir}" \
    --strength "${s}" \
    --seed 42 \
    --tag "${tag}" \
    --skip-metrics
}

run_gen "decode_direct_s40" "${OUT}/decode_vith1024_test_clip_1024.npy" 0.4
run_gen "blend_mem_decode_s40" "${BLEND}" 0.4

echo "===== [8] CLIP/FID + gap report ====="
METRICS="${OUT}/clip_fid_decode_aligner.json"
"${PYTHON}" scripts/nb_adapter/eval_clip_fid.py \
  --gen-root "${OUT}/generation" \
  --tags "decode_direct_s40,blend_mem_decode_s40" \
  --output-json "${METRICS}"

"${PYTHON}" - <<PY
import json
from pathlib import Path
out = Path("${OUT}")
report = json.loads((out / "decode_aligner_report.json").read_text())
metrics = json.loads(Path("${METRICS}").read_text()) if Path("${METRICS}").is_file() else {}
results = metrics.get("results", [])
best = max(results, key=lambda r: r.get("clip_cosine", 0)) if results else None
direct_cos = report.get("best_direct_cos", 0)
gen_clip = best.get("clip_cosine") if best else None
gap = direct_cos - gen_clip if gen_clip else None
summary = {
    "pipeline": "DecodeAligner",
    "best_direct_cos": direct_cos,
    "best_gen": best,
    "decode_gap": gap,
    "baseline_v1": 0.412,
    "report": report,
}
(out / "summary_decode_aligner.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

echo "===== DONE DecodeAligner @ $(date -Iseconds) ====="
