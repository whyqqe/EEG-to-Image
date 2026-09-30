#!/usr/bin/env bash
# DA-Calibrator (SCC) pipeline: probe targets -> train calibrator -> memory -> generation eval.
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
BRAINIT="${BRAINIT:-/project/peilab/why/eeg-brainit}"
OUT="${OUT:-${NB_ROOT}/outputs/nb_scc/sub-08}"
DECODE_ALIGN="${NB_ROOT}/outputs/nb_decode_aligner/sub-08"
V2="${NB_ROOT}/outputs/nb_nmb_sota_v2/sub-08"
CKPT_RN50="${CKPT_RN50:-${NB_ROOT}/results/things_eeg/intra-subjects/20260901-074319-sub-08/checkpoint_test_best.pth}"
WARM_PROBE="${WARM_PROBE:-${DECODE_ALIGN}/checkpoint_decode_aligner_best.pth}"
CLIP_TRAIN="${BRAINIT}/outputs/atm_bridge/clip_img_train_1024.npy"
CLIP_TEST="${BRAINIT}/outputs/atm_bridge/clip_img_test_1024.npy"
DINO_TRAIN="${DINO_TRAIN:-${V2}/targets/dinov2_train.npy}"
DINO_TEST="${DINO_TEST:-${V2}/targets/dinov2_test.npy}"
PROBE_SAMPLES="${PROBE_SAMPLES:-2000}"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"

mkdir -p "${OUT}/probe" "${OUT}/memory" "${OUT}/generation" "${OUT}/blend"
cd "${NB_ROOT}"

echo "===== [0] DINOv2 targets ====="
if [[ ! -f "${DINO_TRAIN}" ]]; then
  export HF_HUB_CACHE="/project/peilab/why/cache/huggingface/hub"
  export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
  "${PYTHON}" scripts/nmb/nmb_build_offline_targets.py --output-dir "${V2}/targets" --device "${DEVICE}"
else
  echo "[SKIP] DINOv2 targets"
fi

echo "===== [1] Probe supervision (reuse or build ${PROBE_SAMPLES}) ====="
PROBE_NPZ="${OUT}/probe/probe_supervision.npz"
if [[ -f "${PROBE_NPZ}" ]]; then
  echo "[SKIP] probe exists"
elif [[ -f "${DECODE_ALIGN}/probe/probe_supervision.npz" ]]; then
  echo "[COPY] decode_aligner probe -> SCC"
  cp "${DECODE_ALIGN}/probe/probe_supervision.npz" "${PROBE_NPZ}"
  cp -f "${DECODE_ALIGN}/probe/probe_build_report.json" "${OUT}/probe/" 2>/dev/null || true
else
  NEIGH="${OUT}/train_neighbor_idx.npy"
  if [[ ! -f "${NEIGH}" ]]; then
    if [[ -f "${DECODE_ALIGN}/train_neighbor_idx.npy" ]]; then
      cp "${DECODE_ALIGN}/train_neighbor_idx.npy" "${NEIGH}"
    else
      "${PYTHON}" - <<PY
import numpy as np
from pathlib import Path
out = Path("${OUT}")
q = np.load("${DECODE_ALIGN}/decode_vith1024_train_clip_1024.npy").astype(np.float32)
g = np.load("${CLIP_TRAIN}").astype(np.float32)
q = q / np.linalg.norm(q, axis=1, keepdims=True).clip(1e-8)
g = g / np.linalg.norm(g, axis=1, keepdims=True).clip(1e-8)
idx = np.argsort(-q @ g.T, axis=1)[:, :5]
np.save(out / "train_neighbor_idx.npy", idx)
PY
    fi
  fi
  EMBED_INIT="${DECODE_ALIGN}/decode_vith1024_train_clip_1024.npy"
  if [[ ! -f "${EMBED_INIT}" ]]; then
    "${PYTHON}" scripts/nmb/nmb_encode_aligner_embeds.py \
      --checkpoint "${CKPT_RN50}" --output-dir "${DECODE_ALIGN}" --device "${DEVICE}"
    EMBED_INIT="${DECODE_ALIGN}/decode_vith1024_train_clip_1024.npy"
  fi
  "${PYTHON}" scripts/nmb/nmb_build_probe_targets.py \
    --embed-npy "${EMBED_INIT}" \
    --neighbor-idx-npy "${NEIGH}" \
    --clip-train-npy "${CLIP_TRAIN}" \
    --output-dir "${OUT}/probe" \
    --max-samples "${PROBE_SAMPLES}" \
    --strength 0.4 \
    --device "${DEVICE}"
fi

echo "===== [2] Train SCC (frozen NB + anchor-residual calibrator) ====="
if [[ ! -f "${OUT}/scc_report.json" ]]; then
  WS_ARG=()
  if [[ -f "${WARM_PROBE}" ]]; then
    WS_ARG=(--warm-probe "${WARM_PROBE}")
  fi
  "${PYTHON}" scripts/nmb/nmb_scc_train.py \
    --checkpoint "${CKPT_RN50}" \
    --dino-train-npy "${DINO_TRAIN}" \
    --dino-test-npy "${DINO_TEST}" \
    --clip-train-npy "${CLIP_TRAIN}" \
    --clip-test-npy "${CLIP_TEST}" \
    --probe-supervision "${PROBE_NPZ}" \
    --output-dir "${OUT}" \
    --num-epochs 50 \
    --batch-size 512 \
    --learning-rate 5e-4 \
    --device "${DEVICE}" \
    "${WS_ARG[@]}"
else
  echo "[SKIP] SCC training"
fi

echo "===== [3] Memory router (proj keys) ====="
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

echo "===== [4] Blend mem + SCC condition ====="
BLEND="${OUT}/blend/mem_scc_a50.npy"
if [[ ! -f "${BLEND}" ]]; then
  "${PYTHON}" scripts/nb_adapter/ensemble_embeds.py \
    --rag-npy "${OUT}/memory/rag_soft5_test_clip_1024.npy" \
    --prior-npy "${OUT}/scc_cond_test_clip_1024.npy" \
    --output-npy "${BLEND}" \
    --alpha 0.5
else
  echo "[SKIP] blend"
fi

echo "===== [5] Generation ====="
NEIGH="${OUT}/memory/rag_soft5_neighbor_idx_test.npy"
SIGMA_NPY="${OUT}/scc_sigma_test.npy"

run_gen() {
  local tag="$1" emb="$2" s="$3" use_sigma="${4:-0}"
  local gdir="${OUT}/generation/${tag}"
  if [[ -f "${gdir}/generated/199.png" ]]; then
    echo "[SKIP] gen ${tag}"
    return 0
  fi
  SIG_ARG=()
  if [[ "${use_sigma}" == "1" && -f "${SIGMA_NPY}" ]]; then
    SIG_ARG=(--strength-npy "${SIGMA_NPY}")
  fi
  "${PYTHON}" scripts/nb_adapter/generate_rag_lowlevel.py \
    --embed-npy "${emb}" \
    --neighbor-idx-npy "${NEIGH}" \
    --output-dir "${gdir}" \
    --strength "${s}" \
  "${SIG_ARG[@]}" \
    --seed 42 \
    --tag "${tag}" \
    --skip-metrics
}

run_gen "scc_direct_s40" "${OUT}/scc_cond_test_clip_1024.npy" 0.4 0
run_gen "scc_direct_sigma" "${OUT}/scc_cond_test_clip_1024.npy" 0.4 1
run_gen "blend_mem_scc_s40" "${BLEND}" 0.4 0

echo "===== [6] CLIP/FID + summary ====="
METRICS="${OUT}/clip_fid_scc.json"
"${PYTHON}" scripts/nb_adapter/eval_clip_fid.py \
  --gen-root "${OUT}/generation" \
  --tags "scc_direct_s40,scc_direct_sigma,blend_mem_scc_s40" \
  --output-json "${METRICS}"

"${PYTHON}" - <<PY
import json
from pathlib import Path
out = Path("${OUT}")
report = json.loads((out / "scc_report.json").read_text())
metrics = json.loads(Path("${METRICS}").read_text()) if Path("${METRICS}").is_file() else {}
results = metrics.get("results", [])
best = max(results, key=lambda r: r.get("clip_cosine", 0)) if results else None
cond_cos = report.get("best_decode_score", report.get("final_cond_cos", 0))
gen_clip = best.get("clip_cosine") if best else None
summary = {
    "pipeline": "DA-Calibrator-SCC",
    "best_cond_cos": report.get("final_cond_cos"),
    "best_decode_score": report.get("best_decode_score"),
    "best_gen": best,
    "decode_gap": float(report.get("final_cond_cos", 0)) - gen_clip if gen_clip else None,
    "baseline_v1": 0.412,
    "baseline_decode_aligner": 0.4182,
    "report": report,
}
(out / "summary_scc.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

echo "===== DONE SCC @ $(date -Iseconds) ====="
