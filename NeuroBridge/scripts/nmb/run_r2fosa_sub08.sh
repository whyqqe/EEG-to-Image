#!/usr/bin/env bash
# R²-FOSA pipeline: D²-FOSA-style FSTDE + Anchor-DDLG + NB retrieval prior (no ERDC).
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
BRAINIT="${BRAINIT:-/project/peilab/why/eeg-brainit}"
OUT="${OUT:-${NB_ROOT}/outputs/nb_r2fosa/sub-08}"
V2="${NB_ROOT}/outputs/nb_nmb_sota_v2/sub-08"
CKPT_RN50="${CKPT_RN50:-${NB_ROOT}/results/things_eeg/intra-subjects/20260901-074319-sub-08/checkpoint_test_best.pth}"
CLIP_TRAIN="${BRAINIT}/outputs/atm_bridge/clip_img_train_1024.npy"
CLIP_TEST="${BRAINIT}/outputs/atm_bridge/clip_img_test_1024.npy"
DINO_TRAIN="${DINO_TRAIN:-${V2}/targets/dinov2_train.npy}"
DINO_TEST="${DINO_TEST:-${V2}/targets/dinov2_test.npy}"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"

mkdir -p "${OUT}/memory" "${OUT}/blend" "${OUT}/generation"
cd "${NB_ROOT}"

echo "===== [0] DINOv2 offline targets ====="
if [[ ! -f "${DINO_TRAIN}" ]]; then
  export HF_HUB_CACHE="/project/peilab/why/cache/huggingface/hub"
  export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
  "${PYTHON}" scripts/nmb/nmb_build_offline_targets.py --output-dir "${V2}/targets" --device "${DEVICE}"
else
  echo "[SKIP] DINOv2 targets"
fi

RETRAIN="${RETRAIN:-0}"
if [[ "${RETRAIN}" == "1" ]]; then
  echo "[RETRAIN] clearing prior R²-FOSA artifacts"
  rm -f "${OUT}/r2fosa_report.json" "${OUT}/summary_r2fosa.json" "${OUT}/clip_fid_r2fosa.json"
  rm -f "${OUT}/r2fosa_best.pth" "${OUT}/r2fosa_phase1_best.pth"
  rm -rf "${OUT}/generation" "${OUT}/blend" "${OUT}/memory"
  rm -f "${OUT}"/r2fosa_*_clip_1024.npy "${OUT}"/e_anchor_*_clip_1024.npy "${OUT}"/z_eeg_proj_*.npy
fi

echo "===== [1] R²-FOSA train + export embeds ====="
if [[ ! -f "${OUT}/r2fosa_report.json" ]]; then
  "${PYTHON}" scripts/nmb/nmb_r2fosa_train.py \
    --checkpoint "${CKPT_RN50}" \
    --clip-train-npy "${CLIP_TRAIN}" \
    --clip-test-npy "${CLIP_TEST}" \
    --dino-train-npy "${DINO_TRAIN}" \
    --dino-test-npy "${DINO_TEST}" \
    --output-dir "${OUT}" \
    --phase1-epochs 20 \
    --phase2-epochs 60 \
    --phase2-min-epochs 30 \
    --batch-size 512 \
    --patience 15 \
    --warm-t-frac 0.35 \
    --val-ddlg-weight 0.7 \
    --device "${DEVICE}"
else
  echo "[SKIP] R²-FOSA training"
fi

echo "===== [2] Memory router (NB proj keys -> soft RAG) ====="
if [[ ! -f "${OUT}/memory/rag_soft5_test_clip_1024.npy" ]]; then
  "${PYTHON}" scripts/nmb/nmb_memory_router.py \
    --embed-dir "${OUT}" \
    --clip-train "${CLIP_TRAIN}" \
    --clip-test "${CLIP_TEST}" \
    --output-dir "${OUT}/memory" \
    --input-key proj \
    --soft-k 5 --soft-tau 0.07
else
  echo "[SKIP] memory router"
fi

echo "===== [3] Blend mem + DDLG anchor embeds ====="
BLEND_A50="${OUT}/blend/mem_ddlg_anchor_a50.npy"
BLEND_A40="${OUT}/blend/mem_ddlg_anchor_a40.npy"
if [[ ! -f "${BLEND_A50}" ]]; then
  "${PYTHON}" scripts/nb_adapter/ensemble_embeds.py \
    --rag-npy "${OUT}/memory/rag_soft5_test_clip_1024.npy" \
    --prior-npy "${OUT}/r2fosa_ddlg_anchor_test_clip_1024.npy" \
    --output-npy "${BLEND_A50}" \
    --alpha 0.5
else
  echo "[SKIP] blend a50"
fi
if [[ ! -f "${BLEND_A40}" ]]; then
  "${PYTHON}" scripts/nb_adapter/ensemble_embeds.py \
    --rag-npy "${OUT}/memory/rag_soft5_test_clip_1024.npy" \
    --prior-npy "${OUT}/r2fosa_ddlg_anchor_test_clip_1024.npy" \
    --output-npy "${BLEND_A40}" \
    --alpha 0.4
else
  echo "[SKIP] blend a40"
fi

echo "===== [4] Generation (no ERDC) ====="
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

# Align head only (ablation)
run_gen "r2fosa_align_s40" "${OUT}/r2fosa_align_test_clip_1024.npy" 0.4
# Main path: Anchor-DDLG warm-start from retrieval top-1
run_gen "r2fosa_ddlg_anchor_s40" "${OUT}/r2fosa_ddlg_anchor_test_clip_1024.npy" 0.4
run_gen "r2fosa_ddlg_anchor_s35" "${OUT}/r2fosa_ddlg_anchor_test_clip_1024.npy" 0.35
# Soft-memory warm DDLG
run_gen "r2fosa_ddlg_mem_s40" "${OUT}/r2fosa_ddlg_mem_test_clip_1024.npy" 0.4
# Blend with memory router
run_gen "blend_mem_ddlg_a50_s40" "${BLEND_A50}" 0.4
run_gen "blend_mem_ddlg_a50_s35" "${BLEND_A50}" 0.35

echo "===== [5] CLIP / FID evaluation ====="
METRICS="${OUT}/clip_fid_r2fosa.json"
"${PYTHON}" scripts/nb_adapter/eval_clip_fid.py \
  --gen-root "${OUT}/generation" \
  --tags "r2fosa_align_s40,r2fosa_ddlg_anchor_s40,r2fosa_ddlg_anchor_s35,r2fosa_ddlg_mem_s40,blend_mem_ddlg_a50_s40,blend_mem_ddlg_a50_s35" \
  --output-json "${METRICS}"

echo "===== [6] Summary ====="
"${PYTHON}" - <<PY
import json
from pathlib import Path
out = Path("${OUT}")
report = json.loads((out / "r2fosa_report.json").read_text())
metrics = json.loads(Path("${METRICS}").read_text()) if Path("${METRICS}").is_file() else {}
results = metrics.get("results", [])
best = max(results, key=lambda r: r.get("clip_cosine", 0)) if results else None
best_fid = min(results, key=lambda r: r.get("fid", 1e9)) if results else None
ddlg_cos = report.get("final_ddlg_cos", 0)
gen_clip = best.get("clip_cosine") if best else None
gap = ddlg_cos - gen_clip if gen_clip else None
summary = {
    "pipeline": "R2-FOSA",
    "no_erdc": True,
    "final_ddlg_cos": ddlg_cos,
    "best_gen_clip": best,
    "best_gen_fid": best_fid,
    "decode_gap": gap,
    "baseline_v1_clip": 0.412,
    "d2_fosa_fid_ref": 146.33,
    "report": report,
}
(out / "summary_r2fosa.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

echo "===== DONE R²-FOSA @ $(date -Iseconds) ====="
