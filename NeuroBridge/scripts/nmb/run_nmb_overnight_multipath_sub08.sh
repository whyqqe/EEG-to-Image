#!/usr/bin/env bash
# Overnight multi-path SOTA sweep: Fusion->ViT-H bridge + proven IP-Adapter decode + Fusion Prior rerank.
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
BRAINIT="${BRAINIT:-/project/peilab/why/eeg-brainit}"
OUT_BASE="${OUT_BASE:-${NB_ROOT}/outputs/nb_nmb_sota/sub-08}"
OUT="${OUT:-${OUT_BASE}}"
BRIDGE="${BRIDGE:-${OUT}/bridge_vith}"
BLEND="${BLEND:-${OUT}/blend_vith}"
GEN="${GEN:-${OUT}/generation_overnight}"
METRICS_JSON="${METRICS_JSON:-${OUT}/clip_fid_overnight.json}"
SUMMARY_JSON="${SUMMARY_JSON:-${OUT}/summary_overnight.json}"

SUBJECT="${SUBJECT:-8}"
MAX_IMAGES="${MAX_IMAGES:-0}"
DEVICE="${DEVICE:-cuda:0}"
PYTHON="${PYTHON:-python}"
IMG2IMG_STRENGTH="${IMG2IMG_STRENGTH:-0.5}"
RERANK_SAMPLES="${RERANK_SAMPLES:-8}"
FUSION_PRIOR="${FUSION_PRIOR:-${OUT}/fusion_prior_finetuned}"

EMBED_DIR="${OUT}/embeds"
MEMORY_DIR="${OUT}/memory"
FUSION_DIR="${OUT}/fusion"
CFT_DIR="${OUT}/cft"
ENSEMBLE_DIR="${OUT}/ensemble"
FP_PHASE0="${NB_ROOT}/outputs/nb_full_phases/sub-08/phase0"
FP_PHASE2="${NB_ROOT}/outputs/nb_full_phases/sub-08/phase2/dual_teacher"
FP_PHASE3="${NB_ROOT}/outputs/nb_full_phases/sub-08/phase3"
CLIP_TRAIN="${CLIP_TRAIN:-${BRAINIT}/outputs/atm_bridge/clip_img_train_1024.npy}"
CLIP_TEST="${CLIP_TEST:-${BRAINIT}/outputs/atm_bridge/clip_img_test_1024.npy}"

MEM_VITH="${MEMORY_DIR}/rag_soft5_test_clip_1024.npy"
NEIGH="${MEMORY_DIR}/rag_soft5_neighbor_idx_test.npy"
if [[ ! -f "${NEIGH}" ]]; then
  NEIGH="${FP_PHASE0}/rag_soft5_neighbor_idx.npy"
fi
RAG5_FP="${FP_PHASE0}/rag_soft5_test_clip_1024.npy"

mkdir -p "${BRIDGE}" "${BLEND}" "${GEN}"
cd "${NB_ROOT}"

run_lowlevel() {
  local tag="$1"
  local embed="$2"
  local strength="${3:-${IMG2IMG_STRENGTH}}"
  local out="${GEN}/${tag}"
  if [[ -f "${out}/generated/$(printf '%03d' 0).png" ]] && [[ "${MAX_IMAGES}" == "0" ]]; then
    local n
    n="$(find "${out}/generated" -maxdepth 1 -name '*.png' | wc -l)"
    if [[ "${n}" -ge 200 ]]; then
      echo "[SKIP] lowlevel ${tag} (${n} images)"
      return 0
    fi
  fi
  echo "[RUN] lowlevel ${tag} strength=${strength}"
  "${PYTHON}" scripts/nb_adapter/generate_rag_lowlevel.py \
    --embed-npy "${embed}" \
    --neighbor-idx-npy "${NEIGH}" \
    --output-dir "${out}" \
    --strength "${strength}" \
    --max-images "${MAX_IMAGES}" \
    --seed 42 \
    --tag "${tag}" \
    --skip-metrics
}

run_txt2img() {
  local tag="$1"
  local embed="$2"
  local out="${GEN}/${tag}"
  if [[ -f "${out}/generated/$(printf '%03d' 0).png" ]] && [[ "${MAX_IMAGES}" == "0" ]]; then
    local n
    n="$(find "${out}/generated" -maxdepth 1 -name '*.png' | wc -l)"
    if [[ "${n}" -ge 200 ]]; then
      echo "[SKIP] txt2img ${tag} (${n} images)"
      return 0
    fi
  fi
  echo "[RUN] txt2img ${tag}"
  "${PYTHON}" scripts/nb_adapter/generate_from_embeds.py \
    --embed-npy "${embed}" \
    --output-dir "${out}" \
    --max-images "${MAX_IMAGES}" \
    --seed 42 \
    --tag "${tag}" \
    --skip-metrics
}

blend_embeds() {
  local tag="$1"
  local rag="$2"
  local prior="$3"
  local alpha="$4"
  local out="${BLEND}/${tag}.npy"
  if [[ -f "${out}" ]]; then
    echo "[SKIP] blend ${tag}"
    return 0
  fi
  "${PYTHON}" scripts/nb_adapter/ensemble_embeds.py \
    --rag-npy "${rag}" \
    --prior-npy "${prior}" \
    --output-npy "${out}" \
    --alpha "${alpha}"
}

echo "===== [1] Fusion -> ViT-H bridge @ $(date -Iseconds) ====="
if [[ ! -f "${BRIDGE}/bridge_report.json" ]]; then
  "${PYTHON}" scripts/nmb/nmb_bridge_fusion_to_vith.py \
    --fusion-train "${FUSION_DIR}/fusion_train.npy" \
    --clip-train "${CLIP_TRAIN}" \
    --clip-test "${CLIP_TEST}" \
    --gallery "${CLIP_TRAIN}" \
    --output-dir "${BRIDGE}" \
    --fusion-test-src \
      "cft:${CFT_DIR}/cft_mlp_test_fusion.npy" \
      "ensemble:${ENSEMBLE_DIR}/ensemble_a0.45_test_fusion.npy" \
      "fusion_mem:${MEMORY_DIR}/fusion_mem_test.npy" \
      "fusion_gt:${FUSION_DIR}/fusion_test.npy" \
    --device "${DEVICE}"
else
  echo "[INFO] bridge exists"
fi

echo "===== [2] ViT-H space blends @ $(date -Iseconds) ====="
blend_embeds "mem_linEns_a40" "${MEM_VITH}" "${BRIDGE}/linear_ensemble_test_clip_1024.npy" 0.40
blend_embeds "mem_linEns_a45" "${MEM_VITH}" "${BRIDGE}/linear_ensemble_test_clip_1024.npy" 0.45
blend_embeds "mem_linEns_a50" "${MEM_VITH}" "${BRIDGE}/linear_ensemble_test_clip_1024.npy" 0.50
blend_embeds "mem_linEns_a55" "${MEM_VITH}" "${BRIDGE}/linear_ensemble_test_clip_1024.npy" 0.55
blend_embeds "mem_linEns_a60" "${MEM_VITH}" "${BRIDGE}/linear_ensemble_test_clip_1024.npy" 0.60
blend_embeds "mem_linCft_a45" "${MEM_VITH}" "${BRIDGE}/linear_cft_test_clip_1024.npy" 0.45
blend_embeds "mem_linCft_a55" "${MEM_VITH}" "${BRIDGE}/linear_cft_test_clip_1024.npy" 0.55
blend_embeds "mem_mlpEns_a50" "${MEM_VITH}" "${BRIDGE}/mlp_ensemble_test_clip_1024.npy" 0.50
blend_embeds "rag5_mlpEns_a50" "${RAG5_FP}" "${BRIDGE}/mlp_ensemble_test_clip_1024.npy" 0.50
blend_embeds "rag5_linEns_a50" "${RAG5_FP}" "${BRIDGE}/linear_ensemble_test_clip_1024.npy" 0.50
blend_embeds "dual_linEns_a50" "${FP_PHASE2}/dual_vith1024_test_clip_1024.npy" "${BRIDGE}/linear_ensemble_test_clip_1024.npy" 0.50

echo "===== [3A] ViT-H IP-Adapter + img2img (proven decode) @ $(date -Iseconds) ====="
# Control / baselines
run_lowlevel "vith_baseline_mem_s50" "${MEM_VITH}" 0.5
run_lowlevel "vith_rag5_fp_s50" "${RAG5_FP}" 0.5
if [[ -f "${FP_PHASE2}/dual_vith1024_test_clip_1024.npy" ]]; then
  run_lowlevel "vith_dual_s50" "${FP_PHASE2}/dual_vith1024_test_clip_1024.npy" 0.5
fi
if [[ -f "${FP_PHASE3}/ensemble_rag_prior_test_clip_1024.npy" ]]; then
  run_lowlevel "vith_ens_rag_prior_s50" "${FP_PHASE3}/ensemble_rag_prior_test_clip_1024.npy" 0.5
fi

# Pure Fusion->ViT-H projections
run_lowlevel "vith_lin_cft_s50" "${BRIDGE}/linear_cft_test_clip_1024.npy" 0.5
run_lowlevel "vith_lin_ens_s50" "${BRIDGE}/linear_ensemble_test_clip_1024.npy" 0.5
run_lowlevel "vith_mlp_cft_s50" "${BRIDGE}/mlp_cft_test_clip_1024.npy" 0.5
run_lowlevel "vith_mlp_ens_s50" "${BRIDGE}/mlp_ensemble_test_clip_1024.npy" 0.5
run_lowlevel "vith_lin_fusiongt_s50" "${BRIDGE}/linear_fusion_gt_test_clip_1024.npy" 0.5

# Blended embeddings (main SOTA candidates)
run_lowlevel "vith_blend_mem_linEns_a45_s50" "${BLEND}/mem_linEns_a45.npy" 0.5
run_lowlevel "vith_blend_mem_linEns_a50_s50" "${BLEND}/mem_linEns_a50.npy" 0.5
run_lowlevel "vith_blend_mem_linEns_a55_s50" "${BLEND}/mem_linEns_a55.npy" 0.5
run_lowlevel "vith_blend_mem_linCft_a45_s50" "${BLEND}/mem_linCft_a45.npy" 0.5
run_lowlevel "vith_blend_mem_mlpEns_a50_s50" "${BLEND}/mem_mlpEns_a50.npy" 0.5
run_lowlevel "vith_blend_rag5_mlpEns_a50_s50" "${BLEND}/rag5_mlpEns_a50.npy" 0.5
run_lowlevel "vith_blend_rag5_linEns_a50_s50" "${BLEND}/rag5_linEns_a50.npy" 0.5
run_lowlevel "vith_blend_dual_linEns_a50_s50" "${BLEND}/dual_linEns_a50.npy" 0.5

# Strength sweep on top blend candidates
for s in 0.4 0.5 0.6; do
  run_lowlevel "vith_blend_mem_linEns_a50_s${s/./}" "${BLEND}/mem_linEns_a50.npy" "${s}"
  run_lowlevel "vith_blend_rag5_mlpEns_a50_s${s/./}" "${BLEND}/rag5_mlpEns_a50.npy" "${s}"
done

echo "===== [3B] ViT-H txt2img (no lowlevel ablation) @ $(date -Iseconds) ====="
run_txt2img "vith_lin_ens_txt2img" "${BRIDGE}/linear_ensemble_test_clip_1024.npy"
run_txt2img "vith_blend_mem_linEns_a50_txt2img" "${BLEND}/mem_linEns_a50.npy"

echo "===== [3C] Fusion Prior + ViT-H rerank @ $(date -Iseconds) ====="
export BRAIN_HIVE="${BRAIN_HIVE:-/project/peilab/why/Brain-HIVE}"
export FUSION_PRIOR

if [[ ! -f "${GEN}/nmb_ens_blend_rerank/generated/199.png" ]]; then
  echo "[RUN] nmb_ens_blend_rerank"
  "${PYTHON}" scripts/nmb/nmb_generate.py \
    --fusion-npy "${ENSEMBLE_DIR}/ensemble_a0.45_test_fusion.npy" \
    --prior-path "${FUSION_PRIOR}" \
    --neighbor-idx-npy "${NEIGH}" \
    --nb-proj-npy "${MEM_VITH}" \
    --output-dir "${GEN}/nmb_ens_blend_rerank" \
    --max-images "${MAX_IMAGES}" \
    --img2img-strength "${IMG2IMG_STRENGTH}" \
    --num-samples "${RERANK_SAMPLES}" \
    --rerank --rerank-mode blend --rerank-alpha 0.5 \
    --tag "nmb_ens_blend_rerank" \
    --seed 42 --device "${DEVICE}"
else
  echo "[SKIP] nmb_ens_blend_rerank"
fi

if [[ ! -f "${GEN}/nmb_ens_vith_rerank/generated/199.png" ]]; then
  echo "[RUN] nmb_ens_vith_rerank"
  "${PYTHON}" scripts/nmb/nmb_generate.py \
    --fusion-npy "${ENSEMBLE_DIR}/ensemble_a0.45_test_fusion.npy" \
    --prior-path "${FUSION_PRIOR}" \
    --neighbor-idx-npy "${NEIGH}" \
    --nb-proj-npy "${MEM_VITH}" \
    --output-dir "${GEN}/nmb_ens_vith_rerank" \
    --max-images "${MAX_IMAGES}" \
    --img2img-strength "${IMG2IMG_STRENGTH}" \
    --num-samples "${RERANK_SAMPLES}" \
    --rerank --rerank-mode nb_proj \
    --tag "nmb_ens_vith_rerank" \
    --seed 42 --device "${DEVICE}"
else
  echo "[SKIP] nmb_ens_vith_rerank"
fi

echo "===== [4] CLIP + FID (all overnight paths) @ $(date -Iseconds) ====="
TAGS="$(find "${GEN}" -mindepth 1 -maxdepth 1 -type d -printf '%f,' | sed 's/,$//')"
"${PYTHON}" scripts/nb_adapter/eval_clip_fid.py \
  --gen-root "${GEN}" \
  --tags "${TAGS}" \
  --output-json "${METRICS_JSON}" \
  --max-images "${MAX_IMAGES}"

echo "===== [5] Overnight summary @ $(date -Iseconds) ====="
"${PYTHON}" - <<PY
import json
from pathlib import Path

out = Path("${OUT}")
gen = Path("${GEN}")
metrics_path = Path("${METRICS_JSON}")
metrics = json.loads(metrics_path.read_text()) if metrics_path.is_file() else {}
results = metrics.get("results", [])
ranked = sorted(results, key=lambda r: r.get("clip_cosine", 0), reverse=True)

summary = {
    "pipeline": "NMB-overnight-multipath",
    "subject": "sub-08",
    "fusion_prior": "${FUSION_PRIOR}",
    "n_paths": len(results),
    "best": ranked[0] if ranked else None,
    "top5": ranked[:5],
    "baseline_ref": {
        "rag_soft5_lowlevel": 0.389,
        "nmb_sota_lowlevel_rerank": 0.365,
        "teacher_vith": 0.635,
    },
    "bridge_report": json.loads((out / "bridge_vith/bridge_report.json").read_text())
        if (out / "bridge_vith/bridge_report.json").is_file() else None,
    "all_results": results,
}
(out / "summary_overnight.json").write_text(json.dumps(summary, indent=2))
print(json.dumps({"best": summary["best"], "top5": summary["top5"]}, indent=2))
PY

echo "===== DONE overnight multipath @ $(date -Iseconds) ====="
