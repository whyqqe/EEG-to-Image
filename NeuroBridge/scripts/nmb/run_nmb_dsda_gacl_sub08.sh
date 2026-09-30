#!/usr/bin/env bash
# DSDA + GACL experiment: dual-space fusion, decode-aligned bridge, adaptive img2img.
# Prunes generated images after each path (keeps KEEP_SAMPLES for figures).
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
BRAINIT="${BRAINIT:-/project/peilab/why/eeg-brainit}"
OUT="${OUT:-${NB_ROOT}/outputs/nb_nmb_dsda_gacl/sub-08}"
BASE="${BASE:-${NB_ROOT}/outputs/nb_nmb_sota/sub-08}"

GACL_DIR="${OUT}/bridge_gacl"
DSDA_DIR="${OUT}/dsda"
GEN="${OUT}/generation"
FIG="${OUT}/figures"
METRICS_JSON="${OUT}/metrics_dsda_gacl.json"
SUMMARY_JSON="${OUT}/summary_dsda_gacl.json"

SUBJECT="${SUBJECT:-8}"
MAX_IMAGES="${MAX_IMAGES:-0}"
DEVICE="${DEVICE:-cuda:0}"
PYTHON="${PYTHON:-python}"
KEEP_SAMPLES="${KEEP_SAMPLES:-8}"
NEIGH="${BASE}/memory/rag_soft5_neighbor_idx_test.npy"
MEM_TEST="${BASE}/memory/rag_soft5_test_clip_1024.npy"
MEM_TRAIN="${BASE}/memory/rag_soft5_train_clip_1024.npy"
FUSION_TRAIN="${BASE}/fusion/fusion_train.npy"
ENSEMBLE_TEST="${BASE}/ensemble/ensemble_a0.45_test_fusion.npy"
CLIP_TRAIN="${CLIP_TRAIN:-${BRAINIT}/outputs/atm_bridge/clip_img_train_1024.npy}"
CLIP_TEST="${CLIP_TEST:-${BRAINIT}/outputs/atm_bridge/clip_img_test_1024.npy}"
LINEAR_ENS="${BASE}/bridge_vith/linear_ensemble_test_clip_1024.npy"
CONTROL_BLEND="${BASE}/blend_vith/mem_linEns_a50.npy"

mkdir -p "${OUT}" "${GACL_DIR}" "${DSDA_DIR}" "${GEN}" "${FIG}"
cd "${NB_ROOT}"

RESULTS_JSON="${OUT}/results_partial.json"
echo "[]" > "${RESULTS_JSON}"

append_result() {
  local tag="$1"
  local eval_json="$2"
  "${PYTHON}" - <<PY
import json
from pathlib import Path
partial = Path("${RESULTS_JSON}")
arr = json.loads(partial.read_text())
data = json.loads(Path("${eval_json}").read_text())
row = data["results"][0] if data.get("results") else {"tag": "${tag}"}
row["tag"] = "${tag}"
arr.append(row)
partial.write_text(json.dumps(arr, indent=2))
print(f"[result] {row.get('tag')} CLIP={row.get('clip_cosine', 'n/a')}")
PY
}

eval_and_prune() {
  local tag="$1"
  local out="${GEN}/${tag}"
  local eval_one="${OUT}/_eval_${tag}.json"
  echo "[EVAL] ${tag}"
  "${PYTHON}" scripts/nb_adapter/eval_clip_fid.py \
    --gen-root "${GEN}" \
    --tags "${tag}" \
    --output-json "${eval_one}" \
    --max-images "${MAX_IMAGES}"
  append_result "${tag}" "${eval_one}"
  rm -f "${eval_one}"
  echo "[PRUNE] ${tag} keep=${KEEP_SAMPLES}"
  "${PYTHON}" scripts/nmb/nmb_prune_generated.py \
    --gen-dir "${out}/generated" \
    --keep "${KEEP_SAMPLES}" \
    --seed 42 \
    --report-json "${FIG}/${tag}_samples.json"
  cp "${out}/generated"/*.png "${FIG}/" 2>/dev/null || true
}

run_lowlevel() {
  local tag="$1"
  local embed="$2"
  local strength="${3:-0.4}"
  local strength_npy="${4:-}"
  local out="${GEN}/${tag}"
  echo "[GEN] ${tag}"
  local -a extra=()
  if [[ -n "${strength_npy}" ]]; then
    extra+=(--strength-npy "${strength_npy}")
  else
    extra+=(--strength "${strength}")
  fi
  "${PYTHON}" scripts/nb_adapter/generate_rag_lowlevel.py \
    --embed-npy "${embed}" \
    --neighbor-idx-npy "${NEIGH}" \
    --output-dir "${out}" \
    --max-images "${MAX_IMAGES}" \
    --seed 42 \
    --tag "${tag}" \
    --skip-metrics \
    "${extra[@]}"
  eval_and_prune "${tag}"
}

echo "===== [1] GACL decode-aligned bridge @ $(date -Iseconds) ====="
if [[ ! -f "${GACL_DIR}/gacl_report.json" ]]; then
  "${PYTHON}" scripts/nmb/nmb_gacl_train_bridge.py \
    --fusion-train "${FUSION_TRAIN}" \
    --clip-train "${CLIP_TRAIN}" \
    --clip-test "${CLIP_TEST}" \
    --mem-train "${MEM_TRAIN}" \
    --gallery "${CLIP_TRAIN}" \
    --output-dir "${GACL_DIR}" \
    --fusion-test-src "ensemble:${ENSEMBLE_TEST}" \
    --gap-beta 2.0 \
    --epochs 60 \
    --patience 10 \
    --device "${DEVICE}"
else
  echo "[SKIP] GACL bridge exists"
fi
GACL_ENS="${GACL_DIR}/gacl_mlp_ensemble_test_clip_1024.npy"

echo "===== [2] DSDA embeddings (linear + GACL) @ $(date -Iseconds) ====="
if [[ ! -f "${DSDA_DIR}/linear/dsda_report.json" ]]; then
  "${PYTHON}" scripts/nmb/nmb_dsda_build.py \
    --mem-npy "${MEM_TEST}" \
    --proj-npy "${LINEAR_ENS}" \
    --output-dir "${DSDA_DIR}/linear" \
    --alpha-base 0.5 --conf-k 1.0 --s-hi 0.45 --s-lo 0.32
fi
if [[ ! -f "${DSDA_DIR}/gacl/dsda_report.json" ]]; then
  "${PYTHON}" scripts/nmb/nmb_dsda_build.py \
    --mem-npy "${MEM_TEST}" \
    --proj-npy "${GACL_ENS}" \
    --output-dir "${DSDA_DIR}/gacl" \
    --alpha-base 0.5 --conf-k 1.0 --s-hi 0.45 --s-lo 0.32
fi

echo "===== [3] Generation paths (eval + prune each) @ $(date -Iseconds) ====="

# A: overnight best control (fixed blend + s=0.4)
if [[ -f "${CONTROL_BLEND}" ]]; then
  run_lowlevel "A_control_mem_linEns_s04" "${CONTROL_BLEND}" 0.4
else
  run_lowlevel "A_control_mem_linEns_s04" "${DSDA_DIR}/linear/dsda_fixed_blend_a50.npy" 0.4
fi

# B: PoE semantic-perceptual fusion (linear proj) + fixed low strength
run_lowlevel "B_dsda_poe_linear_s04" "${DSDA_DIR}/linear/dsda_poe_mem_proj.npy" 0.4

# C: confidence-adaptive blend (linear) + per-sample adaptive strength
run_lowlevel "C_dsda_conf_linear_adapt" \
  "${DSDA_DIR}/linear/dsda_conf_blend.npy" 0.4 \
  "${DSDA_DIR}/linear/dsda_adaptive_strength.npy"

# D: GACL projection + fixed blend strength
run_lowlevel "D_gacl_fixed_blend_s04" "${DSDA_DIR}/gacl/dsda_fixed_blend_a50.npy" 0.4

# E: flagship — GACL + confidence blend + adaptive strength
run_lowlevel "E_gacl_conf_adaptive" \
  "${DSDA_DIR}/gacl/dsda_conf_blend.npy" 0.4 \
  "${DSDA_DIR}/gacl/dsda_adaptive_strength.npy"

echo "===== [4] Summary @ $(date -Iseconds) ====="
"${PYTHON}" - <<PY
import json
from pathlib import Path

out = Path("${OUT}")
rows = json.loads((out / "results_partial.json").read_text())
ranked = sorted(rows, key=lambda r: r.get("clip_cosine", 0), reverse=True)
summary = {
    "pipeline": "DSDA+GACL",
    "subject": "sub-08",
    "n_paths": len(rows),
    "best": ranked[0] if ranked else None,
    "top3": ranked[:3],
    "baseline_ref": {"rag_soft5_lowlevel": 0.389, "overnight_best": 0.412},
    "gacl_report": json.loads((out / "bridge_gacl/gacl_report.json").read_text())
        if (out / "bridge_gacl/gacl_report.json").is_file() else None,
    "all_results": rows,
}
(out / "summary_dsda_gacl.json").write_text(json.dumps(summary, indent=2))
(out / "metrics_dsda_gacl.json").write_text(json.dumps({"results": rows}, indent=2))
print(json.dumps({"best": summary["best"], "top3": summary["top3"]}, indent=2))
PY

echo "===== DONE DSDA+GACL @ $(date -Iseconds) ====="
