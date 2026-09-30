#!/usr/bin/env bash
# Track S eval: S2 embeds -> memory blend -> focused generation -> ERDC fuse.
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
BRAINIT="${BRAINIT:-/project/peilab/why/eeg-brainit}"
SOTA_V1="${SOTA_V1:-${NB_ROOT}/outputs/nb_nmb_sota/sub-08}"
OUT="${OUT:-${NB_ROOT}/outputs/nb_nmb_sota_v2/sub-08}"
S2_DIR="${OUT}/s2_dual"
GEN="${OUT}/generation_s2"
ERDC_OUT="${OUT}/erdc_s2"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"

CLIP_TRAIN="${BRAINIT}/outputs/atm_bridge/clip_img_train_1024.npy"
CLIP_TEST="${BRAINIT}/outputs/atm_bridge/clip_img_test_1024.npy"
MEM_V1="${SOTA_V1}/memory/rag_soft5_test_clip_1024.npy"
NEIGH="${SOTA_V1}/memory/rag_soft5_neighbor_idx_test.npy"
DUAL_EMB="${S2_DIR}/dual_vith1024_test_clip_1024.npy"
BLEND_DIR="${OUT}/blend_s2"

mkdir -p "${GEN}" "${BLEND_DIR}" "${ERDC_OUT}"
cd "${NB_ROOT}"

blend_once() {
  local tag="$1" rag="$2" prior="$3" alpha="$4"
  local out="${BLEND_DIR}/${tag}.npy"
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

run_lowlevel() {
  local tag="$1" embed="$2" strength="$3"
  local out="${GEN}/${tag}"
  local gen_dir="${out}/generated"
  if [[ -f "${gen_dir}/000.png" ]]; then
    local n
    n="$(find "${gen_dir}" -maxdepth 1 -name '*.png' | wc -l)"
    if [[ "${n}" -ge 200 ]]; then
      echo "[SKIP] gen ${tag} (${n} imgs)"
      return 0
    fi
  fi
  echo "[RUN] gen ${tag} strength=${strength}"
  "${PYTHON}" scripts/nb_adapter/generate_rag_lowlevel.py \
    --embed-npy "${embed}" \
    --neighbor-idx-npy "${NEIGH}" \
    --output-dir "${out}" \
    --strength "${strength}" \
    --seed 42 \
    --tag "${tag}" \
    --skip-metrics
}

echo "===== [S2e-1] Blends: mem + dual direct ViT-H ====="
blend_once "mem_dual_a40" "${MEM_V1}" "${DUAL_EMB}" 0.40
blend_once "mem_dual_a45" "${MEM_V1}" "${DUAL_EMB}" 0.45
blend_once "mem_dual_a50" "${MEM_V1}" "${DUAL_EMB}" 0.50

echo "===== [S2e-2] Focused generation (4 paths max) ====="
run_lowlevel "dual_direct_s40" "${DUAL_EMB}" 0.4
run_lowlevel "dual_direct_s45" "${DUAL_EMB}" 0.45
run_lowlevel "mem_dual_a50_s40" "${BLEND_DIR}/mem_dual_a50.npy" 0.4
run_lowlevel "mem_dual_a50_s45" "${BLEND_DIR}/mem_dual_a50.npy" 0.45

echo "===== [S2e-3] CLIP metrics ====="
TAGS="$(find "${GEN}" -mindepth 1 -maxdepth 1 -type d -printf '%f,' | sed 's/,$//')"
METRICS="${OUT}/clip_fid_s2.json"
"${PYTHON}" scripts/nb_adapter/eval_clip_fid.py \
  --gen-root "${GEN}" \
  --tags "${TAGS}" \
  --output-json "${METRICS}"

echo "===== [S2e-4] ERDC pair fuse best S2 vs SOTA v1 ====="
BEST_S2="${GEN}/mem_dual_a50_s40"
PAIR="${ERDC_OUT}/s2_vs_v1"
if [[ -f "${GEN}/mem_dual_a50_s40/generated/000.png" ]] && [[ ! -f "${PAIR}/fuse_run/selected_fused/metrics.json" ]]; then
  "${PYTHON}" "${BRAINIT}/scripts/erdc_pair_bank_fuse.py" \
    --dir-a "${GEN}/mem_dual_a50_s40/generated" \
    --dir-b "${SOTA_V1}/generation_overnight/vith_blend_mem_linEns_a50_s04/generated" \
    --label-a "s2_mem_dual_s40" \
    --label-b "v1_mem_linEns_s04" \
    --eeg-npy "${BLEND_DIR}/mem_dual_a50.npy" \
    --gallery-npy "${CLIP_TRAIN}" \
    --neighbor-npy "${NEIGH}" \
    --lambda-struct 0.25 \
    --output-dir "${PAIR}"
fi

echo "===== [S2e-5] Final summary ====="
"${PYTHON}" - <<PY
import json
from pathlib import Path

out = Path("${OUT}")
metrics = json.loads(Path("${METRICS}").read_text()) if Path("${METRICS}").is_file() else {}
results = metrics.get("results", [])
ranked = sorted(results, key=lambda r: r.get("clip_cosine", 0), reverse=True)

erdc_clip = None
p = out / "erdc_s2" / "s2_vs_v1" / "fuse_run" / "selected_fused" / "metrics.json"
if p.is_file():
    erdc_clip = json.loads(p.read_text()).get("clip_cosine")

summary = {
    "pipeline": "NMB-SOTA-v2-S2-eval",
    "baseline_v1": 0.412,
    "best_gen": ranked[0] if ranked else None,
    "top3_gen": ranked[:3],
    "erdc_s2_vs_v1_clip": erdc_clip,
    "s2_dual_report": json.loads((out / "s2_dual" / "dual_report.json").read_text())
        if (out / "s2_dual" / "dual_report.json").is_file() else None,
}
(out / "summary_s2_eval.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

echo "===== DONE S2 eval @ $(date -Iseconds) ====="
