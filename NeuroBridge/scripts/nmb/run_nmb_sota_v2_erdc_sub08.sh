#!/usr/bin/env bash
# Track E: ERDC on existing SOTA candidates + focused RAS bank (disk-conscious).
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
BRAINIT="${BRAINIT:-/project/peilab/why/eeg-brainit}"
SOTA_V1="${SOTA_V1:-${NB_ROOT}/outputs/nb_nmb_sota/sub-08}"
OUT="${OUT:-${NB_ROOT}/outputs/nb_nmb_sota_v2/sub-08}"
GEN_V1="${SOTA_V1}/generation_overnight"
# overnight images live under generated/ subdirs
GEN_MEM04="${GEN_V1}/vith_blend_mem_linEns_a50_s04/generated"
GEN_RAG5="${GEN_V1}/vith_blend_rag5_mlpEns_a50_s04/generated"
GEN_RERANK="${GEN_V1}/nmb_ens_vith_rerank/generated"
GEN_BASE="${GEN_V1}/vith_baseline_mem_s50/generated"
ERDC_OUT="${OUT}/erdc"
PYTHON="${PYTHON:-python}"

BLEND_EMB="${SOTA_V1}/blend_vith/mem_linEns_a50.npy"
NEIGH="${SOTA_V1}/memory/rag_soft5_neighbor_idx_test.npy"
CLIP_TRAIN="${BRAINIT}/outputs/atm_bridge/clip_img_train_1024.npy"
BRIDGE_DIR="${BRAINIT}/outputs/atm_bridge"

mkdir -p "${ERDC_OUT}"
cd "${NB_ROOT}"

disk_avail_gb() {
  df -BG /project/peilab/why 2>/dev/null | awk 'NR==2 {gsub(/G/,"",$4); print $4}'
}

echo "===== [E0] Pair-bank fuse (reuse overnight images, zero regen) ====="
PAIR_OUT="${ERDC_OUT}/pair_fuse"
mkdir -p "${PAIR_OUT}"

run_pair() {
  local tag="$1" dir_a="$2" dir_b="$3" label_a="$4" label_b="$5"
  local sub="${PAIR_OUT}/${tag}"
  if [[ -f "${sub}/fuse_run/selected_fused/metrics.json" ]]; then
    echo "[SKIP] pair ${tag}"
    return 0
  fi
  echo "[RUN] pair ${tag}"
  "${PYTHON}" "${BRAINIT}/scripts/erdc_pair_bank_fuse.py" \
    --dir-a "${dir_a}" \
    --dir-b "${dir_b}" \
    --label-a "${label_a}" \
    --label-b "${label_b}" \
    --eeg-npy "${BLEND_EMB}" \
    --gallery-npy "${CLIP_TRAIN}" \
    --neighbor-npy "${NEIGH}" \
    --lambda-struct 0.25 \
    --output-dir "${sub}"
}

# Top overnight paths (already on disk)
run_pair "mem04_vs_rag5" \
  "${GEN_MEM04}" \
  "${GEN_RAG5}" \
  "mem_linEns_s04" "rag5_mlpEns_s04"

run_pair "mem04_vs_rerank" \
  "${GEN_MEM04}" \
  "${GEN_RERANK}" \
  "mem_linEns_s04" "nmb_ens_vith_rerank"

run_pair "mem04_vs_baseline" \
  "${GEN_MEM04}" \
  "${GEN_BASE}" \
  "mem_linEns_s04" "baseline_mem_s50"

echo "===== [E1] RAS closed-loop (focused candidate bank) ====="
RAS_OUT="${ERDC_OUT}/ras_blend_mem50"
if [[ ! -f "${RAS_OUT}/metrics.json" ]]; then
  # top_m=1, 3 strengths, no ip_only => 3 candidates/sample (~375MB images)
  "${PYTHON}" "${BRAINIT}/scripts/erdc_ras_closed_loop.py" \
    --subject sub-08 \
    --bridge-dir "${BRIDGE_DIR}" \
    --embed-source bridge_clip \
    --bridge-npy "${BLEND_EMB}" \
    --top-m 1 \
    --strengths "0.40,0.45,0.50" \
    --no-ip-only \
    --gen-steps 4 \
    --gen-guidance 0.0 \
    --gen-size 512 \
    --ip-scale 1.0 \
    --output-dir "${RAS_OUT}" \
    --seed0 42
else
  echo "[SKIP] RAS bank exists"
fi

echo "===== [E2] Fuse reselect on RAS candidates (brain + struct) ====="
FUSE_RAS="${ERDC_OUT}/ras_fuse"
if [[ ! -f "${FUSE_RAS}/selected_fused/metrics.json" ]]; then
  "${PYTHON}" "${BRAINIT}/scripts/erdc_fuse_reselect.py" \
    --cand-dir "${RAS_OUT}/candidates" \
    --eeg-npy "${BLEND_EMB}" \
    --gallery-npy "${CLIP_TRAIN}" \
    --lambda-struct 0.35 \
    --struct-mode retrieve \
    --output-dir "${FUSE_RAS}"
else
  echo "[SKIP] RAS fuse exists"
fi

echo "===== [E3] Prune RAS candidate PNGs (keep feats + selected) ====="
if [[ -d "${RAS_OUT}/candidates" ]]; then
  find "${RAS_OUT}/candidates" -maxdepth 1 -name '*_k*.png' -delete
  echo "[prune] deleted per-candidate PNGs under RAS/candidates"
fi

echo "===== [E4] Summarize ERDC track ====="
"${PYTHON}" - <<PY
import json
from pathlib import Path

out = Path("${OUT}")
erdc = out / "erdc"
rows = []

def load_metrics(p):
    if p.is_file():
        return json.loads(p.read_text())
    return None

for name in ["pair_fuse/mem04_vs_rag5", "pair_fuse/mem04_vs_rerank", "pair_fuse/mem04_vs_baseline"]:
    m = load_metrics(erdc / name / "fuse_run" / "selected_fused" / "metrics.json")
    if m:
        rows.append({"track": name, "clip": m.get("clip_cosine"), "pixcorr": m.get("pixcorr")})

ras = load_metrics(erdc / "ras_blend_mem50" / "metrics.json")
if ras and "selection" in ras:
    for pick, blk in ras["selection"].items():
        m = blk.get("metrics", {})
        rows.append({"track": f"ras/{pick}", "clip": m.get("clip_cosine"), "pixcorr": m.get("pixcorr")})

fuse = load_metrics(erdc / "ras_fuse" / "selected_fused" / "metrics.json")
if fuse:
    rows.append({"track": "ras_fuse/fused", "clip": fuse.get("clip_cosine"), "pixcorr": fuse.get("pixcorr")})

ref = 0.412
ranked = sorted(rows, key=lambda r: r.get("clip") or 0, reverse=True)
summary = {
    "pipeline": "NMB-SOTA-v2-ERDC",
    "baseline_clip": ref,
    "disk_avail_gb": float("${disk_avail_gb:-0}"),
    "best": ranked[0] if ranked else None,
    "top5": ranked[:5],
    "all": rows,
}
(out / "summary_erdc.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

echo "===== DONE ERDC track @ $(date -Iseconds) ====="
