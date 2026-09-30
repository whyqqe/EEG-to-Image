#!/usr/bin/env bash
# ATM official-style metrics (same as eeg-brainit erdc_full_metrics + twoway + fid).
set -euo pipefail

EEG_ROOT=/project/peilab/why/eeg-brainit
OUT_DIR=/project/peilab/why/NeuroBridge/outputs/atm_official_eval_sub08
IMAGES_ROOT=/project/peilab/why/data/images_set

mkdir -p "${OUT_DIR}"

full_metrics() {
  local gen="$1"
  local tag="$2"
  local out="${OUT_DIR}/${tag}.json"
  if [[ -f "${out}" ]]; then
    echo "[SKIP] full ${tag}"
    return 0
  fi
  [[ -d "${gen}" ]] || { echo "[WARN] missing gen dir ${gen}"; return 0; }
  python "${EEG_ROOT}/scripts/erdc_full_metrics.py" \
    --gen-dir "${gen}" \
    --images-root "${IMAGES_ROOT}" \
    --output-json "${out}" \
    --tag "${tag}"
}

twoway() {
  local gen="$1"
  local tag="$2"
  local out="${OUT_DIR}/${tag}_2wc.json"
  if [[ -f "${out}" ]]; then
    echo "[SKIP] 2wc ${tag}"
    return 0
  fi
  [[ -d "${gen}" ]] || return 0
  python "${EEG_ROOT}/scripts/erdc_twoway_metrics.py" \
    --gen-dir "${gen}" \
    --images-root "${IMAGES_ROOT}" \
    --output-json "${out}" \
    --tag "${tag}"
}

fid_metrics() {
  local gen="$1"
  local tag="$2"
  local out="${OUT_DIR}/${tag}_fid.json"
  if [[ -f "${out}" ]]; then
    echo "[SKIP] fid ${tag}"
    return 0
  fi
  [[ -d "${gen}" ]] || return 0
  python "${EEG_ROOT}/scripts/erdc_fid_metrics.py" \
    --gen-dir "${gen}" \
    --images-root "${IMAGES_ROOT}" \
    --output-json "${out}" \
    --tag "${tag}"
}

run_one() {
  local tag="$1"
  local gen="$2"
  echo "======== ${tag} ========"
  full_metrics "${gen}" "${tag}"
  twoway "${gen}" "${tag}"
  fid_metrics "${gen}" "${tag}"
}

# Reference: official ATM flat (already extracted)
run_one "official_atm_gen" \
  "${EEG_ROOT}/outputs/erdc/w7_official_flat"

# NeuroBridge best runs
run_one "nb_decode_aligner" \
  "/project/peilab/why/NeuroBridge/outputs/nb_decode_aligner/sub-08/generation/blend_mem_decode_s40/generated"

run_one "nb_erdc_brain" \
  "/project/peilab/why/NeuroBridge/outputs/nb_nmb_sota_v2/sub-08/erdc/pair_fuse/mem04_vs_rerank/fuse_run/selected_brain"

run_one "nb_v1_linEns" \
  "/project/peilab/why/NeuroBridge/outputs/nb_nmb_sota/sub-08/generation_overnight/vith_blend_mem_linEns_a50_s04/generated"

# Optional: R²-FOSA align gen if present
R2_GEN="/project/peilab/why/NeuroBridge/outputs/nb_r2fosa/sub-08/generation/r2fosa_align_s40/generated"
if [[ -d "${R2_GEN}" ]]; then
  run_one "nb_r2fosa_align" "${R2_GEN}"
fi

python "${EEG_ROOT}/scripts/erdc_paper_table.py" \
  --metrics-dirs "${OUT_DIR}" \
  --tags official_atm_gen nb_decode_aligner nb_erdc_brain nb_v1_linEns nb_r2fosa_align \
  --baseline-tag official_atm_gen \
  --output-md "${OUT_DIR}/summary_table.md" \
  --output-tex "${OUT_DIR}/summary_table.tex"

python - <<'PY'
import json
from pathlib import Path

out_dir = Path("/project/peilab/why/NeuroBridge/outputs/atm_official_eval_sub08")
tags = [
    "official_atm_gen",
    "nb_decode_aligner",
    "nb_erdc_brain",
    "nb_v1_linEns",
    "nb_r2fosa_align",
]
# ATM paper Table 3 (THINGS-EEG, reported values ×100 for 2WC)
paper = {
    "ssim": 0.345,
    "clip_cosine_paired": None,  # paper uses 2WC not paired
    "twoway_clip": 0.786,
    "twoway_alex2": 0.776,
    "twoway_inception": 0.734,
    "twoway_swav": 0.582,
}
rows = []
for tag in tags:
    m_path = out_dir / f"{tag}.json"
    if not m_path.is_file():
        continue
    m = json.loads(m_path.read_text())
    tw_path = out_dir / f"{tag}_2wc.json"
    tw = json.loads(tw_path.read_text()) if tw_path.is_file() else {}
    fid_path = out_dir / f"{tag}_fid.json"
    fid = json.loads(fid_path.read_text()) if fid_path.is_file() else {}
    rows.append({
        "tag": tag,
        "gen_dir": m.get("gen_dir"),
        "pixcorr": m.get("pixcorr"),
        "ssim": m.get("ssim"),
        "clip_cosine": m.get("clip_cosine"),
        "alexnet2": m.get("alexnet2"),
        "alexnet5": m.get("alexnet5"),
        "inception_paired": m.get("inception"),
        "effnet_b1": m.get("effnet_b1"),
        "twoway": tw.get("twoway"),
        "fid": fid.get("fid"),
    })
report = {
    "subject": "sub-08",
    "n_test": 200,
    "protocol": "eeg-brainit erdc_full_metrics + erdc_twoway_metrics (ATM official repro)",
    "atm_paper_table3_reference": paper,
    "methods": rows,
    "notes": [
        "ssim uses ssim_simple (256x256); ATM paper SSIM 0.345 may use different implementation.",
        "2WC metrics are comparable to ATM paper CLIP/Alex2/Inception columns (×100 = %).",
        "SwAV 2WC not implemented in erdc_twoway_metrics; paper SwAV=0.582.",
    ],
}
(out_dir / "full_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(f"[OK] wrote {out_dir / 'full_report.json'} with {len(rows)} methods")
PY

echo "[OK] ATM official eval finished -> ${OUT_DIR}"
