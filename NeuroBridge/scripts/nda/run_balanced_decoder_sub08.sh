#!/usr/bin/env bash
# Balanced decoder: freq-fuse + two-stage refine; CLIP metric = 2-way; end with compare grid.
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
OUT="${OUT:-${NB_ROOT}/outputs/balanced_decoder/sub-08}"
LL_ROOT="${LL_ROOT:-${NB_ROOT}/outputs/lowlevel_decoder/sub-08}"
T1="${T1:-${NB_ROOT}/outputs/top1_structure/sub-08}"
NDA_SS="${NDA_SS:-${NB_ROOT}/outputs/nda_ss/sub-08}"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"
KEEP_TOP_K="${KEEP_TOP_K:-3}"

mkdir -p "${OUT}/generation" "${OUT}/compare" "${OUT}/logs"
cd "${NB_ROOT}"

unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
export HF_HUB_CACHE="${HF_HUB_CACHE:-/project/peilab/why/cache/eeg-brainit/hf/hub}"
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-/project/peilab/why/cache/eeg-brainit/open_clip}"

LL="${LL_ROOT}/vae_head/pred_lowlevel_rgb_512"
SEM="${T1}/generation/pred_coca_ip1.0_cpa/generated"
I2I45="${LL_ROOT}/generation/i2i_s45_cpa/generated"
I2I55="${LL_ROOT}/generation/i2i_s55_cpa/generated"
EMB="${NDA_SS}/blend/mem_decode_a50.npy"
PROMPT="${T1}/prompts/prompts_cpa.json"

test -f "${LL}/000.png" && test -f "${SEM}/199.png"
test -f "${I2I45}/199.png" && test -f "${I2I55}/199.png"
test -f "${EMB}" && test -f "${PROMPT}"

link_ref() {
  local tag="$1" src="$2"
  local dst="${OUT}/generation/${tag}/generated"
  mkdir -p "${dst}"
  "${PYTHON}" - <<PY
from pathlib import Path
src, dst = Path("${src}"), Path("${dst}")
for i in range(200):
    t = dst / f"{i:03d}.png"
    if not t.exists():
        t.symlink_to((src / f"{i:03d}.png").resolve())
print("[OK] linked ${tag}")
PY
}

echo "===== [0] Link references @ $(date -Iseconds) ====="
link_ref "ref_semantic" "${SEM}"
link_ref "ref_lowlevel" "${LL}"
link_ref "ref_i2i_s45" "${I2I45}"
link_ref "ref_i2i_s55" "${I2I55}"

echo "===== [1] Frequency fusion @ $(date -Iseconds) ====="
run_freq() {
  local tag="$1" struct="$2" sigma="$3"
  local gdir="${OUT}/generation/${tag}"
  if [[ -f "${gdir}/generated/199.png" ]]; then echo "[SKIP] ${tag}"; return 0; fi
  "${PYTHON}" scripts/nda/generate_freq_fuse.py \
    --struct-dir "${struct}" --semantic-dir "${SEM}" \
    --output-dir "${gdir}" --tag "${tag}" --sigma "${sigma}"
}
# low-freq from pure VAE / from structure i2i
run_freq "freq_ll_s8" "${LL}" 8.0
run_freq "freq_ll_s12" "${LL}" 12.0
run_freq "freq_i2i45_s8" "${I2I45}" 8.0

echo "===== [2] Two-stage refine @ $(date -Iseconds) ====="
run_ts() {
  local tag="$1"; shift
  local gdir="${OUT}/generation/${tag}"
  if [[ -f "${gdir}/generated/199.png" ]]; then echo "[SKIP] ${tag}"; return 0; fi
  "${PYTHON}" scripts/nda/generate_twostage_refine.py \
    --embed-npy "${EMB}" --prompts-json "${PROMPT}" \
    --output-dir "${gdir}" --tag "${tag}" --seed 42 \
    --gen-steps 28 --gen-guidance 5.0 "$@"
}
# reuse i2i_s45 as Stage A → semantic refine
run_ts "ts_from_i2i45_s65" --stage-a-dir "${I2I45}" --strength-b 0.65 --ip-scale-b 1.0
run_ts "ts_from_i2i45_s75" --stage-a-dir "${I2I45}" --strength-b 0.75 --ip-scale-b 1.0
# full two-stage from lowlevel
run_ts "ts_ll_a45_b70" --lowlevel-dir "${LL}" --strength-a 0.45 --strength-b 0.70

echo "===== [3] Paper metrics (SSIM/FID/PixCorr) @ $(date -Iseconds) ====="
TAGS="ref_semantic,ref_lowlevel,ref_i2i_s45,ref_i2i_s55,freq_ll_s8,freq_ll_s12,freq_i2i45_s8,ts_from_i2i45_s65,ts_from_i2i45_s75,ts_ll_a45_b70"
# only existing
VALID=""
IFS=',' read -ra ARR <<< "${TAGS}"
for t in "${ARR[@]}"; do
  [[ -f "${OUT}/generation/${t}/generated/199.png" ]] && VALID="${VALID:+$VALID,}${t}"
done
"${PYTHON}" scripts/nda/eval_paper_metrics.py \
  --gen-root "${OUT}/generation" \
  --tags "${VALID}" \
  --output-json "${OUT}/paper_metrics.json"

echo "===== [4] CLIP 2-way (PRIMARY semantic metric) @ $(date -Iseconds) ====="
GEN_ARGS=""
IFS=',' read -ra ARR <<< "${VALID}"
for t in "${ARR[@]}"; do
  GEN_ARGS="${GEN_ARGS:+$GEN_ARGS,}${t}=${OUT}/generation/${t}/generated"
done
"${PYTHON}" scripts/nda/eval_clip_2way.py \
  --gen-dirs "${GEN_ARGS}" \
  --output-json "${OUT}/clip_2way_report.json" \
  --device "${DEVICE}" \
  --batch-size 16

echo "===== [5] Compare grid @ $(date -Iseconds) ====="
COLS="sem=${SEM},ll=${LL},i2i45=${I2I45}"
for pair in "freq=${OUT}/generation/freq_i2i45_s8/generated" \
            "freqll=${OUT}/generation/freq_ll_s8/generated" \
            "ts65=${OUT}/generation/ts_from_i2i45_s65/generated" \
            "ts75=${OUT}/generation/ts_from_i2i45_s75/generated" \
            "tsll=${OUT}/generation/ts_ll_a45_b70/generated"; do
  name="${pair%%=*}"; path="${pair#*=}"
  if [[ -f "${path}/000.png" ]]; then COLS="${COLS},${name}=${path}"; fi
done
"${PYTHON}" scripts/nda/make_compare_grid.py \
  --output-dir "${OUT}/compare" \
  --cell 168 \
  --indices "3,12,28,45,67,88,110,133,156,178,190,199" \
  --metrics-json "${OUT}/clip_2way_report.json" \
  --cols "${COLS}"

echo "===== [6] Summary + prune @ $(date -Iseconds) ====="
export KEEP_TOP_K
"${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path
out = Path("/project/peilab/why/NeuroBridge/outputs/balanced_decoder/sub-08")
paper = json.loads((out/"paper_metrics.json").read_text())
tw = json.loads((out/"clip_2way_report.json").read_text())
by_paper = {r["tag"]: r for r in paper.get("results", [])}
by_2way = {r["tag"]: r for r in tw.get("generation_2way", [])}
keep_top_k = int(os.environ.get("KEEP_TOP_K", "3"))

def fid_score(fid):
    # map FID~150→1, FID~320→0 roughly
    return float(max(0.0, min(1.0, (320.0 - fid) / 170.0)))

rows = []
for tag, r in by_paper.items():
    w = by_2way.get(tag, {})
    twoway = float(w.get("clip_2way", 0.0))
    ssim = float(r.get("ssim", 0.0))
    fid = float(r.get("fid", 999.0))
    # primary objective: balance 2way, SSIM, FID
    score = 0.40 * twoway + 0.35 * ssim + 0.25 * fid_score(fid)
    rows.append({
        "tag": tag,
        "clip_2way": twoway,
        "clip_2way_pct": 100.0 * twoway,
        "clip_cosine_paired": w.get("clip_cosine_paired"),
        "ssim": ssim,
        "pixcorr": r.get("pixcorr"),
        "fid": fid,
        "score": score,
    })
rows.sort(key=lambda x: -x["score"])
best = rows[0] if rows else None
best_ssim = max(rows, key=lambda x: x["ssim"]) if rows else None
best_2way = max(rows, key=lambda x: x["clip_2way"]) if rows else None
best_fid = min(rows, key=lambda x: x["fid"]) if rows else None

summary = {
  "pipeline": "balanced_decoder_freq_twostage",
  "clip_metric": "CLIP ViT-L/14 two-way identification (MindEye/Ozcelik)",
  "methods": ["freq_fuse", "twostage_refine"],
  "best_balanced": best,
  "best_ssim": best_ssim,
  "best_2way": best_2way,
  "best_fid": best_fid,
  "all_ranked": rows,
  "targets": {"ssim": 0.28, "clip_2way": 0.88, "fid": 165},
  "compare_grid": str(out/"compare/compare_grid.png"),
}
(out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))

# prune PNGs: keep refs + top-K non-ref by score; always keep compare sources
keep = {"ref_semantic", "ref_lowlevel", "ref_i2i_s45", "ref_i2i_s55"}
nonref = [r for r in rows if not r["tag"].startswith("ref_")]
keep |= {r["tag"] for r in nonref[:keep_top_k]}
keep |= {"freq_i2i45_s8", "ts_from_i2i45_s65", "ts_from_i2i45_s75"}
freed = 0
for d in (out/"generation").iterdir():
    if not d.is_dir() or d.name in keep:
        continue
    for p in (d/"generated").glob("*.png"):
        if p.is_symlink():
            p.unlink(); continue
        try:
            freed += p.stat().st_size
            p.unlink()
        except FileNotFoundError:
            pass
print(json.dumps({"kept": sorted(keep), "bytes_freed": freed}, indent=2))
PY

du -sh "${OUT}" "${OUT}/generation" "${OUT}/compare" 2>/dev/null || true
echo "===== DONE balanced decoder @ $(date -Iseconds) ====="
