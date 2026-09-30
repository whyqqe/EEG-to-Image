#!/usr/bin/env bash
# ============================================================================
# SHB — Structural Hypothesis Branch (sub-08, pure-intra EEG weights)
#
# Architecture after this job:  THREE branches off the frozen EEG backbone
#   branch 1  semantic   : raw -> CLIP-Image (existing, frozen for us)
#   branch 2  structure  : raw -> z_decode_vith -> {Depth CN, VAE-LL} (existing)
#   branch 3  SHB (NEW)  : raw -> StructHead (STRUCTURAL objective) ->
#                          GeoDecoder -> (depth mean, log sigma^2) at 128^2
#                          conditioned on TRIAL-SUBSET hypotheses ->
#                          spatial uncertainty gate on the ControlNet control
#                          image -> image-side hypothesis VERIFICATION/selection
#
# Why a third branch (defects it targets):
#   D1 the structure heads read a *semantic* latent (CLIP cosine+InfoNCE), so
#      spatial information was already destroyed upstream (depth pearson 0.689,
#      VAE pearson 0.332) -> SHB trains a structure-specialised head off `raw`.
#   D2 both existing branches share the backbone output, so they fail TOGETHER
#      and cannot compensate; an ORACLE scalar gate could not move the frontier.
#      -> SHB's evidence is the TRIAL-TO-TRIAL variability, which is independent.
#   D3 the structural condition is a conditional mean: blurred + spatially smooth
#      -> SHB predicts mean AND per-pixel variance (Gaussian NLL) at 128^2.
#   D4 a scalar CN scale can only interpolate a fixed structure<->semantics
#      curve -> SHB gates in SPACE (uncertain pixels -> neutral gray), a new
#      degree of freedom, and adds a DISCRETE choice over hypotheses.
#   D5 the two structure heads are trained independently and can disagree
#      -> SHB's decoder is single-head with explicit uncertainty.
#
# Rows (all standard-7 + FID, strength 0.86, CN 0.40 unless noted):
#   shb_pt_c040_s086   point estimate, UNMASKED        (isolates better geometry)
#   shb_sp_c040_s086   point estimate, masked          (isolates SPATIAL gating)
#   shb_mu_c040_s086   trial-posterior mean, masked
#   shb_h1/h2_c040_s086 subset hypotheses, masked      (mechanism 1)
#   shb_sel_{sem,geo,cons}_c040_s086  verification selection (mechanism 3)
#
# OUTPUT: outputs/shb/sub-08
# ============================================================================
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
BRAINIT="${BRAINIT:-/project/peilab/why/eeg-brainit}"
INTRA="${INTRA:-${NB_ROOT}/outputs/intra_hcma_s/sub-08}"
OUT="${OUT:-${NB_ROOT}/outputs/shb/sub-08}"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"

IMAGES_ROOT="${IMAGES_ROOT:-/project/peilab/why/data/images_set}"
HCMA_PROMPTS="${HCMA_PROMPTS:-${NB_ROOT}/outputs/hcma_10subj/prompts/prompts_full_hcma_test.json}"
CKPT_RN50="${CKPT_RN50:-${NB_ROOT}/results/things_eeg/intra-subjects/20260901-074319-sub-08/checkpoint_test_best.pth}"

# existing pure-intra tower assets (unchanged across rows so only the STRUCTURAL
# channel varies): semantic IP embed, VAE-LL SDEdit init, baseline depth pred
EMB="${EMB:-${INTRA}/blend/mem_decode_a50.npy}"
LL_RGB="${LL_RGB:-${INTRA}/vae_head/pred_lowlevel_rgb_512}"
BASE_DEPTH="${BASE_DEPTH:-${INTRA}/depth/pred_depth_rgb_512}"
BASE_DEPTH_TEST64="${BASE_DEPTH_TEST64:-${INTRA}/depth/pred_depth_test_64.npy}"
GT_DEPTH_TEST_SRC="${GT_DEPTH_TEST_SRC:-${NB_ROOT}/outputs/hcma_s_full10/shared/gt_depth/test_depth_64.npy}"
VAE_CACHE_SRC="${VAE_CACHE_SRC:-${NB_ROOT}/outputs/sdedit_ll_full10/shared/vae_cache}"
GRID="${GRID:-${NB_ROOT}/outputs/intra_hcma_s/sub-08/generation}"

CN="${CN:-0.40}"
STRENGTH="${STRENGTH:-0.86}"

mkdir -p "${OUT}/raw" "${OUT}/struct" "${OUT}/geo" "${OUT}/vae_cache" "${OUT}/gt_depth" \
         "${OUT}/generation" "${OUT}/logs" "${NB_ROOT}/outputs/slurm"
cd "${NB_ROOT}"
if [[ -f "${NB_ROOT}/scripts/nmb/nmb_sota_v2_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${NB_ROOT}/scripts/nmb/nmb_sota_v2_env.sh"
else
  # shellcheck disable=SC1091
  source "${BRAINIT}/scripts/activate.sh"
fi
PYTHON="$(command -v python)"
echo "[INFO] using python: ${PYTHON}"
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
export HF_HUB_CACHE="${HF_HUB_CACHE:-/project/peilab/why/cache/eeg-brainit/hf/hub}"
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-/project/peilab/why/cache/eeg-brainit/open_clip}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/project/peilab/why/cache/xdg}"

echo "{\"pipeline\":\"shb_sub08\",\"started\":\"$(date -Iseconds)\",\"job\":\"${SLURM_JOB_ID:-local}\",\"cn\":\"${CN}\",\"strength\":\"${STRENGTH}\"}" > "${OUT}/job_running.json"

require() { [[ -e "$1" ]] || { echo "[FATAL] missing $1" >&2; exit 1; }; }
require "${CKPT_RN50}"; require "${EMB}"; require "${HCMA_PROMPTS}"
require "${LL_RGB}/000.png"; require "${LL_RGB}/199.png"
require "${BASE_DEPTH}/199.png"; require "${GT_DEPTH_TEST_SRC}"
require "${GRID}/intra_hs_c025_s086/generated/199.png"
require "${GRID}/intra_hs_c040_s086/generated/199.png"

echo "===== [1] GT caches: train depth (image side) + VAE latents + test depth @ $(date -Iseconds) ====="
DTR="${OUT}/gt_depth/train_depth_64.npy"
DTE="${OUT}/gt_depth/test_depth_64.npy"
[[ -f "${DTE}" ]] || ln -sfn "${GT_DEPTH_TEST_SRC}" "${DTE}"
if [[ ! -f "${DTR}" ]]; then
  "${PYTHON}" scripts/nda/build_gt_depth_cache.py \
    --images-root "${IMAGES_ROOT}" --output-dir "${OUT}/gt_depth" \
    --device "${DEVICE}" --splits "train" --batch-size 8
else
  echo "[SKIP] train depth cache"
fi
VT="${OUT}/vae_cache/train_vae_latents_f16.npy"
VE="${OUT}/vae_cache/test_vae_latents_f16.npy"
[[ -f "${VT}" ]] || ln -sfn "${VAE_CACHE_SRC}/train_vae_latents_f16.npy" "${VT}"
[[ -f "${VE}" ]] || ln -sfn "${VAE_CACHE_SRC}/test_vae_latents_f16.npy" "${VE}"
require "${DTR}"; require "${DTE}"; require "${VT}"; require "${VE}"

echo "===== [2] export raw backbone features + trial-level posterior (GATE #0) @ $(date -Iseconds) ====="
if [[ ! -f "${OUT}/raw/trial_posterior_report.json" ]]; then
  "${PYTHON}" scripts/nda/shb_export_raw.py \
    --subject 8 --checkpoint "${CKPT_RN50}" \
    --output-dir "${OUT}/raw" \
    --n-test-groups 8 \
    --gt-depth-test "${DTE}" \
    --pred-depth-test "${BASE_DEPTH_TEST64}" \
    --device "${DEVICE}"
else
  echo "[SKIP] raw export"
fi
require "${OUT}/raw/raw_train.npy"; require "${OUT}/raw/raw_test_sub.npy"

echo "===== [3] StructHead: STRUCTURAL multi-task (depth + SDXL-VAE) off raw @ $(date -Iseconds) ====="
if [[ ! -f "${OUT}/struct/struct_head_report.json" ]]; then
  "${PYTHON}" scripts/nda/shb_train_struct.py \
    --raw-train "${OUT}/raw/raw_train.npy" \
    --raw-test "${OUT}/raw/raw_test.npy" \
    --raw-train-sub "${OUT}/raw/raw_train_sub.npy" \
    --raw-test-sub "${OUT}/raw/raw_test_sub.npy" \
    --depth-train "${DTR}" --depth-test "${DTE}" \
    --vae-train "${VT}" --vae-test "${VE}" \
    --output-dir "${OUT}/struct" \
    --num-epochs 60 --batch-size 256 --lr 1e-3 \
    --device "${DEVICE}"
else
  echo "[SKIP] struct head"
fi
require "${OUT}/struct/struct_test_sub.npy"

echo "===== [4] GeoDecoder: heteroscedastic geometry field + control images @ $(date -Iseconds) ====="
if [[ ! -f "${OUT}/geo/geofield_report.json" ]]; then
  "${PYTHON}" scripts/nda/shb_train_geofield.py \
    --struct-train "${OUT}/struct/struct_train.npy" \
    --struct-test "${OUT}/struct/struct_test.npy" \
    --struct-train-sub "${OUT}/struct/struct_train_sub.npy" \
    --struct-test-sub "${OUT}/struct/struct_test_sub.npy" \
    --depth-train "${DTR}" --depth-test "${DTE}" \
    --baseline-depth-test "${BASE_DEPTH_TEST64}" \
    --output-dir "${OUT}/geo" \
    --res 128 --num-epochs 40 --batch-size 64 --lr 1e-3 \
    --mask-quantile 0.30 --rgb-size 512 \
    --device "${DEVICE}"
else
  echo "[SKIP] geofield"
fi
CTRL="${OUT}/geo/generation_controls"
require "${CTRL}/geo_pt/199.png"; require "${CTRL}/geo_pt_mask/199.png"
require "${CTRL}/geo_mu_mask/199.png"

echo "===== [5] generation: 5 SHB rows (CN ${CN}, strength ${STRENGTH}) @ $(date -Iseconds) ====="
run_row() {
  local tag="$1" cdir="$2"
  local gdir="${OUT}/generation/${tag}"
  if [[ -f "${gdir}/generated/199.png" ]]; then
    echo "[SKIP] ${tag}"
    return 0
  fi
  "${PYTHON}" scripts/nda/generate_hcma_s_decode.py \
    --embed-npy "${EMB}" --prompts-json "${HCMA_PROMPTS}" \
    --depth-rgb-dir "${cdir}" --lowlevel-rgb-dir "${LL_RGB}" \
    --output-dir "${gdir}" --tag "${tag}" \
    --cn-scale "${CN}" --ip-scale 1.0 --strength "${STRENGTH}" \
    --gen-steps 28 --gen-guidance 5.0 --seed 42
}
run_row "shb_pt_c040_s086" "${CTRL}/geo_pt"
run_row "shb_sp_c040_s086" "${CTRL}/geo_pt_mask"
run_row "shb_mu_c040_s086" "${CTRL}/geo_mu_mask"
if [[ -f "${CTRL}/geo_h1_mask/199.png" ]]; then
  run_row "shb_h1_c040_s086" "${CTRL}/geo_h1_mask"
  run_row "shb_h2_c040_s086" "${CTRL}/geo_h2_mask"
fi

echo "===== [6] standard-7 eval pass 1 (SHB generation rows) @ $(date -Iseconds) ====="
STD7="${NB_ROOT}/outputs/standard7_protocol"
cp -f "${STD7}/results.json" "${OUT}/results_std7_backup.json" || true
build_manifest() {
  local outman="$1"; shift
  OUT_EVAL="${OUT}" MAN="${outman}" ROWTAGS="$*" "${PYTHON}" - <<'PY'
import json, os
from pathlib import Path
out = Path(os.environ["OUT_EVAL"])
man = {"protocol": "standard7", "rows": [], "avg_rows": []}
for tag in os.environ["ROWTAGS"].split():
    d = out / "generation" / tag / "generated"
    if (d / "199.png").is_file():
        man["rows"].append({"tag": tag, "display": tag + " (SHB sub-08)", "gen_dir": str(d)})
    else:
        print("[WARN] missing", d)
Path(os.environ["MAN"]).write_text(json.dumps(man, indent=2), encoding="utf-8")
print("[OK] manifest rows", len(man["rows"]))
PY
}
build_manifest "${OUT}/manifest_shb_gen.json" \
  shb_pt_c040_s086 shb_sp_c040_s086 shb_mu_c040_s086 shb_h1_c040_s086 shb_h2_c040_s086
"${PYTHON}" scripts/nda/eval_standard7.py \
  --manifest "${OUT}/manifest_shb_gen.json" --images-root "${IMAGES_ROOT}" \
  --out-dir "${STD7}" --device "${DEVICE}" --batch-size 16
cp -f "${STD7}/results.json" "${OUT}/results_shb_gen.json"

echo "===== [7] image-side verification / hypothesis selection @ $(date -Iseconds) ====="
CANDS="intra_hs_c025_s086=${GRID}/intra_hs_c025_s086/generated"
CANDS="${CANDS},intra_hs_c040_s086=${GRID}/intra_hs_c040_s086/generated"
CANDS="${CANDS},shb_pt_c040_s086=${OUT}/generation/shb_pt_c040_s086/generated"
CANDS="${CANDS},shb_sp_c040_s086=${OUT}/generation/shb_sp_c040_s086/generated"
CANDS="${CANDS},shb_mu_c040_s086=${OUT}/generation/shb_mu_c040_s086/generated"
if [[ -f "${OUT}/generation/shb_h1_c040_s086/generated/199.png" ]]; then
  CANDS="${CANDS},shb_h1_c040_s086=${OUT}/generation/shb_h1_c040_s086/generated"
  CANDS="${CANDS},shb_h2_c040_s086=${OUT}/generation/shb_h2_c040_s086/generated"
fi

"${PYTHON}" scripts/nda/shb_select.py \
  --candidates "${CANDS}" \
  --ip-embed "${EMB}" \
  --geo-ref "${OUT}/geo/pred_depth_mu_128.npy" \
  --output-dir "${OUT}/select" \
  --device "${DEVICE}" --batch-size 8

echo "===== [8] standard-7 eval pass 2 (selection rows) @ $(date -Iseconds) ====="
build_manifest "${OUT}/manifest_shb_sel.json" \
  shb_sel_sem_c040_s086 shb_sel_geo_c040_s086 shb_sel_cons_c040_s086
if [[ "$("${PYTHON}" -c "import json;print(len(json.load(open('${OUT}/manifest_shb_sel.json'))['rows']))")" != "0" ]]; then
  "${PYTHON}" scripts/nda/eval_standard7.py \
    --manifest "${OUT}/manifest_shb_sel.json" --images-root "${IMAGES_ROOT}" \
    --out-dir "${STD7}" --device "${DEVICE}" --batch-size 16
  cp -f "${STD7}/results.json" "${OUT}/results_shb_sel.json"
else
  echo "[WARN] no selection rows; skipping eval pass 2"
fi

echo "===== [9] merge results.json + summary @ $(date -Iseconds) ====="
OUT_M="${OUT}" STD7_M="${STD7}" "${PYTHON}" - <<'PY'
import json, os
from pathlib import Path
out = Path(os.environ["OUT_M"])
std7 = Path(os.environ["STD7_M"])
backup = json.loads((out / "results_std7_backup.json").read_text(encoding="utf-8"))
assert "rows" in backup, "backup results.json is malformed - refusing to overwrite"
by_tag = {r["tag"]: r for r in backup["rows"]}
added = 0
for name in ("results_shb_gen.json", "results_shb_sel.json"):
    p = out / name
    if not p.is_file():
        continue
    for r in json.loads(p.read_text(encoding="utf-8"))["rows"]:
        if r["tag"] not in by_tag:
            added += 1
        by_tag[r["tag"]] = r
merged = dict(backup)
merged["rows"] = list(by_tag.values())
# atomic write: results.json is shared with every other experiment
tmp = std7 / "results.json.shb_tmp"
tmp.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
os.replace(tmp, std7 / "results.json")
print(f"[OK] merged results.json: {len(backup['rows'])} -> {len(merged['rows'])} rows (+{added} SHB)")
PY

# free the bulky train depth cache (struct+geofield are done)
rm -f "${DTR}"

OUT_S="${OUT}" STD7_S="${STD7}" "${PYTHON}" - <<'PY'
import json, os
from pathlib import Path
out = Path(os.environ["OUT_S"])
std7 = Path(os.environ["STD7_S"])
res = json.loads((std7 / "results.json").read_text(encoding="utf-8"))
by = {r["tag"]: r for r in res["rows"]}
want = ["shb_pt_c040_s086", "shb_sp_c040_s086", "shb_mu_c040_s086", "shb_h1_c040_s086",
        "shb_h2_c040_s086", "shb_sel_sem_c040_s086", "shb_sel_geo_c040_s086",
        "shb_sel_cons_c040_s086"]
rows = [by[t] for t in want if t in by]
anchors = {}
for t in ("sdedit_ll_s082", "intra_sdedit_ll_s082", "intra_hs_c025_s082", "intra_hs_c040_s082",
          "intra_hs_c025_s086", "intra_hs_c032_s086", "intra_hs_c040_s086", "official_atm_sub08"):
    if t in by:
        anchors[t] = by[t]
def load(p):
    p = out / p
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None
summary = {
  "pipeline": "shb_sub08",
  "core_claim": "third branch: structure-specialised objective off raw + trial-level posterior "
                "+ spatially-resolved uncertainty gate + image-side hypothesis verification",
  "gate0_trial_posterior": load("raw/trial_posterior_report.json"),
  "struct_head": load("struct/struct_head_report.json"),
  "geofield": load("geo/geofield_report.json"),
  "selection": load("select/selection_report.json"),
  "rows": rows,
  "anchor_rows": anchors,
}
(out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({"rows": [r["tag"] for r in rows], "anchors": list(anchors)}, indent=2, ensure_ascii=False))
PY

du -sh "${OUT}" 2>/dev/null || true
echo "{\"pipeline\":\"shb_sub08\",\"finished\":\"$(date -Iseconds)\"}" > "${OUT}/job_done.json"
echo "===== DONE shb sub08 @ $(date -Iseconds) ====="
