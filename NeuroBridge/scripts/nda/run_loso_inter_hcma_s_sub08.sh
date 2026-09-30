#!/usr/bin/env bash
# ============================================================================
# LOSO-INTER HCMA-S (sub-08) — INTER-SUBJECT SEMANTIC under INTRA STRUCTURE
# A strict controlled companion to job 564261 (intra_hcma_s_sub08).
#
# CONTROLLED VARIABLE DESIGN:
#   * Semantic tower:  LOSO fold holdout_08 MG-Flow (trained on the OTHER 9
#                       subjects; sub-08 NEVER in pretraining) -> pure-forward
#                       inter embed (already produced by inter_ll_full10 /
#                       mg_flow_inter_encode.py on hcma_loso_fid129 fold ckpt).
#   * Structure tower: IDENTICAL to the pure-intra job 564261 — the VAE-LL and
#                       Depth-CN heads it trains on sub-08-only data.
#   => Difference vs 564261 is ONLY the semantic embed (inter LOSO vs intra).
#
# Runs AFTER 564261 completes (sbatch --dependency=afterok:564261) because it
# reuses 564261's structure heads.
#
# OUTPUT: outputs/loso_inter_hcma_s/sub-08  (+ std7 merged rows loso_inter_*)
# ============================================================================
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
BRAINIT="${BRAINIT:-/project/peilab/why/eeg-brainit}"
INTRA="${INTRA:-${NB_ROOT}/outputs/intra_hcma_s/sub-08}"
OUT="${OUT:-${NB_ROOT}/outputs/loso_inter_hcma_s/sub-08}"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"

IMAGES_ROOT="${IMAGES_ROOT:-/project/peilab/why/data/images_set}"
HCMA_PROMPTS="${HCMA_PROMPTS:-${NB_ROOT}/outputs/hcma_10subj/prompts/prompts_full_hcma_test.json}"

# --- INTER LOSO semantic embed (pure-forward, sub-08 never seen) ---
# Produced earlier by inter_ll_full10 via mg_flow_inter_encode.py on the
# hcma_loso_fid129/folds/holdout_08 MG-Flow ckpt (9-subject pretrain, NO FT).
INTER_EMB="${INTER_EMB:-${NB_ROOT}/outputs/inter_ll_full10/sub-08/inter_embeds/embeds/blend_nda_cfm_f_a40_test.npy}"

# --- INTRA structure heads produced by job 564261 (pure sub-08) ---
LL_RGB="${INTRA}/vae_head/pred_lowlevel_rgb_512"
DEPTH_RGB="${INTRA}/depth/pred_depth_rgb_512"

mkdir -p "${OUT}/generation" "${OUT}/metrics" "${OUT}/logs" "${NB_ROOT}/outputs/slurm"
cd "${NB_ROOT}"
# source project venv (diffusers 0.31 / transformers 4.46) — REQUIRED for SDXL decode
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

echo "{\"pipeline\":\"loso_inter_hcma_s_sub08\",\"started\":\"$(date -Iseconds)\",\"depends_on\":\"564261(intra structure heads)\",\"semantic\":\"LOSO 9-subject MG-Flow pure-forward (sub-08 unseen)\",\"structure\":\"intra sub-08 heads from 564261\"}" > "${OUT}/job_running.json"

require() { [[ -e "$1" ]] || { echo "[FATAL] missing $1" >&2; exit 1; }; }
require "${HCMA_PROMPTS}"; require "${INTER_EMB}"
echo "[WAIT] waiting for intra structure heads from 564261 ..."
for k in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
  if [[ -f "${LL_RGB}/000.png" && -f "${DEPTH_RGB}/000.png" && -f "${LL_RGB}/199.png" && -f "${DEPTH_RGB}/199.png" ]]; then
    echo "[OK] intra structure heads ready after ${k}0s"
    break
  fi
  sleep 10
done
require "${LL_RGB}/000.png"; require "${DEPTH_RGB}/000.png"
require "${LL_RGB}/199.png"; require "${DEPTH_RGB}/199.png"

echo "===== [1] generation: sdedit-LL baseline + HCMA-S dual grid @ $(date -Iseconds) ====="
SDIR="${OUT}/generation/loso_inter_sdedit_ll"
if [[ ! -f "${SDIR}/generated/199.png" ]]; then
  "${PYTHON}" scripts/nda/generate_atm_aligned_decode.py \
    --mode sdedit --embed-npy "${INTER_EMB}" \
    --prompts-json "${HCMA_PROMPTS}" \
    --output-dir "${SDIR}" --tag "loso_inter_sdedit_ll" \
    --lowlevel-rgb-dir "${LL_RGB}" \
    --strength 0.82 --ip-scale 1.0 --gen-steps 28 --gen-guidance 5.0 --seed 42
else
  echo "[SKIP] loso_inter_sdedit_ll"
fi

run_dual() {
  local tag="$1" cn="$2" strength="$3"
  local gdir="${OUT}/generation/${tag}"
  [[ -f "${gdir}/generated/199.png" ]] && echo "[SKIP] ${tag}" && return 0
  "${PYTHON}" scripts/nda/generate_hcma_s_decode.py \
    --embed-npy "${INTER_EMB}" \
    --prompts-json "${HCMA_PROMPTS}" \
    --depth-rgb-dir "${DEPTH_RGB}" \
    --lowlevel-rgb-dir "${LL_RGB}" \
    --output-dir "${gdir}" --tag "${tag}" \
    --cn-scale "${cn}" --ip-scale 1.0 --strength "${strength}" \
    --gen-steps 28 --gen-guidance 5.0 --seed 42
}
for cn in 0.25 0.32 0.40; do
  for s in 0.82 0.86 0.88; do
    ctag=$(echo "$cn" | tr -d .); stag=$(echo "$s" | tr -d .)
    run_dual "loso_inter_hs_c${ctag}_s${stag}" "$cn" "$s"
  done
done

echo "===== [2] standard-7 @ $(date -Iseconds) ====="
STD7="${NB_ROOT}/outputs/standard7_protocol"
cp -f "${STD7}/results.json" "${OUT}/results_std7_backup.json" || true
MAN="${OUT}/manifest_loso_inter.json"
OUT_EVAL="${OUT}" MAN="${MAN}" "${PYTHON}" - <<'PY'
import json, os
from pathlib import Path
out = Path(os.environ["OUT_EVAL"])
man = {"protocol": "standard7", "rows": [], "avg_rows": []}
def add(tag, display, gdir):
    man["rows"].append({"tag": tag, "display": display, "gen_dir": gdir})
add("loso_inter_sdedit_ll_s082", "loso-inter sdedit_ll sub-08", str(out / "generation/loso_inter_sdedit_ll/generated"))
for cn in ("025","032","040"):
    for s in ("082","086","088"):
        t = f"loso_inter_hs_c{cn}_s{s}"
        d = out / "generation" / t / "generated"
        if (d / "199.png").exists():
            add(t, t, str(d))
Path(os.environ["MAN"]).write_text(json.dumps(man, indent=2), encoding="utf-8")
print("[OK] manifest rows", len(man["rows"]))
PY
"${PYTHON}" scripts/nda/eval_standard7.py \
  --manifest "${MAN}" --images-root "${IMAGES_ROOT}" \
  --out-dir "${STD7}" --device "${DEVICE}" --batch-size 16
cp -f "${STD7}/results.json" "${OUT}/results_loso_inter.json"
"${PYTHON}" - <<PY
import json
from pathlib import Path
backup = json.loads(Path("${OUT}/results_std7_backup.json").read_text(encoding="utf-8"))
loso = json.loads(Path("${OUT}/results_loso_inter.json").read_text(encoding="utf-8"))
by_tag = {r["tag"]: r for r in backup["rows"]}
for r in loso["rows"]:
    by_tag[r["tag"]] = r
merged = dict(backup)
merged["rows"] = [by_tag[t] for t in by_tag]
Path("${STD7}/results.json").write_text(json.dumps(merged, indent=2), encoding="utf-8")
print(f"[OK] merged results.json: {len(backup['rows'])} -> {len(merged['rows'])} rows")
PY

echo "===== [3] summary @ $(date -Iseconds) ====="
"${PYTHON}" - <<PY
import json
from pathlib import Path
out = Path("${OUT}")
std7 = Path("${STD7}/results.json")
res = json.loads(std7.read_text(encoding="utf-8"))["rows"]
by = {r["tag"]: r for r in res}
want = [r["tag"] for r in json.loads((out/"manifest_loso_inter.json").read_text(encoding="utf-8"))["rows"]]
rows = [by[t] for t in want if t in by]
# anchor comparisons
anchors = {}
for t in ("sdedit_ll_s082","hs_c040_s082","intra_sdedit_ll_s082","official_atm_sub08"):
    if t in by:
        anchors[t] = by[t]
summary = {
  "pipeline": "loso_inter_hcma_s_sub08",
  "design": "controlled: semantic = LOSO 9-subject MG-Flow pure-forward (sub-08 unseen); structure = intra sub-08 heads (same as 564261)",
  "rows": rows,
  "anchor_rows": anchors,
}
(out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

du -sh "${OUT}" 2>/dev/null || true
echo "{\"pipeline\":\"loso_inter_hcma_s_sub08\",\"finished\":\"$(date -Iseconds)\"}" > "${OUT}/job_done.json"
echo "===== DONE loso_inter_hcma_s sub08 @ $(date -Iseconds) ====="
