#!/usr/bin/env bash
# ============================================================================
# GCC P0 + P1 (sub-08) — protocol/alpha ablation + cross-subject geometry fix
#
# Built on measured defects, not hypotheses (see gcc_build_conditions.py):
#   D1 intra: RAG memory degrades the IP condition. alpha was never ablated:
#             a00 Top-1 0.350 -> a25 0.320 -> a50 0.215 (what we ship today)
#   D2 LOSO : pipeline blends the 5th-best of 9 available components
#             (blend_nda_cfm_f_a40, Top-1 0.055) while z_s_f/z_s_c
#             (2-way 0.910/0.925, Top-1 0.160) are never used; hubness 3x.
#             label-free SAW pushes T1 0.160 -> 0.235 and hub_skew 3.00 -> 0.63
#   D3 both : the prompt carries the GT concept name, so image metrics stay
#             flat while the real EEG condition collapses 6x. We add the
#             deployable protocols (predicted concept / prompt-free).
#
# Everything except the explicit oracle reference row is label-free, and NO
# training happens here: we reuse intra 564261's structural heads and the
# inter_ll_full10 LOSO semantic embeds.
#
# OUTPUT: outputs/gcc_p0p1/sub-08  (+ std7 merged rows gcc_*)
# ============================================================================
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
BRAINIT="${BRAINIT:-/project/peilab/why/eeg-brainit}"
INTRA="${INTRA:-${NB_ROOT}/outputs/intra_hcma_s/sub-08}"
LOSO_EMB="${LOSO_EMB:-${NB_ROOT}/outputs/inter_ll_full10/sub-08/inter_embeds/embeds}"
OUT="${OUT:-${NB_ROOT}/outputs/gcc_p0p1/sub-08}"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"

IMAGES_ROOT="${IMAGES_ROOT:-/project/peilab/why/data/images_set}"
GALLERY_CLIP="${GALLERY_CLIP:-${BRAINIT}/outputs/atm_bridge/clip_img_test_1024.npy}"
CONCEPTS="${CONCEPTS:-${NB_ROOT}/outputs/nda_ss/sub-08/clip_text/test/concept_phrases.json}"
ORACLE_PROMPTS="${ORACLE_PROMPTS:-${NB_ROOT}/outputs/hcma_10subj/prompts/prompts_full_hcma_test.json}"

# structural heads: intra (pure sub-08), reused by BOTH phases so that only the
# semantic condition varies -> controlled comparison.
LL_RGB="${INTRA}/vae_head/pred_lowlevel_rgb_512"
DEPTH_RGB="${INTRA}/depth/pred_depth_rgb_512"

COND="${OUT}/conditions"
STD7="${NB_ROOT}/outputs/standard7_protocol"

require() { [[ -e "$1" ]] || { echo "[FATAL] missing $1" >&2; exit 1; }; }

mkdir -p "${OUT}/generation" "${OUT}/metrics" "${OUT}/logs" "${NB_ROOT}/outputs/slurm"
cd "${NB_ROOT}"

# project venv (diffusers 0.31 / transformers 4.46) — REQUIRED for SDXL decode
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
export TOKENIZERS_PARALLELISM=false

require "${GALLERY_CLIP}"; require "${CONCEPTS}"; require "${ORACLE_PROMPTS}"
require "${LL_RGB}/000.png"; require "${LL_RGB}/199.png"
require "${DEPTH_RGB}/000.png"; require "${DEPTH_RGB}/199.png"
require "${INTRA}/memory/rag_soft5_test_clip_1024.npy"
require "${INTRA}/train/z_decode_vith_test.npy"
for f in z_s_f z_s_c blend_nda_cfm_f_a40; do require "${LOSO_EMB}/${f}_test.npy"; done
echo "[OK] all inputs present"

# ---------------------------------------------------------------------------
echo "===== [1] build conditions + label-free retrieval diagnostics @ $(date -Iseconds) ====="
"${PYTHON}" scripts/nda/gcc_build_conditions.py \
  --intra-root "${INTRA}" \
  --loso-embeds "${LOSO_EMB}" \
  --gallery-clip "${GALLERY_CLIP}" \
  --concept-phrases "${CONCEPTS}" \
  --oracle-prompts "${ORACLE_PROMPTS}" \
  --out-dir "${COND}" \
  --cn-scale 0.25 --strength 0.86
require "${COND}/rows.tsv"

# ---------------------------------------------------------------------------
echo "===== [2] generation (HCMA-S dual, CN=0.25 s=0.86, seed 42) @ $(date -Iseconds) ====="
while IFS=$'\t' read -r tag embed prompts cn strength gdir; do
  [[ -z "${tag}" ]] && continue
  if [[ -f "${gdir}/generated/199.png" && -f "${gdir}/generated/000.png" ]]; then
    echo "[SKIP] ${tag}"
    continue
  fi
  require "${embed}"; require "${prompts}"
  echo "[GEN ] ${tag}  cn=${cn} s=${strength}"
  "${PYTHON}" scripts/nda/generate_hcma_s_decode.py \
    --embed-npy "${embed}" \
    --prompts-json "${prompts}" \
    --depth-rgb-dir "${DEPTH_RGB}" \
    --lowlevel-rgb-dir "${LL_RGB}" \
    --output-dir "${gdir}" --tag "${tag}" \
    --cn-scale "${cn}" --ip-scale 1.0 --strength "${strength}" \
    --gen-steps 28 --gen-guidance 5.0 --seed 42
done < "${COND}/rows.tsv"

# ---------------------------------------------------------------------------
echo "===== [3] standard-7 (P0 then P1) @ $(date -Iseconds) ====="
cp -f "${STD7}/results.json" "${OUT}/results_std7_backup.json" || true

eval_phase() {
  local phase="$1"   # p0 | p1
  local man="${OUT}/manifest_${phase}.json"
  OUT_EVAL="${OUT}" MAN="${man}" PHASE="${phase}" COND="${COND}" "${PYTHON}" - <<'PY'
import json, os
from pathlib import Path
out = Path(os.environ["OUT_EVAL"])
cond = Path(os.environ["COND"])
rows_all = json.loads((cond / "conditions_manifest.json").read_text(encoding="utf-8"))["rows"]
phase = os.environ["PHASE"]
man = {"protocol": "standard7", "rows": [], "avg_rows": []}
for r in rows_all:
    if not r["tag"].startswith(f"gcc_{phase}_"):
        continue
    d = Path(r["gen_dir"]) / "generated"
    if (d / "199.png").exists() and (d / "000.png").exists():
        man["rows"].append({"tag": r["tag"], "display": r["tag"], "gen_dir": str(d)})
    else:
        print(f"[WARN] missing images for {r['tag']} -> excluded")
Path(os.environ["MAN"]).write_text(json.dumps(man, indent=2), encoding="utf-8")
print(f"[OK] {phase} manifest rows={len(man['rows'])}")
PY
  if [[ ! -s "${man}" ]] || [[ "$("${PYTHON}" -c "import json,sys;print(len(json.load(open(sys.argv[1]))['rows']))" "${man}")" == "0" ]]; then
    echo "[SKIP] eval ${phase}: no rows"
    return 0
  fi
  "${PYTHON}" scripts/nda/eval_standard7.py \
    --manifest "${man}" --images-root "${IMAGES_ROOT}" \
    --out-dir "${STD7}" --device "${DEVICE}" --batch-size 16
  cp -f "${STD7}/results.json" "${OUT}/results_${phase}.json"
  OUT_PHASE="${OUT}" PHASE="${phase}" "${PYTHON}" - <<'PY'
import json, os
from pathlib import Path
out = Path(os.environ["OUT_PHASE"]); phase = os.environ["PHASE"]
backup = json.loads((out / "results_std7_backup.json").read_text(encoding="utf-8"))
new = json.loads((out / f"results_{phase}.json").read_text(encoding="utf-8"))
by_tag = {r["tag"]: r for r in backup["rows"]}
added = []
for r in new["rows"]:
    by_tag[r["tag"]] = r
    added.append(r["tag"])
merged = dict(backup)
merged["rows"] = [by_tag[t] for t in by_tag]
p = Path("/project/peilab/why/NeuroBridge/outputs/standard7_protocol/results.json")
tmp = p.with_suffix(".json.tmp")
tmp.write_text(json.dumps(merged, indent=2), encoding="utf-8")
tmp.replace(p)
print(f"[OK] merged {phase}: +{len(added)} rows, total {len(merged['rows'])}")
PY
  # keep the backup refreshed so the second phase does not clobber phase 1
  cp -f "${STD7}/results.json" "${OUT}/results_std7_backup.json"
}

eval_phase p0
eval_phase p1

# ---------------------------------------------------------------------------
echo "===== [4] summary @ $(date -Iseconds) ====="
OUT_SUM="${OUT}" COND="${COND}" "${PYTHON}" - <<'PY'
import json, os
from pathlib import Path
out = Path(os.environ["OUT_SUM"])
cond = Path(os.environ["COND"])
diag = json.loads((cond / "retrieval_diag.json").read_text(encoding="utf-8"))
rows_all = json.loads((cond / "conditions_manifest.json").read_text(encoding="utf-8"))
res = {r["tag"]: r for r in json.loads(
    Path("/project/peilab/why/NeuroBridge/outputs/standard7_protocol/results.json").read_text(encoding="utf-8"))["rows"]}
KEYS = ["pixcorr","ssim","alex2","alex5","inception","clip","swav","effnet","fid"]
summary = {
    "pipeline": "gcc_p0p1_sub08",
    "purpose": "P0 = alpha/prompt protocol ablation (intra); P1 = cross-subject conditioning + SAW (LOSO)",
    "prompt_concept_accuracy_vs_gt": rows_all["prompt_concept_accuracy_vs_gt"],
    "retrieval_diag": diag["conditions"],
    "rows": [],
}
print(f"{'tag':<20}{'prompt':<12}{'T1':>7}" + "".join(f"{k:>10}" for k in KEYS))
for r in rows_all["rows"]:
    m = res.get(r["tag"])
    entry = {"tag": r["tag"], "prompt_protocol": r["prompt_protocol"],
             "cond": r["cond"], "retrieval": r["retrieval"], "metrics": m}
    summary["rows"].append(entry)
    t1 = r["retrieval"]["top1"]
    if m:
        print(f"{r['tag']:<20}{r['prompt_protocol']:<12}{t1:>7.3f}" +
              "".join(f"{m.get(k, float('nan')):>10.4f}" if k != "fid" else f"{m.get(k, float('nan')):>10.2f}" for k in KEYS))
    else:
        print(f"{r['tag']:<20}{r['prompt_protocol']:<12}{t1:>7.3f}   (no metrics)")
for t in ("intra_hs_c025_s086", "official_atm_sub08", "sdedit_ll_s082"):
    if t in res:
        m = res[t]
        print(f"{t:<20}{'[anchor]':<12}{float('nan'):>7.3f}" +
              "".join(f"{m.get(k, float('nan')):>10.4f}" if k != "fid" else f"{m.get(k, float('nan')):>10.2f}" for k in KEYS))
(out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
PY

du -sh "${OUT}" 2>/dev/null || true
echo "{\"pipeline\":\"gcc_p0p1_sub08\",\"finished\":\"$(date -Iseconds)\"}" > "${OUT}/job_done.json"
echo "===== DONE gcc_p0p1 sub08 @ $(date -Iseconds) ====="
