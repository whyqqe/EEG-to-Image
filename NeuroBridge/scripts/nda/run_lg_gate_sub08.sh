#!/usr/bin/env bash
# ============================================================================
# LG-GATE sub-08 (learnable structural gate on HCMA-S)
#
# Goal: let HCMA-S beat sdedit_LL by END-TO-END LEARNING a per-sample
# structural-confidence router u = MLP(z_decode_vith) that controls the
# ControlNet scale at decode time. High u (structure decodable from EEG) =>
# strong CN (HCMA-S dual path); low u => CN off => HCMA semantics dominate
# (sdedit-LL behavior). This makes the EEG structural tower ADAPTIVE instead
# of the fixed cn=0.25/0.32/0.40 grid that hurt semantics.
#
# Supervision is leakage-free:
#   u_label_i = 0.5+0.5*pearson( DepthHead_fold(z_i), GT_depth_i )   (train)
# computed OUT-OF-FOLD (K-fold) with fresh sub-08-only depth heads so the
# label never comes from a head that saw sample i. Router trained on EEG only.
#
# Reuses the PURE-INTRA sub-08 tower (outputs/intra_hcma_s/sub-08):
#   - semantic embed      : blend/mem_decode_a50.npy (NDA-v2 dual sub-08 + RAG)
#   - LL SDEdit init      : vae_head/pred_lowlevel_rgb_512 (pure-intra VAE head)
#   - Depth-CN condition  : depth/pred_depth_rgb_512 (pure-intra Depth head)
#   - test prompts        : hcma_10subj prompts (image-side texts)
#   - test GT depth cache : reused (image-side)
# Only NEW trained weights: K-fold label depth heads + router (sub-08 only).
#
# OUTPUT: outputs/lg_gate/sub-08  (+ std7 merged rows lg_router_s082/s086,
#                                   lg_oracle_s082 diagnostic upper bound)
# ============================================================================
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
BRAINIT="${BRAINIT:-/project/peilab/why/eeg-brainit}"
INTRA="${INTRA:-${NB_ROOT}/outputs/intra_hcma_s/sub-08}"
OUT="${OUT:-${NB_ROOT}/outputs/lg_gate/sub-08}"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"

IMAGES_ROOT="${IMAGES_ROOT:-/project/peilab/why/data/images_set}"
HCMA_PROMPTS="${HCMA_PROMPTS:-${NB_ROOT}/outputs/hcma_10subj/prompts/prompts_full_hcma_test.json}"
GT_DEPTH_TEST="${GT_DEPTH_TEST:-${NB_ROOT}/outputs/hcma_s_full10/shared/gt_depth/test_depth_64.npy}"

# --- CN policy ---
LG_CN_MIN="${LG_CN_MIN:-0.0}"          # low-u fallback: CN off (== sdedit-LL style)
LG_CN_MAX="${LG_CN_MAX:-0.40}"         # high-u ceiling (max observed safe in grid)
LG_STRENGTHS="${LG_STRENGTHS:-0.82 0.86}"
LG_RUN_ORACLE="${LG_RUN_ORACLE:-1}"    # 1 = also generate oracle-gated (upper bound, diag)

# --- pure-intra tower inputs (from outputs/intra_hcma_s/sub-08) ---
Z_TR="${INTRA}/train/z_decode_vith_train.npy"
Z_TE="${INTRA}/train/z_decode_vith_test.npy"
EMB="${INTRA}/blend/mem_decode_a50.npy"
LL_RGB="${INTRA}/vae_head/pred_lowlevel_rgb_512"
DEPTH_RGB="${INTRA}/depth/pred_depth_rgb_512"

mkdir -p "${OUT}/gt_depth" "${OUT}/router" "${OUT}/generation" \
         "${OUT}/metrics" "${OUT}/logs" "${NB_ROOT}/outputs/slurm"
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
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/project/peilab/why/cache/xdg}"

echo "{\"pipeline\":\"lg_gate_sub08\",\"started\":\"$(date -Iseconds)\",\"job\":\"${SLURM_JOB_ID:-local}\",\"cn_min\":\"${LG_CN_MIN}\",\"cn_max\":\"${LG_CN_MAX}\",\"strengths\":\"${LG_STRENGTHS}\",\"oracle\":\"${LG_RUN_ORACLE}\"}" > "${OUT}/job_running.json"

require() { [[ -e "$1" ]] || { echo "[FATAL] missing $1" >&2; exit 1; }; }
require "${Z_TR}"; require "${Z_TE}"; require "${EMB}"; require "${HCMA_PROMPTS}"
require "${LL_RGB}/000.png"; require "${LL_RGB}/199.png"
require "${DEPTH_RGB}/000.png"; require "${DEPTH_RGB}/199.png"

echo "===== [1] GT depth cache (train; image-side Depth-Anything) @ $(date -Iseconds) ====="
DTR="${OUT}/gt_depth/train_depth_64.npy"
DTE="${OUT}/gt_depth/test_depth_64.npy"
if [[ ! -f "${DTE}" ]]; then
  ln -sfn "${GT_DEPTH_TEST}" "${DTE}"
fi
if [[ ! -f "${DTR}" ]]; then
  "${PYTHON}" scripts/nda/build_gt_depth_cache.py \
    --images-root "${IMAGES_ROOT}" --output-dir "${OUT}/gt_depth" \
    --device "${DEVICE}" --splits "train" --batch-size 8
else
  echo "[SKIP] train depth cache"
fi
require "${DTR}"; require "${DTE}"

echo "===== [2] Router: K-fold OOF structural-confidence labels + train + predict @ $(date -Iseconds) ====="
if [[ ! -f "${OUT}/router/router_report.json" ]]; then
  "${PYTHON}" scripts/nda/lg_gate_router.py \
    --z-train-npy "${Z_TR}" \
    --z-test-npy "${Z_TE}" \
    --depth-gt-train-npy "${DTR}" \
    --depth-gt-test-npy "${DTE}" \
    --output-dir "${OUT}/router" \
    --num-folds 3 --fold-epochs 30 --final-epochs 25 \
    --router-epochs 80 --batch-size 256 --lr 1e-3 \
    --device "${DEVICE}" --seed 42
else
  echo "[SKIP] router"
fi
require "${OUT}/router/u_hat_test.npy"; require "${OUT}/router/u_true_test.npy"
require "${OUT}/router/router_report.json"

echo "===== [3] LG-gated generation (router per-sample cn) @ $(date -Iseconds) ====="
run_lg() {
  local tag="$1" u_npy="$2" strength="$3"
  local gdir="${OUT}/generation/${tag}"
  if [[ -f "${gdir}/generated/199.png" ]]; then
    echo "[SKIP] ${tag}"
  else
    "${PYTHON}" scripts/nda/generate_lg_gate_decode.py \
      --embed-npy "${EMB}" \
      --prompts-json "${HCMA_PROMPTS}" \
      --depth-rgb-dir "${DEPTH_RGB}" \
      --lowlevel-rgb-dir "${LL_RGB}" \
      --u-npy "${u_npy}" \
      --output-dir "${gdir}" --tag "${tag}" \
      --cn-min "${LG_CN_MIN}" --cn-max "${LG_CN_MAX}" \
      --ip-scale 1.0 --strength "${strength}" \
      --gen-steps 28 --gen-guidance 5.0 --seed 42
  fi
}
for s in ${LG_STRENGTHS}; do
  stag=$(echo "$s" | tr -d .)
  run_lg "lg_router_s${stag}" "${OUT}/router/u_hat_test.npy" "$s"
done
if [[ "${LG_RUN_ORACLE}" == "1" ]]; then
  run_lg "lg_oracle_s082" "${OUT}/router/u_true_test.npy" "0.82"
fi
# free bulky train depth cache (router done; only test ordering stays for potential reruns)
rm -f "${DTR}"

echo "===== [4] standard-7 (incl. per-row FID) @ $(date -Iseconds) ====="
STD7="${NB_ROOT}/outputs/standard7_protocol"
cp -f "${STD7}/results.json" "${OUT}/results_std7_backup.json" || true
MAN="${OUT}/manifest_lg_gate.json"
OUT_EVAL="${OUT}" MAN="${MAN}" "${PYTHON}" - <<'PY'
import json, os
from pathlib import Path
out = Path(os.environ["OUT_EVAL"])
man = {"protocol": "standard7", "rows": [], "avg_rows": []}
def add(tag, display, gdir):
    man["rows"].append({"tag": tag, "display": display, "gen_dir": gdir})
for cand in ("lg_router_s082", "lg_router_s086", "lg_oracle_s082"):
    d = out / "generation" / cand / "generated"
    if (d / "199.png").is_file():
        add(cand, cand + " (LG-Gate sub-08)", str(d))
Path(os.environ["MAN"]).write_text(json.dumps(man, indent=2), encoding="utf-8")
print("[OK] manifest rows", len(man["rows"]))
PY
if [[ "$(python -c "import json,sys;print(len(json.load(open('${MAN}'))['rows']))" 2>/dev/null)" == "0" ]]; then
  echo "[FATAL] no lg rows generated" >&2; exit 1
fi
"${PYTHON}" scripts/nda/eval_standard7.py \
  --manifest "${MAN}" --images-root "${IMAGES_ROOT}" \
  --out-dir "${STD7}" --device "${DEVICE}" --batch-size 16
cp -f "${STD7}/results.json" "${OUT}/results_lg.json"
OUT_LG="${OUT}" STD7_DIR="${STD7}" "${PYTHON}" - <<'PY'
import json, os
from pathlib import Path
out = Path(os.environ["OUT_LG"])
std7 = Path(os.environ["STD7_DIR"])
backup = json.loads(Path(out / "results_std7_backup.json").read_text(encoding="utf-8"))
lg = json.loads(Path(out / "results_lg.json").read_text(encoding="utf-8"))
by_tag = {r["tag"]: r for r in backup["rows"]}
for r in lg["rows"]:
    by_tag[r["tag"]] = r
merged = dict(backup)
merged["rows"] = [by_tag[t] for t in by_tag]
(std7 / "results.json").write_text(json.dumps(merged, indent=2), encoding="utf-8")
print(f"[OK] merged results.json: {len(backup['rows'])} -> {len(merged['rows'])} rows")
PY

echo "===== [5] summary + comparison vs anchors @ $(date -Iseconds) ====="
OUT_S="${OUT}" STD7_S="${STD7}" "${PYTHON}" - <<'PY'
import json, os
from pathlib import Path
out = Path(os.environ["OUT_S"])
std7 = Path(os.environ["STD7_S"])
res = json.loads((std7 / "results.json").read_text(encoding="utf-8"))
by = {r["tag"]: r for r in res["rows"]}
want = [r["tag"] for r in json.loads((out / "manifest_lg_gate.json").read_text(encoding="utf-8"))["rows"]]
rows = [by[t] for t in want if t in by]
anchors = {}
for t in ("sdedit_ll_s082", "hs_c040_s082", "intra_sdedit_ll_s082", "intra_hs_c040_s082",
          "loso_inter_sdedit_ll_s082", "official_atm_sub08"):
    if t in by:
        anchors[t] = by[t]
report = json.loads((out / "router/router_report.json").read_text(encoding="utf-8"))
summary = {
  "pipeline": "lg_gate_sub08",
  "core_claim": "learnable structural gate: EEG->u router sets per-sample CN scale; high-u trust structure, low-u keep semantic-safe sdedit-LL behavior",
  "router_diagnostics": report,
  "cn_policy": {"min": float(os.environ.get("LG_CN_MIN", "0.0")), "max": float(os.environ.get("LG_CN_MAX", "0.40"))},
  "rows": rows,
  "anchor_rows": anchors,
}
(out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

du -sh "${OUT}" 2>/dev/null || true
echo "{\"pipeline\":\"lg_gate_sub08\",\"finished\":\"$(date -Iseconds)\"}" > "${OUT}/job_done.json"
echo "===== DONE lg_gate sub08 @ $(date -Iseconds) ====="
