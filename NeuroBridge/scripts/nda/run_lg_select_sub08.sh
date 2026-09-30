#!/usr/bin/env bash
# ============================================================================
# LG-SELECT (sub-08): validation of per-sample CN gating by composing EXISTING
# pure-intra grid images (NO re-generation). Cheapest decisive test of whether
# two-level / four-level gating (CN 0 <-> 0.25/0.32/0.40) can beat both the
# fixed grid AND sdedit-LL under the standard-7 protocol.
#
# Rows (all standard-7, pure-intra sub-08):
#   lgsel_r2_s082 / lgsel_o2_s082 : 2-level at strength 0.82 (router/oracle)
#   lgsel_r4_s082 / lgsel_o4_s082 : 4-level at strength 0.82 (router/oracle)
#   lgsel_r2_s086 / lgsel_o2_s086 : 2-level at strength 0.86 (router/oracle)
#   r* deployable (router u_hat); o* oracle u_true upper bound (diagnostic).
#
# Uses (kept after LG-Gate cleanup): lg_gate/sub-08/router/u_hat_test.npy +
# u_true_test.npy (200 each), intra_hcma_s/sub-08/generation/* (grid images).
# ============================================================================
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
BRAINIT="${BRAINIT:-/project/peilab/why/eeg-brainit}"
LG="${LG:-${NB_ROOT}/outputs/lg_gate/sub-08}"
GRID="${GRID:-${NB_ROOT}/outputs/intra_hcma_s/sub-08/generation}"
OUT="${OUT:-${NB_ROOT}/outputs/lg_select/sub-08}"
PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda:0}"

IMAGES_ROOT="${IMAGES_ROOT:-/project/peilab/why/data/images_set}"
STD7="${NB_ROOT}/outputs/standard7_protocol"

mkdir -p "${OUT}/generation" "${OUT}/logs" "${NB_ROOT}/outputs/slurm"
cd "${NB_ROOT}"
# source project venv (GPU features) 
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

echo "{\"pipeline\":\"lg_select_sub08\",\"started\":\"$(date -Iseconds)\",\"job\":\"${SLURM_JOB_ID:-local}\"}" > "${OUT}/job_running.json"

require() { [[ -e "$1" ]] || { echo "[FATAL] missing $1" >&2; exit 1; }; }
require "${LG}/router/u_hat_test.npy"; require "${LG}/router/u_true_test.npy"
require "${GRID}/sdedit_ll_intra/generated/199.png"
require "${GRID}/intra_hs_c025_s082/generated/199.png"
require "${GRID}/intra_hs_c040_s082/generated/199.png"
require "${GRID}/intra_hs_c025_s086/generated/199.png"
require "${GRID}/intra_hs_c040_s086/generated/199.png"

echo "===== [1] compose selection rows (symlink only, no re-gen) @ $(date -Iseconds) ====="
"${PYTHON}" scripts/nda/compose_lg_selection.py \
  --u-hat "${LG}/router/u_hat_test.npy" \
  --u-true "${LG}/router/u_true_test.npy" \
  --grid-root "${GRID}" \
  --output-dir "${OUT}"

echo "===== [2] standard-7 eval @ $(date -Iseconds) ====="
cp -f "${STD7}/results.json" "${OUT}/results_std7_backup.json" || true
"${PYTHON}" scripts/nda/eval_standard7.py \
  --manifest "${OUT}/manifest_lgsel.json" --images-root "${IMAGES_ROOT}" \
  --out-dir "${STD7}" --device "${DEVICE}" --batch-size 16
cp -f "${STD7}/results.json" "${OUT}/results_lgsel.json"
OUT_LG="${OUT}" STD7_DIR="${STD7}" "${PYTHON}" - <<'PY'
import json, os
from pathlib import Path
out = Path(os.environ["OUT_LG"])
std7 = Path(os.environ["STD7_DIR"])
backup = json.loads(Path(out / "results_std7_backup.json").read_text(encoding="utf-8"))
sel = json.loads(Path(out / "results_lgsel.json").read_text(encoding="utf-8"))
by_tag = {r["tag"]: r for r in backup["rows"]}
for r in sel["rows"]:
    by_tag[r["tag"]] = r
merged = dict(backup)
merged["rows"] = [by_tag[t] for t in by_tag]
(std7 / "results.json").write_text(json.dumps(merged, indent=2), encoding="utf-8")
print(f"[OK] merged: {len(backup['rows'])} -> {len(merged['rows'])} rows")
PY

echo "===== [3] summary @ $(date -Iseconds) ====="
OUT_S="${OUT}" STD7_S="${STD7}" "${PYTHON}" - <<'PY'
import json, os
from pathlib import Path
out = Path(os.environ["OUT_S"])
std7 = Path(os.environ["STD7_S"])
res = json.loads((std7 / "results.json").read_text(encoding="utf-8"))
by = {r["tag"]: r for r in res["rows"]}
want = [r["tag"] for r in json.loads((out / "manifest_lgsel.json").read_text(encoding="utf-8"))["rows"]]
rows = [by[t] for t in want if t in by]
anchors = {}
for t in ("sdedit_ll_s082", "intra_sdedit_ll_s082", "intra_hs_c025_s082", "intra_hs_c032_s082",
          "intra_hs_c040_s082", "intra_hs_c025_s086", "intra_hs_c032_s086", "intra_hs_c040_s086",
          "official_atm_sub08"):
    if t in by:
        anchors[t] = by[t]
summary = {
  "pipeline": "lg_select_sub08",
  "claim_test": "two/four-level per-sample CN gating composed from existing grid images; r*=router(deployable), o*=oracle u_true upper bound",
  "candidates": "CN=0 sdedit-LL | CN=.25/.32/.40 grid at strength .82/.86 (pure intra)",
  "rows": rows,
  "anchor_rows": anchors,
}
(out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

du -sh "${OUT}" 2>/dev/null || true
echo "{\"pipeline\":\"lg_select_sub08\",\"finished\":\"$(date -Iseconds)\"}" > "${OUT}/job_done.json"
echo "===== DONE lg_select sub08 @ $(date -Iseconds) ====="
