#!/usr/bin/env bash
# CF-MSF selection-rule probe, sub-08, one job.
#
# PURPOSE: price the head/route SELECTION RULE against an oracle ceiling, using the
# frozen arm's export from job 581704 (so the numbers are directly comparable with the
# 0.3815 inductive / 0.5445 transductive figures already reported).
#
# WHY THIS RUN EXISTS
#   `cfmsf_select_audit.py` (read-only, already run) measured over 260 route-fits:
#       Spearman(val_top1, test_top1) = +0.509
#       sd(val_top1) = 0.0075  vs  sd(test_top1) = 0.0591   (8x coarser)
#   and that `--fuse-topk 4`'s val-based route choice loses to an oracle pick in 20/20
#   (subject, arm) pairs, missing the single best route in 12/20.  Meanwhile
#   `cfmsf_train.py:train_route` keeps only the last 5 epochs of history, so the
#   per-epoch curve -- the thing that would show whether a bad epoch was selected -- has
#   never been recorded for ANY route in this project.  So the claim "the selector is the
#   bottleneck" could not be tested against existing artefacts; this run records the
#   curve and tests it.
#
# WHAT IT TESTS (all on the SAME encoder, SAME heads, SAME data, SAME seeds)
#   full_top1  the current rule (1654-way gallery top-1 on val_b)
#   mini_top1  val-concept gallery (83 columns) top-1        -- same question, resolvable
#   mini_csls  as above + CSLS
#   two_way    correct-vs-64-distractors
#   inst_cos   the auxiliary cosine the head already optimises
#   swaK       weight averaging over the last K epochs  -- DELETES the selection problem
#   fixedE     a fixed early epoch                      -- trivial baseline
#   ORACLE     best epoch that exists                   -- ceiling, NOT a result
#
# The oracle column requires the test set, so it is computed for diagnosis only and is
# never an input to any rule.  Every reported number in the summary table comes from a
# rule that saw val data alone.
set -uo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
cd "${NB_ROOT}"

PYTHON=/project/peilab/why/eeg-brainit/.venv/bin/python
[[ -x "${PYTHON}" ]] || { echo "[FATAL] missing ${PYTHON}"; exit 1; }
export HF_HOME=/project/peilab/why/cache/eeg-brainit/hf
export HF_HUB_CACHE=/project/peilab/why/cache/eeg-brainit/hf/hub
export TORCH_HOME=/project/peilab/why/cache/eeg-brainit/torch

OUT="${SEL_OUT:-${NB_ROOT}/outputs/cfmsf_sel/sub-08}"
DEVICE="${DEVICE:-cuda:0}"
# the frozen arm from job 581704: same encoder as the 0.3815/0.5445 numbers
Z_ROOT="${Z_ROOT:-${NB_ROOT}/outputs/cfmsf_all/sub-08/frozen/enc}"
EPOCHS="${EPOCHS:-80}"
# The route set spans the range the audit found interesting: the four multi-level
# aggregate targets that dominate the test ranking, the two extremes (plain image, the
# weakest structural route), and two single perturbations for contrast. Kept to 8 so the
# full per-epoch curve is affordable for every one of them.
ROUTES="${ROUTES:-vith_cat5,vith_mixall,vith_levels_mean,vith_cat3,vith_gaussiannoise,vith_lowresolution,vith_image,depth_clip}"
SWA_K="${SWA_K:-5,10,20}"
FIXED_EPOCHS="${FIXED_EPOCHS:-5,9,15,30}"
mkdir -p "${OUT}/logs"

log() { echo "[$(date -Iseconds)] $*"; }
require() { [[ -e "$1" ]] || { echo "[FATAL] missing $1"; return 1; }; }

require "${Z_ROOT}/sub-08/shared_r_train.npy" || exit 1
require "${Z_ROOT}/sub-08/shared_r_test.npy" || exit 1
require outputs/leakfree/split.json || exit 1
require scripts/nda/cfmsf_sel_probe.py || exit 1

"${PYTHON}" scripts/nda/device_audit.py scripts/nda/cfmsf_sel_probe.py \
  || { echo "[FATAL] device audit failed"; exit 1; }

# `vith_highres` does not exist -- it is caught here rather than 40 minutes into the run
# because the route list is user-editable and the probe's own validation happens late.
log "validating route list: ${ROUTES}"
"${PYTHON}" - <<PY || exit 1
import sys
sys.path.insert(0, "${NB_ROOT}/scripts/nda")
sys.path.insert(0, "${NB_ROOT}")
from types import SimpleNamespace
from cfmsf_route_probe import build_targets
args = SimpleNamespace(cond_cache="${NB_ROOT}/outputs/gem/cond_cache")
have = set(build_targets(args))
want = [s.strip() for s in "${ROUTES}".split(",") if s.strip()]
missing = [r for r in want if r not in have]
if missing:
    print(f"[FATAL] unknown routes {missing}; available: {sorted(have)}")
    sys.exit(1)
print(f"[ok] {len(want)} routes present: {want}")
PY

log "===== selection probe: routes=${ROUTES} epochs=${EPOCHS} z_root=${Z_ROOT} ====="
if [[ -f "${OUT}/sel_report.json" && "${FORCE:-0}" != "1" ]]; then
  log "[skip] sel_report.json present (set FORCE=1 to rerun)"
else
  "${PYTHON}" scripts/nda/cfmsf_sel_probe.py \
      --out "${OUT}" --test-subject 8 --z-root "${Z_ROOT}" \
      --routes "${ROUTES}" --epochs "${EPOCHS}" \
      --swa-k "${SWA_K}" --fixed-epochs "${FIXED_EPOCHS}" --device "${DEVICE}" \
      > "${OUT}/logs/sel_probe.log" 2>&1 \
    || { echo "[FATAL] selection probe failed; tail:"; tail -n 25 "${OUT}/logs/sel_probe.log"; exit 1; }
fi
require "${OUT}/sel_report.json" || exit 1

log "===== RULE vs ORACLE (per route, then pooled) ====="
"${PYTHON}" - <<PY | tee "${OUT}/logs/summary.log"
import json
import numpy as np
from pathlib import Path
r = json.loads(Path("${OUT}/sel_report.json").read_text())
print(f"subject={r['subject']}  routes={len(r['routes'])}  epochs={r['params']['epochs']}")
print(f"selection set = {r['protocol']['selection']}   confirmation = "
      f"{r['protocol']['confirmation']}   test = oracle column only")
print()
print(f"{'route':<22}{'oracle t1':>10}{'oracle ep':>10}"
      f"{'full t1':>9}{'full ep':>8}{'mini t1':>9}{'mini ep':>8}{'swa t1':>9}")
for name, v in r["routes"].items():
    fu = v["selectors"].get("full_top1", {})
    mi = v["selectors"].get("mini_top1", {})
    sw = v["swa"].get("swa" + "${SWA_K}".split(",")[-1], {})
    print(f"{name:<22}{v['oracle']['best_test_top1']:>10.4f}"
          f"{v['oracle']['best_epoch']:>10}{fu.get('top1', float('nan')):>9.4f}"
          f"{fu.get('epoch', -1):>8}{mi.get('top1', float('nan')):>9.4f}"
          f"{mi.get('epoch', -1):>8}{sw.get('top1', float('nan')):>9.4f}")
print()
print("POOLED OVER ROUTES")
print(f"{'rule':<14}{'mean test t1':>14}{'regret vs oracle':>18}{'n':>5}"
      f"{'|ep-oracle|>10':>16}")
for rule, s in sorted(r["summary"]["per_rule"].items(),
                      key=lambda kv: -kv[1]["mean_test_top1"]):
    print(f"{rule:<14}{s['mean_test_top1']:>14.4f}{s['mean_regret_vs_oracle']:>+18.4f}"
          f"{s['n']:>5}{s['far_epoch_gt10']:>16}")
print(f"{'ORACLE':<14}{r['summary']['oracle_mean_top1']:>14.4f}{0.0:>+18.4f}")
print()
print("The ORACLE row is a ceiling, not an achievable number: it is the best epoch that")
print("exists and it was found using the test set. The question this table answers is how")
print("much of that ceiling a legitimately-selected rule can reach.")
print()
# the single most decision-relevant comparison, stated as a number
fu = r["summary"]["per_rule"].get("full_top1", {}).get("mean_test_top1", float("nan"))
mi = r["summary"]["per_rule"].get("mini_top1", {}).get("mean_test_top1", float("nan"))
oc = r["summary"]["oracle_mean_top1"]
print(f"full_top1 = {fu:.4f}   mini_top1 = {mi:.4f}   oracle = {oc:.4f}")
print(f"  mini_top1 recovers {100*(mi-fu)/max(oc-fu,1e-9):.0f}% of the full->oracle gap"
      if oc > fu else "  no gap to recover; the selector is NOT the bottleneck")
PY

log "===== done ====="
du -sh "${OUT}" | sed 's/^/[disk] /'
