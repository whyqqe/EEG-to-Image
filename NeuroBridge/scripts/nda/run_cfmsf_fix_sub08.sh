#!/usr/bin/env bash
# CF-MSF FIX round, sub-08, one job.
#
# Trains the five arms that separate the three defects measured in job 581652, then
# re-runs the (UNCHANGED) 13-route probe on each encoder space so the fix is judged by
# the same 200-way number as every previous stage.
#
# IMPORTANT -- WHY THIS DOES NOT TOUCH THE RUNNING 10-SUBJECT JOB
#   Job 581652 invokes `cfmsf_joint_train.py` and `cfmsf_route_probe.py` once per
#   subject.  Editing either file mid-run would change the code that the remaining
#   subjects execute, so the chain's provenance would be broken and its arms would no
#   longer be comparable.  This round therefore adds ONLY new files and reuses the
#   probe read-only.  The `enc_aligned/` directory trick is what makes that possible:
#   the probe always reads `<z-root>/sub-XX/shared_r_{train,test}.npy`, so writing the
#   projection output under that filename lets the unmodified probe measure the space
#   the loss actually optimised.
#
# PROBE PLAN (sub-08)
#   frozen        -> enc            the control (== job 581602's setting)
#   joint         -> enc            the failing arm as previously measured
#   joint_aligned -> aligned        the space `joint` actually optimised  <- tests defect 1
#   direct        -> enc            no projection: optimised space IS the exported space
#   disc          -> enc            discriminative LRs (defect 2, LR fix)
#   lora          -> enc            rank-8 frozen-base adapters (defect 2, structural fix)
set -uo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
cd "${NB_ROOT}"

PYTHON=/project/peilab/why/eeg-brainit/.venv/bin/python
[[ -x "${PYTHON}" ]] || { echo "[FATAL] missing ${PYTHON}"; exit 1; }
export HF_HOME=/project/peilab/why/cache/eeg-brainit/hf
export HF_HUB_CACHE=/project/peilab/why/cache/eeg-brainit/hf/hub
export TORCH_HOME=/project/peilab/why/cache/eeg-brainit/torch

OUT="${FIX_OUT:-${NB_ROOT}/outputs/cfmsf_fix/sub-08}"
DEVICE="${DEVICE:-cuda:0}"
TARGET="${TARGET:-levels_mean}"
ARMS="${ARMS:-frozen,joint,direct,disc,lora}"
EPOCHS="${EPOCHS:-40}"
PROBE_EPOCHS="${PROBE_EPOCHS:-80}"
mkdir -p "${OUT}"/logs

log() { echo "[$(date -Iseconds)] $*"; }
require() { [[ -e "$1" ]] || { echo "[FATAL] missing $1"; return 1; }; }

require outputs/ocf/intra_enc/sub-08/checkpoint_ss_calib_best.pth || exit 1
require outputs/leakfree/split.json || exit 1
require scripts/nda/cfmsf_fix_train.py || exit 1
require scripts/nda/cfmsf_route_probe.py || exit 1

"${PYTHON}" scripts/nda/device_audit.py \
    scripts/nda/cfmsf_fix_train.py scripts/nda/cfmsf_route_probe.py \
  || { echo "[FATAL] device audit failed"; exit 1; }

log "===== fix round: train arms [${ARMS}] (target=${TARGET}, epochs=${EPOCHS}) ====="
if [[ -f "${OUT}/fix_report.json" ]]; then
  log "[skip] fix_report.json present (resume)"
else
  "${PYTHON}" scripts/nda/cfmsf_fix_train.py \
      --out "${OUT}" --test-subject 8 --target "${TARGET}" --arms "${ARMS}" \
      --epochs "${EPOCHS}" --device "${DEVICE}" \
      > "${OUT}/logs/fix_train.log" 2>&1 \
    || { echo "[FATAL] fix training failed; tail:"; tail -n 20 "${OUT}/logs/fix_train.log"; exit 1; }
fi
require "${OUT}/fix_report.json" || exit 1

# ---- probes.  (arm, space) pairs; each needs <space>/sub-08/shared_r_*.npy -------
probe_one() {
  local arm="$1" space="$2" tag="$3"
  local enc="${OUT}/${arm}/${space}/sub-08"
  if [[ ! -f "${enc}/shared_r_train.npy" ]]; then
    log "[skip] ${tag}: no encoder at ${enc}"
    return 0
  fi
  if [[ -f "${OUT}/${tag}/probe/route_probe.json" ]]; then
    log "[skip] ${tag}: probe present"
    return 0
  fi
  log "----- probe ${tag} (${arm}/${space}) -----"
  if "${PYTHON}" scripts/nda/cfmsf_route_probe.py \
        --out "${OUT}/${tag}/probe" --test-subject 8 --z-root "${OUT}/${arm}/${space}" \
        --epochs "${PROBE_EPOCHS}" --device "${DEVICE}" \
        > "${OUT}/logs/probe_${tag}.log" 2>&1; then
    log "probe ${tag} OK"
  else
    log "[FAIL] probe ${tag}; tail:"
    tail -n 12 "${OUT}/logs/probe_${tag}.log"
  fi
}

probe_one frozen enc        frozen
probe_one joint  enc        joint
probe_one joint  enc_aligned joint_aligned
probe_one direct enc        direct
probe_one disc   enc        disc
probe_one lora   enc        lora

log "===== summary over arms ====="
"${PYTHON}" scripts/nda/cfmsf_joint_summary.py \
    --root "${OUT}" --out "${OUT}/summary.json" --pick lvl5+agg --tol 0.10 \
    2>&1 | tee "${OUT}/logs/summary.log"

log "===== fix-round detail ====="
"${PYTHON}" - <<'PY'
import json, os
from pathlib import Path
OUT = Path(os.environ["FIX_OUT"])
r = json.loads((OUT / "fix_report.json").read_text())
print(f"{'arm':<10}{'lora':>6}{'r-direct':>9}{'sel epoch':>10}"
      f"{'valA 2way':>10}{'valA marg':>10}{'test200 t1':>12}{'t5':>8}{'rank':>8}")
for a, v in r["arms"].items():
    s = v["selectors"]["chosen_val_a"]
    print(f"{a:<10}{v['n_lora']:>6}{str(v['encode_r_directly']):>9}"
          f"{v['selectors']['picks']['margin']['epoch']:>10}"
          f"{s['two_way']:>10.4f}{s['margin']:>10.4f}"
          f"{v['test200_top1']:>12.4f}{v['test200_top5']:>8.4f}"
          f"{v['test200_mean_rank']:>8.1f}")
print("\n选择器分歧（同一臂内 margin / 2way / top1 各自会挑哪一轮）:")
for a, v in r["arms"].items():
    p = v["selectors"]["picks"]
    print(f"  {a:<8} margin->ep{p['margin']['epoch']:<3} "
          f"two_way->ep{p['two_way']['epoch']:<3} top1->ep{p['top1']['epoch']:<3} "
          f"| 不同轮次数={len(v['selectors']['disagreement'])}")
print("\n训练 vs 最后轮（判断是否还需要早停/正则）:")
for a, v in r["arms"].items():
    b, l = v["selectors"]["chosen_val_a"], v["selectors"]["last_epoch_val_a"]
    print(f"  {a:<8} 选中 ep{v['selectors']['picks']['margin']['epoch']:<3} "
          f"2way={b['two_way']:.4f} | 末轮 2way={l['two_way']:.4f} "
          f"delta={l['two_way']-b['two_way']:+.4f}")
PY
log "===== done ====="
du -sh "${OUT}" | sed 's/^/[disk] /'
