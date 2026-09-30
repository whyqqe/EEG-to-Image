#!/usr/bin/env bash
# NeuroWeave Stage-1 ROUND 2 (sub-08): remove the two round-1 failures.
#
# ROUND 1 RESULT (jobs 586588/586589, outputs/neuroweave/sub-08)
#   arm            test200  best_single_csls  fuse_csls
#   frozen          0.3650  0.4900            0.4800
#   lora            0.3900  0.5000            0.4950   <- best single, +0.015
#   direct          0.1100  0.4000            0.3950   <- destroys the space
#   multi_head      0.3700  0.4800            0.4800   <- hierarchy did NOT win
#   anytime_train   0.3300  0.4950            0.4600
#   causal_stage    0.3900  0.4750            0.4700
#   anytime curve @150ms: anytime_train 0.1300 vs next-best 0.0450 (~3x)
#                         but @1000ms it DROPPED to 0.3300 vs 0.3900
#
# TWO THINGS THIS ROUND FIXES, each with its own pre-registered bar
#   FIX-1 (anytime full-window regression).  Round 1's uniform schedule saw a
#     complete trial on only 25% of steps.  Three schedules are separated because
#     they fail differently:
#       anytime_soft    -- P(full)=0.5, never out of distribution at test time
#       anytime_curric  -- truncation ramps 0 -> 0.6 over 20 epochs
#       anytime_consist -- soft schedule PLUS a truncated-vs-full representation
#                          consistency loss (the progressive-decoding objective)
#     BAR: full-window >= lora-0.01 AND @150ms >= 0.13 (keep the win, drop the cost).
#   FIX-2 (hierarchy).  three heads lost to one head, so hierarchy is now
#     expressed as auxiliary early/mid losses on the SINGLE exported head:
#       hier_aux
#     BAR: >= lora + 0.02, else the hierarchy claim is dead and must be dropped
#          from the paper rather than re-parameterised again.
#
# Also: cycle consistency is now scored in THREE visual spaces including two
# structural ones, with a hard GT-ordering control (see neuroweave_cycle_score.py).
#
# Round 1's arms are re-run here as in-job controls (training is ~30 s per arm),
# so no comparison crosses jobs.
set -uo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
cd "${NB_ROOT}"

PYTHON="${PYTHON:-/project/peilab/why/eeg-brainit/.venv/bin/python}"
[[ -x "${PYTHON}" ]] || { echo "[FATAL] missing ${PYTHON}"; exit 1; }
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/project/peilab/why/cache/eeg-brainit/hf/hub}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/project/peilab/why/cache/xdg}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-/project/peilab/why/cache/eeg-brainit/open_clip}"

OUT="${NWEAVE2_OUT:-${NB_ROOT}/outputs/neuroweave2/sub-08}"
DEVICE="${DEVICE:-cuda:0}"
ARMS="${ARMS:-frozen,lora,anytime_train,anytime_soft,anytime_curric,anytime_consist,hier_aux}"
EPOCHS="${EPOCHS:-40}"
PROBE_EPOCHS="${PROBE_EPOCHS:-40}"
mkdir -p "${OUT}/logs"

log() { echo "[$(date -Iseconds)] $*"; }
require() { [[ -e "$1" ]] || { echo "[FATAL] missing $1"; return 1; }; }

require outputs/ocf/intra_enc/sub-08/checkpoint_ss_calib_best.pth || exit 1
require outputs/leakfree/split.json || exit 1
require scripts/nda/neuroweave_s1_train.py || exit 1
require scripts/nda/neuroweave_cycle_score.py || exit 1

log "===== device audit ====="
"${PYTHON}" scripts/nda/device_audit.py \
    scripts/nda/neuroweave_s1_train.py \
    scripts/nda/neuroweave_anytime_eval.py \
    scripts/nda/neuroweave_cycle_score.py \
  || { echo "[FATAL] device audit failed"; exit 1; }

log "===== Stage-1 round 2: arms [${ARMS}] epochs=${EPOCHS} ====="
if [[ -f "${OUT}/s1_report.json" ]]; then
  log "[skip] s1_report.json present (resume)"
else
  "${PYTHON}" scripts/nda/neuroweave_s1_train.py \
      --out "${OUT}" --test-subject 8 --target levels_mean --arms "${ARMS}" \
      --epochs "${EPOCHS}" --device "${DEVICE}" \
      > "${OUT}/logs/s1_train.log" 2>&1 \
    || { echo "[FATAL] s1 train failed; tail:"; tail -n 40 "${OUT}/logs/s1_train.log"; exit 1; }
fi
require "${OUT}/s1_report.json" || exit 1

log "===== anytime curves (free: reuses the chosen checkpoints) ====="
if [[ -f "${OUT}/anytime_report.json" ]]; then
  log "[skip] anytime_report.json present"
else
  "${PYTHON}" scripts/nda/neuroweave_anytime_eval.py \
      --root "${OUT}" --arms "${ARMS}" --device "${DEVICE}" \
      > "${OUT}/logs/anytime.log" 2>&1 \
    || { echo "[FATAL] anytime failed; tail:"; tail -n 30 "${OUT}/logs/anytime.log"; exit 1; }
fi

# FULL 13-route probe, not the 3-route subset round 1 used, so the fused number
# is directly comparable to the historical 13-route reference (0.5350 CSLS).
probe_one() {
  local arm="$1"
  local enc="${OUT}/${arm}/enc"
  local pout="${OUT}/${arm}/probe"
  if [[ ! -f "${enc}/sub-08/shared_r_train.npy" ]]; then
    log "[skip] probe ${arm}: no enc export"; return 0
  fi
  if [[ -f "${pout}/route_probe.json" ]]; then
    log "[skip] probe ${arm}: present"; return 0
  fi
  log "----- probe ${arm} (13 routes, ${PROBE_EPOCHS} ep) -----"
  if "${PYTHON}" scripts/nda/cfmsf_route_probe.py \
        --out "${pout}" --test-subject 8 --z-root "${enc}" \
        --epochs "${PROBE_EPOCHS}" --device "${DEVICE}" \
        --fuse-topk 4 --fuse-by mini_csls --select-by mini_csls \
        > "${OUT}/logs/probe_${arm}.log" 2>&1; then
    log "probe ${arm} OK"
  else
    log "[FAIL] probe ${arm}; tail:"; tail -n 20 "${OUT}/logs/probe_${arm}.log"
  fi
}

IFS=',' read -ra ARM_ARR <<< "${ARMS}"
for a in "${ARM_ARR[@]}"; do
  a="$(echo "$a" | xargs)"; [[ -n "$a" ]] || continue
  probe_one "$a"
done

log "===== cycle: 3 visual spaces + GT ordering control ====="
if [[ -f "${OUT}/cycle/cycle_report.json" ]]; then
  log "[skip] cycle_report present"
else
  SETS=""
  ATM="outputs/atm_aligned_decode/sub-08/generation/combo_d40_luma_pc_a060/generated"
  MB1="outputs/mb_s08/gen/mb_p1_i1/generated"
  MBCN="outputs/mb_s08/gen/mb_p3_i3_cn/generated"
  [[ -d "$ATM" ]]  && SETS="atm_aligned=${ATM}"
  [[ -d "$MB1" ]]  && SETS="${SETS:+${SETS},}mb_p1_i1=${MB1}"
  [[ -d "$MBCN" ]] && SETS="${SETS:+${SETS},}mb_p3_i3_cn=${MBCN}"
  if [[ -z "$SETS" ]]; then
    log "[warn] no generation dirs; skipping cycle"
  else
    "${PYTHON}" scripts/nda/neuroweave_cycle_score.py \
        --out "${OUT}/cycle" --sets "${SETS}" --device "${DEVICE}" \
        > "${OUT}/logs/cycle.log" 2>&1 \
      || { echo "[FAIL] cycle; tail:"; tail -n 40 "${OUT}/logs/cycle.log"; }
  fi
fi

log "===== summary + pre-registered verdicts ====="
"${PYTHON}" scripts/nda/neuroweave_summary.py --root "${OUT}" \
  2>&1 | tee "${OUT}/logs/summary.log"

log "===== done ====="
du -sh "${OUT}" | sed 's/^/[disk] /'
