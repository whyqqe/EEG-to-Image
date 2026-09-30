#!/usr/bin/env bash
# NeuroWeave Stage-1 on sub-08: hierarchical objective + temporal choices + cycle variants.
#
# ARMS (tested in one job so comparisons cannot drift)
#   frozen / lora / direct / multi_head / anytime_train / causal_stage
#
# THEN
#   anytime eval curve on every arm (150/350/700/1000 ms)
#   quick route probe (vith_cat5 + vith_levels_mean) on every arm
#   full route probe on frozen + probe winner
#   cycle raw vs disc on existing generation dirs (no diffusion)
#   pre-registered verdicts via neuroweave_summary.py
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

OUT="${NWEAVE_OUT:-${NB_ROOT}/outputs/neuroweave/sub-08}"
DEVICE="${DEVICE:-cuda:0}"
ARMS="${ARMS:-frozen,lora,direct,multi_head,anytime_train,causal_stage}"
EPOCHS="${EPOCHS:-40}"
PROBE_EPOCHS="${PROBE_EPOCHS:-40}"
PROBE_ONLY="${PROBE_ONLY:-vith_cat5,vith_levels_mean,vith_image}"
mkdir -p "${OUT}/logs"

log() { echo "[$(date -Iseconds)] $*"; }
require() { [[ -e "$1" ]] || { echo "[FATAL] missing $1"; return 1; }; }

require outputs/ocf/intra_enc/sub-08/checkpoint_ss_calib_best.pth || exit 1
require outputs/leakfree/split.json || exit 1
require scripts/nda/neuroweave_s1_train.py || exit 1
require scripts/nda/cfmsf_route_probe.py || exit 1

log "===== device audit ====="
"${PYTHON}" scripts/nda/device_audit.py \
    scripts/nda/neuroweave_s1_train.py \
    scripts/nda/neuroweave_anytime_eval.py \
    scripts/nda/cfmsf_route_probe.py \
  || { echo "[FATAL] device audit failed"; exit 1; }

log "===== Stage-1 train arms [${ARMS}] epochs=${EPOCHS} ====="
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

log "===== anytime eval curve ====="
if [[ -f "${OUT}/anytime_report.json" ]]; then
  log "[skip] anytime_report.json present"
else
  "${PYTHON}" scripts/nda/neuroweave_anytime_eval.py \
      --root "${OUT}" --arms "${ARMS}" --device "${DEVICE}" \
      > "${OUT}/logs/anytime.log" 2>&1 \
    || { echo "[FATAL] anytime failed; tail:"; tail -n 30 "${OUT}/logs/anytime.log"; exit 1; }
fi

probe_one() {
  local arm="$1"
  local enc="${OUT}/${arm}/enc"
  local pout="${OUT}/${arm}/probe"
  if [[ ! -f "${enc}/sub-08/shared_r_train.npy" ]]; then
    log "[skip] probe ${arm}: no enc export"
    return 0
  fi
  if [[ -f "${pout}/route_probe.json" ]]; then
    log "[skip] probe ${arm}: present"
    return 0
  fi
  log "----- quick probe ${arm} (${PROBE_ONLY}, ${PROBE_EPOCHS} ep) -----"
  if "${PYTHON}" scripts/nda/cfmsf_route_probe.py \
        --out "${pout}" --test-subject 8 --z-root "${enc}" \
        --only "${PROBE_ONLY}" --epochs "${PROBE_EPOCHS}" --device "${DEVICE}" \
        --fuse-topk 3 --fuse-by mini_csls --select-by mini_csls \
        > "${OUT}/logs/probe_${arm}.log" 2>&1; then
    log "probe ${arm} OK"
  else
    log "[FAIL] probe ${arm}; tail:"
    tail -n 20 "${OUT}/logs/probe_${arm}.log"
  fi
}

IFS=',' read -ra ARM_ARR <<< "${ARMS}"
for a in "${ARM_ARR[@]}"; do
  a="$(echo "$a" | xargs)"
  [[ -n "$a" ]] || continue
  probe_one "$a"
done

log "===== cycle raw vs disc on existing generations ====="
if [[ -f "${OUT}/cycle/cycle_report.json" ]]; then
  log "[skip] cycle_report present"
else
  # Canonical 200-png dirs from prior jobs (headline semantic vs structure arms).
  SETS=""
  ATM="outputs/atm_aligned_decode/sub-08/generation/combo_d40_luma_pc_a060/generated"
  MB1="outputs/mb_s08/gen/mb_p1_i1/generated"
  MBCN="outputs/mb_s08/gen/mb_p3_i3_cn/generated"
  [[ -d "$ATM" ]]  && SETS="atm_aligned=${ATM}"
  [[ -d "$MB1" ]]  && SETS="${SETS:+${SETS},}mb_p1_i1=${MB1}"
  [[ -d "$MBCN" ]] && SETS="${SETS:+${SETS},}mb_p3_i3_cn=${MBCN}"
  if [[ -z "$SETS" ]]; then
    log "[warn] no generation dirs found; skipping cycle"
  else
    log "cycle sets: ${SETS}"
    "${PYTHON}" scripts/nda/neuroweave_cycle_score.py \
        --out "${OUT}/cycle" --sets "${SETS}" --device "${DEVICE}" \
        > "${OUT}/logs/cycle.log" 2>&1 \
      || { echo "[FAIL] cycle; tail:"; tail -n 30 "${OUT}/logs/cycle.log"; }
  fi
fi

log "===== summary + pre-registered verdicts ====="
"${PYTHON}" scripts/nda/neuroweave_summary.py --root "${OUT}" \
  2>&1 | tee "${OUT}/logs/summary.log"

log "===== done ====="
du -sh "${OUT}" | sed 's/^/[disk] /'
