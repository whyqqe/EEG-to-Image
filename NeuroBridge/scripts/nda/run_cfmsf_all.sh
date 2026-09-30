#!/usr/bin/env bash
# CF-MSF full pipeline, ALL 10 SUBJECTS, one job.
#
# Per subject, in order:
#   1) joint-train the encoder against the multi-level target  (arms: joint, frozen)
#   2) route probe on each resulting encoder                    (13 target spaces)
# then, across subjects:
#   3) per-subject table + mean/std + paired Wilcoxon/sign test  -> cfmsf_aggregate.py
#
# WHY ALL TEN AND NOT ONE
#   Every number so far is sub-08 only, and the literature reports the MEAN OVER 10
#   SUBJECTS (73.5% CORTIVA / 86.3% multi-blur+EVNet / 91.3% SAMGA).  A single subject
#   cannot distinguish "the method works" from "sub-08 is easy", and it gives no power
#   for a paired test.  Ten subjects in ONE chain is also what makes the joint-vs-frozen
#   comparison paired: every subject sees both arms with the same data, same loss,
#   same epochs, differing only in whether the encoder is in the gradient path.
#
# ORDER: sub-08 FIRST.  It is the subject whose numbers the whole project is calibrated
#   against (jobs 581546 / 581602), so it doubles as the live correctness check -- if
#   the chain is wired wrong it fails on subject one instead of after nine.
#
# RESUMABLE: each subject's stages are skipped when their output already exists, so a
#   wall-clock timeout costs at most the subject in flight, not the whole run.
#
# FAILURE POLICY: a subject that fails is recorded and the chain CONTINUES.  Nine good
#   subjects should not be discarded because one diverged.  The job still exits non-zero
#   at the end and the aggregate stage prints which subjects are missing.
set -uo pipefail   # NOTE: no -e; per-subject failures are handled explicitly below

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
cd "${NB_ROOT}"

PYTHON=/project/peilab/why/eeg-brainit/.venv/bin/python
[[ -x "${PYTHON}" ]] || { echo "[FATAL] missing ${PYTHON}"; exit 1; }
export HF_HOME=/project/peilab/why/cache/eeg-brainit/hf
export HF_HUB_CACHE=/project/peilab/why/cache/eeg-brainit/hf/hub
export TORCH_HOME=/project/peilab/why/cache/eeg-brainit/torch

OUT="${ALL_OUT:-${NB_ROOT}/outputs/cfmsf_all}"
DEVICE="${DEVICE:-cuda:0}"
TARGET="${TARGET:-levels_mean}"
ARMS="${ARMS:-joint,frozen}"
EPOCHS="${EPOCHS:-40}"
PROBE_EPOCHS="${PROBE_EPOCHS:-80}"
SUBJECTS="${SUBJECTS:-8,1,2,3,4,5,6,7,9,10}"
mkdir -p "${OUT}/logs"

log() { echo "[$(date -Iseconds)] $*"; }
require() { [[ -e "$1" ]] || { echo "[FATAL] missing $1"; return 1; }; }

require outputs/ocf/intra_enc/sub-08/checkpoint_ss_calib_best.pth || exit 1
require outputs/leakfree/split.json || exit 1
require outputs/nda_ss/sub-08/clip_text/train/concept_phrases.json || exit 1

# The GPU job re-checks the device invariant: a script edited between submit and start
# would otherwise burn GPU time on a bug a CPU smoke provably cannot detect (job 581629).
"${PYTHON}" scripts/nda/device_audit.py \
    scripts/nda/cfmsf_joint_train.py scripts/nda/cfmsf_route_probe.py \
    scripts/nda/cfmsf_joint_summary.py scripts/nda/cfmsf_aggregate.py \
  || { echo "[FATAL] device audit failed"; exit 1; }

declare -a FAILED=()
START_ALL=$(date +%s)

for sid in ${SUBJECTS//,/ }; do
  sid2=$(printf "%02d" "${sid}")
  SDIR="${OUT}/sub-${sid2}"
  mkdir -p "${SDIR}/logs"

  # ---- guard: the per-subject init encoder must exist and be PURE intra ----------
  INIT="${NB_ROOT}/outputs/ocf/intra_enc/sub-${sid2}/checkpoint_ss_calib_best.pth"
  if [[ ! -f "${INIT}" ]]; then
    log "[SKIP] sub-${sid2}: no init encoder ${INIT}"
    FAILED+=("sub-${sid2}: no init encoder")
    continue
  fi

  # ---- 1. joint encoder training -------------------------------------------------
  if [[ -f "${SDIR}/joint_report.json" ]]; then
    log "[skip] sub-${sid2}: joint_report.json present"
  else
    log "===== sub-${sid2} 1/2 joint encoder training (target=${TARGET} arms=${ARMS}) ====="
    if "${PYTHON}" scripts/nda/cfmsf_joint_train.py \
          --out "${SDIR}" --test-subject "${sid}" --target "${TARGET}" --arms "${ARMS}" \
          --epochs "${EPOCHS}" --device "${DEVICE}" \
          > "${SDIR}/logs/joint_train.log" 2>&1; then
      log "sub-${sid2} joint training OK"
    else
      log "[FAIL] sub-${sid2} joint training (tail follows)"
      tail -n 15 "${SDIR}/logs/joint_train.log"
      FAILED+=("sub-${sid2}: joint training failed")
      continue
    fi
  fi

  # ---- 2. route probe on each resulting encoder ----------------------------------
  for arm in ${ARMS//,/ }; do
    # TWO DIFFERENT VALUES, AND CONFLATING THEM COST JOB 581652 ALL 20 PROBES.
    #   ENC    = what `--z-root` must be.  The probe itself appends `sub-<sid>`, so
    #            `--z-root <...>/enc` makes it read `<...>/enc/sub-<sid>/shared_r_*.npy`.
    #            Passing `ENCSUB` here yields `<...>/enc/sub-<sid>/sub-<sid>/...`, which
    #            is exactly the FileNotFoundError that aborted every probe.
    #   ENCSUB = where THIS script can see the exported files, i.e. one level deeper.
    ENC="${SDIR}/${arm}/enc"
    ENCSUB="${ENC}/sub-${sid2}"
    if [[ ! -f "${ENCSUB}/shared_r_train.npy" ]]; then
      log "[FAIL] sub-${sid2}: missing encoder export ${ENCSUB}"
      FAILED+=("sub-${sid2}/${arm}: no encoder export")
      continue
    fi
    if [[ -f "${SDIR}/${arm}/probe/route_probe.json" ]]; then
      log "[skip] sub-${sid2}/${arm}: probe present"
      continue
    fi
    log "----- sub-${sid2} 2/2 route probe on '${arm}' -----"
    if "${PYTHON}" scripts/nda/cfmsf_route_probe.py \
          --out "${SDIR}/${arm}/probe" --test-subject "${sid}" \
          --z-root "${ENC}" --epochs "${PROBE_EPOCHS}" --device "${DEVICE}" \
          > "${SDIR}/logs/probe_${arm}.log" 2>&1; then
      log "sub-${sid2}/${arm} probe OK"
    else
      log "[FAIL] sub-${sid2}/${arm} probe (tail follows)"
      tail -n 15 "${SDIR}/logs/probe_${arm}.log"
      FAILED+=("sub-${sid2}/${arm}: probe failed")
    fi
  done

  N_ARMS=$(awk -F, '{print NF}' <<<"${ARMS}")
  DONE=$(ls "${SDIR}"/*/probe/route_probe.json 2>/dev/null | wc -l)
  log "sub-${sid2} complete: ${DONE}/${N_ARMS} probe outputs; elapsed $(( $(date +%s) - START_ALL ))s"
done

# ---- 3. cross-subject aggregation --------------------------------------------------
log "===== cross-subject aggregation (rule=lvl5+agg) ====="
"${PYTHON}" scripts/nda/cfmsf_aggregate.py \
    --root "${OUT}" --out "${OUT}/aggregate.json" --subjects "${SUBJECTS}" \
    --arms "${ARMS}" --pick lvl5+agg 2>&1 | tee "${OUT}/logs/aggregate.log"

log "===== done at $(( ($(date +%s) - START_ALL) / 60 )) min ====="
if (( ${#FAILED[@]} )); then
  echo "[FAILURES] ${#FAILED[@]}"
  printf '  %s\n' "${FAILED[@]}"
  du -sh "${OUT}" | sed 's/^/[disk] /'
  exit 1
fi
echo "[OK] all subjects completed"
du -sh "${OUT}" | sed 's/^/[disk] /'
