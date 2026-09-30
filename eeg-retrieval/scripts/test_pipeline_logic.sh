#!/usr/bin/env bash
# Unit tests for scripts/lib_pipeline_guard.sh and the arm-done guard.
#
# These run on the login node: no GPU, no data, no model downloads. Their job is
# to catch the class of bug that killed the first submission -- a mistyped
# `${remaining}` that `set -u` turned into a crash one second into a 24h job,
# after the allocation had already been granted.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
source "${HERE}/lib_pipeline_guard.sh"

PY="${PYTHON:-/project/peilab/why/eeg-brainit/.venv/bin/python}"
PASS=0; FAIL=0
ok()   { echo "  PASS  $1"; PASS=$((PASS+1)); }
bad()  { echo "  FAIL  $1"; FAIL=$((FAIL+1)); }
check(){ if [[ "$2" == "$3" ]]; then ok "$1"; else bad "$1 (got '$2', want '$3')"; fi; }

echo "== guard::remaining / have_time_for =="
guard::init 3600 600 "$(guard::now)"
check "fresh budget, remaining ~3600" "$(guard::remaining)" "3600"
guard::have_time_for 1200 && ok "1200s arm fits in 3600-600" || bad "1200s arm should fit"
guard::have_time_for 5000 && bad "5000s arm must not fit" || ok "5000s arm correctly rejected"

# Genuinely exhausted: the job started 100s ago but only had a 10s budget.
guard::init 10 5 "$(( $(guard::now) - 100 ))"
check "exhausted budget goes negative" "$(guard::remaining)" "-90"
guard::have_time_for 1 && bad "exhausted budget must reject even a 1s arm" \
                       || ok "exhausted budget rejects even a 1s arm"

# The reserve is only honoured if it actually blocks an arm that would otherwise
# fit: usable = 1000 - 900 = 100s, so a 150s arm must be refused.
guard::init 1000 900 "$(guard::now)"
guard::have_time_for 50  && ok "50s fits in the 100s usable window" || bad "50s should fit"
guard::have_time_for 150 && bad "150s must be blocked by the 900s reserve" \
                         || ok "150s correctly blocked by the 900s reserve"

check "elapsed_h has 2dp" "$(guard::elapsed_h | grep -cE '^[0-9]+\.[0-9]{2}$')" "1"

echo
echo "== arm_done guard (resumability) =="
TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT
printf '%s' '{"test":{"n":200,"top1":50.0}}' > "${TMP}/complete_result.json"
printf '%s' '{"best_val":{"top1":1.0}}'      > "${TMP}/no_test_result.json"
printf '%s' 'not json{{{'                     > "${TMP}/bad_result.json"

arm_done() {
  "${PY}" - "$1" <<'PY2' 2>/dev/null
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    ok = "test" in d and d["test"].get("n", 0) > 0
except Exception:
    ok = False
sys.exit(0 if ok else 1)
PY2
}
arm_done "${TMP}/complete_result.json" && ok "complete result -> skip" || bad "complete result should skip"
arm_done "${TMP}/no_test_result.json" && bad "result without test block must rerun" || ok "no test block -> rerun"
arm_done "${TMP}/bad_result.json"     && bad "corrupt json must rerun"         || ok "corrupt json -> rerun"
arm_done "${TMP}/nope_result.json"    && bad "missing file must rerun"         || ok "missing file -> rerun"

echo
echo "== dispatcher ==="
ARMS_OUT="$(ARM=list bash "${ROOT}/scripts/run_arm.sh" | tr '\n' ' ')"
check "arm list is non-empty" "$([[ -n "${ARMS_OUT// /}" ]] && echo yes)" "yes"
for a in ${ARMS_OUT}; do
  # Every arm must resolve to a descriptive tag and a note. run_arm.sh refuses to
  # train without a tag, and the pipeline uses the tag to decide whether the arm is
  # already done -- an arm with no mapping is a typo that only surfaces hours in.
  tag="$(ARM=tag PROBE_LAYER=block26 bash "${ROOT}/scripts/run_arm.sh" "${a}")"
  [[ -n "${tag}" ]] && ok "arm ${a} -> tag ${tag}" || bad "arm ${a} has no tag mapping"
  note="$(ARM=note bash "${ROOT}/scripts/run_arm.sh" "${a}")"
  [[ -n "${note}" ]] && ok "arm ${a} has a note" || bad "arm ${a} has no note"
done

# Tag collision is the dangerous one: the pipeline skips any arm whose result file
# already exists, so two arms sharing a tag means the second one silently never runs
# and the leaderboard reports a missing axis as though it were measured.
ALL_TAGS="$(for a in ${ARMS_OUT}; do
  ARM=tag PROBE_LAYER=block26 bash "${ROOT}/scripts/run_arm.sh" "${a}"; done)"
check "tags are unique across all arms" \
  "$(echo "${ALL_TAGS}" | wc -l)" "$(echo "${ALL_TAGS}" | sort -u | wc -l)"

# The two layer axes must not be conflated. --target-layer names a layer of the frozen
# image tower; --layers names the EEG encoder's depth. Different experiments, so they
# must not collapse onto one tag.
check "PA/T22/T28 (the target axis) get three distinct tags" \
  "$(for a in PA T22 T28; do ARM=tag PROBE_LAYER=block26 bash "${ROOT}/scripts/run_arm.sh" "${a}"; done \
     | sort -u | wc -l)" "3"

# PA's tag depends on the probe result, so a missing PROBE_LAYER has to be visible
# rather than silently producing 'align_unset' and re-running an arm that is done.
check "PA tag tracks the probe result" \
  "$(ARM=tag PROBE_LAYER=block26 bash "${ROOT}/scripts/run_arm.sh" PA)" "align_block26"
check "PA tag without a probe result is 'align_unset'" \
  "$(env -u PROBE_LAYER bash -c "ARM=tag bash '${ROOT}/scripts/run_arm.sh' PA")" "align_unset"
check "an unknown arm maps to an empty tag" \
  "$(ARM=tag bash "${ROOT}/scripts/run_arm.sh" NOPE_ARM | tr -d '[:space:]')" ""
ARM=NOPE_ARM bash "${ROOT}/scripts/run_arm.sh" >/dev/null 2>&1 \
  && bad "unknown arm must be refused" || ok "unknown arm is refused with a non-zero exit"
# The internal budget must stay BELOW the wall clock. If it does not, the guard never
# fires and Slurm kills the job mid-arm instead of the pipeline stopping cleanly after
# one -- which is exactly what the first submission did.
SB="${ROOT}/slurm/nwret_sub08.sbatch"
WALL_H="$(grep -oP '^#SBATCH --time=\K[0-9]+' "${SB}" | head -1)"
BUDGET_H="$(grep -oP '^export PIPELINE_HOURS="\$\{PIPELINE_HOURS:-\K[0-9]+' "${SB}" | head -1)"
if [[ -n "${WALL_H}" && -n "${BUDGET_H}" ]]; then
  (( BUDGET_H < WALL_H )) \
    && ok "pipeline budget ${BUDGET_H}h is below the ${WALL_H}h wall clock" \
    || bad "budget ${BUDGET_H}h must be below wall ${WALL_H}h or the guard never fires"
else
  bad "could not read wall=${WALL_H:-?} / budget=${BUDGET_H:-?} from the sbatch"
fi

echo
echo "== arm flag validation (dry run, no GPU) =="
# Exercises each arm's REAL flag list through train.py's argparse and config
# resolver. This is what catches a typo in an arm's flags before submission instead
# of hours into a job. It deliberately does not keep a copy of the flag lists: a
# duplicated list is how such a check drifts out of sync and starts passing vacuously.
for a in T22 T28 MTmean MTrouted E04 E08 FEU FER K49 K98; do
  if DRY_RUN=1 ARM="${a}" PROBE_LAYER=block26 PYTHON="${PY}" \
       bash "${ROOT}/scripts/run_arm.sh" >"/tmp/nwret_dry_${a}.log" 2>&1; then
    ok "arm ${a} flags validate"
  else
    bad "arm ${a} flags FAILED to validate -- see /tmp/nwret_dry_${a}.log"
  fi
done

# The checks above only mean something if a broken arm would actually fail.
if DRY_RUN=1 ARM=E04 PROBE_LAYER=block26 PYTHON="${PY}" EXTRA="--fusion-mode bogus" \
     bash "${ROOT}/scripts/run_arm.sh" >/dev/null 2>&1; then
  bad "an invalid choice must be rejected"
else
  ok "an invalid choice is rejected (so the checks above are not vacuous)"
fi

# Semantic checks: the axes must actually be varied, not merely accepted. A flag
# that parses but changes nothing is the quiet version of this whole class of bug.
check "K49 really yields 49 tokens" \
  "$(grep -o 'tokens=[0-9]*' "/tmp/nwret_dry_K49.log" | tail -1)" "tokens=49"
check "K98 really yields 98 tokens" \
  "$(grep -o 'tokens=[0-9]*' "/tmp/nwret_dry_K98.log" | tail -1)" "tokens=98"
grep -q "target-fusion=mean" "/tmp/nwret_dry_MTmean.log" \
  && ok "MTmean resolves to mean fusion" || bad "MTmean did not resolve to mean fusion"
grep -q "target-fusion=routed" "/tmp/nwret_dry_MTrouted.log" \
  && ok "MTrouted resolves to routed fusion" || bad "MTrouted did not resolve to routed"
grep -q "fusion=uniform" "/tmp/nwret_dry_FEU.log" \
  && ok "FEU uses uniform EEG-layer fusion" || bad "FEU is not uniform"
grep -q "fusion=routed" "/tmp/nwret_dry_FER.log" \
  && ok "FER uses routed EEG-layer fusion" || bad "FER is not routed"
check "MTmean blends 4 target layers" \
  "$(grep -o "targets=\['block[0-9]*', 'block[0-9]*', 'block[0-9]*', 'block[0-9]*'\]" \
     "/tmp/nwret_dry_MTmean.log" | wc -l)" "1"
check "E04 aligns to block26 at EEG depth 4" \
  "$(grep -c "eeg-layers=\[4\].*'block26'" "/tmp/nwret_dry_E04.log")" "1"

echo
echo "==================================================="
echo "  passed ${PASS}   failed ${FAIL}"
echo "==================================================="
[[ "${FAIL}" -eq 0 ]]
