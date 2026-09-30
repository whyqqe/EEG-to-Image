#!/usr/bin/env bash
#SBATCH --job-name=nv-gates
#SBATCH --account=peilab
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/project/peilab/why/eeg-retrieval/alignment/outputs/slurm/%x-%j.out
#SBATCH --error=/project/peilab/why/eeg-retrieval/alignment/outputs/slurm/%x-%j.err
#
# =============================================================================
# ONE JOB that runs the whole critical path on CPU.
# =============================================================================
# Stages, in order, each of which can halt the rest:
#
#   0. selfcheck_lib.py        -- validates the CCA implementation against known
#                                 answers (the implementation was rewritten after a
#                                 derivation error, so nothing may run until this
#                                 passes).  ~20 s
#   1. lib_windows + detect_onset
#                              -- re-derives the stimulus onset from the data.
#                                 info.json advertises times = -0.2..1.0 s but the
#                                 stored arrays start AT onset; trusting the
#                                 metadata shifts every window by 200 ms and turns
#                                 the early evoked response into a fake
#                                 "pre-stimulus" interval.  This has already
#                                 fooled this pipeline once.  ~1 min
#   2. a0_synthetic.py         -- GATE 1. Recovers a KNOWN neural-visible rank from
#                                 simulated data.  If this fails, no EEG result is
#                                 interpretable.  ~8 min
#   3. a1_controls.py          -- GATE 2. Control battery: shuffled EEG, shuffled
#                                 CLIP, circular time shift, each must give k*=0,
#                                 while early evoked windows must give k*>0.  A
#                                 pipeline that always answers zero would pass a
#                                 lone negative control, hence the pairing.  ~15 min
#   4. a2_ceiling_effdim.py    -- A2/A3. Noise ceiling (split-half RDM reliability)
#                                 and EEG effective dimensionality.  These bound
#                                 every downstream claim and no model can lift
#                                 them.  Also decides the paper branch.  ~15 min
#
# Deliberately CPU-only: nothing here needs a GPU, so it cannot idle a GPU node.
# Feature extraction is NOT included: the shipped ViT-H-14 / RN50 features suffice
# for the gates, and burning GPU jobs before the gates pass would be wasteful.
# Submit that separately once the gates are green.
#
# Usage:  sbatch slurm/run_all_gates.sbatch
# =============================================================================
set -uo pipefail

ROOT=/project/peilab/why/eeg-retrieval/alignment
PY=${PY:-/project/peilab/why/eeg-brainit/.venv/bin/python}
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
cd "${ROOT}"
mkdir -p outputs/{slurm,a0,a1,a2,axis}

LOG="${ROOT}/outputs/slurm/nv-gates-${SLURM_JOB_ID:-local}.summary"
STATUS="${ROOT}/outputs/pipeline/gate_status.json"
mkdir -p "$(dirname "${STATUS}")"

echo "started: $(date -Iseconds)"  | tee "${LOG}"
echo "job_id : ${SLURM_JOB_ID:-local}" | tee -a "${LOG}"
echo "host   : $(hostname)"        | tee -a "${LOG}"
echo "python : ${PY}"              | tee -a "${LOG}"
echo "omp    : ${OMP_NUM_THREADS}" | tee -a "${LOG}"

STAGE_RESULTS=()
halted=""

run_stage() {
  # $1 = label, $2 = json key, rest = command
  local label="$1"; local key="$2"; shift 2
  echo ""                                   | tee -a "${LOG}"
  echo "==============================================================" | tee -a "${LOG}"
  echo "[stage] ${label}   $(date +%H:%M:%S)" | tee -a "${LOG}"
  echo "==============================================================" | tee -a "${LOG}"
  local t0=$SECONDS
  set +e
  "$@" 2>&1 | tee -a "${LOG}"
  local rc=${PIPESTATUS[0]}
  set -e
  local dt=$(( SECONDS - t0 ))
  STAGE_RESULTS+=("${key}:${rc}:${dt}")
  echo "[stage] ${label} -> rc=${rc}  (${dt}s)" | tee -a "${LOG}"
  if [[ $rc -ne 0 ]]; then
    halted="${label} (rc=${rc})"
    echo "[stage] *** HALTING: ${label} failed; downstream stages not run ***" | tee -a "${LOG}"
  fi
  return $rc
}

set -e

# ---------------------------- 0. implementation -----------------------------
run_stage "0. lib_rsca self-check" selfcheck \
  "${PY}" "${ROOT}/scripts/selfcheck_lib.py" || true

# ------------------------------ 1. time axis -------------------------------
if [[ -z "$halted" ]]; then
  run_stage "1a. window/axis module" axis \
    "${PY}" "${ROOT}/scripts/lib_windows.py" || true
fi
if [[ -z "$halted" ]]; then
  run_stage "1b. onset detection" onset \
    "${PY}" "${ROOT}/scripts/detect_onset.py" --out "${ROOT}/outputs/axis/onset" \
       --subs sub-01 sub-02 sub-03 || true
fi

# -------------------------------- 2. GATE 1 --------------------------------
if [[ -z "$halted" ]]; then
  run_stage "2. GATE 1 synthetic rank recovery" a0 \
    "${PY}" "${ROOT}/scripts/a0_synthetic.py" --out "${ROOT}/outputs/a0" \
       --seeds 5 --n-list 500 1654 --k-true-list 0 1 3 8 \
       --snr-list 0.01 0.10 0.50 --p 15750 --q 1024 --D 400 --K 32 --n-perm 1000 || true
fi

# -------------------------------- 3. GATE 2 --------------------------------
if [[ -z "$halted" ]]; then
  run_stage "3. GATE 2 control battery" a1 \
    "${PY}" "${ROOT}/scripts/a1_controls.py" --out "${ROOT}/outputs/a1" \
       --subs sub-01 sub-02 sub-03 sub-04 sub-05 --clip ViT-H-14 \
       --D 256 --K 24 --n-perm 500 --seeds 2 || true
fi

# ------------------------------ 4. bounds ----------------------------------
if [[ -z "$halted" ]]; then
  run_stage "4. A2/A3 ceiling + effective dim" a2 \
    "${PY}" "${ROOT}/scripts/a2_ceiling_effdim.py" --out "${ROOT}/outputs/a2" \
       --subs sub-01 sub-02 sub-03 sub-04 sub-05 --clips RN50 ViT-H-14 \
       --n-perm 200 || true
fi

# -------------------- 5. reproducible dimension + coverage ------------------
# Deliberately NOT a single number.  The first version of this estimator
# reported PR(psd(S_obs - S_noise/T)) and it grew with the analysis dimension
# budget instead of saturating (23% of D at D=100, 19% at D=1500), and a
# held-out screening count of the same quantity behaved the same way.  So this
# stage sweeps D to MEASURE the budget dependence, and runs the circular-shift
# control to separate stimulus-locked reproducible dimensions from the rest.
# See slurm/run_a3b.sbatch for the full rationale.
if [[ -z "$halted" ]]; then
  run_stage "5. A3b/A6 reproducible dim + coverage" a3b \
    "${PY}" "${ROOT}/scripts/a3b_reproducible_dim.py" --out "${ROOT}/outputs/a3b" \
       --subs sub-01 sub-02 sub-03 sub-04 sub-05 --which train \
       --windows w0_000_100ms w1_100_200ms w2_200_300ms w3_300_500ms \
                 w4_500_800ms w5_800_1000ms full_000_800ms full_000_1000ms \
       --clips ViT-H-14 RN50 \
       --d-grid 400 150 300 900 1400 --shrink 0.05 \
       --n-rep 8 --n-sim-null 400 --n-boot 200 --n-perm 200 --n-shift 2 || true
fi

# ------------------------------ summary ------------------------------------
"${PY}" - <<PYEOF
import json, os, datetime
root = "${ROOT}"
status = {
    "job_id": "${SLURM_JOB_ID:-local}",
    "finished": datetime.datetime.now().isoformat(timespec="seconds"),
    "halted_after": "${halted}" or None,
    "stages": {},
}
for item in """${STAGE_RESULTS[*]:-}""".split():
    parts = item.split(":")
    if len(parts) == 3:
        status["stages"][parts[0]] = {"rc": int(parts[1]), "seconds": int(parts[2])}
for g, f in [("axis", "axis/gate.json"), ("a0", "a0/gate.json"),
             ("a1", "a1/gate.json"), ("a2", "a2/gate.json")]:
    p = os.path.join(root, "outputs", f)
    if os.path.exists(p):
        try:
            status["stages"].setdefault(g, {}).update(json.load(open(p)))
        except Exception:
            pass
os.makedirs(os.path.dirname("${STATUS}"), exist_ok=True)
json.dump(status, open("${STATUS}", "w"), indent=2)
print(json.dumps(status, indent=2))
PYEOF

echo "" | tee -a "${LOG}"
echo "finished: $(date -Iseconds)" | tee -a "${LOG}"
echo "gate status -> ${STATUS}" | tee -a "${LOG}"
echo "summary     -> ${LOG}" | tee -a "${LOG}"
if [[ -n "$halted" ]]; then
  echo "A GATE FAILED (${halted}); the pipeline is designed to stop there." | tee -a "${LOG}"
  exit 1
fi
echo "ALL STAGES COMPLETED." | tee -a "${LOG}"
