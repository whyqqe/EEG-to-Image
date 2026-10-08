#!/bin/bash
# =====================================================================================
# v6 CHAINED PIPELINE -- ONE job, every promising path to a SOTA-beating number, run
# SEQUENTIALLY in a single allocation (docs/eeg2image_v6_architecture.md).
#
# WHY ONE JOB AND NOT SIX.  The cluster is shared, and a six-arm fan-out whose arms each
# re-load the same fold's EEG and target stacks is a queue hog whose parts cannot be
# compared until all of them land.  More importantly a chained job makes the SEQUENCE the
# unit of record: each later step is a single-variable change from the one before it, so
# the log is the ablation, and there is no window in which "arm C landed but the baseline
# it is measured against did not".
#
# THE LADDER, in the order it runs (each step skip-guarded, see `should_skip`):
#
#   0  route caches   build the missing dinov2 TRAIN split (the eval split already exists)
#   1  gates          smoke_test.py (legacy regressions) + smoke_v6.py (v6 mechanisms)
#   2  V0  faithful RAT      objective=v4, in-loop FULL deploy stack (whiten->CSLS->NN->Procrustes)
#   3  V1  multi-route       objective=v6, three visual routes, score-level fusion
#   4  eval V0        full calibration ladder + recovery + C2 repetition refinement
#   5  eval V1        per-route rows AND the fused rows + C2, so a collapsed route is visible
#   6  summarize      one table across every step
#
# WHAT EACH STEP IS FOR, AND WHAT IT IS NOT
#   * V0 tests pillar A: does training the encoder inside the EXACT deployment stack
#     (`in_loop_full: true`) beat `v5-ep2`, which ran CSLS(k=20)+recovery while deployment
#     runs whiten+recovery(k=10)+CSLS(k=10)?  `v5-ep2` gained +3.00 raw and still lost
#     2.0pp on deployment, which is the hypothesis this arm exists to falsify.
#   * V1 tests pillars B+C: three visual geometries (semantic InternViT-6B / self-supervised
#     DINOv2-L / low-level 24x24 RGB) over ONE 868k-parameter EEG trunk, fused at the SCORE
#     level with uniform weights.  Axiom A5: concatenating embeddings first would impose a
#     single similarity geometry and discard the routes' disagreement, which is the reason
#     to have more than one.
#   * The C2 row is label-free transductive refinement on the UNAVERAGED test repetitions.
#     It never reads the pairing (`smoke_v6` test 9 asserts equivariance to a row permutation).
#
# Submit (this is the ONLY job this pipeline needs):
#     sbatch slurm/v6_chained_pipeline.sbatch
# =====================================================================================
#SBATCH --job-name=v6-pipeline
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --output=/project/peilab/why/CLIP/outputs/slurm/v6-pipeline-%j.out
#SBATCH --error=/project/peilab/why/CLIP/outputs/slurm/v6-pipeline-%j.err

set -euo pipefail

ROOT="/project/peilab/why/CLIP"
PY="${ROOT}/.venv/bin/python"
TARGET_SUBJECT="${TARGET_SUBJECT:-8}"
mkdir -p "${ROOT}/outputs/slurm"

# --- caches (AGENTS.md §2.4): MUST be exported before any torch/HF import ------------
export PYTHONNOUSERSITE=1
export HF_HOME=/project/peilab/why/cache/huggingface
export HF_HUB_CACHE=/project/peilab/why/cache/huggingface/hub
export TRANSFORMERS_CACHE=/project/peilab/why/cache/huggingface
export OPENCLIP_CACHE_DIR=/project/peilab/why/cache/open_clip
export TORCH_HOME=/project/peilab/why/cache/torch
export XDG_CACHE_HOME=/project/peilab/why/cache/xdg
export PIP_CACHE_DIR=/project/peilab/why/cache/pip
# TMPDIR is NODE-LOCAL on purpose: NFS + multiprocessing cannot rmtree its own per-worker
# temp dirs (unlink-while-open is silly-renamed to `.nfs*`, which stays visible -> EBUSY),
# which buried every job here under thousands of false tracebacks.
export TMPDIR="/tmp/clip-${SLURM_JOB_ID:-local}"
export XDG_CONFIG_HOME="${ROOT}/scratch/config"
export MPLCONFIGDIR="${ROOT}/scratch/mpl"
mkdir -p "${TMPDIR}" "${XDG_CONFIG_HOME}" "${MPLCONFIGDIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"

LOG_DIR="${ROOT}/outputs/slurm"
STAMP="$(date +%Y%m%d-%H%M%S)"
MASTER="${LOG_DIR}/v6-pipeline-${SLURM_JOB_ID}.log"

log() { echo "[$(date -Is)] $*" | tee -a "${MASTER}"; }

log "=== v6 chained pipeline job=${SLURM_JOB_ID} host=$(hostname) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
co_tenants=$(squeue -w "$(hostname)" -h -o '%i %u %j' | grep -v "${SLURM_JOB_ID}" || true)
if [[ -n "${co_tenants}" ]]; then
  log "shared-node co-tenants:"
  echo "${co_tenants}" | tee -a "${MASTER}"
fi

# --- hard CUDA gate: fail fast rather than silently running 3 trainings on CPU --------
"${PY}" -c "import sys,torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { log "[FATAL] CUDA unavailable on $(hostname)"; exit 1; }

# =====================================================================================
# Skip guard. A step that has already produced an output NEWER than every input that
# could have changed it is skipped, so a wall-clock kill or a preemption costs the
# current step rather than the whole pipeline (AGENTS.md §3.5). The input set is the
# config, the src tree and this script, because those are exactly what "the code that
# produced this artefact" means.
# =====================================================================================
should_skip() {
  local out="$1" cfg="$2"
  [[ -s "${out}" ]] || return 1
  local newest=""
  newest="$(find "${ROOT}/src" "${ROOT}/scripts" "${cfg}" \
            "${ROOT}/slurm/v6_chained_pipeline.sbatch" \
            -type f -newer "${out}" 2>/dev/null | head -1 || true)"
  [[ -z "${newest}" ]]
}

FAILED_STEPS=()
run_step() {                       # run_step <name> <logfile> <cmd...>
  local name="$1" lf="$2"; shift 2
  log "--- STEP ${name} : $* ---"
  local rc=0
  # The `|| rc=$?` form, NOT `if "$@"; then ... fi`: an `if` whose condition is false
  # returns 0, so reading `$?` after the `fi` reports success for a failed step. That is
  # exactly the silent pass this pipeline must not have.
  "$@" >"${lf}" 2>&1 || rc=$?
  if [[ ${rc} -eq 0 ]]; then
    log "    STEP ${name} OK  (log ${lf})"
    return 0
  fi
  log "    STEP ${name} FAILED rc=${rc}  (log ${lf})"
  tail -n 40 "${lf}" | tee -a "${MASTER}" || true
  FAILED_STEPS+=("${name}")
  return 0                          # keep chaining; the final rc reflects the failures
}

# =====================================================================================
# STEP 0 -- route caches
# =====================================================================================
# `dinov2_l14` is the only route whose features are not already on the shared tree. Both
# splits are requested: the builder `[SKIP]`s a file that exists, so this is idempotent
# when both are present and self-healing when one is missing or was written by an
# interrupted job. The builder also VERIFIES the new cache's concept grouping against
# `internvit_multilevel` -- a slot permutation between the EEG targets and the image
# targets loads fine, trains fine, and silently scores a different task.
step0() {
  "${PY}" scripts/build_route_cache.py --only dinov2_l14 --splits train test
}
ROUTE_DIR="${ROOT}/data/routes/dinov2_l14"
ROUTE_TRAIN="${ROUTE_DIR}/image_train_layer0.npy"
ROUTE_TEST="${ROUTE_DIR}/image_test_layer0.npy"
# Both splits are asked for, not just `train`: the builder `[SKIP]`s a file that is
# already there, so asking for both is idempotent when both exist and self-healing when
# one is missing (a killed build, or a file damaged by an interrupted write). Guarding on
# `train` alone would leave a missing/corrupt TEST cache to fail later inside
# `resolve_routes`, after the fold's EEG is already resident.
if should_skip "${ROUTE_TRAIN}" "${ROOT}/scripts/build_route_cache.py" \
   && should_skip "${ROUTE_TEST}" "${ROOT}/scripts/build_route_cache.py"; then
  log "--- STEP route-cache : [SKIP] both dinov2_l14 splits are up to date ---"
else
  run_step route-cache "${LOG_DIR}/v6-step0-routecache-${SLURM_JOB_ID}.log" step0 || true
fi

# DEGRADATION, LOUD. If the gamma route's TRAIN cache is still missing, the multi-route
# step runs with alpha+beta only and a GENERATED config records that. The alternative --
# aborting the whole pipeline -- would discard a finished V0 and a working two-route
# fusion over a cache that is orthogonal to both, and the alternative -- silently dropping
# the route -- is the misattribution this project keeps punishing. So the generated config
# is written, its route list is printed, and it is what the run record shows.
MULTIROUTE_CFG="${ROOT}/configs/loso_sub08_v6_multiroute.yaml"
GENERATED_CFG="${ROOT}/configs/loso_sub08_v6_multiroute_generated.yaml"
if [[ ! -s "${ROUTE_TRAIN}" || ! -s "${ROUTE_TEST}" ]]; then
  log "[WARN] the dinov2_l14 cache is incomplete (train=${ROUTE_TRAIN##*/} "
  log "       test=${ROUTE_TEST##*/}); writing a 2-route config (alpha+beta) to"
  log "       ${GENERATED_CFG}. The reported number is then a TWO-route fusion."
  "${PY}" - "${ROOT}/configs/loso_sub08_v6_multiroute.yaml" "${GENERATED_CFG}" <<'PY'
import sys, re
src, dst = sys.argv[1], sys.argv[2]
text = open(src).read()
text = re.sub(r"\n  - name: gamma\n(?:    .*\n)+", "\n", text)
text = "# GENERATED at submit/run time -- the gamma (DINOv2) route was DROPPED because its\n" \
       "# TRAIN feature cache was missing. This is a 2-route (alpha+beta) fusion.\n" + text
open(dst, "w").write(text)
print("wrote", dst)
PY
  MULTIROUTE_CFG="${GENERATED_CFG}"
fi

# =====================================================================================
# STEP 1 -- regression gates (cheap, CPU, and they run BEFORE the expensive work)
# =====================================================================================
if should_skip "${LOG_DIR}/v6-gates.ok" "${ROOT}/src/samclip/train.py"; then
  log "--- STEP gates : [SKIP] already green for this src tree ---"
else
  step1() { "${PY}" scripts/smoke_test.py && "${PY}" scripts/smoke_v6.py; }
  if run_step gates "${LOG_DIR}/v6-step1-gates-${SLURM_JOB_ID}.log" step1; then
    touch "${LOG_DIR}/v6-gates.ok"
  else
    # A red gate is a REAL stop: every arm below would be measured on a code path that
    # fails its own properties, and a number from it cannot be attributed to the treatment.
    log "[FATAL] regression gates failed; refusing to train on an unverified code path."
    exit 1
  fi
fi

# =====================================================================================
# STEP 2 -- V0 : faithful RAT (pillar A)
# =====================================================================================
V0_CFG="${ROOT}/configs/loso_sub08_v6_rat.yaml"
V0_OUT="${ROOT}/outputs/stage1/v6-rat"
step2() {
  "${PY}" scripts/run_stage1.py --config "${V0_CFG}" --target-subject "${TARGET_SUBJECT}" \
      --out-dir "${V0_OUT}" --seed 2025 --mvnn train --channel-set all63
}
if should_skip "${V0_OUT}/last.pt" "${V0_CFG}"; then
  log "--- STEP V0-rat : [SKIP] ${V0_OUT}/last.pt is up to date ---"
else
  run_step V0-rat "${LOG_DIR}/v6-step2-rat-${SLURM_JOB_ID}.log" step2 || true
fi

# =====================================================================================
# STEP 3 -- V1 : multi-route + score fusion (pillars B+C)
# =====================================================================================
V1_OUT="${ROOT}/outputs/stage1/v6-multiroute"
step3() {
  "${PY}" scripts/run_stage1.py --config "${MULTIROUTE_CFG}" \
      --target-subject "${TARGET_SUBJECT}" --out-dir "${V1_OUT}" --seed 2025 \
      --mvnn train --channel-set all63
}
if should_skip "${V1_OUT}/last.pt" "${MULTIROUTE_CFG}"; then
  log "--- STEP V1-multiroute : [SKIP] ${V1_OUT}/last.pt is up to date ---"
else
  run_step V1-multiroute "${LOG_DIR}/v6-step3-multiroute-${SLURM_JOB_ID}.log" step3 || true
fi

# =====================================================================================
# STEPS 4-6 -- evaluation
# =====================================================================================
# The full ladder is printed for every checkpoint (`scripts/run_eval.py`), and `--recovery`
# and `--reps` add the two deployment rungs that the v6 plan turns on. `--reps` (C2) runs
# here rather than in training because it is LABEL-FREE test-time refinement -- it is a
# deployment lever, not a training arm, so it must be measured on a frozen checkpoint.
CKPTS=()
[[ -s "${V0_OUT}/last.pt" ]] && CKPTS+=("${V0_OUT}/last.pt")
[[ -s "${V1_OUT}/last.pt" ]] && CKPTS+=("${V1_OUT}/last.pt")
if [[ ${#CKPTS[@]} -eq 0 ]]; then
  log "[FATAL] no checkpoint was produced; nothing to evaluate."
  exit 1
fi

EVAL_OUT="${ROOT}/outputs/eval/v6-chained-sub-$(printf '%02d' "${TARGET_SUBJECT}").json"
step4() {
  "${PY}" scripts/run_eval.py --ckpts "${CKPTS[@]}" --target-subject "${TARGET_SUBJECT}" \
      --out "${EVAL_OUT}" --recovery --reps --csls-k 10 --rho 0.1 --dump-ranks
}
if should_skip "${EVAL_OUT}" "${V0_CFG}"; then
  log "--- STEP eval : [SKIP] ${EVAL_OUT} is up to date ---"
else
  run_step eval "${LOG_DIR}/v6-step4-eval-${SLURM_JOB_ID}.log" step4 || true
fi

log "=== summary ==="
if [[ -s "${EVAL_OUT}" ]]; then
  "${PY}" - "${EVAL_OUT}" <<'PY' | tee -a "${MASTER}"
import json, sys
r = json.load(open(sys.argv[1]))
print(f"target sub-{r['target_subject']:02d}   reference (SAMGA official, this fold) "
      f"{r['reference']['top1']:.2f} / {r['reference']['top5']:.2f}")
print(f"{'checkpoint':<34} {'row':<28} {'Top-1':>8} {'Top-5':>8} {'Δref':>8}")
print("-" * 92)
best = None
for tag, res in r["checkpoints"].items():
    for name, m in res["rows"].items():
        print(f"{tag:<34} {name:<28} {m['top1']:>8.2f} {m['top5']:>8.2f} "
              f"{m['top1'] - r['reference']['top1']:>+8.2f}")
        if best is None or m["top1"] > best[0]:
            best = (m["top1"], tag, name, m["top5"])
    for route, rows in (res.get("route_rows") or {}).items():
        for name, m in rows.items():
            print(f"{tag + ' [' + route + ']':<34} {name:<28} {m['top1']:>8.2f} "
                  f"{m['top5']:>8.2f} {m['top1'] - r['reference']['top1']:>+8.2f}")
print("-" * 92)
if best:
    print(f"BEST: {best[0]:.2f} / {best[3]:.2f}  ({best[1]} | {best[2]})")
    print(f"      vs reference {r['reference']['top1']:.2f}: "
          f"{best[0] - r['reference']['top1']:+.2f}pp")
PY
else
  log "[WARN] no eval report at ${EVAL_OUT}"
fi

if [[ ${#FAILED_STEPS[@]} -gt 0 ]]; then
  log "FINISHED WITH FAILURES: ${FAILED_STEPS[*]}  (master log ${MASTER})"
  exit 1
fi
log "ALL STEPS OK  (master log ${MASTER})"
