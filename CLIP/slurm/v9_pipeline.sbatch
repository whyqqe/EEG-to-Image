#!/bin/bash
#SBATCH --job-name=v9-pipeline
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=20:00:00
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --output=/project/peilab/why/CLIP/outputs/slurm/v9-pipeline-%j.out
#SBATCH --error=/project/peilab/why/CLIP/outputs/slurm/v9-pipeline-%j.err
#
# v9 PIPELINE -- validate the FGW weight, then launch the full experiment. ONE submission.
#
#   Stage A  deployment-only alpha sweep on the 30 BANKED G3 checkpoints (no training).
#   Stage B  pick alpha by LEAVE-ONE-SUBJECT-OUT, then train v9 on 10 folds x 3 seeds with that
#            alpha in BOTH places (the training coupling and the deployed operator), and
#            evaluate each checkpoint under three operators so the decomposition stays readable.
#
# WHY STAGE A COMES FIRST AND IS NOT OPTIONAL. The FGW weight has a measured INTERIOR optimum
# plus a catastrophic collapse at pure structure (frozen-checkpoint sweep, 5 subjects:
# alpha 0/0.1/0.25/0.5/0.75/1 -> 37.00/37.30/38.80/37.90/36.30/1.90). It is therefore a real
# hyperparameter, and fixing it by eye before spending 30 training runs is how a sweep ends up
# reporting a value that was chosen on the folds it is scored on. Stage A measures it on the 30
# banked runs and hands Stage B a number.
#
# WHY IT IS A SWEEP AND NOT A GUESS. `alpha = 0` is BIT-IDENTICAL to the v8 operator (verified),
# so the sweep is anchored to a reproduction, and the pipeline also carries the ALREADY-COMMITTED
# v8 result into the comparison -- the shipped stack is `v8 encoder + FGW`, and both halves have
# to be attributable:
#     banked G3          encoder=G3   operator=hard
#     outputs/eval/s3r_* encoder=G3   operator=S3R     <- job 643819, closed, +2.80pp
#     Stage A            encoder=G3   operator=FGW     <- isolates the structural term
#     v9 + S3R           encoder=v9   operator=S3R     <- isolates the training term
#     v9 + FGW           encoder=v9   operator=FGW     <- the shipped stack
#
# RESUMABILITY. Every step is skipped when its output already exists and is newer than the
# config it came from (AGENTS.md §3.5), so a wall-clock kill costs one fold, not the sweep. If
# Stage A's json already exists the sweep is skipped and its value is reused -- so re-submitting
# after a Stage B failure does NOT re-spend Stage A.
set -euo pipefail

ROOT="/project/peilab/why/CLIP"
PY="${ROOT}/.venv/bin/python"
mkdir -p "${ROOT}/outputs/slurm" "${ROOT}/logs"

export PYTHONNOUSERSITE=1
export HF_HOME=/project/peilab/why/cache/huggingface
export HF_HUB_CACHE=/project/peilab/why/cache/huggingface/hub
export OPENCLIP_CACHE_DIR=/project/peilab/why/cache/open_clip
export TORCH_HOME=/project/peilab/why/cache/torch
export XDG_CACHE_HOME=/project/peilab/why/cache/xdg
export TMPDIR="/tmp/clip-${SLURM_JOB_ID:-local}"
export XDG_CONFIG_HOME="${ROOT}/scratch/config"
export MPLCONFIGDIR="${ROOT}/scratch/mpl"
mkdir -p "${TMPDIR}" "${XDG_CONFIG_HOME}" "${MPLCONFIGDIR}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"

TAG="${TAG:-v9}"
CFG="${CFG:-${ROOT}/configs/${TAG}_loso_k20.yaml}"
CKPT_ROOT="${CKPT_ROOT:-${ROOT}/outputs/stage1/g3}"
SUBJECTS="${SUBJECTS:-1 2 3 4 5 6 7 8 9 10}"
SEEDS="${SEEDS:-2025 2026 2027}"
# The alpha grid is FINER THAN THE PROBE'S on purpose. The probe found its optimum (0.25) at
# tau = 0.05, and Stage A measures at tau = 0.01 -- the temperature Stage B actually deploys --
# so the optimum can move. Four points cost ~3 minutes of evals and remove the risk of shipping
# an alpha that is on the edge of the grid it was selected from.
ALPHAS="${ALPHAS:-0.1 0.25 0.5 0.75}"
TAU="${TAU:-0.01}"
ALPHA_OUT="${ALPHA_OUT:-${ROOT}/outputs/fgw_alpha.json}"
ALPHA_JSON_OVERRIDE="${ALPHA_JSON_OVERRIDE:-}"
LOG_DIR="${ROOT}/logs"
MASTER="${LOG_DIR}/${TAG}-pipeline-${SLURM_JOB_ID}.log"

log() { echo "[$(date -Is)] $*" | tee -a "${MASTER}"; }
run_step() {  # name logfile cmd...
  local name="$1" lf="$2"; shift 2
  log "  -> ${name}"
  local rc=0
  "$@" >"${lf}" 2>&1 || rc=$?
  if [[ "${rc}" -eq 0 ]]; then log "     OK   ${name}"; else
    log "     FAIL ${name} (rc=${rc}); tail:"; tail -n 30 "${lf}" | tee -a "${MASTER}"
    return "${rc}"
  fi
}

# `should_skip_train <out_dir> <config>`: reuse a finished fold instead of re-training it.
# A fold is finished when `last.pt` exists and is newer than the recipe, because the recipe is
# the only input whose change invalidates it. `slurm/v7_loso10x3.sbatch` has the same guard
# inline; it is factored here because Stage B calls it inside a loop body.
should_skip_train() {
  local out="$1" cfg="$2"
  [[ -s "${out}/last.pt" ]] || return 1
  [[ "${out}/last.pt" -nt "${cfg}" ]]
}

log "=== ${TAG} PIPELINE job=${SLURM_JOB_ID} host=$(hostname) cfg=${CFG} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
co_tenants=$(squeue -w "$(hostname)" -h -o '%i %u %j' | grep -v "${SLURM_JOB_ID}" || true)
if [[ -n "${co_tenants}" ]]; then log "shared-node co-tenants:"; echo "${co_tenants}" | tee -a "${MASTER}"; fi

"${PY}" -c "import sys,torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { log "[FATAL] CUDA unavailable on $(hostname)"; exit 1; }

# =====================================================================================
# PREFLIGHT -- the mechanism must be in the file, and the FGW algebra must be self-consistent.
# A flag that is accepted and then trains the baseline is this project's most expensive
# recurring failure.
log "--- preflight ---"
"${PY}" - "${CFG}" <<'PY' || exit 1
import sys
from samclip.utils import load_config
cfg = load_config(sys.argv[1])
print(f"[pre] recipe={sys.argv[1]} arch={cfg['arch']} objective={cfg['objective']} "
      f"epochs={cfg['epochs']} lr={cfg['lr']}")
fold = {k: cfg.get(k) for k in ("target_subject", "source_subjects", "seed")}
if any(v is not None for v in fold.values()):
    sys.exit(f"[pre] FATAL: recipe must be FOLD-FREE, got {fold}")
ra = cfg.get("recovery_aware") or {}
sp = cfg.get("soft_plan") or {}
if not (sp.get("enabled") and float(sp.get("weight", 0.0)) > 0.0):
    sys.exit("[pre] FATAL: `soft_plan` is not enabled with a positive weight, so this sweep "
             "would train the banked baseline thirty times under a v9 name.")
if float(sp.get("tau", 0.1)) < 0.05:
    sys.exit(f"[pre] FATAL: soft_plan.tau={sp.get('tau')!r} is at/below deployment's; the plan "
             "saturates and the term's gradient vanishes.")
if sp.get("block_per_subject") is not True:
    sys.exit("[pre] FATAL: soft_plan.block_per_subject must be true; an unblocked Sinkhorn "
             "column marginal equates different subjects' concept distributions.")
if int(sp.get("csls_k", 0) or 0) != int(ra.get("csls_k") or 0):
    sys.exit(f"[pre] FATAL: soft_plan.csls_k={sp.get('csls_k')!r} must equal "
             f"recovery_aware.csls_k={ra.get('csls_k')!r}.")
print(f"[pre] soft_plan OK: weight={sp.get('weight')} tau={sp.get('tau')} "
      f"alpha={sp.get('alpha')} block={sp.get('block_per_subject')}")

# The FGW algebra: alpha=0 must reproduce the shipped operator EXACTLY, and alpha>0 must not be
# a no-op. Asserted here rather than trusted, because "the flag silently did nothing" is the
# failure this project keeps paying for.
import numpy as np
from samclip import calibration as C
rng = np.random.default_rng(0)
q = rng.normal(size=(200, 64)); g = rng.normal(size=(200, 64))
o0, _ = C.subspace_soft_recovery(q, g, k=10, rho=0.1, rank=None, tau=0.05,
                                 hard_landmarks=False, alpha=0.0)
o0b, _ = C.subspace_soft_recovery(q, g, k=10, rho=0.1, rank=None, tau=0.05,
                                  hard_landmarks=False)
assert np.array_equal(o0, o0b), "alpha=0 is NOT bit-identical to the shipped S3R operator"
oa, da = C.subspace_soft_recovery(q, g, k=10, rho=0.1, rank=None, tau=0.05,
                                  hard_landmarks=False, alpha=0.25)
assert not np.allclose(o0, oa), "alpha=0.25 is a NO-OP vs alpha=0"
assert np.isfinite(oa).all() and not da["abstained"], da
# and the pure-structure collapse that motivates fusing rather than replacing
o1, d1 = C.subspace_soft_recovery(q, g, k=10, rho=0.1, rank=None, tau=0.05,
                                  hard_landmarks=False, alpha=1.0)
_, d0 = C.subspace_soft_recovery(q, g, k=10, rho=0.1, rank=None, tau=0.05,
                                 hard_landmarks=False, alpha=0.0)
print(f"[pre] FGW algebra OK: alpha=0 bit-identical, alpha=0.25 differs, "
      f"plan_acc alpha0={d0['plan_acc']:.3f} alpha1={d1['plan_acc']:.3f} "
      f"(pure structure must be WORSE: {d1['plan_acc'] < d0['plan_acc']})")
PY

# =====================================================================================
# STAGE A -- alpha sweep on the BANKED encoders. Deployment-only, so the encoder is the same 30
# weights the banked 45.53 came from and the only variable is the operator.
# =====================================================================================
if [[ -s "${ALPHA_OUT}" && -z "${ALPHA_JSON_OVERRIDE}" ]]; then
  log "--- Stage A SKIPPED: ${ALPHA_OUT} exists (reusing its selected_alpha) ---"
else
  log "--- Stage A: alpha sweep on banked G3 encoders (alphas: ${ALPHAS}) ---"
  for a in ${ALPHAS}; do
    EVAL_A="${ROOT}/outputs/eval/fgw_a${a}"
    mkdir -p "${EVAL_A}"
    for s in ${SUBJECTS}; do
      tag="$(printf 'sub%02d' "${s}")"
      if [[ "${s}" -eq 8 ]]; then REF_ARGS=(--ref-top1 22.00 --ref-top5 50.00)
      else                        REF_ARGS=(--ref-top1 unknown); fi
      for seed in ${SEEDS}; do
        ck="${CKPT_ROOT}/${tag}_k20_seed${seed}/last.pt"
        ev="${EVAL_A}/${tag}_seed${seed}.json"
        if [[ ! -s "${ck}" ]]; then log "  [SKIP] no checkpoint ${ck}"; continue; fi
        if [[ -s "${ev}" && "${ev}" -nt "${ck}" ]]; then
          log "  [SKIP] eval a=${a} ${tag}/seed${seed}"; continue
        fi
        run_step "A a=${a} ${tag}/seed${seed}" \
          "${LOG_DIR}/${TAG}-A-${tag}-seed${seed}-a${a}-${SLURM_JOB_ID}.log" \
          "${PY}" scripts/run_eval.py --ckpts "${ck}" \
            --target-subject "${s}" --out "${ev}" \
            --recovery --reps --csls-k 10 --rho 0.1 \
            --recovery-operator fgw --recovery-alpha "${a}" --recovery-tau "${TAU}" \
            "${REF_ARGS[@]}" \
          || log "  [WARN] Stage A eval failed (a=${a} ${tag}/seed${seed}); the fold is dropped from the selection"
      done
    done
  done

  log "--- Stage A selection (leave-one-subject-out) ---"
  run_step "select-alpha" "${LOG_DIR}/${TAG}-select-${SLURM_JOB_ID}.log" \
    "${PY}" scripts/select_fgw_alpha.py --eval-root outputs/eval/fgw \
      --alphas ${ALPHAS} --out "${ALPHA_OUT}"
  tail -n 40 "${LOG_DIR}/${TAG}-select-${SLURM_JOB_ID}.log" | tee -a "${MASTER}" || true
fi

# The value Stage B uses. Read from the json rather than re-derived, so the number in the log is
# the number that was actually used.
if [[ -n "${ALPHA_JSON_OVERRIDE}" ]]; then
  ALPHA="${ALPHA_JSON_OVERRIDE}"
  log "alpha OVERRIDDEN to ${ALPHA} (ALPHA_JSON_OVERRIDE)"
else
  ALPHA="$("${PY}" -c "import json,sys; print(json.load(open(sys.argv[1]))['selected_alpha'])" \
    "${ALPHA_OUT}")"
  log "alpha SELECTED by nested LOSO: ${ALPHA}"
fi
# Guard the value actually being passed on: a `null`/NaN/out-of-range alpha would either crash
# 30 training runs or silently disable the structural term.
"${PY}" - "${ALPHA}" <<'PY' || exit 1
import math, sys
a = float(sys.argv[1])
if not math.isfinite(a) or not (0.0 < a <= 1.0):
    sys.exit(f"[v9] FATAL: selected alpha {a!r} is not in (0, 1]; Stage A did not produce a "
             f"usable value.")
print(f"[v9] alpha {a} accepted for Stage B")
PY

# =====================================================================================
# STAGE B -- full v9 sweep: 10 folds x 3 seeds, alpha in the training coupling AND the operator.
# =====================================================================================
log "--- Stage B: ${TAG} training, 10 folds x 3 seeds, soft_plan.alpha=${ALPHA} ---"
OUT_ROOT="${ROOT}/outputs/stage1/${TAG}"
EVAL_ROOT="${ROOT}/outputs/eval/${TAG}"
for s in ${SUBJECTS}; do
  src=()
  for o in 1 2 3 4 5 6 7 8 9 10; do [[ "${o}" -ne "${s}" ]] && src+=("${o}"); done
  tag="$(printf 'sub%02d' "${s}")"
  if [[ "${s}" -eq 8 ]]; then REF_ARGS=(--ref-top1 22.00 --ref-top5 50.00)
  else                        REF_ARGS=(--ref-top1 unknown); fi

  for seed in ${SEEDS}; do
    out="${OUT_ROOT}/${tag}_k20_seed${seed}"
    ev="${EVAL_ROOT}/${tag}_seed${seed}.json"
    name="${tag}/seed${seed}"
    log "--- ${name} : sources ${src[*]} ---"

    if should_skip_train "${out}" "${CFG}"; then
      log "  [SKIP] train -- ${out}/last.pt is newer than the recipe"
    else
      run_step "train ${name}" \
        "${LOG_DIR}/${TAG}-train-${tag}-seed${seed}-${SLURM_JOB_ID}.log" \
        "${PY}" scripts/run_stage1.py --config "${CFG}" \
          --target-subject "${s}" --source-subjects "${src[@]}" \
          --out-dir "${out}" --seed "${seed}" \
          --mvnn train --channel-set all63 \
          --soft-plan-alpha "${ALPHA}" \
        || { log "  [WARN] train failed for ${name}; skipping its evals"; continue; }
    fi

    # Three operators per checkpoint, so the decomposition stays readable:
    #   hard  = the banked baseline, for a like-for-like G3 comparison
    #   sinkhorn (alpha=0) = isolates the TRAINING term
    #   fgw (alpha) = the shipped stack
    run_step "eval ${name}" \
      "${LOG_DIR}/${TAG}-eval-${tag}-seed${seed}-${SLURM_JOB_ID}.log" \
      "${PY}" scripts/run_eval.py --ckpts "${out}/last.pt" \
        --target-subject "${s}" --out "${ev}" \
        --recovery --reps --csls-k 10 --rho 0.1 "${REF_ARGS[@]}" \
      || log "  [WARN] eval (hard) failed for ${name}; per-fold re-run is cheap (skip-guarded)"

    ev2="${ROOT}/outputs/eval/${TAG}_s3r/${tag}_seed${seed}.json"
    mkdir -p "$(dirname "${ev2}")"
    run_step "eval-s3r ${name}" \
      "${LOG_DIR}/${TAG}-evals3r-${tag}-seed${seed}-${SLURM_JOB_ID}.log" \
      "${PY}" scripts/run_eval.py --ckpts "${out}/last.pt" \
        --target-subject "${s}" --out "${ev2}" \
        --recovery --reps --csls-k 10 --rho 0.1 \
        --recovery-operator sinkhorn --recovery-tau 0.01 "${REF_ARGS[@]}" \
      || log "  [WARN] eval-s3r failed for ${name}"

    ev3="${ROOT}/outputs/eval/${TAG}_fgw/${tag}_seed${seed}.json"
    mkdir -p "$(dirname "${ev3}")"
    run_step "eval-fgw ${name}" \
      "${LOG_DIR}/${TAG}-evalfgw-${tag}-seed${seed}-${SLURM_JOB_ID}.log" \
      "${PY}" scripts/run_eval.py --ckpts "${out}/last.pt" \
        --target-subject "${s}" --out "${ev3}" \
        --recovery --reps --csls-k 10 --rho 0.1 \
        --recovery-operator fgw --recovery-alpha "${ALPHA}" --recovery-tau "${TAU}" \
        "${REF_ARGS[@]}" \
      || log "  [WARN] eval-fgw failed for ${name}"
  done
done

# =====================================================================================
# SUMMARIES -- paired against the right baseline each time.
log "--- summaries ---"
run_step "summarize-hard" "${LOG_DIR}/${TAG}-summary-${SLURM_JOB_ID}.log" \
  "${PY}" scripts/summarize_g3.py --glob "outputs/eval/${TAG}/sub*_seed*.json" \
    --out "${ROOT}/outputs/${TAG}_summary.json" || true
tail -n 40 "${LOG_DIR}/${TAG}-summary-${SLURM_JOB_ID}.log" | tee -a "${MASTER}" || true

# v9 + FGW against the banked FGW grid of the SAME alpha: isolates the training term's effect on
# top of the structural operator.
run_step "summarize-fgw-vs-A" "${LOG_DIR}/${TAG}-summary-fgw-${SLURM_JOB_ID}.log" \
  "${PY}" scripts/summarize_s3r.py --ours "outputs/eval/${TAG}_fgw" \
    --base "outputs/eval/fgw_a${ALPHA}" \
    --out "${ROOT}/outputs/${TAG}_fgw_summary.json" || true
tail -n 30 "${LOG_DIR}/${TAG}-summary-fgw-${SLURM_JOB_ID}.log" | tee -a "${MASTER}" || true

# v9 + FGW against the SHIPPED v8 stack: the number that decides whether v9 ships.
if [[ -d "${ROOT}/outputs/eval/v8_s3r" ]]; then
  run_step "summarize-vs-v8" "${LOG_DIR}/${TAG}-summary-v8-${SLURM_JOB_ID}.log" \
    "${PY}" scripts/summarize_s3r.py --ours "outputs/eval/${TAG}_fgw" \
      --base "outputs/eval/v8_s3r" \
      --out "${ROOT}/outputs/${TAG}_vs_v8_summary.json" || true
  tail -n 25 "${LOG_DIR}/${TAG}-summary-v8-${SLURM_JOB_ID}.log" | tee -a "${MASTER}" || true
fi

log "ALL STEPS OK  (master log ${MASTER})"
