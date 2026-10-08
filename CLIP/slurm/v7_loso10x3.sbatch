#!/bin/bash
# =====================================================================================
# 10-fold inter-subject LOSO x 3 seeds, chained into ONE job. DRIVEN BY THE RECIPE.
#
# One chain, two mechanisms under test, selected by environment variable so the two sweeps
# cannot drift apart:
#
#     TAG=v7  CFG=configs/v7_loso_k20.yaml    cross-trial repetition collapse (T2'')
#     TAG=v7b CFG=configs/v7b_loso_k20.yaml   G3 + SCORE source-only recovery episode
#     TAG=v8  CFG=configs/v8_loso_k20.yaml    G3 + Sinkhorn soft-plan alignment (v8)
#
# S3R_TAU=<tau> (e.g. 0.01) additionally evaluates every checkpoint under the deployed
# soft-plan recovery operator into outputs/eval/${TAG}_s3r and prints a PAIRED comparison
# against the banked grid at outputs/eval/s3r_tau<tau>. Leave empty for the hard operator only.
#
# `TAG` also names the output roots, so the two sweeps can never overwrite each other. The
# preflight below asserts that THE MECHANISM UNDER TEST IS PRESENT IN THE FILE, per
# mechanism -- see its comments for why "differs from G3 somehow" is not a sufficient check.
#
# v7b is the primary candidate. SCORE (arXiv 2608.19134, 53.23/83.55 THINGS-EEG2 Top-1/Top-5)
# names two mechanisms in its own abstract: recovery-aware source training and source-only
# episodes. The first is already in G3 (csls_k=20, measured +5.5pp raw 3/3 seeds); the
# episode is the one component we never shipped. v7 (the repetition-collapse term) was
# falsified on fold 8 and is kept only as a runnable attribution arm.
#
# Everything else -- the fold identity handling, the skip guards, the eval calling
# convention, the summary -- is deliberately IDENTICAL across both, because the comparison
# is recipe-minus-G3 and any other difference would be an unmeasured second variable.
#
# WHY THIS JOB EXISTS. Every performance number this project held was a sub-08 number. The
# published cells we compare against -- 26.22/57.98 for a SAMGA encoder and 53.23/83.55 for
# SCORE -- are TEN-FOLD MEANS, and the last round of analysis produced direct evidence that
# sub-08 is a HARD fold (two structurally unrelated encoders both land on 35.50 under this
# ladder). So a sub-08 comparison is not a weak 10-fold comparison; it is a different
# question. This job measures the right one.
#
# WHAT IS RUN. 30 runs: 10 folds x seeds {2025, 2026, 2027}, final-epoch checkpoints,
# evaluated with the full deployment ladder plus the T2 repetition cloud.
#
# WHY CHAINED AND NOT 30 JOBS. The cluster is shared, 30 allocations each re-reading the
# same image-target stacks is a queue hog, and a chained job makes the SEQUENCE the unit of
# record. Each (fold, seed) is skip-guarded on its own outputs, so a wall-clock kill costs
# one unit rather than the sweep (AGENTS.md §3.5).
#
# COST. Measured: a 50-epoch stage-1 run is ~3 min on an H800 and one eval is ~10 s on CPU
# (the retrieval set is 200-way). The v7b episode adds one deployment-ladder pass per
# mini-batch, measured at ~2x the stage-1 wall time on the v6b attribution job. The only
# other non-obvious cost is that the first eval OF EACH FOLD builds two caches on demand:
# the held-out subject's MVNN-test whitener (7 of 10 are not cached) and its repetition
# cloud (`test_subNN_all63_reps_mvnntest_std.npy`, ~1 GB). Those are built lazily and
# cached, so seeds 2 and 3 of the same fold reuse them.
#
# THE FOLD IDENTITY IS PASSED EXPLICITLY, EVERY TIME. `--source-subjects` is computed here
# from the fold and handed to every run. It is not left to the config: the recipe files
# deliberately null `target_subject`/`source_subjects`, because the chain they would
# otherwise inherit ends in `v4.yaml`, which carries `target_subject: 8`. A forgotten
# override would train fold 3 on a source list containing sub-03 -- `load_loso` raises on
# that ("that is not a holdout") rather than leaking, but the fold should be auditable from
# the artefact, not from remembering what was on the command line.
#
# Submit v7b: TAG=v7b CFG="${PWD}/configs/v7b_loso_k20.yaml" sbatch slurm/v7_loso10x3.sbatch
# Submit v7 : sbatch slurm/v7_loso10x3.sbatch
# Smoke:      TAG=v7b SUBJECTS=8 SEEDS=2025 sbatch ...   (single fold, ~6 min)
# =====================================================================================
#SBATCH --job-name=v7-loso10x3
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --output=/project/peilab/why/CLIP/outputs/slurm/v7-loso10x3-%j.out
#SBATCH --error=/project/peilab/why/CLIP/outputs/slurm/v7-loso10x3-%j.err

set -euo pipefail

ROOT="/project/peilab/why/CLIP"
PY="${ROOT}/.venv/bin/python"
# The recipe is an ENVIRONMENT variable, not a hard-coded path, so one chain drives both
# mechanisms under test instead of two near-identical scripts drifting apart. `TAG` names
# the output roots so the two sweeps can never overwrite each other's checkpoints.
TAG="${TAG:-v7}"
CFG="${CFG:-${ROOT}/configs/${TAG}_loso_k20.yaml}"
mkdir -p "${ROOT}/outputs/slurm"

# --- caches (AGENTS.md §2.4): MUST be exported before any torch/HF import -------------
export PYTHONNOUSERSITE=1
export HF_HOME=/project/peilab/why/cache/huggingface
export HF_HUB_CACHE=/project/peilab/why/cache/huggingface/hub
export TRANSFORMERS_CACHE=/project/peilab/why/cache/huggingface
export OPENCLIP_CACHE_DIR=/project/peilab/why/cache/open_clip
export TORCH_HOME=/project/peilab/why/cache/torch
export XDG_CACHE_HOME=/project/peilab/why/cache/xdg
export PIP_CACHE_DIR=/project/peilab/why/cache/pip
# NODE-LOCAL, not NFS: multiprocessing's per-worker temp dirs cannot be rmtree'd on NFS
# (unlink-while-open becomes a visible `.nfs*` -> EBUSY), which buried jobs under
# thousands of false tracebacks (AGENTS.md §2.4).
export TMPDIR="/tmp/clip-${SLURM_JOB_ID:-local}"
export XDG_CONFIG_HOME="${ROOT}/scratch/config"
export MPLCONFIGDIR="${ROOT}/scratch/mpl"
mkdir -p "${TMPDIR}" "${XDG_CONFIG_HOME}" "${MPLCONFIGDIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"

LOG_DIR="${ROOT}/outputs/slurm"
MASTER="${LOG_DIR}/${TAG}-loso10x3-${SLURM_JOB_ID}.log"
OUT_ROOT="${ROOT}/outputs/stage1/${TAG}"
EVAL_ROOT="${ROOT}/outputs/eval/${TAG}"
mkdir -p "${OUT_ROOT}" "${EVAL_ROOT}"

log() { echo "[$(date -Is)] $*" | tee -a "${MASTER}"; }

log "=== ${TAG} 10-fold x 3-seed job=${SLURM_JOB_ID} host=$(hostname) cfg=${CFG} ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
co_tenants=$(squeue -w "$(hostname)" -h -o '%i %u %j' | grep -v "${SLURM_JOB_ID}" || true)
if [[ -n "${co_tenants}" ]]; then log "shared-node co-tenants:"; echo "${co_tenants}" | tee -a "${MASTER}"; fi

# --- hard CUDA gate: fail fast rather than silently training 30 runs on CPU ----------
"${PY}" -c "import sys,torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { log "[FATAL] CUDA unavailable on $(hostname)"; exit 1; }

# Fail before the first of 30 trainings if the recipe has no fold identity -- a null
# `target_subject` would otherwise surface as a TypeError from an f-string three frames in.
# The checks below then verify that THE MECHANISM UNDER TEST IS ACTUALLY IN THE FILE. They
# are per-mechanism rather than generic because "the config differs from G3 somehow" is not
# a check: the failure mode this guards is a 30-run sweep that silently reproduces G3 and
# gets written up as a legitimate negative result.
"${PY}" - "${CFG}" <<'PY' || exit 1
import sys
from samclip.utils import load_config
cfg = load_config(sys.argv[1])
print(f"[sweep] recipe {sys.argv[1]}: arch={cfg['arch']} objective={cfg['objective']} "
      f"epochs={cfg['epochs']} lr={cfg['lr']} recovery_aware={cfg['recovery_aware']}")
fold_keys = {k: cfg.get(k) for k in ("target_subject", "source_subjects", "seed")}
if any(v is not None for v in fold_keys.values()):
    sys.exit(f"[sweep] FATAL: the recipe must be FOLD-FREE, got {fold_keys}. A fold baked "
             f"into the config is a fold that can be inherited by the wrong run.")

ra = cfg.get("recovery_aware") or {}
concept = cfg.get("concept") or {}

if ra.get("recovery_episode"):
    # ---- v7b: G3 + SCORE's source-only recovery episode -------------------------------
    if not ra.get("in_loop_full"):
        sys.exit("[sweep] FATAL: `recovery_episode` without `in_loop_full` runs a SUBSET of "
                 "the deployment ladder, so the episode stops being a simulation of "
                 "deployment and the 'matched proxy' argument no longer holds.")
    if ra.get("episode_whiten") is not False:
        sys.exit("[sweep] FATAL: this sweep must run `episode_whiten: false`. G3's own "
                 "10-fold ladder shows the non-whitened rung WINS (+CSLS+recovery 35.72 vs "
                 "+whiten+CSLS+recovery 33.98), so whitening inside the episode trains the "
                 "encoder for a ladder deployment does not run -- and the measured cost of "
                 "getting that wrong was 9.00pp unexplained with raw cosine down 8.50pp "
                 "(v6-rat, job 642512).")
    if int(ra.get("episode_csls_k", ra.get("csls_k", 0))) != 10:
        sys.exit("[sweep] FATAL: the episode must select landmarks at deployment's k=10, "
                 f"got {ra.get('episode_csls_k')!r}. A k mismatch is exactly the defect that "
                 "made v5-ep2 lose (-2.00pp) while its raw number rose.")
    print(f"[sweep] EPISODE ON: k={ra.get('episode_csls_k')} rho={ra.get('episode_rho')} "
          f"whiten={ra.get('episode_whiten')} min_landmarks={ra.get('episode_min_landmarks')}")
elif concept.get("enabled") and float(concept.get("reps_weight", 0.0)) > 0.0:
    # ---- v7: cross-trial repetition collapse (T2'' / T2') ----------------------------
    if concept.get("reps_group") != "stimulus":
        sys.exit(f"[sweep] FATAL: this sweep must run `reps_group: stimulus`; got "
                 f"{concept.get('reps_group')!r}. `row` is the rejected anti-T1 arm (the "
                 f"sampler tiles 9 subjects per stimulus, so its negatives include other "
                 f"subjects' views of the SAME image).")
    if not concept.get("reps_detach_gate", False):
        sys.exit("[sweep] FATAL: this sweep must run `reps_detach_gate: true`; without it "
                 "the term displaces the SMN scale gate (~5pp raw; measured gate "
                 "0.498 -> 0.203 -> 0.000 for weight 0 -> 0.1 -> 0.5).")
    print(f"[sweep] T2'' ON: weight={concept['reps_weight']} group={concept['reps_group']} "
          f"n_stim={concept.get('reps_n_stim')} subj_sel={concept.get('reps_subj_sel')} "
          f"r_use={concept.get('reps_r_use')}")
elif cfg.get("soft_plan") and bool((cfg.get("soft_plan") or {}).get("enabled")) \
        and float((cfg.get("soft_plan") or {}).get("weight", 0.0)) > 0.0:
    # ---- v8: soft-plan alignment (the operator-aligned term) --------------------------
    sp = cfg["soft_plan"]
    # `tau` IS NOT FREE HERE. Deployment's tau is 0.01-0.05, and it is tempting to reuse it.
    # At that value the plan is a near-permutation whose Jacobian through the Sinkhorn loop
    # vanishes, so the term would be reported as active while contributing nothing -- and the
    # run would come back reading as "the idea did not help" when nothing was tested.
    if float(sp.get("tau", 0.1)) < 0.05:
        sys.exit("[sweep] FATAL: `soft_plan.tau` "
                 f"{sp.get('tau')!r} is at or below deployment's; the plan saturates to a "
                 "near-permutation and the term's gradient vanishes. Training tau must be "
                 "looser than deployment's (0.1 measured |grad| 1.6e1 vs 2.2e1 at 0.05).")
    # The block is a structural requirement, not a regulariser: deployment always scores ONE
    # held-out subject, and an unblocked plan was measured to spend 54% of its mass
    # cross-subject.
    if sp.get("block_per_subject") is not True:
        sys.exit("[sweep] FATAL: this sweep must run `soft_plan.block_per_subject: true`. "
                 "An unblocked Sinkhorn column marginal equates different subjects' concept "
                 "distributions, a constraint with no basis in the task (measured: 54% of "
                 "plan mass cross-subject on an 8-row toy block).")
    if int(sp.get("csls_k", 0) or 0) != int((ra.get("csls_k") or 0)):
        sys.exit("[sweep] FATAL: `soft_plan.csls_k` "
                 f"{sp.get('csls_k')!r} must equal `recovery_aware.csls_k` "
                 f"{ra.get('csls_k')!r}; the plan has to be built in the same metric the "
                 "deployment ladder builds it in, or the term shapes a plan nobody computes.")
    print(f"[sweep] SOFT_PLAN ON: weight={sp.get('weight')} tau={sp.get('tau')} "
          f"iters={sp.get('iters')} csls_k={sp.get('csls_k')} "
          f"block={sp.get('block_per_subject')}  "
          f"(as the DECLARED candidate against banked sinkhorn 48.13/48.12 at tau 0.01/0.05)")
else:
    sys.exit(f"[sweep] FATAL: the recipe enables NO mechanism under test (neither "
             f"`recovery_aware.recovery_episode`, nor `concept.enabled` with a positive "
             f"weight, nor `soft_plan.enabled` with a positive weight). This job would train "
             f"G3 thirty times under the '{sys.argv[1]}' name -- the one failure mode that "
             f"makes the whole sweep worthless while looking like a legitimate negative "
             f"result.")
PY

# --- skip guard: outputs newer than everything that could have changed them -----------
should_skip() {
  local out="$1"
  [[ -s "${out}" ]] || return 1
  local newest
  newest="$(find "${ROOT}/src" "${ROOT}/scripts" "${CFG}" \
                 "${BASH_SOURCE[0]}" \
                 -type f -newer "${out}" 2>/dev/null | head -1 || true)"
  [[ -z "${newest}" ]]
}

FAILED_STEPS=()
run_step() {                       # run_step <name> <logfile> <cmd...>
  local name="$1" lf="$2"; shift 2
  log "  -> ${name}"
  local rc=0
  # `|| rc=$?`, NOT `if "$@"; then`: an `if` whose condition is false returns 0, so
  # reading `$?` after `fi` reports success for a failed step -- the silent pass this
  # pipeline must not have.
  "$@" >"${lf}" 2>&1 || rc=$?
  if [[ ${rc} -eq 0 ]]; then
    log "     OK   ${name}"
    return 0
  fi
  log "     FAILED rc=${rc}  ${name}  (log ${lf})"
  tail -n 30 "${lf}" | tee -a "${MASTER}" || true
  FAILED_STEPS+=("${name}")
  return 0                         # keep chaining; the final rc reflects the failures
}

SUBJECTS="${SUBJECTS:-1 2 3 4 5 6 7 8 9 10}"
SEEDS="${SEEDS:-2025 2026 2027}"
log "folds: ${SUBJECTS}"
log "seeds: ${SEEDS}"

for s in ${SUBJECTS}; do
  # The other nine subjects, computed rather than listed: a hand-written list is exactly
  # how a fold ends up training on itself.
  src=()
  for o in 1 2 3 4 5 6 7 8 9 10; do [[ "${o}" -ne "${s}" ]] && src+=("${o}"); done
  tag="$(printf 'sub%02d' "${s}")"

  # The SAMGA-official reference was measured by us on sub-08 ONLY. Passing it there keeps
  # the historical cell comparable; everywhere else `unknown` makes the report print `n/a`
  # instead of a delta against a baseline from a different subject (see run_eval REF_TOP1).
  if [[ "${s}" -eq 8 ]]; then REF_ARGS=(--ref-top1 22.00 --ref-top5 50.00)
  else                        REF_ARGS=(--ref-top1 unknown); fi

  for seed in ${SEEDS}; do
    out="${OUT_ROOT}/${tag}_k20_seed${seed}"
    ev="${EVAL_ROOT}/${tag}_seed${seed}.json"
    name="${tag}/seed${seed}"
    log "--- ${name} : sources ${src[*]} ---"

    if should_skip "${out}/last.pt"; then
      log "  [SKIP] train -- ${out}/last.pt is up to date"
    else
      run_step "train ${name}" "${LOG_DIR}/${TAG}-train-${tag}-seed${seed}-${SLURM_JOB_ID}.log" \
        "${PY}" scripts/run_stage1.py --config "${CFG}" \
          --target-subject "${s}" --source-subjects "${src[@]}" \
          --out-dir "${out}" --seed "${seed}" \
          --mvnn train --channel-set all63
    fi

    # An eval is stale when its checkpoint is newer than it, even if no source file moved:
    # the `.json` and the `.pt` are the pair, and re-running only train would otherwise
    # leave a report from the previous checkpoint on disk under the current name.
    if [[ -s "${ev}" ]] && [[ "${ev}" -nt "${out}/last.pt" ]] && should_skip "${ev}"; then
      log "  [SKIP] eval -- ${ev} is newer than its checkpoint"
    elif [[ ! -s "${out}/last.pt" ]]; then
      log "  [SKIP] eval -- no checkpoint at ${out}/last.pt (its training failed above)"
    else
      run_step "eval ${name}" "${LOG_DIR}/${TAG}-eval-${tag}-seed${seed}-${SLURM_JOB_ID}.log" \
        "${PY}" scripts/run_eval.py --ckpts "${out}/last.pt" \
          --target-subject "${s}" --out "${ev}" \
          --recovery --reps --csls-k 10 --rho 0.1 "${REF_ARGS[@]}"

      # ---- the DEPLOYED operator, evaluated in the same job ----------------------------
      # `S3R_TAU` (empty = skip) adds a second evaluation of the same checkpoint with the
      # soft-plan recovery operator. Two reasons it belongs here rather than in a follow-up
      # job. First, the arm being shipped is `encoder + operator`, so a training run that
      # reports only the operator it will not use is reporting the wrong number -- and
      # re-deriving it later means re-loading 30 checkpoints. Second, evaluating both in one
      # job makes the two numbers come from one artefact set, so a comparison between this
      # arm and the banked `outputs/eval/s3r_tau*` grid (job 643819) is paired by
      # construction rather than by a join that could silently drop runs.
      #
      # Cheap: `run_eval` re-embeds from the checkpoint, and a fold costs seconds on a GPU.
      if [[ -n "${S3R_TAU:-}" ]]; then
        ev2="${EVAL_ROOT}_s3r/${tag}_seed${seed}.json"
        mkdir -p "$(dirname "${ev2}")"
        run_step "eval-s3r ${name}" \
          "${LOG_DIR}/${TAG}-evals3r-${tag}-seed${seed}-${SLURM_JOB_ID}.log" \
          "${PY}" scripts/run_eval.py --ckpts "${out}/last.pt" \
            --target-subject "${s}" --out "${ev2}" \
            --recovery --reps --csls-k 10 --rho 0.1 \
            --recovery-operator sinkhorn --recovery-tau "${S3R_TAU}" "${REF_ARGS[@]}"
      fi
    fi
  done
done

# =====================================================================================
# summary -- reads whatever landed and says so if the grid is incomplete
# =====================================================================================
log "=== summary ==="
run_step "summarize" "${LOG_DIR}/${TAG}-summary-${SLURM_JOB_ID}.log" \
  "${PY}" scripts/summarize_g3.py --glob "outputs/eval/${TAG}/sub*_seed*.json" \
    --out "${ROOT}/outputs/${TAG}_summary.json"
tail -n 60 "${LOG_DIR}/${TAG}-summary-${SLURM_JOB_ID}.log" | tee -a "${MASTER}" || true

# The deployed-operator summary is a PAIRED comparison against the banked sinkhorn grid
# (`outputs/eval/s3r_tau*`, job 643819, which is G3's own 30 checkpoints under this operator).
# Paired because recovery gain varies by several pp across folds (+3.62 +- 1.72 over 30 runs),
# so the between-fold spread dominates the effect and an unpaired difference of means would be
# the weaker test -- the same lesson §7.5 records.
if [[ -n "${S3R_TAU:-}" && -d "${ROOT}/outputs/eval/s3r_tau${S3R_TAU}" ]]; then
  run_step "summarize-s3r" "${LOG_DIR}/${TAG}-summary-s3r-${SLURM_JOB_ID}.log" \
    "${PY}" scripts/summarize_s3r.py --ours "outputs/eval/${TAG}_s3r" \
      --base "outputs/eval/s3r_tau${S3R_TAU}" \
      --out "${ROOT}/outputs/${TAG}_s3r_summary.json" \
    || log "  (s3r summary failed; per-fold reports are still on disk)"
  tail -n 30 "${LOG_DIR}/${TAG}-summary-s3r-${SLURM_JOB_ID}.log" | tee -a "${MASTER}" || true
fi

if [[ ${#FAILED_STEPS[@]} -gt 0 ]]; then
  log "FINISHED WITH FAILURES (${#FAILED_STEPS[@]}): ${FAILED_STEPS[*]}"
  log "re-submit to retry only the failed units -- every step is skip-guarded."
  exit 1
fi
log "ALL STEPS OK  (master log ${MASTER})"
