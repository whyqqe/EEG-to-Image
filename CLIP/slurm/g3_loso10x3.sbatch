#!/bin/bash
# =====================================================================================
# G3 -- 10-fold inter-subject LOSO x 3 seeds, chained into ONE job.
#
# WHY THIS JOB EXISTS. Every performance number this project holds is a sub-08 number
# (raw 34.00 -> the `+ CSLS + recovery` rung 35.50/73.50, and now 41.50/78.00 with the
# repetition cloud). The published cells we are trying to compare against -- 26.22/57.98
# for a SAMGA encoder and 53.23/83.55 for SCORE -- are TEN-FOLD MEANS. We cannot tell
# whether sub-08 is an easy or a hard fold, and the last round of analysis produced direct
# evidence that it is a HARD one: two structurally unrelated encoders (SAMGA's official
# 2.63M TSConv and our 1.13M ATM-S) both land on exactly 35.50/46.7%-landmark under this
# deployment ladder, i.e. the fold's mutual-NN landmark accuracy (~47%) caps what any
# coordinate recovery can do on it. So a sub-08 comparison is not a weak version of a
# 10-fold comparison, it is a different question. This job measures the right one.
#
# WHAT IS RUN. The best-validated recipe (configs/g3_loso_k20.yaml = `v5-a1-k20`), 30
# times: 10 folds x seeds {2025, 2026, 2027}, final-epoch checkpoints, evaluated with the
# full deployment ladder plus the T2 repetition cloud. 30 runs is what makes the MEAN
# comparable; 3 seeds is what makes the fold-level spread separable from seed noise.
#
# WHY CHAINED AND NOT 30 JOBS. Same reason as v6: the cluster is shared, 30 allocations
# each re-reading the same image-target stacks is a queue hog, and a chained job makes the
# SEQUENCE the unit of record. Each (fold, seed) is skip-guarded on its own outputs, so a
# wall-clock kill costs one unit rather than the sweep (AGENTS.md §3.5).
#
# COST. Measured: a 50-epoch stage-1 run is ~3 min on an H800 (job 641981, 12:10:41 ->
# 12:13:41) and one eval is ~10 s on CPU (the retrieval set is 200-way). The only
# non-obvious cost is that the first eval OF EACH FOLD builds two caches on demand: the
# held-out subject's MVNN-test whitener (7 of 10 are not cached) and its repetition cloud
# (`test_subNN_all63_reps_mvnntest_std.npy`, ~1 GB, `load_test_reps`). Those are built
# lazily and cached, so seeds 2 and 3 of the same fold reuse them. Deliberately NOT a
# step 0: the lazy path has the identical total cost, and a pre-build step would be one
# more moving part in the chain for no time saved.
#
# THE FOLD IDENTITY IS PASSED EXPLICITLY, EVERY TIME. `--source-subjects` is computed here
# from the fold and handed to every run. It is not left to the config: `configs/g3_loso_k20.yaml`
# deliberately nulls `target_subject`/`source_subjects`, because the chain it would
# otherwise inherit ends in `v4.yaml`, which carries `target_subject: 8`. A forgotten
# override would then train fold 3 on a source list containing sub-03 -- `load_loso` raises
# on that ("that is not a holdout") rather than leaking, but the point is that the fold
# should be auditable from the artefact, not from remembering what was on the command line.
#
# Submit:   sbatch slurm/g3_loso10x3.sbatch
# Smoke:    SUBJECTS=8 SEEDS=2025 sbatch ...   (reuses the banked fold-8 checkpoint)
# =====================================================================================
#SBATCH --job-name=g3-loso10x3
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --output=/project/peilab/why/CLIP/outputs/slurm/g3-loso10x3-%j.out
#SBATCH --error=/project/peilab/why/CLIP/outputs/slurm/g3-loso10x3-%j.err

set -euo pipefail

ROOT="/project/peilab/why/CLIP"
PY="${ROOT}/.venv/bin/python"
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
MASTER="${LOG_DIR}/g3-loso10x3-${SLURM_JOB_ID}.log"
CFG="${ROOT}/configs/g3_loso_k20.yaml"
OUT_ROOT="${ROOT}/outputs/stage1/g3"
EVAL_ROOT="${ROOT}/outputs/eval/g3"
mkdir -p "${OUT_ROOT}" "${EVAL_ROOT}"

log() { echo "[$(date -Is)] $*" | tee -a "${MASTER}"; }

log "=== G3 10-fold x 3-seed LOSO job=${SLURM_JOB_ID} host=$(hostname) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
co_tenants=$(squeue -w "$(hostname)" -h -o '%i %u %j' | grep -v "${SLURM_JOB_ID}" || true)
if [[ -n "${co_tenants}" ]]; then log "shared-node co-tenants:"; echo "${co_tenants}" | tee -a "${MASTER}"; fi

# --- hard CUDA gate: fail fast rather than silently training 30 runs on CPU ----------
"${PY}" -c "import sys,torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { log "[FATAL] CUDA unavailable on $(hostname)"; exit 1; }

# Fail before the first of 30 trainings if the recipe has no fold identity -- a null
# `target_subject` would otherwise surface as a TypeError from an f-string three frames in.
"${PY}" - "${CFG}" <<'PY' || exit 1
import json, sys
from samclip.utils import load_config
cfg = load_config(sys.argv[1])
print(f"[g3] recipe {sys.argv[1]}: arch={cfg['arch']} objective={cfg['objective']} "
      f"epochs={cfg['epochs']} lr={cfg['lr']} schedule={cfg['schedule']['coarse_to_fine']} "
      f"recovery_aware={cfg['recovery_aware']}")
fold_keys = {k: cfg.get(k) for k in ("target_subject", "source_subjects", "seed")}
if any(v is not None for v in fold_keys.values()):
    sys.exit(f"[g3] FATAL: the recipe must be FOLD-FREE, got {fold_keys}. A fold baked "
             f"into the config is a fold that can be inherited by the wrong run.")
PY

# --- skip guard: outputs newer than everything that could have changed them -----------
should_skip() {
  local out="$1"
  [[ -s "${out}" ]] || return 1
  local newest
  newest="$(find "${ROOT}/src" "${ROOT}/scripts" "${CFG}" \
                 "${ROOT}/slurm/g3_loso10x3.sbatch" \
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
      run_step "train ${name}" "${LOG_DIR}/g3-train-${tag}-seed${seed}-${SLURM_JOB_ID}.log" \
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
      run_step "eval ${name}" "${LOG_DIR}/g3-eval-${tag}-seed${seed}-${SLURM_JOB_ID}.log" \
        "${PY}" scripts/run_eval.py --ckpts "${out}/last.pt" \
          --target-subject "${s}" --out "${ev}" \
          --recovery --reps --csls-k 10 --rho 0.1 "${REF_ARGS[@]}"
    fi
  done
done

# =====================================================================================
# summary -- reads whatever landed and says so if the grid is incomplete
# =====================================================================================
log "=== summary ==="
run_step "summarize" "${LOG_DIR}/g3-summary-${SLURM_JOB_ID}.log" \
  "${PY}" scripts/summarize_g3.py --glob 'outputs/eval/g3/sub*_seed*.json' \
    --out "${ROOT}/outputs/g3_summary.json"
tail -n 60 "${LOG_DIR}/g3-summary-${SLURM_JOB_ID}.log" | tee -a "${MASTER}" || true

if [[ ${#FAILED_STEPS[@]} -gt 0 ]]; then
  log "FINISHED WITH FAILURES (${#FAILED_STEPS[@]}): ${FAILED_STEPS[*]}"
  log "re-submit to retry only the failed units -- every step is skip-guarded."
  exit 1
fi
log "ALL STEPS OK  (master log ${MASTER})"
