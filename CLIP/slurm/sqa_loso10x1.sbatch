#!/bin/bash
# =====================================================================================
# SQA -- 10-fold inter-subject LOSO x 1 seed, the GROUP-ACTION TWIN, chained in ONE job.
#
# WHAT IS NEW HERE, AND WHY IT IS THE PART THAT NEEDED TRAINING.  SQA (Symmetry-Quotient
# Alignment) has three levels; two of them already ship inside this recipe and are NOT
# retrained here --
#   * level 1 (representation / invariant anchor): the temporal factor that volume
#     conduction does not touch is already the axis the deployment's repetition cloud
#     exploits (`--reps`); the closed-form gate measured the ERP-window temporal subspace at
#     0.6247 affinity vs chance 0.02, while the spatial topographies sit AT chance 0.0794.
#   * level 3 (decision / routing): score-level fusion already exists as the `v6` multi-route
#     arm (`MultiRouteSAMCLIP`) and was measured on sub-08 (23.5 vs the single route's 25.5),
#     i.e. it did not win, so it is not re-run here on a 10-fold budget.
# The ONE level that is a TRAINING-side change -- and therefore the only one a retrain can
# test -- is level 2: quotient the spatial group `GL(C)` by manufacturing its orbit,
# `X -> (I + mixing*G) X`, so the encoder is asked to be invariant to the channel
# recombination that separates a held-out subject from the nine sources. `cross` can only
# ever teach invariance to the mixing PRESENT in the sources; this manufactures the rest.
#
# WHAT IS RUN.  `configs/sqa_loso_k20.yaml` (mixing=0.05) and its byte-identical twin
# `configs/sqa_loso_k20_noaug.yaml` (augment off), 10 folds x seed 2025 = 20 trainings, each
# evaluated with the DEPLOYED best cell (`--recovery --reps` + FGW structure ensemble
# fuse=16), the cell that holds 54.83/83.45 for the current best recipe. The arms differ in
# exactly one boolean, so the comparison is PAIRED and the fold spread cancels -- which is
# what removes the need for the 3 seeds G3 needed.
#
# WHY 0.05 AND NOT "MORE IS BETTER".  `I + mixing*G` must stay invertible or it is not a
# group action. 200 draws at mixing=0.05 give cond(M) <= 3.5 and min singular value 0.44; at
# 0.1 the condition number occasionally reaches ~500 (min singular value 0.004), where the
# augmentation stops relocating the orbit and starts destroying the trial. 0.05 also perturbs
# each channel by ~40%, the same order as the stock noise/gain defaults rather than above them.
#
# COST / RESUME.  A 50-epoch run is ~3 min on an H800 (job 641981) and an eval is seconds;
# 20 runs is ~1 GPU-hour. Every (fold, arm) is skip-guarded on its own outputs, so a
# wall-clock kill costs one unit rather than the sweep (AGENTS.md 3.5).
#
# Submit:   sbatch slurm/sqa_loso10x1.sbatch
# Smoke:    SUBJECTS=8 sbatch slurm/sqa_loso10x1.sbatch
# =====================================================================================
#SBATCH --job-name=sqa-loso10x1
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --output=/project/peilab/why/CLIP/outputs/slurm/sqa-loso10x1-%j.out
#SBATCH --error=/project/peilab/why/CLIP/outputs/slurm/sqa-loso10x1-%j.err

set -euo pipefail

ROOT="/project/peilab/why/CLIP"
PY="${ROOT}/.venv/bin/python"
mkdir -p "${ROOT}/outputs/slurm"

# --- caches (AGENTS.md 2.4): MUST be exported before any torch/HF import -------------
export PYTHONNOUSERSITE=1
export HF_HOME=/project/peilab/why/cache/huggingface
export HF_HUB_CACHE=/project/peilab/why/cache/huggingface/hub
export TRANSFORMERS_CACHE=/project/peilab/why/cache/huggingface
export OPENCLIP_CACHE_DIR=/project/peilab/why/cache/open_clip
export TORCH_HOME=/project/peilab/why/cache/torch
export XDG_CACHE_HOME=/project/peilab/why/cache/xdg
export PIP_CACHE_DIR=/project/peilab/why/cache/pip
# NODE-LOCAL, not NFS (AGENTS.md 2.4): multiprocessing's per-worker temp dirs cannot be
# rmtree'd on NFS (unlink-while-open -> visible `.nfs*` -> EBUSY), which buries real errors.
export TMPDIR="/tmp/clip-${SLURM_JOB_ID:-local}"
export XDG_CONFIG_HOME="${ROOT}/scratch/config"
export MPLCONFIGDIR="${ROOT}/scratch/mpl"
mkdir -p "${TMPDIR}" "${XDG_CONFIG_HOME}" "${MPLCONFIGDIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"

LOG_DIR="${ROOT}/outputs/slurm"
MASTER="${LOG_DIR}/sqa-loso10x1-${SLURM_JOB_ID}.log"
OUT_ROOT="${ROOT}/outputs/stage1"          # .../{sqa,sqa_noaug}/subNN_k20_seed2025
EVAL_ROOT="${ROOT}/outputs/eval"
FUSE=16
REPS_ARGS=(--reps --csls-k 10 --rho 0.1 \
           --recovery-operator fgw --recovery-tau 0.03 --recovery-alpha 0.75 \
           --rep-shrink 0.1 --fgw-struct-fuse "${FUSE}")

log() { echo "[$(date -Is)] $*" | tee -a "${MASTER}"; }

log "=== SQA group-action twin 10-fold x 1-seed job=${SLURM_JOB_ID} host=$(hostname) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
co_tenants=$(squeue -w "$(hostname)" -h -o '%i %u %j' | grep -v "${SLURM_JOB_ID}" || true)
if [[ -n "${co_tenants}" ]]; then log "shared-node co-tenants:"; echo "${co_tenants}" | tee -a "${MASTER}"; fi

# --- hard CUDA gate: fail fast rather than silently training 20 runs on CPU ----------
"${PY}" -c "import sys,torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { log "[FATAL] CUDA unavailable on $(hostname)"; exit 1; }

# Fail before the first of 20 trainings if either recipe lost its fold-free identity, and
# print the ONE key that is the experiment: the augment block. A silently-unaugmented `sqa`
# arm is indistinguishable from the twin until the numbers come back identical.
"${PY}" - "${ROOT}/configs/sqa_loso_k20.yaml" "${ROOT}/configs/sqa_loso_k20_noaug.yaml" <<'PY' || exit 1
import sys
from samclip.utils import load_config
from samclip.data.augment import unwrap
for path in sys.argv[1:]:
    cfg = load_config(path)
    a = unwrap(cfg)
    print(f"[sqa] {path}: arch={cfg['arch']} epochs={cfg['epochs']} lr={cfg['lr']} "
          f"augment={a}")
    fold_keys = {k: cfg.get(k) for k in ("target_subject", "source_subjects", "seed")}
    if any(v is not None for v in fold_keys.values()):
        sys.exit(f"[sqa] FATAL: {path} must be FOLD-FREE, got {fold_keys}")
# the two arms must differ ONLY in the augment block (paranoia against a base drift)
import copy
c_on, c_off = load_config(sys.argv[1]), load_config(sys.argv[2])
on, off = {k: v for k, v in c_on.items() if k != "augment"}, \
          {k: v for k, v in c_off.items() if k != "augment"}
if on != off:
    diff = {k for k in set(on) | set(off) if on.get(k) != off.get(k)}
    sys.exit(f"[sqa] FATAL: arms differ outside `augment`: {sorted(diff)}")
if unwrap(c_on)["mixing"] <= 0.0:
    sys.exit("[sqa] FATAL: the group arm has mixing <= 0; that is the twin, not the treatment.")
print("[sqa] OK: arms differ only in `augment`, group arm mixing = "
      f"{unwrap(c_on)['mixing']}")
PY

# --- skip guard: outputs newer than everything that could have changed them -----------
should_skip() {
  local out="$1"
  [[ -s "${out}" ]] || return 1
  local newest
  newest="$(find "${ROOT}/src" "${ROOT}/scripts" \
                 "${ROOT}/configs/sqa_loso_k20.yaml" "${ROOT}/configs/sqa_loso_k20_noaug.yaml" \
                 "${ROOT}/slurm/sqa_loso10x1.sbatch" \
                 -type f -newer "${out}" 2>/dev/null | head -1 || true)"
  [[ -z "${newest}" ]]
}

FAILED_STEPS=()
run_step() {                       # run_step <name> <logfile> <cmd...>
  local name="$1" lf="$2"; shift 2
  log "  -> ${name}"
  local rc=0
  "$@" >"${lf}" 2>&1 || rc=$?
  if [[ ${rc} -eq 0 ]]; then
    log "     OK   ${name}"
    return 0
  fi
  log "     FAILED rc=${rc}  ${name}  (log ${lf})"
  tail -n 30 "${lf}" | tee -a "${MASTER}" || true
  FAILED_STEPS+=("${name}")
  return 0
}

SUBJECTS="${SUBJECTS:-1 2 3 4 5 6 7 8 9 10}"
SEED=2025
# (arm dir, arm config); order matters only for reading the log
ARMS=("sqa:${ROOT}/configs/sqa_loso_k20.yaml" "sqa_noaug:${ROOT}/configs/sqa_loso_k20_noaug.yaml")

log "folds: ${SUBJECTS}   seed: ${SEED}   fuse: ${FUSE}   arms: sqa,sqa_noaug"

for s in ${SUBJECTS}; do
  src=()
  for o in 1 2 3 4 5 6 7 8 9 10; do [[ "${o}" -ne "${s}" ]] && src+=("${o}"); done
  tag="$(printf 'sub%02d' "${s}")"
  # The SAMGA-official reference was measured on sub-08 only; elsewhere `unknown` prints n/a
  # rather than a delta against a baseline from a different subject (run_eval REF_TOP1).
  if [[ "${s}" -eq 8 ]]; then REF_ARGS=(--ref-top1 22.00 --ref-top5 50.00)
  else                        REF_ARGS=(--ref-top1 unknown); fi

  for arm in "${ARMS[@]}"; do
    name_dir="${arm%%:*}"; cfg="${arm#*:}"
    out="${OUT_ROOT}/${name_dir}/${tag}_k20_seed${SEED}"
    ev="${EVAL_ROOT}/${name_dir}/${tag}_seed${SEED}.json"
    name="${name_dir}/${tag}"
    mkdir -p "${OUT_ROOT}/${name_dir}" "${EVAL_ROOT}/${name_dir}"
    log "--- ${name} : sources ${src[*]} ---"

    if should_skip "${out}/last.pt"; then
      log "  [SKIP] train -- ${out}/last.pt is up to date"
    else
      run_step "train ${name}" "${LOG_DIR}/sqa-train-${name_dir}-${tag}-${SLURM_JOB_ID}.log" \
        "${PY}" scripts/run_stage1.py --config "${cfg}" \
          --target-subject "${s}" --source-subjects "${src[@]}" \
          --out-dir "${out}" --seed "${SEED}" \
          --mvnn train --channel-set all63
    fi

    if [[ -s "${ev}" ]] && [[ "${ev}" -nt "${out}/last.pt" ]] && should_skip "${ev}"; then
      log "  [SKIP] eval -- ${ev} is newer than its checkpoint"
    elif [[ ! -s "${out}/last.pt" ]]; then
      log "  [SKIP] eval -- no checkpoint at ${out}/last.pt (its training failed above)"
    else
      run_step "eval ${name}" "${LOG_DIR}/sqa-eval-${name_dir}-${tag}-${SLURM_JOB_ID}.log" \
        "${PY}" scripts/run_eval.py --ckpts "${out}/last.pt" \
          --target-subject "${s}" --out "${ev}" \
          --recovery "${REPS_ARGS[@]}" "${REF_ARGS[@]}"
    fi
  done
done

# =====================================================================================
# paired summary -- reads whatever landed and says so if the grid is incomplete
# =====================================================================================
log "=== summary ==="
run_step "summarize" "${LOG_DIR}/sqa-summary-${SLURM_JOB_ID}.log" \
  "${PY}" scripts/summarize_sqa.py --glob 'outputs/eval/sqa*/*.json' \
    --out "${ROOT}/outputs/sqa_summary.json"
tail -n 60 "${LOG_DIR}/sqa-summary-${SLURM_JOB_ID}.log" | tee -a "${MASTER}" || true

if [[ ${#FAILED_STEPS[@]} -gt 0 ]]; then
  log "FINISHED WITH FAILURES (${#FAILED_STEPS[@]}): ${FAILED_STEPS[*]}"
  log "re-submit to retry only the failed units -- every step is skip-guarded."
  exit 1
fi
log "ALL STEPS OK  (master log ${MASTER})"
