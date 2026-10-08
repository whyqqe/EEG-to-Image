#!/bin/bash
# =====================================================================================
# v12 TWIN -- the decisive experiment of the locked claim (docs/eeg2image_v12_core_claim.md).
#
# THREE ARMS, PAIRED, ONE CHAIN. Every arm trains the SAME recipe (configs/v12_loso_k20.yaml,
# which is banked v8 + the v12 metric terms) with the SAME fold, seed and data order, and
# differs ONLY in the v12 weights:
#
#   base   --v12-distill 0   --v12-consistency 0   --reps-weight 0     (v8 + reps loaded, inert)
#   v12    --v12-distill 1.0 --v12-consistency 0.5 --reps-weight 0     (the treatment)
#   v12d   --v12-distill 1.0 --v12-consistency 0.5 --reps-weight 0.1   (treatment + T2' stimulus)
#
# WHY `base` IS TRAINED AND NOT REUSED FROM outputs/stage1/v8. Its recipe is bit-identical to v8
# (`concept.enabled` only loads `batch['reps']`; `reps_weight: 0` skips the T2' branch), but "should
# be bit-identical" is exactly the class of claim this project keeps having to MEASURE. Training it
# in the same chain makes the twin paired by construction instead of by an argument.
#
# The `base vs v12` difference is Part C of the claim made trainable: the teacher is the v11
# reliability-fused consensus metric (job 645697, permutation control ~0.000) and the student is a
# SINGLE-repetition metric. The R-curve (job 645719) is why this and not another contrastive term:
# Top-1 runs 4.0 -> 54.5 from R=1 to R=80 and is STILL climbing +3.33pp in the last doubling, so
# the query metric is estimation-limited at the deployed R.
#
# `v12d vs v12` isolates the documented T2' grouping fix (`row` is the measured anti-T1 arm,
# sub-08 42.0 -> 30.0); it is a third arm rather than a shared setting so that it cannot confound
# the headline twin.
#
# COST. A 50-epoch stage-1 run is ~3 min on an H800 (v7 note); the v12 terms add ~0.75x per-step
# encoder passes (288 extra rows x 4 reps vs a 384-row batch; the TRAIN cloud is only R=4 deep,
# the test cloud R=80). 30 runs, chained, skip-guarded.
#
# Smoke (one fold, one seed, ~10 min): SUBJECTS=8 SEEDS=2025 ARMS="base v12" sbatch ...
# =====================================================================================
#SBATCH --job-name=v12-twin
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --output=/project/peilab/why/CLIP/outputs/slurm/v12-twin-%j.out
#SBATCH --error=/project/peilab/why/CLIP/outputs/slurm/v12-twin-%j.err

set -euo pipefail

ROOT="/project/peilab/why/CLIP"
PY="${ROOT}/.venv/bin/python"
CFG="${CFG:-${ROOT}/configs/v12_loso_k20.yaml}"
ARMS="${ARMS:-base v12 v12d}"
SUBJECTS="${SUBJECTS:-1 2 3 4 5 6 7 8 9 10}"
SEEDS="${SEEDS:-2025}"
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
# NODE-LOCAL, not NFS (see AGENTS.md §2.4 for the EBUSY traceback this prevents).
export TMPDIR="/tmp/clip-${SLURM_JOB_ID:-local}"
export XDG_CONFIG_HOME="${ROOT}/scratch/config"
export MPLCONFIGDIR="${ROOT}/scratch/mpl"
mkdir -p "${TMPDIR}" "${XDG_CONFIG_HOME}" "${MPLCONFIGDIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"

LOG_DIR="${ROOT}/outputs/slurm"
MASTER="${LOG_DIR}/v12-twin-${SLURM_JOB_ID}.log"
log() { echo "[$(date -Is)] $*" | tee -a "${MASTER}"; }

log "=== v12 twin job=${SLURM_JOB_ID} host=$(hostname) cfg=${CFG} arms='${ARMS}' ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

# hard CUDA gate: fail fast rather than silently training on CPU for hours
"${PY}" -c "import sys,torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { log "[FATAL] CUDA unavailable on $(hostname)"; exit 1; }

# --- preflight: the recipe must carry the v12 mechanism, and must be FOLD-FREE ---------
"${PY}" - "${CFG}" <<'PY' || exit 1
import sys
from samclip.utils import load_config
cfg = load_config(sys.argv[1])
print(f"[v12] recipe {sys.argv[1]}: arch={cfg['arch']} objective={cfg['objective']} "
      f"epochs={cfg['epochs']} lr={cfg['lr']} soft_plan={(cfg.get('soft_plan') or {}).get('enabled')}")
fold = {k: cfg.get(k) for k in ("target_subject", "source_subjects", "seed")}
if any(v is not None for v in fold.values()):
    sys.exit(f"[v12] FATAL: the recipe must be FOLD-FREE, got {fold}.")
v12 = cfg.get("v12") or {}
conc = cfg.get("concept") or {}
if not conc.get("enabled"):
    sys.exit("[v12] FATAL: `concept.enabled` must be true -- it is what loads `batch['reps']`, "
             "which the v12 metric terms consume. Without it the treatment arm would be a "
             "silent no-op reported as a mechanism.")
if conc.get("reps_group") != "stimulus":
    sys.exit(f"[v12] FATAL: `concept.reps_group` must be 'stimulus', got "
             f"{conc.get('reps_group')!r} (`row` is the rejected anti-T1 arm).")
if int(v12.get("r_use", 0)) < 3:
    sys.exit(f"[v12] FATAL: `v12.r_use` {v12.get('r_use')!r} < 3 -- the teacher would have no "
             "independent view and the distillation would be a no-op.")
print(f"[v12] base recipe OK: reps_group={conc.get('reps_group')} r_use={v12.get('r_use')} "
      f"student_reps={v12.get('student_reps')} teacher_blocks={v12.get('teacher_blocks')} "
      f"n_stim={v12.get('n_stim')} subj_sel={v12.get('subj_sel')}")
PY

# --- skip guard: an output is fresh iff nothing that could change it is newer ----------
should_skip() {
  local out="$1"
  [[ -s "${out}" ]] || return 1
  local newest
  newest="$(find "${ROOT}/src" "${ROOT}/scripts" "${CFG}" "${BASH_SOURCE[0]}" \
                 -type f -newer "${out}" 2>/dev/null | head -1 || true)"
  [[ -z "${newest}" ]]
}

FAILED_STEPS=()
run_step() {                       # run_step <name> <logfile> <cmd...>
  local name="$1" lf="$2"; shift 2
  log "  -> ${name}"
  local rc=0
  "$@" >"${lf}" 2>&1 || rc=$?
  if [[ ${rc} -eq 0 ]]; then log "     OK   ${name}"; return 0; fi
  log "     FAILED rc=${rc}  ${name}  (log ${lf})"
  tail -n 30 "${lf}" | tee -a "${MASTER}" || true
  FAILED_STEPS+=("${name}")
  return 0
}

# Per-arm flags. `base` is the PAIRED TWIN and MUST keep both v12 weights at 0 -- if this
# mapping drifts, the headline comparison silently becomes treatment-vs-treatment.
arm_flags() {
  case "$1" in
    base) echo "--v12-distill 0 --v12-consistency 0 --reps-weight 0" ;;
    v12)  echo "--v12-distill 1.0 --v12-consistency 0.5 --reps-weight 0" ;;
    v12d) echo "--v12-distill 1.0 --v12-consistency 0.5 --reps-weight 0.1" ;;
    *) echo ""; return 1 ;;
  esac
}

for arm in ${ARMS}; do
  FLAGS="$(arm_flags "${arm}")" || { log "[FATAL] unknown arm '${arm}'"; exit 1; }
  OUT_ROOT="${ROOT}/outputs/stage1/v12/${arm}"
  EVAL_ROOT="${ROOT}/outputs/eval/v12/${arm}"
  mkdir -p "${OUT_ROOT}" "${EVAL_ROOT}"
  for s in ${SUBJECTS}; do
    src=()
    for o in 1 2 3 4 5 6 7 8 9 10; do [[ "${o}" -ne "${s}" ]] && src+=("${o}"); done
    tag="$(printf 'sub%02d' "${s}")"
    # The SAMGA-official reference was measured on sub-08 ONLY; elsewhere `unknown` prints n/a
    # rather than a delta against a baseline from a different subject (see run_eval REF_TOP1).
    if [[ "${s}" -eq 8 ]]; then REF_ARGS=(--ref-top1 22.00 --ref-top5 50.00)
    else                        REF_ARGS=(--ref-top1 unknown); fi
    for seed in ${SEEDS}; do
      out="${OUT_ROOT}/${tag}_k20_seed${seed}"
      ev="${EVAL_ROOT}/${tag}_seed${seed}.json"
      name="${arm}/${tag}/seed${seed}"
      log "--- ${name} : sources ${src[*]} ---"

      if should_skip "${out}/last.pt"; then
        log "  [SKIP] train -- ${out}/last.pt is up to date"
      else
        run_step "train ${name}" "${LOG_DIR}/v12-train-${arm}-${tag}-seed${seed}-${SLURM_JOB_ID}.log" \
          ${PY} scripts/run_stage1.py --config "${CFG}" \
            --target-subject "${s}" --source-subjects "${src[@]}" \
            --out-dir "${out}" --seed "${seed}" \
            --mvnn train --channel-set all63 ${FLAGS}
      fi

      if [[ ! -s "${out}/last.pt" ]]; then
        log "  [SKIP] eval -- no checkpoint at ${out}/last.pt (its training failed above)"
        continue
      fi
      if [[ -s "${ev}" ]] && [[ "${ev}" -nt "${out}/last.pt" ]] && should_skip "${ev}"; then
        log "  [SKIP] eval -- ${ev} is newer than its checkpoint"
      else
        # The HEADLINE operator: the deployed FGW recovery at tau 0.03 / alpha 0.75, rep-shrink
        # 0.1 -- the exact convention the banked 54.25 (v11) and the R-curve (v12 claim Part C)
        # are computed under, so the v12 arms are comparable to both.
        run_step "eval-fgw ${name}" \
          "${LOG_DIR}/v12-evalfgw-${arm}-${tag}-seed${seed}-${SLURM_JOB_ID}.log" \
          "${PY}" scripts/run_eval.py --ckpts "${out}/last.pt" \
            --target-subject "${s}" --out "${ev}" \
            --recovery --reps --csls-k 10 --rho 0.1 \
            --recovery-operator fgw --recovery-tau 0.03 --recovery-alpha 0.75 \
            --rep-shrink 0.1 "${REF_ARGS[@]}"
      fi

      # Second readout under the HARD operator, matching the banked v8 eval convention
      # (`+ T2 reps` = 47.20 over 10 folds). Cheap, and it separates "the metric got better" from
      # "the metric got better under the deployed operator".
      evh="${EVAL_ROOT}/${tag}_hard${seed}.json"
      if [[ -s "${evh}" ]] && [[ "${evh}" -nt "${out}/last.pt" ]] && should_skip "${evh}"; then
        log "  [SKIP] eval-hard -- ${evh} is newer than its checkpoint"
      else
        run_step "eval-hard ${name}" \
          "${LOG_DIR}/v12-evalhard-${arm}-${tag}-seed${seed}-${SLURM_JOB_ID}.log" \
          "${PY}" scripts/run_eval.py --ckpts "${out}/last.pt" \
            --target-subject "${s}" --out "${evh}" \
            --recovery --reps --csls-k 10 --rho 0.1 "${REF_ARGS[@]}"
      fi
    done
  done
done

log "=== summarize ==="
for arm in ${ARMS}; do
  run_step "summary ${arm}" "${LOG_DIR}/v12-summary-${arm}-${SLURM_JOB_ID}.log" \
    "${PY}" scripts/summarize_g3.py --glob "outputs/eval/v12/${arm}/sub*_seed*.json" \
      --out "${ROOT}/outputs/v12_${arm}_summary.json" || true
  tail -n 40 "${LOG_DIR}/v12-summary-${arm}-${SLURM_JOB_ID}.log" 2>/dev/null | tee -a "${MASTER}" || true
done

if [[ ${#FAILED_STEPS[@]} -gt 0 ]]; then
  log "FINISHED WITH FAILURES (${#FAILED_STEPS[@]}): ${FAILED_STEPS[*]}"
  log "re-submit to retry only the failed units -- every step is skip-guarded."
  exit 1
fi
log "ALL STEPS OK  (master log ${MASTER})"
