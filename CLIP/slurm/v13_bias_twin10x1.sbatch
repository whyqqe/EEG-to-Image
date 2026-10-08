#!/bin/bash
# =====================================================================================
# v13 / CROMA -- the order-2 BIAS-ANCHOR twin on the full 10-fold LOSO protocol.
#
# THE EXPERIMENT. Two arms that differ in EXACTLY ONE key:
#
#   ON   configs/v13_bias_loso_k20.yaml       bias_anchor.enabled: true,  weight 2.0
#   OFF  configs/v13_bias_off_loso_k20.yaml   bias_anchor.enabled: false, weight 0.0
#
# Both arms keep `concept.enabled: true, reps_weight: 0.0`, so the DATA PATH -- sampler, augmentation
# stream, per-step encoder passes -- is identical and the only difference is the loss term.
# `summarize_v13.py` ASSERTS this by loading both configs and diffing them; a pair that differs in
# two places could not attribute anything.
#
# WHY THIS TERM AND NOT OTHERS (measurement, job 650135 -> docs/eeg2image_v13_croma.md §3):
#   * variance is EXHAUSTED -- repetition-error rho <= 0 on 10/10 folds, K_eff = R = 80, so pooling
#     already removes every independent component. No variance term is added, because one would be
#     decoration whose loss curve cannot fall;
#   * bias is LIVE and SUBJECT-SPECIFIC -- a 9-subject pooled EEG metric (9 independent ENCODINGS,
#     ~720 reps) reaches 0.796 against the gallery while the target's own 80 reps reach 0.616:
#     +0.180, paired t = +22.2, 10/10 folds, its own shuffle control at -0.0005. The ceiling
#     hypothesis is dead; the subject bias is the binding constraint.
#
# THE READOUT IS PAIRED AND PRE-REGISTERED. Headline = the DEPLOYED cell
# `+ T2 R=80,a=0.75,t=0.03,fuse=16` (the cell holding 54.83/83.45 for the banked recipe), plus the
# bias-curve probe on both arms, so the mechanism is read independently of the accuracy:
#     PASS <=> d(corr(D_R80, D_g)) > 0  AND  d(corr(D_R1, D_g)) >= d(corr(R80)) > 0.
# The second clause is the discriminator: a lift that lives only at R=1 means the encoder got less
# NOISY, and pooling already removes noise -- so the deployment cell would not move.
#
# GUARDS (a "win" failing any of these is booked as a collapse, not a success):
#   * effrank(D_eeg) must not fall (v12 collapsed to rank 2-3, -13.85pp raw cosine);
#   * the concept-shuffle control must stay dead in both arms (else it is M1 leakage);
#   * single-variable config diff.
#
# COST. 20 trainings x ~3 min + 20 evals + 20 cheap probes. A 50-epoch stage-1 run is ~3 min on an
# H800 (job 641981); the first eval of each fold also builds the target's MVNN-test whitener and
# repetition cloud (~1 GB) lazily and caches them, so seeds/folds after the first reuse them.
#
# Submit:  sbatch slurm/v13_bias_twin10x1.sbatch
# Smoke:   SUBJECTS=8 SEEDS=2025 sbatch slurm/v13_bias_twin10x1.sbatch
# =====================================================================================
#SBATCH --job-name=v13-bias-twin
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --output=/project/peilab/why/CLIP/outputs/slurm/v13-bias-twin-%j.out
#SBATCH --error=/project/peilab/why/CLIP/outputs/slurm/v13-bias-twin-%j.err

set -euo pipefail

ROOT="/project/peilab/why/CLIP"
PY="${ROOT}/.venv/bin/python"
LOG_DIR="${ROOT}/outputs/slurm"
mkdir -p "${LOG_DIR}"

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

MASTER="${LOG_DIR}/v13-bias-twin-${SLURM_JOB_ID}.log"
log() { echo "[$(date -Is)] $*" | tee -a "${MASTER}"; }

log "=== v13 bias-anchor twin job=${SLURM_JOB_ID} host=$(hostname) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
co_tenants=$(squeue -w "$(hostname)" -h -o '%i %u %j' | grep -v "${SLURM_JOB_ID}" || true)
if [[ -n "${co_tenants}" ]]; then log "shared-node co-tenants:"; echo "${co_tenants}" | tee -a "${MASTER}"; fi

# hard CUDA gate: fail fast rather than silently training 20 runs on CPU
"${PY}" -c "import sys,torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { log "[FATAL] CUDA unavailable on $(hostname)"; exit 1; }

# Fail BEFORE the first training if the pair is not single-variable, or if `concept.enabled` is off
# (the loader would then never pass `batch["reps"]` and the ON arm would silently be the OFF arm).
"${PY}" - "${ROOT}/configs/v13_bias_loso_k20.yaml" "${ROOT}/configs/v13_bias_off_loso_k20.yaml" <<'PY' || exit 1
import sys
from samclip.utils import load_config

def flat(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from flat(v, f"{p}.{k}" if p else str(k))
    else:
        yield p, o

on, off = (dict(flat(load_config(p))) for p in sys.argv[1:3])
keys = sorted(set(on) | set(off))
diffs = {k: (on.get(k), off.get(k)) for k in keys if on.get(k) != off.get(k)}
print(f"[v13] config diff ({len(diffs)} key(s)): {diffs}")
if len(diffs) != 1 or "bias_anchor.enabled" not in diffs:
    sys.exit(f"[v13] FATAL: the arms must differ only in `bias_anchor.enabled`, got {sorted(diffs)}")
if not on.get("concept.enabled"):
    sys.exit("[v13] FATAL: the ON arm needs `concept.enabled: true` or `batch['reps']` is never "
             "passed and the term would silently never fire.")
if float(on.get("concept.reps_weight", 0.0)) != 0.0:
    sys.exit("[v13] FATAL: `concept.reps_weight` must be 0 in both arms -- a nonzero value would "
             "add the (separately falsified) T2' term to BOTH arms and confound the pair.")
for p in sys.argv[1:3]:
    c = load_config(p)
    fold = {k: c.get(k) for k in ("target_subject", "source_subjects", "seed")}
    if any(v is not None for v in fold.values()):
        sys.exit(f"[v13] FATAL: {p} must be FOLD-FREE, got {fold}")
    print(f"[v13] recipe {p}: epochs={c['epochs']} lr={c['lr']} "
          f"soft_plan={ (c.get('soft_plan') or {}).get('enabled') } "
          f"bias_anchor={c.get('bias_anchor')}")
PY

should_skip() {                       # should_skip <artifact> [extra deps...]
  local out="$1"; shift
  [[ -s "${out}" ]] || return 1
  local newest
  newest="$(find "${ROOT}/src" "${ROOT}/scripts" "$@" -type f -newer "${out}" 2>/dev/null | head -1 || true)"
  [[ -z "${newest}" ]]
}

FAILED_STEPS=()
run_step() {                          # run_step <name> <logfile> <cmd...>
  local name="$1" lf="$2"; shift 2
  log "  -> ${name}"
  local rc=0
  # `|| rc=$?`, NOT `if "$@"; then`: an `if` whose condition is false returns 0, so reading `$?`
  # after `fi` reports success for a failed step -- the silent pass this pipeline must not have.
  "$@" >"${lf}" 2>&1 || rc=$?
  if [[ ${rc} -eq 0 ]]; then log "     OK   ${name}"; return 0; fi
  log "     FAILED rc=${rc}  ${name}  (log ${lf})"
  tail -n 30 "${lf}" | tee -a "${MASTER}" || true
  FAILED_STEPS+=("${name}")
  return 0                            # keep chaining; the final rc reflects the failures
}

SUBJECTS="${SUBJECTS:-1 2 3 4 5 6 7 8 9 10}"
SEEDS="${SEEDS:-2025}"

# arm name -> (config, stage1 out root, eval root, probe root)
declare -A CFG OF OUT ED PR
CFG[on]="${ROOT}/configs/v13_bias_loso_k20.yaml"
CFG[off]="${ROOT}/configs/v13_bias_off_loso_k20.yaml"
OUT[on]="${ROOT}/outputs/stage1/v13_bias"
OUT[off]="${ROOT}/outputs/stage1/v13_bias_off"
ED[on]="${ROOT}/outputs/eval/v13_bias"
ED[off]="${ROOT}/outputs/eval/v13_bias_off"
PR[on]="${ROOT}/outputs/probe/v13_bias"
PR[off]="${ROOT}/outputs/probe/v13_bias_off"
for a in on off; do mkdir -p "${OUT[$a]}" "${ED[$a]}" "${PR[$a]}"; done

log "folds: ${SUBJECTS}   seeds: ${SEEDS}   arms: on off"

for s in ${SUBJECTS}; do
  src=()
  for o in 1 2 3 4 5 6 7 8 9 10; do [[ "${o}" -ne "${s}" ]] && src+=("${o}"); done
  tag="$(printf 'sub%02d' "${s}")"
  if [[ "${s}" -eq 8 ]]; then REF_ARGS=(--ref-top1 22.00 --ref-top5 50.00)
  else                        REF_ARGS=(--ref-top1 unknown); fi

  for seed in ${SEEDS}; do
    for a in on off; do
      out="${OUT[$a]}/${tag}_k20_seed${seed}"
      ev="${ED[$a]}/${tag}_seed${seed}.json"
      name="${a}/${tag}/seed${seed}"
      log "--- ${name} : sources ${src[*]} ---"

      if should_skip "${out}/last.pt" "${CFG[$a]}"; then
        log "  [SKIP] train -- up to date"
      else
        run_step "train ${name}" "${LOG_DIR}/v13-train-${a}-${tag}-seed${seed}-${SLURM_JOB_ID}.log" \
          "${PY}" scripts/run_stage1.py --config "${CFG[$a]}" \
            --target-subject "${s}" --source-subjects "${src[@]}" \
            --out-dir "${out}" --seed "${seed}" \
            --mvnn train --channel-set all63
      fi

      if [[ -s "${ev}" ]] && [[ "${ev}" -nt "${out}/last.pt" ]] \
         && should_skip "${ev}" "${CFG[$a]}"; then
        log "  [SKIP] eval -- newer than its checkpoint"
      elif [[ ! -s "${out}/last.pt" ]]; then
        log "  [SKIP] eval -- no checkpoint (training failed above)"
      else
        # THE DEPLOYED CELL, identical for both arms: whiten+CSLS+recovery cloud, FGW structure
        # ensemble fuse=0,16, satc calibration. Emitting the paired fuse=0 row too keeps the
        # fusion contribution readable in the same file.
        run_step "eval ${name}" "${LOG_DIR}/v13-eval-${a}-${tag}-seed${seed}-${SLURM_JOB_ID}.log" \
          "${PY}" scripts/run_eval.py --ckpts "${out}/last.pt" \
            --target-subject "${s}" --out "${ev}" \
            --recovery --reps --csls-k 10 --rho 0.1 \
            --recovery-operator fgw --recovery-tau 0.03 --recovery-alpha 0.75 \
            --rep-shrink 0.1 --fgw-struct-fuse 0,16 \
            "${REF_ARGS[@]}"
      fi

      # THE MECHANISM READOUT. The probe is ~15 s/fold and needs no labels: it re-embeds the
      # target's repetition cloud and measures corr(D_R', D_g) and the effective rank, so the
      # discriminating prediction (does the lever reach R=80, or only R=1?) and the collapse guard
      # are read independently of any retrieval number.
      if should_skip "${PR[$a]}/${tag}_seed${seed}.json" "${CFG[$a]}"; then
        log "  [SKIP] probe -- up to date"
      elif [[ ! -s "${out}/last.pt" ]]; then
        log "  [SKIP] probe -- no checkpoint"
      else
        run_step "probe ${name}" "${LOG_DIR}/v13-probe-${a}-${tag}-seed${seed}-${SLURM_JOB_ID}.log" \
          "${PY}" scripts/probe_view_independence.py \
            --stage1-root "${OUT[$a]}" --tag-fmt "${tag}_k20_seed${seed}" \
            --folds "${s}" --seed "${seed}" --fuse 16 --rep-shrink 0.1 --n-split 20 \
            --src-metric-root "${ROOT}/outputs/src_metric/v8" \
            --outdir "${PR[$a]}"
      fi
    done
  done
done

# =====================================================================================
# the signed verdict, with the pre-registered criterion and the guards
# =====================================================================================
log "=== summary (paired ON vs OFF) ==="
run_step "summarize_v13" "${LOG_DIR}/v13-summary-${SLURM_JOB_ID}.log" \
  "${PY}" scripts/summarize_v13.py \
    --on-eval "${ED[on]}" --off-eval "${ED[off]}" \
    --on-probe "${PR[on]}" --off-probe "${PR[off]}" \
    --on-cfg "${CFG[on]}" --off-cfg "${CFG[off]}" \
    --out "${ROOT}/outputs/v13_twin_summary.json"
tail -n 40 "${LOG_DIR}/v13-summary-${SLURM_JOB_ID}.log" | tee -a "${MASTER}" || true

if [[ ${#FAILED_STEPS[@]} -gt 0 ]]; then
  log "FINISHED WITH FAILURES (${#FAILED_STEPS[@]}): ${FAILED_STEPS[*]}"
  log "re-submit to retry only the failed units -- every step is skip-guarded."
  exit 1
fi
log "ALL STEPS OK  (master log ${MASTER})"
