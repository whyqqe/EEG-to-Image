#!/bin/bash
# =====================================================================================
# v14 / O2-MVE -- the frame fix on the training side, plus the multi-view estimator on the
# deployment side. Full train -> eval -> probe on the 10-fold LOSO protocol.
#
# THE EXPERIMENT. Three arms that form TWO single-variable pairs:
#
#   off    configs/v14_anchor_off.yaml    bias_anchor.enabled: false          (baseline)
#   raw    configs/v14_anchor_raw.yaml    bias_anchor.enabled: true, frame: raw
#   whit   configs/v14_anchor_whit.yaml   bias_anchor.enabled: true, frame: whitened
#
#   whit vs off  differ in EXACTLY `bias_anchor.enabled`  -> the term's effect
#   whit vs raw  differ in EXACTLY `bias_anchor.frame`     -> v14's actual hypothesis
#
# The gate below asserts both diffs BEFORE the first training, and `summarize_v13.py` re-asserts
# them from the configs after the runs. The `raw` arm exists because v13 was early-stopped at n=2
# (job 650254): quoting those two folds against a 10-fold mean would not be a paired test and a
# fold-selection artefact could read as a frame effect.
#
# WHY `frame` IS THE HYPOTHESIS (docs/eeg2image_v13_croma.md §6). v13's gain survived only in the raw
# encoder frame -- dR80 +0.025..+0.041 raw vs +0.001..+0.011 whitened (the DEPLOYED frame) -- while
# costing 9-13pp of headline Top-1. A structural gain cannot vanish under a change of basis; one that
# does is ANISOTROPY, i.e. gradient spent on the direction the deployment whitener deletes. The loss
# was scored in one frame and the operator reads another. `whit` scores it in the operator's frame,
# which makes `D_student` invariant to per-subject invertible linear maps (smoke §22: 2e-6 vs 4.8e-3,
# ~2400x) -- the exact nuisance class that was absorbing v13's gradient.
#
# THE DEPLOYMENT SIDE (eval only, no training, rides on the same checkpoints). `--mve-partitions`
# pools SEVERAL partition families of the repetition cloud as independent metric views into ONE
# structural reference, reliability-weighted. This is licensed by this project's own measurement that
# rho <= 0 on 10/10 folds: K_eff = K/(1+(K-1)rho) = K, so each added view cuts the estimator variance
# and does NOT saturate. `cont` alone is the shipped `fuse=B` object; `stride` (repetitions b::B)
# spreads slow equipment drift across every block instead of giving each block its own offset, which
# makes it a genuinely different view rather than a relabelling; `rand` is the seeded control.
# `--mve-fuse` emits the cont-ONLY cell, which must reproduce the shipped `fuse=16` -- if it does not,
# the pipeline is broken and nothing from this job may be read (see the kill switch below).
#
# GUARDS. A "win" failing any of these is booked as a collapse, not a success:
#   * effrank(D_eeg) must not fall (v12 collapsed to 2-3 and lost 13.85pp raw cosine);
#   * the concept-shuffle control must stay dead in every arm (else it is M1 leakage);
#   * single-variable config diffs (asserted, not assumed);
#   * the MVE structural-off twin must stay BIT-FLAT (the reference can only reach the output
#     through the structural term, so a moving twin is a leak);
#   * `cont-only` must reproduce the shipped cell (the kill switch).
#
# COST. 30 trainings x ~7 min + 60 evals (base + MVE) + 30 cheap probes. The eval split is
# DELIBERATE: the proven base ladder and the NEW MVE ladder write to SEPARATE files, so a bug in the
# new estimator cannot destroy the headline numbers -- the failure modes are independent.
#
# Submit:  sbatch slurm/v14_anchor10x1.sbatch
# Smoke:   SUBJECTS=8 SEEDS=2025 sbatch slurm/v14_anchor10x1.sbatch
# =====================================================================================
#SBATCH --job-name=v14-anchor
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --output=/project/peilab/why/CLIP/outputs/slurm/v14-anchor-%j.out
#SBATCH --error=/project/peilab/why/CLIP/outputs/slurm/v14-anchor-%j.err

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
# (unlink-while-open becomes a visible `.nfs*` -> EBUSY), which buried jobs under thousands of
# false tracebacks (AGENTS.md §2.4).
export TMPDIR="/tmp/clip-${SLURM_JOB_ID:-local}"
export XDG_CONFIG_HOME="${ROOT}/scratch/config"
export MPLCONFIGDIR="${ROOT}/scratch/mpl"
mkdir -p "${TMPDIR}" "${XDG_CONFIG_HOME}" "${MPLCONFIGDIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"

MASTER="${LOG_DIR}/v14-anchor-${SLURM_JOB_ID}.log"
log() { echo "[$(date -Is)] $*" | tee -a "${MASTER}"; }

log "=== v14 anchor job=${SLURM_JOB_ID} host=$(hostname) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
co_tenants=$(squeue -w "$(hostname)" -h -o '%i %u %j' | grep -v "${SLURM_JOB_ID}" || true)
if [[ -n "${co_tenants}" ]]; then log "shared-node co-tenants:"; echo "${co_tenants}" | tee -a "${MASTER}"; fi

# hard CUDA gate: fail fast rather than silently training 30 runs on CPU
"${PY}" -c "import sys,torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { log "[FATAL] CUDA unavailable on $(hostname)"; exit 1; }

# ---- PREFLIGHT: every declared pair single-variable, and the frame actually reached the loss ----
# A dropped config key is this project's most expensive recurring bug (an argument that dies in a
# `**kw` leaves a cell bit-identical to its twin and READS AS A RESULT). So the declared diffs are
# asserted here, before any GPU time, and `concept.enabled` is required on every arm because
# otherwise the loader never passes `batch["reps"]` and the terms silently never fire.
"${PY}" - "${ROOT}/configs/v14_anchor_off.yaml" "${ROOT}/configs/v14_anchor_raw.yaml" \
          "${ROOT}/configs/v14_anchor_whit.yaml" "${ROOT}/configs/v14_anchor_whit05.yaml" <<'PY' \
  || exit 1
import sys
from samclip.utils import load_config

def flat(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from flat(v, f"{p}.{k}" if p else str(k))
    else:
        yield p, o

#: name -> flattened effective config, in the order the argv paths were given
names = ["off", "raw", "whit", "whit05"]
cfgs = {n: dict(flat(load_config(p))) for n, p in zip(names, sys.argv[1:5])}

#: the DECLARED single-variable pairs. Each is a claim that the two arms differ in exactly this key
#: set -- asserting it turns "I forgot a key" into a startup failure instead of a result.
PAIRS = [
    (("whit", "off"), {"bias_anchor.enabled"}),    # the term's effect
    (("whit", "raw"), {"bias_anchor.frame"}),      # v14's hypothesis: the FRAME
    (("whit05", "whit"), {"bias_anchor.weight"}),  # the magnitude, isolated from the frame
]
for (na, nb), expected in PAIRS:
    a, b = cfgs[na], cfgs[nb]
    d = {k: (a.get(k), b.get(k)) for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}
    print(f"[v14] {na} vs {nb}: {len(d)} key(s) {d}")
    if set(d) != expected:
        sys.exit(f"[v14] FATAL: {na} vs {nb} must differ in exactly {sorted(expected)}, "
                 f"got {sorted(d)}")
for nm in ("whit", "whit05"):
    if str(cfgs[nm].get("bias_anchor.frame")) != "whitened":
        sys.exit(f"[v14] FATAL: arm {nm}'s frame did not reach the config; the fix is not applied")
if str(cfgs["raw"].get("bias_anchor.frame")) != "raw":
    sys.exit("[v14] FATAL: the `raw` control's frame is not `raw`; whit-vs-raw would be a no-op")
for nm in ("whit", "whit05"):
    if float(cfgs[nm].get("bias_anchor.whiten_shrink", -1)) != 0.0:
        sys.exit(f"[v14] FATAL: arm {nm} needs whiten_shrink 0.0 -- shrinkage breaks the exact "
                 f"frame invariance that is the whole point (smoke §22 pins it).")
for nm in names:
    c = cfgs[nm]
    if not c.get("concept.enabled"):
        sys.exit(f"[v14] FATAL: arm {nm} needs `concept.enabled: true` or `batch['reps']` is never "
                 f"passed and the order-2 terms silently never fire.")
    if float(c.get("concept.reps_weight", 0.0)) != 0.0:
        sys.exit(f"[v14] FATAL: arm {nm} has a nonzero `concept.reps_weight`, which would add the "
                 f"(separately falsified) T2' term to ALL arms and confound every pair.")
    fold = {k: c.get(k) for k in ("target_subject", "source_subjects", "seed")}
    if any(v is not None for v in fold.values()):
        sys.exit(f"[v14] FATAL: arm {nm} must be FOLD-FREE, got {fold}")
# The shared repetition block is encoded at the FULL train R and then sliced to `r_use`, so if
# `r_use` were smaller than the train R the encoder would see more points than the v13 `raw` arm did
# and any batch-level SMN difference would make `raw` a non-reproduction. The train split has R=4.
for nm in names:
    if int(cfgs[nm].get("bias_anchor.r_use", 4)) != 4:
        print(f"[v14] NOTE arm {nm}: bias_anchor.r_use != 4; the `raw` arm's bit-compatibility with "
              f"the v13 twin relies on r_use == the train repetition count.")
for nm in names:
    print(f"[v14] {nm}: epochs={cfgs[nm].get('epochs')} lr={cfgs[nm].get('lr')} "
          f"bias_anchor={ {k.split('.', 1)[1]: v for k, v in cfgs[nm].items() if k.startswith('bias_anchor.')} }")
print("[v14] preflight OK: all declared pairs are single-variable, frames reached the configs")
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
ARMS="${ARMS:-off raw whit whit05}"

# arm -> config / stage1 root / base-eval root / MVE-eval root / probe root
declare -A CFG OUT ED MV PR
for a in off raw whit whit05; do
  CFG[$a]="${ROOT}/configs/v14_anchor_${a}.yaml"
  OUT[$a]="${ROOT}/outputs/stage1/v14_anchor_${a}"
  ED[$a]="${ROOT}/outputs/eval/v14_anchor_${a}"
  MV[$a]="${ROOT}/outputs/eval/v14_mve_${a}"
  PR[$a]="${ROOT}/outputs/probe/v14_anchor_${a}"
  mkdir -p "${OUT[$a]}" "${ED[$a]}" "${MV[$a]}" "${PR[$a]}"
done

log "folds: ${SUBJECTS}   seeds: ${SEEDS}   arms: ${ARMS}"

for s in ${SUBJECTS}; do
  src=()
  for o in 1 2 3 4 5 6 7 8 9 10; do [[ "${o}" -ne "${s}" ]] && src+=("${o}"); done
  tag="$(printf 'sub%02d' "${s}")"
  if [[ "${s}" -eq 8 ]]; then REF_ARGS=(--ref-top1 22.00 --ref-top5 50.00)
  else                        REF_ARGS=(--ref-top1 unknown); fi

  for seed in ${SEEDS}; do
    for a in ${ARMS}; do
      out="${OUT[$a]}/${tag}_k20_seed${seed}"
      ev="${ED[$a]}/${tag}_seed${seed}.json"
      mv="${MV[$a]}/${tag}_seed${seed}.json"
      name="${a}/${tag}/seed${seed}"
      log "--- ${name} : sources ${src[*]} ---"

      if should_skip "${out}/last.pt" "${CFG[$a]}"; then
        log "  [SKIP] train -- up to date"
      else
        run_step "train ${name}" "${LOG_DIR}/v14-train-${a}-${tag}-seed${seed}-${SLURM_JOB_ID}.log" \
          "${PY}" scripts/run_stage1.py --config "${CFG[$a]}" \
            --target-subject "${s}" --source-subjects "${src[@]}" \
            --out-dir "${out}" --seed "${seed}" \
            --mvnn train --channel-set all63
      fi

      # ---- (1) the PROVEN base ladder. Unchanged flags from the shipped recipe, so a headline
      # number here is comparable to the banked 54.83/83.45. It is written FIRST and to its own
      # file so that a failure in the new MVE ladder below cannot cost us the headline.
      if [[ -s "${ev}" ]] && [[ "${ev}" -nt "${out}/last.pt" ]] \
         && should_skip "${ev}" "${CFG[$a]}"; then
        log "  [SKIP] eval(base) -- newer than its checkpoint"
      elif [[ ! -s "${out}/last.pt" ]]; then
        log "  [SKIP] eval(base) -- no checkpoint (training failed above)"
      else
        run_step "eval ${name}" "${LOG_DIR}/v14-eval-${a}-${tag}-seed${seed}-${SLURM_JOB_ID}.log" \
          "${PY}" scripts/run_eval.py --ckpts "${out}/last.pt" \
            --target-subject "${s}" --out "${ev}" \
            --recovery --reps --csls-k 10 --rho 0.1 \
            --recovery-operator fgw --recovery-tau 0.03 --recovery-alpha 0.75 \
            --rep-shrink 0.1 --fgw-struct-fuse 0,16 \
            "${REF_ARGS[@]}"
      fi

      # ---- (2) the NEW MVE ladder. `--mve-partitions cont,stride,rand` pools three partition
      # families into one reliability-weighted structural reference; `--mve-fuse` additionally
      # emits the cont-only cell, which is the KILL SWITCH: it must reproduce the shipped
      # `fuse=16` measurement, and if it does not then this pipeline is broken and no number from
      # this job may be read.
      if [[ -s "${mv}" ]] && [[ "${mv}" -nt "${out}/last.pt" ]] \
         && should_skip "${mv}" "${CFG[$a]}" "${ROOT}/src/samclip/calibration.py" \
                        "${ROOT}/scripts/run_eval.py"; then
        log "  [SKIP] eval(mve) -- newer than its checkpoint"
      elif [[ ! -s "${out}/last.pt" ]]; then
        log "  [SKIP] eval(mve) -- no checkpoint"
      else
        run_step "eval-mve ${name}" "${LOG_DIR}/v14-mve-${a}-${tag}-seed${seed}-${SLURM_JOB_ID}.log" \
          "${PY}" scripts/run_eval.py --ckpts "${out}/last.pt" \
            --target-subject "${s}" --out "${mv}" \
            --recovery --reps --csls-k 10 --rho 0.1 \
            --recovery-operator fgw --recovery-tau 0.03 --recovery-alpha 0.75 \
            --rep-shrink 0.1 \
            --mve-partitions cont,stride,rand --mve-blocks 16 --mve-fuse \
            "${REF_ARGS[@]}"
      fi

      # ---- (3) the MECHANISM READOUT. ~15 s/fold, no labels: it re-embeds the target's
      # repetition cloud and measures corr(D_R', D_g) in BOTH the whitened and the raw frame, plus
      # the effective rank and the shuffle control. This is where the v14 hypothesis is read: the
      # fix is real only if the WHITENED-frame delta moves, since that is the frame the operator
      # consumes.
      if should_skip "${PR[$a]}/${tag}_seed${seed}.json" "${CFG[$a]}"; then
        log "  [SKIP] probe -- up to date"
      elif [[ ! -s "${out}/last.pt" ]]; then
        log "  [SKIP] probe -- no checkpoint"
      else
        run_step "probe ${name}" "${LOG_DIR}/v14-probe-${a}-${tag}-seed${seed}-${SLURM_JOB_ID}.log" \
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
# the two signed verdicts, each with its own pre-registered criterion and guards
# =====================================================================================
# `summarize_v13.py` is reused DELIBERATELY rather than reimplemented: its criterion, its guard
# checks and its single-variable assertion are already written and reviewed, and a second copy of a
# verdict rule is how a rule quietly becomes two different rules.
log "=== verdict 1/2: whit vs off (the TERM's effect) ==="
run_step "summary whit-vs-off" "${LOG_DIR}/v14-summary-whit-off-${SLURM_JOB_ID}.log" \
  "${PY}" scripts/summarize_v13.py \
    --on-eval "${ED[whit]}" --off-eval "${ED[off]}" \
    --on-probe "${PR[whit]}" --off-probe "${PR[off]}" \
    --on-cfg "${CFG[whit]}" --off-cfg "${CFG[off]}" \
    --out "${ROOT}/outputs/v14_summary_whit_vs_off.json"
tail -n 25 "${LOG_DIR}/v14-summary-whit-off-${SLURM_JOB_ID}.log" | tee -a "${MASTER}" || true

log "=== verdict 2/2: whit vs raw (v14's actual hypothesis: the FRAME) ==="
run_step "summary whit-vs-raw" "${LOG_DIR}/v14-summary-whit-raw-${SLURM_JOB_ID}.log" \
  "${PY}" scripts/summarize_v13.py \
    --on-eval "${ED[whit]}" --off-eval "${ED[raw]}" \
    --on-probe "${PR[whit]}" --off-probe "${PR[raw]}" \
    --on-cfg "${CFG[whit]}" --off-cfg "${CFG[raw]}" \
    --out "${ROOT}/outputs/v14_summary_whit_vs_raw.json"
tail -n 25 "${LOG_DIR}/v14-summary-whit-raw-${SLURM_JOB_ID}.log" | tee -a "${MASTER}" || true

# The third pair isolates MAGNITUDE from FRAME: the two arms agree on frame=whitened and differ only
# in weight, so a difference here is a dose effect and not a change of objective.
log "=== verdict 2b: whit05 vs whit (the WEIGHT, with the frame held fixed) ==="
run_step "summary whit05-vs-whit" "${LOG_DIR}/v14-summary-whit05-whit-${SLURM_JOB_ID}.log" \
  "${PY}" scripts/summarize_v13.py \
    --on-eval "${ED[whit05]}" --off-eval "${ED[whit]}" \
    --on-probe "${PR[whit05]}" --off-probe "${PR[whit]}" \
    --on-cfg "${CFG[whit05]}" --off-cfg "${CFG[whit]}" \
    --out "${ROOT}/outputs/v14_summary_whit05_vs_whit.json"
tail -n 25 "${LOG_DIR}/v14-summary-whit05-whit-${SLURM_JOB_ID}.log" | tee -a "${MASTER}" || true

log "=== verdict 3/3: O2-MVE (the deployment-side estimator) ==="
run_step "summary mve" "${LOG_DIR}/v14-summary-mve-${SLURM_JOB_ID}.log" \
  "${PY}" scripts/summarize_v14_mve.py \
    --off "${MV[off]}" --raw "${MV[raw]}" --whit "${MV[whit]}" --whit05 "${MV[whit05]}" \
    --base "${ED[off]}" \
    --out "${ROOT}/outputs/v14_summary_mve.json"
tail -n 30 "${LOG_DIR}/v14-summary-mve-${SLURM_JOB_ID}.log" | tee -a "${MASTER}" || true

if [[ ${#FAILED_STEPS[@]} -gt 0 ]]; then
  log "FINISHED WITH FAILURES (${#FAILED_STEPS[@]}): ${FAILED_STEPS[*]}"
  log "re-submit to retry only the failed units -- every step is skip-guarded."
  exit 1
fi
log "ALL STEPS OK  (master log ${MASTER})"
