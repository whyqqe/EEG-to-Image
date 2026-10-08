#!/bin/bash
#SBATCH --job-name=s3r-eval
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --output=/project/peilab/why/CLIP/outputs/slurm/s3r-%j.out
#SBATCH --error=/project/peilab/why/CLIP/outputs/slurm/s3r-%j.err
#
# S3R-EVAL -- the soft-plan recovery operator, swept over the BANKED G3 grid. NO TRAINING.
#
# WHY THIS IS ITS OWN SUBMISSION AND WHY IT IS THE PRIORITY ONE.
#
# The recovery operator is applied at DEPLOYMENT only, so a change to it is measurable on the
# 30 checkpoints that already exist. That makes this the one experiment on the table whose
# result cannot be invalidated by a training run: the encoder weights are held fixed and are
# the same 30 weights the banked 45.53 came from, so the only variable is the estimator.
#
# The probe (`scripts/probe_s3r.py`, outputs/probe/s3r_full.json) established the effect on
# the same 30 runs first: the fused headline moves +3.20 +- 2.01 pp, 28/30 runs positive,
# t = 8.72, 9/10 folds. It also verified its own faithfulness -- its `deployed` arm reproduces
# the banked grid exactly (`+ T2 reps` = 45.53 over 30 runs, and sub-08/seed-2025 with
# recovery = 35.50, the banked value), so a delta against that arm is a delta against G3.
#
# This job is what re-derives those numbers through the REAL evaluation path
# (`scripts/run_eval.py`, not the probe's short path) and writes per-fold artefacts beside the
# banked ones, so the claim is auditable from reports rather than from a probe's stdout.
#
# THE OPERATOR IS DEPLOYMENT-ONLY, so there is no encoder to absorb it (Scaffold Absorption,
# the mechanism that killed v5-ep2 / v6-rat / v7 T2' / v7b) -- the failure mode that makes
# in-forward operators fail is structurally absent here. That is why this arm is expected to
# survive where the four in-forward pillars did not.
#
# TWO TEMPERATURES, because the plateau is a claim that needs replicating AT A SECOND POINT
# rather than asserting: `tau` is 0.01 below, then 0.05, and the two must agree. A single
# temperature would leave "the gain is a plateau" untested (a lucky setting of one knob is
# indistinguishable from a mechanism), while the sweep already showed the arm degrading
# monotonically above 0.1 -- so 0.05 is off-peak on purpose.
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

CKPT_ROOT="${CKPT_ROOT:-${ROOT}/outputs/stage1/g3}"
EVAL_TAG_ROOT="${EVAL_TAG_ROOT:-s3r}"
SUBJECTS="${SUBJECTS:-1 2 3 4 5 6 7 8 9 10}"
SEEDS="${SEEDS:-2025 2026 2027}"
TAUS="${TAUS:-0.01 0.05}"
LOG_DIR="${ROOT}/logs"
MASTER="${LOG_DIR}/s3r-eval-${SLURM_JOB_ID}.log"

log() { echo "[$(date -Is)] $*" | tee -a "${MASTER}"; }
run_step() {  # name logfile cmd...
  local name="$1" lf="$2"; shift 2
  log "  -> ${name}"
  local rc=0
  "$@" >"${lf}" 2>&1 || rc=$?
  if [[ "${rc}" -eq 0 ]]; then log "     OK   ${name}"; else
    log "     FAIL ${name} (rc=${rc}); tail:"; tail -n 25 "${lf}" | tee -a "${MASTER}"
    return "${rc}"
  fi
}

log "=== S3R eval-only sweep job=${SLURM_JOB_ID} host=$(hostname) date=$(date -Is) ==="
log "ckpt root: ${CKPT_ROOT}"
log "subjects:  ${SUBJECTS}"
log "seeds:     ${SEEDS}"
log "taus:      ${TAUS}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
co_tenants=$(squeue -w "$(hostname)" -h -o '%i %u %j' | grep -v "${SLURM_JOB_ID}" || true)
if [[ -n "${co_tenants}" ]]; then log "shared-node co-tenants:"; echo "${co_tenants}" | tee -a "${MASTER}"; fi

# --- hard CUDA gate: a silent CPU run would take ~10x longer and still "succeed" ----------
"${PY}" -c "import sys,torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { log "[FATAL] CUDA unavailable on $(hostname)"; exit 1; }

# --- the operator must be importable AND must actually differ from the deployed one ------
# A flag that is accepted and then produces the baseline is this project's most expensive
# recurring failure (`--reps` did exactly that once). Assert both: the operator exists, and
# it moves a synthetic score matrix.
"${PY}" - <<'PY' || { log "[FATAL] soft recovery operator failed its self-test"; exit 1; }
import numpy as np, sys
sys.path.insert(0, "src")
from samclip import calibration as C
rng = np.random.default_rng(0)
q = rng.normal(size=(200, 64)); g = rng.normal(size=(200, 64))
a, _ = C.coordinate_recovery(q, g, k=10, rho=0.1)
b, d = C.subspace_soft_recovery(q, g, k=10, rho=0.1, rank=None, tau=0.01,
                               hard_landmarks=False)
assert a.shape == b.shape == (200, 64), (a.shape, b.shape)
assert not np.allclose(a, b), "sinkhorn operator is a no-op vs the deployed operator"
assert not d["abstained"], d
assert np.isfinite(b).all()
print("[preflight] operator OK  n_effective=%.0f plan_acc=%.3f orth=%.1e"
      % (d["n_effective"], d["plan_acc"], d["orthogonality_err"]))
PY

# --- checkpoints present BEFORE spending an hour failing on the last one ------------------
missing=0
for s in ${SUBJECTS}; do
  for seed in ${SEEDS}; do
    ck="${CKPT_ROOT}/$(printf 'sub%02d' "${s}")_k20_seed${seed}/last.pt"
    [[ -s "${ck}" ]] || { log "  [MISS] ${ck}"; missing=$((missing + 1)); }
  done
done
if [[ "${missing}" -gt 0 ]]; then log "[FATAL] ${missing} checkpoints missing"; exit 1; fi
log "all $(( $(echo ${SUBJECTS} | wc -w) * $(echo ${SEEDS} | wc -w) )) checkpoints present"

# =====================================================================================
for tau in ${TAUS}; do
  EVAL_ROOT="${ROOT}/outputs/eval/${EVAL_TAG_ROOT}_tau${tau}"
  mkdir -p "${EVAL_ROOT}"
  log "--- tau=${tau}  ->  ${EVAL_ROOT} ---"
  for s in ${SUBJECTS}; do
    tag="$(printf 'sub%02d' "${s}")"
    # The SAMGA-official reference was measured on sub-08 only; elsewhere `unknown` makes
    # the report print n/a instead of a delta against a different subject.
    if [[ "${s}" -eq 8 ]]; then REF_ARGS=(--ref-top1 22.00 --ref-top5 50.00)
    else                        REF_ARGS=(--ref-top1 unknown); fi
    for seed in ${SEEDS}; do
      ck="${CKPT_ROOT}/${tag}_k20_seed${seed}/last.pt"
      ev="${EVAL_ROOT}/${tag}_seed${seed}.json"
      if [[ -s "${ev}" && "${ev}" -nt "${ck}" ]]; then
        log "  [SKIP] eval -- ${ev} is newer than its checkpoint"; continue
      fi
      run_step "eval ${tag}/seed${seed} tau=${tau}" \
        "${LOG_DIR}/s3r-eval-${tag}-seed${seed}-tau${tau}-${SLURM_JOB_ID}.log" \
        "${PY}" scripts/run_eval.py --ckpts "${ck}" \
          --target-subject "${s}" --out "${ev}" \
          --recovery --reps --csls-k 10 --rho 0.1 \
          --recovery-operator sinkhorn --recovery-tau "${tau}" "${REF_ARGS[@]}"
    done
  done
done

# =====================================================================================
log "=== paired summary vs the banked G3 grid ==="
run_step "summarize" "${LOG_DIR}/s3r-summary-${SLURM_JOB_ID}.log" \
  "${PY}" scripts/summarize_s3r.py --ours "outputs/eval/${EVAL_TAG_ROOT}_tau*" \
    --base outputs/eval/g3 --out "${ROOT}/outputs/${EVAL_TAG_ROOT}_summary.json" \
  || log "  (summary failed; per-fold reports are still on disk)"

log "=== finished rc=$? at $(date -Is) ==="
