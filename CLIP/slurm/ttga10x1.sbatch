#!/bin/bash
# =====================================================================================
# TTGA -- TEST-TIME GROUP AVERAGING over BANKED checkpoints, 10 folds, ZERO retraining.
#
# WHAT THIS TESTS, AND WHY IT IS THE SHARPEST AVAILABLE TEST OF THE THEORY.
# SQA says volume conduction is a `GL(C)` action and the concept is `G`-invariant, so the
# representation should be the Reynolds average `P_G z(x) = E_{M~G} z(M x)`. Training can
# only ESTIMATE that average (one draw per epoch); inference can compute it exactly. Two
# checkpoints bracket the prediction:
#
#   sqa       trained WITH the group augmentation -> should be near-invariant -> TTGA helps
#   v8        the SHIPPED best encoder, trained with NO augmentation -> should NOT be
#             invariant -> TTGA HURTS. (sqa_noaug is included as a third point.)
#
# So the falsifiable prediction is a CROSSOVER: `Reynolds residual` (E_M ||z(Mx)-z(x)||) is
# smaller for `sqa` than for `v8`, and the paired dTop1 as a function of K is POSITIVE for
# `sqa` and NEGATIVE for `v8`. If instead TTGA helps every encoder equally, the mechanism is
# generic test-time augmentation and NOT the group structure -- and the claim dies cleanly.
#
# WHY THE ACCURACY IS GUARANTEED. Every report contains `gamma = 0`, which is EXACTLY the
# shipped cell (verified bit-for-bit: `probe_ttga.py` on v8/sub01 reproduced
# `outputs/eval/v11_fuse/sub01_seed2025.json`'s 64.50/88.50). TTGA cells are ADDITIVE rows
# on an interpolation grid z(g) = normalize((1-g)*z_base + g*z_TTGA), so nothing is removed
# and no cell is selected on the test set. v8's shipped 54.83/55.10 stays on disk whatever
# this job finds.
#
# WHAT IS APPLIED. The deployed cell scores the REPETITION CLOUD (`--reps`, R=80) as the
# query side, so TTGA is applied THERE, i.e. the Reynolds average of `z_reps` over K group
# elements drawn from the SAME distribution the training augmentation used (dense, matched
# by ||M - I||_F = 3.15; the match is asserted in `--check-mixing`).
#
# COST. No training. Per (fold, ckpt): K_max embeddings of the (200x80) repetition cloud
# plus a handful of FGW cells. ~2 GPU-hours for the whole grid.
#
# Submit:   sbatch slurm/ttga10x1.sbatch
# Smoke:    SUBJECTS=1 sbatch slurm/ttga10x1.sbatch
# =====================================================================================
#SBATCH --job-name=ttga10x1
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --output=/project/peilab/why/CLIP/outputs/slurm/ttga10x1-%j.out
#SBATCH --error=/project/peilab/why/CLIP/outputs/slurm/ttga10x1-%j.err

set -euo pipefail

ROOT="/project/peilab/why/CLIP"
PY="${ROOT}/.venv/bin/python"
mkdir -p "${ROOT}/outputs/slurm"

export PYTHONNOUSERSITE=1
export HF_HOME=/project/peilab/why/cache/huggingface
export HF_HUB_CACHE=/project/peilab/why/cache/huggingface/hub
export TRANSFORMERS_CACHE=/project/peilab/why/cache/huggingface
export OPENCLIP_CACHE_DIR=/project/peilab/why/cache/open_clip
export TORCH_HOME=/project/peilab/why/cache/torch
export XDG_CACHE_HOME=/project/peilab/why/cache/xdg
export PIP_CACHE_DIR=/project/peilab/why/cache/pip
export TMPDIR="/tmp/clip-${SLURM_JOB_ID:-local}"     # node-local, NOT NFS (AGENTS.md 2.4)
export XDG_CONFIG_HOME="${ROOT}/scratch/config"
export MPLCONFIGDIR="${ROOT}/scratch/mpl"
mkdir -p "${TMPDIR}" "${XDG_CONFIG_HOME}" "${MPLCONFIGDIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"

LOG_DIR="${ROOT}/outputs/slurm"
MASTER="${LOG_DIR}/ttga10x1-${SLURM_JOB_ID}.log"
EVAL_ROOT="${ROOT}/outputs/probe/ttga"
mkdir -p "${EVAL_ROOT}"

KG="${KG:-1,2,4,8}"
GAMMAS="${GAMMAS:-0,0.5,1.0}"
MIXING="${MIXING:-3.15}"
MODE="${MODE:-dense}"

log() { echo "[$(date -Is)] $*" | tee -a "${MASTER}"; }

log "=== TTGA over banked ckpts, 10 folds, job=${SLURM_JOB_ID} host=$(hostname) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
co_tenants=$(squeue -w "$(hostname)" -h -o '%i %u %j' | grep -v "${SLURM_JOB_ID}" || true)
if [[ -n "${co_tenants}" ]]; then log "shared-node co-tenants:"; echo "${co_tenants}" | tee -a "${MASTER}"; fi

"${PY}" -c "import sys,torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { log "[FATAL] CUDA unavailable on $(hostname)"; exit 1; }

FAILED_STEPS=()
run_step() {
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

SUBJECTS="${SUBJECTS:-1 2 3 4 5 6 7 8 9 10}"
# (tag, checkpoint path template) -- all BANKED, nothing is trained here.
CKPTS=("v8:${ROOT}/outputs/stage1/v8/%s_k20_seed2025/last.pt"
       "sqa:${ROOT}/outputs/stage1/sqa/%s_k20_seed2025/last.pt"
       "sqa_noaug:${ROOT}/outputs/stage1/sqa_noaug/%s_k20_seed2025/last.pt")

log "folds: ${SUBJECTS}   K grid: ${KG}   gammas: ${GAMMAS}   mode: ${MODE} mixing: ${MIXING}"

for s in ${SUBJECTS}; do
  tag="$(printf 'sub%02d' "${s}")"
  for spec in "${CKPTS[@]}"; do
    name="${spec%%:*}"; tmpl="${spec#*:}"
    # shellcheck disable=SC2059
    ck="$(printf "${tmpl}" "${tag}")"
    out="${EVAL_ROOT}/${name}/${tag}.json"
    mkdir -p "${EVAL_ROOT}/${name}"
    if [[ ! -s "${ck}" ]]; then
      log "  [SKIP] ${name}/${tag} -- no banked checkpoint at ${ck}"
      continue
    fi
    if [[ -s "${out}" ]] && [[ "${out}" -nt "${ck}" ]]; then
      log "  [SKIP] ${name}/${tag} -- up to date"
      continue
    fi
    run_step "ttga ${name}/${tag}" "${LOG_DIR}/ttga-${name}-${tag}-${SLURM_JOB_ID}.log" \
      "${PY}" scripts/probe_ttga.py --ckpts "${ck}" --target-subject "${s}" \
        --out "${out}" --ttga-K "${KG}" --ttga-gammas "${GAMMAS}" \
        --ttga-mode "${MODE}" --ttga-mixing "${MIXING}" --check-mixing --rep-blocks 16
  done
done

log "=== summary ==="
run_step "summarize" "${LOG_DIR}/ttga-summary-${SLURM_JOB_ID}.log" \
  "${PY}" scripts/summarize_ttga.py --glob 'outputs/probe/ttga/*/sub*.json' \
    --out "${ROOT}/outputs/ttga_summary.json"
tail -n 60 "${LOG_DIR}/ttga-summary-${SLURM_JOB_ID}.log" | tee -a "${MASTER}" || true

if [[ ${#FAILED_STEPS[@]} -gt 0 ]]; then
  log "FINISHED WITH FAILURES (${#FAILED_STEPS[@]}): ${FAILED_STEPS[*]}"
  exit 1
fi
log "ALL STEPS OK  (master log ${MASTER})"
