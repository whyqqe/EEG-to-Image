#!/bin/bash
# =====================================================================================
# COMMET GATES G0 + G1 -- the two cheap, no-training gates before any full grid.
#
# WHY A GATE AT ALL. Six training-side attempts have already been falsified on this project
# (T2', T2'', SCORE-episode, G-a, and v12's two metric terms), and v12 looked healthy for
# half its epochs before collapsing. So nothing is submitted to a 30-run grid until a
# 10-fold probe says the mechanism moves the deployed number.
#
#   G0 -- the METRIC ADAPTER (I5). A linear map fitted on SOURCE subjects to suppress
#         within-stimulus noise in the frozen encoder's per-trial embeddings, with an
#         unsupervised control (`total`) and a label-destroying control (`shuffled`).
#         No training loop: the BEST-CASE LINEAR adapter is closed-form, so if it does not
#         move the number, no learned phi will. `scripts/probe_g0_adapter.py`.
#
#   G1 -- the BILATERAL GALLERY VIEW (I4). v11 pools the QUERY's own repetition blocks;
#         this pools the GALLERY's own K layer-views by the same reliability weights and
#         injects the fused metric into the FGW structural term. Each cell carries a
#         structural-off twin that MUST stay flat, because the injected reference can only
#         route through the structural term. `--fgw-gallery-fuse` in `run_eval.py`.
#
# NEITHER STEP TRAINS. Both read the banked 10-fold v8 checkpoints, so this costs a GPU
# slot for one eval + one embedding pass per fold and nothing else.
#
# PREDICTIONS THIS JOB CANNOT BE RE-INTERPRETED AFTER THE FACT
#   * G0 passes iff `within` > `identity` AND `within` > `total` AND `within` > `shuffled`,
#     on a majority of folds (read by `summarize_commet_gates.py`, not by eye).
#   * G1 passes iff `...,fuse=16,gv` > `...,fuse=16` on a majority of folds AND the
#     structural-off twin moves by < 0.5pp.
#   * If neither passes, G3 is NOT submitted: the architecture is unchanged and the doc's
#     I4/I5 rows get the falsification instead of a claim.
#
# Submit: sbatch slurm/commet_gates.sbatch
# Smoke:  SUBJECTS=8 sbatch slurm/commet_gates.sbatch
# =====================================================================================
#SBATCH --job-name=commet-gates
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --output=/project/peilab/why/CLIP/outputs/slurm/commet-gates-%j.out
#SBATCH --error=/project/peilab/why/CLIP/outputs/slurm/commet-gates-%j.err

set -euo pipefail

ROOT="/project/peilab/why/CLIP"
PY="${ROOT}/.venv/bin/python"
SUBJECTS="${SUBJECTS:-1 2 3 4 5 6 7 8 9 10}"
SEED="${SEED:-2025}"

cd "${ROOT}"
mkdir -p outputs/slurm outputs/probe/g0 outputs/eval/g1

export PYTHONNOUSERSITE=1
export HF_HOME=/project/peilab/why/cache/huggingface
export HF_HUB_CACHE=/project/peilab/why/cache/huggingface/hub
export TRANSFORMERS_CACHE=/project/peilab/why/cache/huggingface
export OPENCLIP_CACHE_DIR=/project/peilab/why/cache/open_clip
export TORCH_HOME=/project/peilab/why/cache/torch
export XDG_CACHE_HOME=/project/peilab/why/cache/xdg
export PIP_CACHE_DIR=/project/peilab/why/cache/pip
# NODE-LOCAL, not NFS (AGENTS.md §2.4): multiprocessing temp dirs cannot be rmtree'd on NFS.
export TMPDIR="/tmp/clip-${SLURM_JOB_ID:-local}"
export XDG_CONFIG_HOME="${ROOT}/scratch/config"
export MPLCONFIGDIR="${ROOT}/scratch/mpl"
mkdir -p "${TMPDIR}" "${XDG_CONFIG_HOME}" "${MPLCONFIGDIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

MASTER="${ROOT}/outputs/slurm/commet-gates-${SLURM_JOB_ID}.log"
log() { echo "[$(date -Is)] $*" | tee -a "${MASTER}"; }

log "=== COMMET gates G0+G1 job=${SLURM_JOB_ID} host=$(hostname) ==="
log "folds: ${SUBJECTS}  seed: ${SEED}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
co_tenants=$(squeue -w "$(hostname)" -h -o '%i %u %j' | grep -v "${SLURM_JOB_ID}" || true)
[[ -n "${co_tenants}" ]] && { log "shared-node co-tenants:"; echo "${co_tenants}" | tee -a "${MASTER}"; }

# --- hard CUDA gate: fail fast rather than silently embedding 16000 trials on CPU ------
"${PY}" -c "import sys,torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { log "[FATAL] CUDA unavailable on $(hostname)"; exit 1; }

# The gate scripts must actually be importable/executable before the 20-step chain starts.
"${PY}" -c "import samclip.calibration as c; import inspect; \
assert 'gallery_di_ref' in inspect.signature(c.rep_cloud_scores).parameters, 'di_ref not wired'; \
assert 'di_ref' in inspect.signature(c._fgw_plan).parameters, 'di_ref missing in _fgw_plan'" \
  || { log "[FATAL] calibration.py is missing the G1 plumbing"; exit 1; }

FAILED=()
run_step() {                       # run_step <name> <logfile> <cmd...>
  local name="$1" lf="$2"; shift 2
  local rc=0
  "$@" >"${lf}" 2>&1 || rc=$?
  if [[ ${rc} -eq 0 ]]; then log "     OK   ${name}"; return 0; fi
  log "     FAILED rc=${rc}  ${name}  (log ${lf})"
  tail -n 25 "${lf}" | tee -a "${MASTER}" || true
  FAILED+=("${name}")
}

for s in ${SUBJECTS}; do
  tag="$(printf 'sub%02d' "${s}")"
  ck="${ROOT}/outputs/stage1/v8/${tag}_k20_seed${SEED}/last.pt"
  if [[ ! -s "${ck}" ]]; then log "  [skip] ${tag}: no checkpoint ${ck}"; continue; fi
  src=(); for o in 1 2 3 4 5 6 7 8 9 10; do [[ "${o}" -ne "${s}" ]] && src+=("${o}"); done
  if [[ "${s}" -eq 8 ]]; then REF=(--ref-top1 22.00 --ref-top5 50.00); else REF=(--ref-top1 unknown); fi
  log "--- ${tag}/seed${SEED} : sources ${src[*]} ---"

  # ---- G0 -----------------------------------------------------------------
  g0="outputs/probe/g0/${tag}_seed${SEED}.json"
  if [[ -s "${g0}" ]]; then
    log "  [SKIP] G0 ${tag}"
  else
    run_step "g0 ${tag}" "${ROOT}/outputs/slurm/commet-g0-${tag}-${SLURM_JOB_ID}.log" \
      "${PY}" scripts/probe_g0_adapter.py --ckpts "${ck}" --target-subject "${s}" \
        --source-subjects "${src[@]}" --out "${g0}" \
        --fit-reps 20 --gamma 1e-2 --lda-k 16 --fuse 16 --embed-batch 128
  fi

  # ---- G1 -----------------------------------------------------------------
  g1="outputs/eval/g1/${tag}_seed${SEED}.json"
  if [[ -s "${g1}" ]]; then
    log "  [SKIP] G1 ${tag}"
  else
    run_step "g1 ${tag}" "${ROOT}/outputs/slurm/commet-g1-${tag}-${SLURM_JOB_ID}.log" \
      "${PY}" scripts/run_eval.py --ckpts "${ck}" --target-subject "${s}" --out "${g1}" \
        --reps --csls-k 10 --rho 0.1 \
        --recovery-operator fgw --recovery-tau 0.03 --recovery-alpha 0.75 \
        --rep-shrink 0.1 --fgw-struct-fuse 16 --fgw-gallery-fuse "${REF[@]}"
  fi
done

log "=== summary ==="
run_step "summarize" "${ROOT}/outputs/slurm/commet-summary-${SLURM_JOB_ID}.log" \
  "${PY}" scripts/summarize_commet_gates.py \
    --g0 'outputs/probe/g0/*.json' --g1 'outputs/eval/g1/*.json' \
    --out "${ROOT}/outputs/commet_gates_summary.json"
tail -n 70 "${ROOT}/outputs/slurm/commet-summary-${SLURM_JOB_ID}.log" | tee -a "${MASTER}" || true

if [[ ${#FAILED[@]} -gt 0 ]]; then
  log "FINISHED WITH FAILURES (${#FAILED[@]}): ${FAILED[*]}"
  log "re-submit to retry only the failed units -- every step is skip-guarded."
  exit 1
fi
log "ALL STEPS OK  (master log ${MASTER})"
