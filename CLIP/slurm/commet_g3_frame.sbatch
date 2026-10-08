#!/bin/bash
# =====================================================================================
# G3 -- MULTI-SUBJECT CONCEPT-FRAME TRANSDUCTION, 10 folds x 3 seeds, on the CANONICAL G3
# checkpoints. This is the full grid of the method the user adopted (option `m1_direct`).
#
# THE METHOD, IN ONE PARAGRAPH. Every subject is tested on the SAME 200 concepts the gallery
# indexes, in the same order, so each subject's concept metric is a noisy view of ONE shared
# metric in ONE shared frame. The method pools the TARGET's own repetition-cloud metric with
# the 9 SOURCE subjects' metrics (all in the shared frame) and hands the pooled metric to the
# FGW structural term. Operationally: `rep_cloud_scores(..., src_means=<9x200x64>, src_mix=1)`.
#
# HONEST STATUS (read before quoting the number). `docs/eeg2image_v10_m1_theory.md` §9 measured
# this at +15.37pp and REJECTED it, because a concept-axis permutation flipped it to -14.5pp.
# That control destroys TWO correspondences at once -- source<->gallery (the answer key) AND
# source<->target (the legitimate shared metric, measured corr +0.565) -- so it does not by
# itself separate them. The user has adopted the arm with the caveat attached. TWO things this
# job DOES establish and pins:
#   * the whole gain routes through the STRUCTURAL term: the structural-off twin (alpha=0) is
#     bit-identical between m=1 and m=0, so nothing leaks through the cross-modal cost;
#   * the effect is 30/30 folds with a minimum of +6.0pp, so it is not a fold artefact.
#
# PROTOCOL CAVEAT, CARRIED INTO THE SUMMARY. Using the SOURCE subjects' unlabelled test EEG is
# a stronger transductive setting than the published cross-subject cells use (SCORE, SATTC take
# only the held-out subject's test EEG). The comparison to 53.23/83.55 is therefore
# CROSS-PROTOCOL and is labelled as such in every artefact this job writes.
#
# WHY IT RUNS AGAIN ON THE g3 CHECKPOINTS. The 30-cell grid already exists on the v8
# checkpoints (`outputs/eval/v10_m1`, same recipe, +15.37pp). Re-running on the canonical
# `outputs/stage1/g3` checkpoints makes it a genuine second independent grid (different runs of
# the same recipe) AND puts the headline on the same checkpoint set as the rest of G3.
#
# Submit:   sbatch slurm/commet_g3_frame.sbatch
# Smoke:    SUBJECTS=8 SEEDS=2025 sbatch slurm/commet_g3_frame.sbatch
# =====================================================================================
#SBATCH --job-name=commet-g3-frame
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=20:00:00
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --output=/project/peilab/why/CLIP/outputs/slurm/commet-g3-frame-%j.out
#SBATCH --error=/project/peilab/why/CLIP/outputs/slurm/commet-g3-frame-%j.err

set -euo pipefail

ROOT="/project/peilab/why/CLIP"
PY="${ROOT}/.venv/bin/python"
SUBJECTS="${SUBJECTS:-1 2 3 4 5 6 7 8 9 10}"
SEEDS="${SEEDS:-2025 2026 2027}"

cd "${ROOT}"
mkdir -p outputs/slurm outputs/eval/commet_g3 outputs/src_metric/g3

export PYTHONNOUSERSITE=1
export HF_HOME=/project/peilab/why/cache/huggingface
export HF_HUB_CACHE=/project/peilab/why/cache/huggingface/hub
export TRANSFORMERS_CACHE=/project/peilab/why/cache/huggingface
export OPENCLIP_CACHE_DIR=/project/peilab/why/cache/open_clip
export TORCH_HOME=/project/peilab/why/cache/torch
export XDG_CACHE_HOME=/project/peilab/why/cache/xdg
export PIP_CACHE_DIR=/project/peilab/why/cache/pip
# NODE-LOCAL, not NFS (AGENTS.md §2.4)
export TMPDIR="/tmp/clip-${SLURM_JOB_ID:-local}"
export XDG_CONFIG_HOME="${ROOT}/scratch/config"
export MPLCONFIGDIR="${ROOT}/scratch/mpl"
mkdir -p "${TMPDIR}" "${XDG_CONFIG_HOME}" "${MPLCONFIGDIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

MASTER="${ROOT}/outputs/slurm/commet-g3-frame-${SLURM_JOB_ID}.log"
log() { echo "[$(date -Is)] $*" | tee -a "${MASTER}"; }

log "=== G3 concept-frame transduction job=${SLURM_JOB_ID} host=$(hostname) ==="
log "folds: ${SUBJECTS}  seeds: ${SEEDS}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
co_tenants=$(squeue -w "$(hostname)" -h -o '%i %u %j' | grep -v "${SLURM_JOB_ID}" || true)
[[ -n "${co_tenants}" ]] && { log "shared-node co-tenants:"; echo "${co_tenants}" | tee -a "${MASTER}"; }

"${PY}" -c "import sys,torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { log "[FATAL] CUDA unavailable on $(hostname)"; exit 1; }
"${PY}" -c "import samclip.calibration as c, inspect; \
assert 'src_means' in inspect.signature(c.rep_cloud_scores).parameters" \
  || { log "[FATAL] calibration.rep_cloud_scores has no src_means knob"; exit 1; }

FAILED=()
run_step() {
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
  if [[ "${s}" -eq 8 ]]; then REF=(--ref-top1 22.00 --ref-top5 50.00); else REF=(--ref-top1 unknown); fi
  for seed in ${SEEDS}; do
    ck="${ROOT}/outputs/stage1/g3/${tag}_k20_seed${seed}/last.pt"
    sm="${ROOT}/outputs/src_metric/g3/${tag}_seed${seed}.npz"
    ev="${ROOT}/outputs/eval/commet_g3/${tag}_seed${seed}.json"
    [[ -s "${ck}" ]] || { log "  [skip] ${tag}/seed${seed}: no checkpoint"; continue; }
    if [[ -s "${ev}" ]] && [[ "${ev}" -nt "${ck}" ]] && [[ "${ev}" -nt "${sm}" ]]; then
      log "  [SKIP] ${tag}/seed${seed} (eval newer than both its checkpoint and its template)"
      continue
    fi
    log "--- ${tag}/seed${seed} ---"
    # ONE PASS produces the template AND consumes it: `--dump-src-metric` runs inside
    # `_evaluate_checkpoint` BEFORE the repetition block reads `--fgw-src-metric`, so pointing
    # both at the same path writes the template and then reads it back within the same call.
    # Producing it in the fold that owns the checkpoint is what stops channels/mvnn/feature-set
    # from drifting from the eval path (the same guarantee `slurm/v10_m1.sbatch` relies on).
    #
    # The pre-registered grid: the mix enters as a LIST so the whole curve comes from ONE
    # embedding and m=0 / m=1 are paired cell-by-cell at every R.
    run_step "eval ${tag}/${seed}" \
      "${ROOT}/outputs/slurm/commet-g3-eval-${tag}-${seed}-${SLURM_JOB_ID}.log" \
      "${PY}" scripts/run_eval.py --ckpts "${ck}" --target-subject "${s}" --out "${ev}" \
        --dump-src-metric "${sm}" \
        --reps --csls-k 10 --rho 0.1 \
        --recovery-operator fgw --recovery-tau 0.03 --recovery-alpha 0.75 \
        --rep-shrink 0.1 --fgw-sweep --rep-subsample 80 \
        --fgw-src-metric "${sm}" --fgw-src-mix "0.0,1.0" "${REF[@]}"
  done
done

log "=== summary ==="
run_step "summarize" "${ROOT}/outputs/slurm/commet-g3-summary-${SLURM_JOB_ID}.log" \
  "${PY}" scripts/summarize_frame_g3.py \
    --glob 'outputs/eval/commet_g3/sub*_seed*.json' \
    --out "${ROOT}/outputs/commet_g3_frame_summary.json"
tail -n 40 "${ROOT}/outputs/slurm/commet-g3-summary-${SLURM_JOB_ID}.log" | tee -a "${MASTER}" || true

if [[ ${#FAILED[@]} -gt 0 ]]; then
  log "FINISHED WITH FAILURES (${#FAILED[@]}): ${FAILED[*]}"
  log "re-submit to retry only the failed units -- every step is skip-guarded."
  exit 1
fi
log "ALL STEPS OK  (master log ${MASTER})"
