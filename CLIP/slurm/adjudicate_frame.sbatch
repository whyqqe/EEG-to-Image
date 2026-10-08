#!/bin/bash
# =====================================================================================
# ADJUDICATION — is the concept-frame transduction (+15pp) a neural leak, or does it need no
# EEG at all? 10 folds x 1 seed (2025), four arms, ONE argument apart (`--fgw-src-ref-mode`).
#
#   eeg     : the source subjects' EEG metric  -> the arm under adjudication (+15pp)
#   gallery : the PUBLIC image-gallery metric  -> if it reproduces the gain, zero EEG is needed
#   self    : the target's own metric          -> a NO-OP; must equal the baseline bit-for-bit
#   rand    : fixed-seed random symmetric metric -> the floor any real geometry must beat
#
# No training, no new encoder: it re-reads the banked g3 checkpoints and the src_metric templates
# that `slurm/commet_g3_frame.sbatch` already wrote. ~8 cells x 4 arms x 2 mix values = cheap.
#
# Submit:  sbatch slurm/adjudicate_frame.sbatch
# Smoke:   SUBJECTS=1 MODES=eeg sbatch slurm/adjudicate_frame.sbatch   # adds a `self` sanity too
# =====================================================================================
#SBATCH --job-name=commet-adjudicate
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --output=/project/peilab/why/CLIP/outputs/slurm/adjudicate-%j.out
#SBATCH --error=/project/peilab/why/CLIP/outputs/slurm/adjudicate-%j.err

set -euo pipefail

ROOT="/project/peilab/why/CLIP"
PY="${ROOT}/.venv/bin/python"
SUBJECTS="${SUBJECTS:-1 2 3 4 5 6 7 8 9 10}"
MODES="${MODES:-eeg gallery self rand}"
SEED="${SEED:-2025}"

cd "${ROOT}"
mkdir -p outputs/slurm outputs/eval/adjudicate

export PYTHONNOUSERSITE=1
export HF_HOME=/project/peilab/why/cache/huggingface
export HF_HUB_CACHE=/project/peilab/why/cache/huggingface/hub
export TRANSFORMERS_CACHE=/project/peilab/why/cache/huggingface
export OPENCLIP_CACHE_DIR=/project/peilab/why/cache/open_clip
export TORCH_HOME=/project/peilab/why/cache/torch
export XDG_CACHE_HOME=/project/peilab/why/cache/xdg
export PIP_CACHE_DIR=/project/peilab/why/cache/pip
export TMPDIR="/tmp/clip-${SLURM_JOB_ID:-local}"
export XDG_CONFIG_HOME="${ROOT}/scratch/config"
export MPLCONFIGDIR="${ROOT}/scratch/mpl"
mkdir -p "${TMPDIR}" "${XDG_CONFIG_HOME}" "${MPLCONFIGDIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

MASTER="${ROOT}/outputs/slurm/adjudicate-${SLURM_JOB_ID}.log"
log() { echo "[$(date -Is)] $*" | tee -a "${MASTER}"; }

log "=== ADJUDICATE job=${SLURM_JOB_ID} host=$(hostname) ==="
log "subjects: ${SUBJECTS}  modes: ${MODES}  seed: ${SEED}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

"${PY}" -c "import sys,torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { log "[FATAL] CUDA unavailable on $(hostname)"; exit 1; }
"${PY}" -c "import samclip.calibration as c, inspect; \
assert 'src_ref_mode' in inspect.signature(c.rep_cloud_scores).parameters" \
  || { log "[FATAL] calibration.rep_cloud_scores has no src_ref_mode knob"; exit 1; }

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
  ck="${ROOT}/outputs/stage1/g3/${tag}_k20_seed${SEED}/last.pt"
  sm="${ROOT}/outputs/src_metric/g3/${tag}_seed${SEED}.npz"
  [[ -s "${ck}" ]] || { log "  [skip] ${tag}: no checkpoint"; continue; }
  [[ -s "${sm}" ]] || { log "  [skip] ${tag}: no src_metric template"; continue; }
  for mode in ${MODES}; do
    ev="${ROOT}/outputs/eval/adjudicate/${tag}_${mode}.json"
    if [[ -s "${ev}" ]] && [[ "${ev}" -nt "${ck}" ]]; then
      log "  [SKIP] ${tag}/${mode} (eval newer than checkpoint)"; continue
    fi
    log "--- ${tag}/${mode} ---"
    # `--fgw-src-metric` is still supplied for every mode because the >0-mix guard requires it;
    # `gallery`/`self`/`rand` ignore its CONTENTS and only use the slot it fills.
    run_step "eval ${tag}/${mode}" \
      "${ROOT}/outputs/slurm/adjudicate-${tag}-${mode}-${SLURM_JOB_ID}.log" \
      "${PY}" scripts/run_eval.py --ckpts "${ck}" --target-subject "${s}" --out "${ev}" \
        --reps --csls-k 10 --rho 0.1 \
        --recovery-operator fgw --recovery-tau 0.03 --recovery-alpha 0.75 \
        --rep-shrink 0.1 --fgw-sweep --rep-subsample 80 \
        --fgw-src-metric "${sm}" --fgw-src-mix "0.0,1.0" \
        --fgw-src-ref-mode "${mode}" "${REF[@]}"
  done
done

log "=== summary ==="
run_step "summarize" "${ROOT}/outputs/slurm/adjudicate-summary-${SLURM_JOB_ID}.log" \
  "${PY}" scripts/summarize_adjudication.py \
    --dir "${ROOT}/outputs/eval/adjudicate" \
    --out "${ROOT}/outputs/adjudication_summary.json"
tail -n 40 "${ROOT}/outputs/slurm/adjudicate-summary-${SLURM_JOB_ID}.log" | tee -a "${MASTER}" || true

if [[ ${#FAILED[@]} -gt 0 ]]; then
  log "FINISHED WITH FAILURES (${#FAILED[@]}): ${FAILED[*]}"
  exit 1
fi
log "ALL STEPS OK  (master log ${MASTER})"
