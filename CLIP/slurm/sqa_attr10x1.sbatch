#!/bin/bash
# =====================================================================================
# SQA ATTRIBUTION -- 10-fold LOSO x 1 seed, three mixing subgroups of GL(63), chained.
#
# THE QUESTION. The SQA group-action arm beat its augmentation-off twin by +2.10pp Top-1
# (8/10 folds, t=1.98) / +1.70pp Top-5 (t=3.39). But "add noise to the input" helps almost
# any encoder, so the win is only evidence for SQA IF it comes from the GROUP structure --
# an arbitrary INVERTIBLE channel recombination -- and not from generic per-channel
# perturbation (which the recipe can already do with `gain`). This job decides which.
#
# THE PANEL. Three arms, same folds, same seed, same code; only `augment.mixing_mode`
# differs, and every mode is rescaled to the SAME ||M - I||_F = 3.15 so the perturbation the
# encoder sees is matched by construction (E ||Mx-x||/||x|| = ||M-I||_F/sqrt(63) = 0.397):
#
#   dense  M = I + 3.15 * G/||G||_F                    full GL(63): mixes + rescales
#   diag   M = diag(exp(3.15 * g/||g||_2))             diagonal subgroup ONLY: no channel
#                                                      ever mixes -> the `gain` direction
#   orth   M = exp(3.15 * (S-S^T)/||S-S^T||_F)         orthogonal subgroup: mixes but is
#                                                      norm-preserving -> no amplitude effect
#
# THE READING, FIXED BEFORE THE NUMBERS ARRIVE:
#   diag reproduces dense's gain            -> generic perturbation; the group claim is not
#                                              what earned it, and SQA's level-2 claim fails.
#   diag ~ 0, dense/orth carry it           -> the OFF-DIAGONAL recombination is the source,
#                                              i.e. the group action proper.
#   orth ~ dense (both mix, only one can inflate amplitude)
#                                           -> the MIXING is the mechanism, not scale.
#
# BASELINE. The augmentation-off twin (`sqa_noaug`, already banked, Top-1 51.95) is reused
# rather than re-run: its augment block is INACTIVE, so it never enters the mixing code this
# commit changed, and it ran in the SAME job as the banked dense arm.
#
# COST. 30 trainings x ~3 min + 30 evals = ~1.7 GPU-hours. (The banked dense arm is re-run
# here so the panel is one code version, not two.)
#
# Submit:   sbatch slurm/sqa_attr10x1.sbatch
# Smoke:    SUBJECTS=8 sbatch slurm/sqa_attr10x1.sbatch
# =====================================================================================
#SBATCH --job-name=sqa-attr10x1
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --output=/project/peilab/why/CLIP/outputs/slurm/sqa-attr10x1-%j.out
#SBATCH --error=/project/peilab/why/CLIP/outputs/slurm/sqa-attr10x1-%j.err

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
MASTER="${LOG_DIR}/sqa-attr10x1-${SLURM_JOB_ID}.log"
OUT_ROOT="${ROOT}/outputs/stage1/sqa_attr"
EVAL_ROOT="${ROOT}/outputs/eval/sqa_attr"
REPS_ARGS=(--reps --csls-k 10 --rho 0.1 \
           --recovery-operator fgw --recovery-tau 0.03 --recovery-alpha 0.75 \
           --rep-shrink 0.1 --fgw-struct-fuse 16)

log() { echo "[$(date -Is)] $*" | tee -a "${MASTER}"; }

log "=== SQA attribution 10-fold x 1-seed job=${SLURM_JOB_ID} host=$(hostname) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
co_tenants=$(squeue -w "$(hostname)" -h -o '%i %u %j' | grep -v "${SLURM_JOB_ID}" || true)
if [[ -n "${co_tenants}" ]]; then log "shared-node co-tenants:"; echo "${co_tenants}" | tee -a "${MASTER}"; fi

"${PY}" -c "import sys,torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { log "[FATAL] CUDA unavailable on $(hostname)"; exit 1; }

# Print the panel and assert the design invariant that makes the numbers interpretable:
# every arm must hit the SAME ||M - I||_F, or the comparison is a magnitude artefact.
"${PY}" - <<'PY' || exit 1
import numpy as np
from samclip.utils import load_config
from samclip.data.augment import build_augment, unwrap, MIXING_MODES
arms = {
    "dense": "configs/sqa_loso_k20.yaml",
    "diag":  "configs/sqa_loso_k20_diag.yaml",
    "orth":  "configs/sqa_loso_k20_orth.yaml",
}
c = 63
xs = [np.random.default_rng(k).normal(size=(c, 250)).astype(np.float32) for k in range(40)]
seen = {}
for name, path in arms.items():
    cfg = load_config(path); a = unwrap(cfg)
    assert a["mixing_mode"] == name, (name, a["mixing_mode"])
    if a["mixing"] <= 0:
        raise SystemExit(f"[sqa-attr] FATAL: arm {name} has mixing <= 0 (that is the twin)")
    aug = build_augment(cfg)
    rels = [float(np.linalg.norm(aug(x, np.random.default_rng(1000 + i)) - x)
                  / np.linalg.norm(x)) for i, x in enumerate(xs)]
    seen[name] = float(np.mean(rels))
    print(f"[sqa-attr] {name:6s} mixing_mode={a['mixing_mode']:5s} mixing={a['mixing']} "
          f"||M-I||_F-matched -> E||Mx-x||/||x|| = {seen[name]:.3f}")
    fold_keys = {k: cfg.get(k) for k in ("target_subject", "source_subjects", "seed")}
    if any(v is not None for v in fold_keys.values()):
        raise SystemExit(f"[sqa-attr] FATAL: {path} must be FOLD-FREE, got {fold_keys}")
spread = max(seen.values()) - min(seen.values())
if spread > 0.08:
    raise SystemExit(f"[sqa-attr] FATAL: arms are not magnitude-matched (spread {spread:.3f})")
print(f"[sqa-attr] OK: perturbation magnitude matched to within {spread:.3f}")
PY

should_skip() {
  local out="$1"
  [[ -s "${out}" ]] || return 1
  local newest
  newest="$(find "${ROOT}/src" "${ROOT}/scripts" \
                 "${ROOT}/configs/sqa_loso_k20.yaml" \
                 "${ROOT}/configs/sqa_loso_k20_diag.yaml" \
                 "${ROOT}/configs/sqa_loso_k20_orth.yaml" \
                 "${ROOT}/slurm/sqa_attr10x1.sbatch" \
                 -type f -newer "${out}" 2>/dev/null | head -1 || true)"
  [[ -z "${newest}" ]]
}

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
SEED=2025
ARMS=("dense:${ROOT}/configs/sqa_loso_k20.yaml"
      "diag:${ROOT}/configs/sqa_loso_k20_diag.yaml"
      "orth:${ROOT}/configs/sqa_loso_k20_orth.yaml")

log "folds: ${SUBJECTS}   seed: ${SEED}   arms: dense,diag,orth   baseline: sqa_noaug (banked)"

for s in ${SUBJECTS}; do
  src=()
  for o in 1 2 3 4 5 6 7 8 9 10; do [[ "${o}" -ne "${s}" ]] && src+=("${o}"); done
  tag="$(printf 'sub%02d' "${s}")"
  if [[ "${s}" -eq 8 ]]; then REF_ARGS=(--ref-top1 22.00 --ref-top5 50.00)
  else                        REF_ARGS=(--ref-top1 unknown); fi

  for arm in "${ARMS[@]}"; do
    name_dir="${arm%%:*}"; cfg="${arm#*:}"
    out="${OUT_ROOT}/${name_dir}/${tag}_k20_seed${SEED}"
    ev="${EVAL_ROOT}/${name_dir}/${tag}_seed${SEED}.json"
    mkdir -p "${OUT_ROOT}/${name_dir}" "${EVAL_ROOT}/${name_dir}"
    log "--- ${name_dir}/${tag} : sources ${src[*]} ---"

    if should_skip "${out}/last.pt"; then
      log "  [SKIP] train -- up to date"
    else
      run_step "train ${name_dir}/${tag}" \
        "${LOG_DIR}/sqa-attr-train-${name_dir}-${tag}-${SLURM_JOB_ID}.log" \
        "${PY}" scripts/run_stage1.py --config "${cfg}" \
          --target-subject "${s}" --source-subjects "${src[@]}" \
          --out-dir "${out}" --seed "${SEED}" --mvnn train --channel-set all63
    fi

    if [[ -s "${ev}" ]] && [[ "${ev}" -nt "${out}/last.pt" ]] && should_skip "${ev}"; then
      log "  [SKIP] eval -- newer than its checkpoint"
    elif [[ ! -s "${out}/last.pt" ]]; then
      log "  [SKIP] eval -- no checkpoint (training failed above)"
    else
      run_step "eval ${name_dir}/${tag}" \
        "${LOG_DIR}/sqa-attr-eval-${name_dir}-${tag}-${SLURM_JOB_ID}.log" \
        "${PY}" scripts/run_eval.py --ckpts "${out}/last.pt" \
          --target-subject "${s}" --out "${ev}" \
          --recovery "${REPS_ARGS[@]}" "${REF_ARGS[@]}"
    fi
  done
done

log "=== summary ==="
run_step "summarize" "${LOG_DIR}/sqa-attr-summary-${SLURM_JOB_ID}.log" \
  "${PY}" scripts/summarize_sqa_attribution.py \
    --arms "dense@outputs/eval/sqa_attr/dense/*.json,diag@outputs/eval/sqa_attr/diag/*.json,orth@outputs/eval/sqa_attr/orth/*.json" \
    --baseline sqa_noaug --baseline-glob 'outputs/eval/sqa_noaug/*.json' \
    --out "${ROOT}/outputs/sqa_attribution.json"
tail -n 40 "${LOG_DIR}/sqa-attr-summary-${SLURM_JOB_ID}.log" | tee -a "${MASTER}" || true

if [[ ${#FAILED_STEPS[@]} -gt 0 ]]; then
  log "FINISHED WITH FAILURES (${#FAILED_STEPS[@]}): ${FAILED_STEPS[*]}"
  exit 1
fi
log "ALL STEPS OK  (master log ${MASTER})"
