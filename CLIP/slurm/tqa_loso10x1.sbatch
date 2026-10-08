#!/bin/bash
# =====================================================================================
# TQA -- theory-unified 10-fold LOSO x 1 seed, TWO arms, chained in ONE job.
#
# THE COMPOSITION THIS JOB IS BUILT ON. Two facts were measured SEPARATELY and never
# together:
#   * `configs/v8_loso_k20.yaml` (g3 + the soft-plan term) is the shipped best encoder:
#     55.10 Top-1 on the deployed cell at seed 2025 (54.83 over 3 seeds) vs SCORE's 53.23.
#   * the group-action augmentation `x -> (I + m G/||G||_F) x` is worth a PAIRED
#     +2.10pp Top-1 / +1.70pp Top-5 (8/10 folds, t=1.98 / t=3.39) on the g3 recipe (job 648728).
# Their composition is the single change with the best evidence behind it and lands ~57.
#
#   Arm A  tqa_v8       = v8 + group-action augmentation            (the SOTA arm)
#   Arm B  tqa_v8_gqf   = Arm A + GQF (row-whitening front-end)      (the architecture arm)
#
# Arm B differs from Arm A by ONE config key (`gqf`), so the GQF contribution is a paired
# difference. If it loses, Arm A is untouched and still carries the claim.
#
# EVAL IS THE DEPLOYED CELL. `+ T2 R=80,a=0.75,t=0.03,fuse=16` -- the FGW structure
# ensemble on the repetition cloud, i.e. the exact row that holds 55.10 for v8 and 53.23's
# comparison. `--recovery-operator fgw --recovery-alpha 0.75 --recovery-tau 0.03
# --fgw-struct-fuse 16` reproduces it; `probe_ttga.py` verified that call sequence is
# bit-identical to the banked `outputs/eval/v11_fuse/` cell.
#
# COST / RESUME. ~3-4 min per 50-epoch run on this node + seconds per eval; 20 runs is
# ~1.5 GPU-hours. Every (fold, arm) is skip-guarded on its own outputs (AGENTS.md 3.5).
#
# Submit:   sbatch slurm/tqa_loso10x1.sbatch
# Smoke:    SUBJECTS=1 sbatch slurm/tqa_loso10x1.sbatch
# =====================================================================================
#SBATCH --job-name=tqa-loso10x1
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=10:00:00
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --output=/project/peilab/why/CLIP/outputs/slurm/tqa-loso10x1-%j.out
#SBATCH --error=/project/peilab/why/CLIP/outputs/slurm/tqa-loso10x1-%j.err

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
MASTER="${LOG_DIR}/tqa-loso10x1-${SLURM_JOB_ID}.log"
OUT_ROOT="${ROOT}/outputs/stage1"
EVAL_ROOT="${ROOT}/outputs/eval"
REPS_ARGS=(--reps --csls-k 10 --rho 0.1 \
           --recovery-operator fgw --recovery-tau 0.03 --recovery-alpha 0.75 \
           --rep-shrink 0.1 --fgw-struct-fuse 16)

log() { echo "[$(date -Is)] $*" | tee -a "${MASTER}"; }

log "=== TQA 10-fold x 1-seed (2 arms) job=${SLURM_JOB_ID} host=$(hostname) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
co_tenants=$(squeue -w "$(hostname)" -h -o '%i %u %j' | grep -v "${SLURM_JOB_ID}" || true)
if [[ -n "${co_tenants}" ]]; then log "shared-node co-tenants:"; echo "${co_tenants}" | tee -a "${MASTER}"; fi

"${PY}" -c "import sys,torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { log "[FATAL] CUDA unavailable on $(hostname)"; exit 1; }

# Pre-flight: the arms must differ ONLY in `gqf`, and both must be fold-free and augmented.
"${PY}" - "${ROOT}/configs/tqa_v8_loso_k20.yaml" "${ROOT}/configs/tqa_v8_gqf_loso_k20.yaml" <<'PY' || exit 1
import sys
from samclip.utils import load_config
from samclip.data.augment import unwrap
from samclip.models.samclip import build_model
import torch
ca, cb = load_config(sys.argv[1]), load_config(sys.argv[2])
for p, c in ((sys.argv[1], ca), (sys.argv[2], cb)):
    a = unwrap(c)
    print(f"[tqa] {p}: soft_plan={bool((c.get('soft_plan') or {}).get('enabled'))} "
          f"gqf={c.get('gqf')} augment={a['mixing']}/{a['mixing_mode']}")
    fk = {k: c.get(k) for k in ("target_subject", "source_subjects", "seed")}
    if any(v is not None for v in fk.values()):
        sys.exit(f"[tqa] FATAL: {p} must be FOLD-FREE, got {fk}")
    if not a["enabled"] or a["mixing"] <= 0:
        sys.exit(f"[tqa] FATAL: {p} is not group-augmented -- that is the point of this job")
if bool(ca.get("soft_plan", {}).get("enabled")) is False:
    sys.exit("[tqa] FATAL: base v8 lost its soft_plan term")
diff = {k for k in set(ca) | set(cb) if ca.get(k) != cb.get(k)}
if diff != {"gqf"}:
    sys.exit(f"[tqa] FATAL: arms must differ only in `gqf`, got {sorted(diff)}")
for p, c in ((sys.argv[1], ca), (sys.argv[2], cb)):
    m = build_model(c, 5, 3200)
    with torch.no_grad():
        out = m(torch.randn(4, 63, 250), torch.randn(4, 5, 3200))
    assert torch.isfinite(out["z_eeg"]).all(), p
    print(f"[tqa] {p}: model.gqf={m.gqf} forward finite OK")
print("[tqa] OK: arms differ only in `gqf`")
PY

should_skip() {
  local out="$1"
  [[ -s "${out}" ]] || return 1
  local newest
  newest="$(find "${ROOT}/src" "${ROOT}/scripts" \
                 "${ROOT}/configs/tqa_v8_loso_k20.yaml" \
                 "${ROOT}/configs/tqa_v8_gqf_loso_k20.yaml" \
                 "${ROOT}/slurm/tqa_loso10x1.sbatch" \
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
ARMS=("tqa_v8:${ROOT}/configs/tqa_v8_loso_k20.yaml"
      "tqa_v8_gqf:${ROOT}/configs/tqa_v8_gqf_loso_k20.yaml")

log "folds: ${SUBJECTS}   seed: ${SEED}   arms: tqa_v8 (SOTA), tqa_v8_gqf (arch)"

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
        "${LOG_DIR}/tqa-train-${name_dir}-${tag}-${SLURM_JOB_ID}.log" \
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
        "${LOG_DIR}/tqa-eval-${name_dir}-${tag}-${SLURM_JOB_ID}.log" \
        "${PY}" scripts/run_eval.py --ckpts "${out}/last.pt" \
          --target-subject "${s}" --out "${ev}" \
          --recovery "${REPS_ARGS[@]}" "${REF_ARGS[@]}"
    fi
  done
done

log "=== summary ==="
run_step "summarize" "${LOG_DIR}/tqa-summary-${SLURM_JOB_ID}.log" \
  "${PY}" scripts/summarize_tqa.py \
    --arms "tqa_v8@outputs/eval/tqa_v8/*.json,tqa_v8_gqf@outputs/eval/tqa_v8_gqf/*.json" \
    --baseline v8 --baseline-glob 'outputs/eval/v11_fuse/*_seed2025.json' \
    --out "${ROOT}/outputs/tqa_summary.json"
tail -n 60 "${LOG_DIR}/tqa-summary-${SLURM_JOB_ID}.log" | tee -a "${MASTER}" || true

if [[ ${#FAILED_STEPS[@]} -gt 0 ]]; then
  log "FINISHED WITH FAILURES (${#FAILED_STEPS[@]}): ${FAILED_STEPS[*]}"
  exit 1
fi
log "ALL STEPS OK  (master log ${MASTER})"
