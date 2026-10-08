#!/bin/bash
#SBATCH --job-name=v7b-ep-probe
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --output=/project/peilab/why/CLIP/outputs/slurm/v7b-ep-probe-%j.out
#SBATCH --error=/project/peilab/why/CLIP/outputs/slurm/v7b-ep-probe-%j.err

# v7b GATE -- does G3 + the source-only episode beat G3, on the fold it must?
#
# The episode is the ONE component of SCORE's published recipe our 45.53 stack never
# shipped (the other three -- recovery-aware training, CSLS/hubness-corrected matching,
# orthogonal recovery -- are already in G3). So the only question this job answers is
# whether adding it to G3 improves the deployed rung, measured on the SAME ladder and the
# same two folds the T2'' probe used, so the two candidate mechanisms are compared on
# identical ground rather than across two reports.
#
# The gate is pre-registered here, before the numbers exist:
#
#   fold 8 (anchor)   baseline best rung 40.5   -> PASS needs >= 40.5
#   fold 4 (worst)    baseline best rung 31.5   -> PASS needs >= 31.5
#
# and PASS on BOTH is required to ship a 30-run sweep. Fold 8 alone is not enough: the v7
# argument is that this mechanism should help where recovery fails, so a term that only
# moves the anchor has not earned the sweep.
#
# WHY fold 8 IS QUOTED AGAINST 40.5 AND NOT 42.0. The baseline was measured twice on this
# fold today under two different code revisions (archive 643149 = 40.5, re-run 643644 =
# 42.0), so the honest comparison is against the same-job `g3_baseline` arm of the T2''
# probe, also 40.5. Using the more favourable 42.0 would make the gate pass on a run that
# did not beat it.

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
export TMPDIR="/tmp/clip-${SLURM_JOB_ID:-local}"
export XDG_CONFIG_HOME="${ROOT}/scratch/config"
export MPLCONFIGDIR="${ROOT}/scratch/mpl"
mkdir -p "${TMPDIR}" "${XDG_CONFIG_HOME}" "${MPLCONFIGDIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"

LOG_DIR="${ROOT}/outputs/slurm"
MASTER="${LOG_DIR}/v7b-ep-probe-${SLURM_JOB_ID}.log"
OUT_ROOT="${ROOT}/outputs/stage1/v7b_ep"
EVAL_ROOT="${ROOT}/outputs/eval/v7b_ep"
mkdir -p "${OUT_ROOT}" "${EVAL_ROOT}"

log() { echo "[$(date -Is)] $*" | tee -a "${MASTER}"; }
log "=== v7b episode gate job=${SLURM_JOB_ID} host=$(hostname) date=$(date -Is) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

"${PY}" -c "import sys,torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { log "[FATAL] CUDA unavailable on $(hostname)"; exit 1; }

SEED=2025
for fold in 8 4; do
  src=()
  for o in 1 2 3 4 5 6 7 8 9 10; do [[ "${o}" -ne "${fold}" ]] && src+=("${o}"); done
  out="${OUT_ROOT}/sub$(printf '%02d' ${fold})_v7b"
  ev="${EVAL_ROOT}/sub$(printf '%02d' ${fold})_v7b.json"
  log "--- v7b fold ${fold} (sources ${src[*]}) ---"
  if [[ -s "${out}/last.pt" ]]; then
    log "  [SKIP] train"
  elif "${PY}" scripts/run_stage1.py --config "${ROOT}/configs/v7b_loso_k20.yaml" \
         --target-subject "${fold}" --source-subjects "${src[@]}" \
         --out-dir "${out}" --seed "${SEED}" --mvnn train --channel-set all63 \
         >"${LOG_DIR}/v7b-train-f${fold}-${SLURM_JOB_ID}.log" 2>&1; then
    log "     OK   train"
  else
    log "     FAILED train fold ${fold}"
    tail -30 "${LOG_DIR}/v7b-train-f${fold}-${SLURM_JOB_ID}.log" | tee -a "${MASTER}"
    continue
  fi
  "${PY}" scripts/run_eval.py --ckpts "${out}/last.pt" --target-subject "${fold}" \
      --out "${ev}" --recovery --reps --csls-k 10 --rho 0.1 --ref-top1 unknown \
      >"${LOG_DIR}/v7b-eval-f${fold}-${SLURM_JOB_ID}.log" 2>&1 \
    && log "     OK   eval" \
    || { log "     FAILED eval fold ${fold}"; tail -20 "${LOG_DIR}/v7b-eval-f${fold}-${SLURM_JOB_ID}.log" | tee -a "${MASTER}"; }
done

# --- the gate, evaluated against the SAME-JOB baseline where one exists, else the archive.
"${PY}" - <<'PY' 2>&1 | tee -a "${MASTER}"
import json, os

BASELINE = {8: 40.5, 4: 31.5}          # g3_baseline arm, job 643656 (matched code revision)
GATE = 0.0                             # must not LOSE on either fold

def read(f):
    ck = list(json.load(open(f))["checkpoints"].values())[0]
    r = ck["rows"]
    raw = r.get("raw cosine", {}).get("top1")
    best = max((v.get("top1"), k) for k, v in r.items()
               if isinstance(v, dict) and k.startswith("+"))
    return raw, best[0], r[best[1]].get("top5"), ck.get("smn_gate"), best[1]

print("\n============ v7b (G3 + source-only episode, no whitening) ============")
print(f"{'fold':<5} {'raw':>6} {'best':>6} {'bestT5':>7} {'gate':>6}  {'base':>6} {'delta':>7}  rung")
ok = True
for fold in (8, 4):
    p = f"outputs/eval/v7b_ep/sub{fold:02d}_v7b.json"
    if not os.path.exists(p):
        print(f"{fold:<5} (no report -- arm failed)")
        ok = False
        continue
    raw, b, t5, g, k = read(p)
    base = BASELINE[fold]
    d = b - base
    ok &= (d >= GATE)
    print(f"{fold:<5} {raw:>6.1f} {b:>6.1f} {t5:>7.1f} {g:>6.3f}  {base:>6.1f} {d:>+7.1f}  {k}")

print("\nVERDICT:", "PASS -- ship the 10-fold sweep" if ok else
      "FAIL -- do NOT ship; the episode does not beat G3 on the folds that matter")
# and the log line that says whether the episode actually ran
for fold in (8, 4):
    p = f"outputs/stage1/v7b_ep/sub{fold:02d}_v7b"
    if os.path.isdir(p):
        print(f"   fold {fold}: episode diagnostics are in the training log "
              f"(fires/abstains + landmark_rate)")
print("=" * 70)
PY

log "gate finished at $(date -Is)"
