#!/bin/bash
#SBATCH --job-name=v7-t2b-probe
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --output=/project/peilab/why/CLIP/outputs/slurm/v7-t2b-probe-%j.out
#SBATCH --error=/project/peilab/why/CLIP/outputs/slurm/v7-t2b-probe-%j.err

# T2'' ATTRIBUTION PROBE -- the fixed cross-trial term, 4 arms x 2 folds.
#
# WHAT WENT WRONG THE FIRST TIME, AND WHAT THIS SEPARATES. v7 v1 (`reps_group: row`, w=0.5)
# lost to G3 on sub-08/seed2025: best deployment rung 30.0 vs 42.0, raw 22.0 vs 27.0, with a
# monotone dose-response (gate 0.498 -> 0.203 -> 0.000 for w 0 -> 0.1 -> 0.5). Two candidate
# causes, and they need different fixes:
#
#   (1) `row` grouping is ANTI-T1. The sampler tiles 9 subjects per stimulus, so a row's
#       negatives include other subjects' views of the SAME image; the term pushes apart what
#       `cross_subject_loss` pulls together. Fix: group by STIMULUS.
#   (2) The term DISPLACES the SMN's scale gate (both remove within-concept scatter). The
#       gate is worth ~5pp raw, so winning that competition is a net loss. Fix: cut the
#       term's gradient to the gate only (`reps_detach_gate`).
#
# The 4 arms are a 2x2 over exactly those two switches at one weight, so each fix is
# attributed rather than assumed:
#
#   arm                    grouping    gate
#   g3_baseline            -           -      (matched in-job baseline)
#   stimulus_gatefree      stimulus    free   -> isolates fix (1) alone
#   stimulus_gatelock      stimulus    locked -> the shipped config
#   row_gatelock           row         locked -> isolates fix (2) alone
#
# FOLDS. 8 AND 4. Fold 8 is the anchor every number in this project was developed on; fold 4
# is the WORST fold of the G3 sweep (32.17 vs the 45.53 mean). A term that helps the anchor
# and nothing else is not a candidate for a 10-fold sweep, and the whole point of the v7
# argument is that it should help the folds where recovery fails -- so both are measured, and
# fold 4 is the one that matters for the headline.

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
MASTER="${LOG_DIR}/v7-t2b-probe-${SLURM_JOB_ID}.log"
OUT_ROOT="${ROOT}/outputs/stage1/v7_t2b"
EVAL_ROOT="${ROOT}/outputs/eval/v7_t2b"
mkdir -p "${OUT_ROOT}" "${EVAL_ROOT}"

log() { echo "[$(date -Is)] $*" | tee -a "${MASTER}"; }
log "=== v7 T2'' attribution probe job=${SLURM_JOB_ID} host=$(hostname) date=$(date -Is) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

"${PY}" -c "import sys,torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { log "[FATAL] CUDA unavailable on $(hostname)"; exit 1; }

SEED=2025
FOLDS="${FOLDS:-8 4}"
# tag:config pairs. `row_gatelock` also pins `reps_n_rows` because the row-grouped path
# ignores `reps_n_stim`/`reps_subj_sel` (there is no stimulus design to balance).
ARMS="g3_baseline:${ROOT}/configs/g3_loso_k20.yaml
stimulus_gatefree:${ROOT}/configs/probe_t2_gatefree.yaml
stimulus_gatelock:${ROOT}/configs/v7_loso_k20.yaml
row_gatelock:${ROOT}/configs/probe_t2_row_lock.yaml"

for fold in ${FOLDS}; do
  src=()
  for o in 1 2 3 4 5 6 7 8 9 10; do [[ "${o}" -ne "${fold}" ]] && src+=("${o}"); done
  while IFS=: read -r tag cfg; do
    [[ -z "${tag}" ]] && continue
    out="${OUT_ROOT}/sub$(printf '%02d' ${fold})_${tag}"
    ev="${EVAL_ROOT}/sub$(printf '%02d' ${fold})_${tag}.json"
    log "--- ${tag} fold ${fold} (sources ${src[*]}) ---"
    if [[ -s "${out}/last.pt" ]]; then
      log "  [SKIP] train"
    else
      if "${PY}" scripts/run_stage1.py --config "${cfg}" \
           --target-subject "${fold}" --source-subjects "${src[@]}" \
           --out-dir "${out}" --seed "${SEED}" \
           --mvnn train --channel-set all63 \
           >"${LOG_DIR}/v7-t2b-${tag}-f${fold}-${SLURM_JOB_ID}.log" 2>&1; then
        log "     OK   train"
      else
        log "     FAILED train ${tag} fold ${fold}"
        tail -25 "${LOG_DIR}/v7-t2b-${tag}-f${fold}-${SLURM_JOB_ID}.log" | tee -a "${MASTER}"
        continue
      fi
    fi
    if [[ "${fold}" -eq 8 ]]; then REF=(--ref-top1 22.00 --ref-top5 50.00)
    else                         REF=(--ref-top1 unknown); fi
    if "${PY}" scripts/run_eval.py --ckpts "${out}/last.pt" \
         --target-subject "${fold}" --out "${ev}" \
         --recovery --reps --csls-k 10 --rho 0.1 "${REF[@]}" \
         >"${LOG_DIR}/v7-t2b-eval-${tag}-f${fold}-${SLURM_JOB_ID}.log" 2>&1; then
      log "     OK   eval"
    else
      log "     FAILED eval ${tag} fold ${fold}"
      tail -25 "${LOG_DIR}/v7-t2b-eval-${tag}-f${fold}-${SLURM_JOB_ID}.log" | tee -a "${MASTER}"
    fi
  done <<< "${ARMS}"
done

# ------------------------------------------------------------------- verdict table
"${PY}" - <<'PY' 2>&1 | tee -a "${MASTER}"
import glob, json, os

# G3's 10-fold-measured headline for the two folds, as the external reference point.
BANKED = {8: (25.5, 40.5), 4: None}

def read(f):
    ck = list(json.load(open(f))["checkpoints"].values())[0]
    r = ck["rows"]
    raw = r.get("raw cosine", {}).get("top1")
    best = max((v.get("top1"), k) for k, v in r.items()
               if isinstance(v, dict) and k.startswith("+"))
    return raw, best[0], r[best[1]].get("top5"), ck.get("smn_gate"), best[1]

print("\n================ T2'' attribution -- seed 2025 ================")
print(f"{'fold':<5} {'arm':<18} {'raw':>6} {'best':>6} {'bestT5':>7} {'gate':>6}  rung")
for f in sorted(glob.glob("outputs/eval/v7_t2b/*.json")):
    base = os.path.basename(f)[:-5]
    fold = int(base[3:5])
    tag = base[6:]
    raw, b, t5, g, k = read(f)
    print(f"{fold:<5} {tag:<18} {raw:>6.1f} {b:>6.1f} {t5:>7.1f} "
          f"{(g if g is not None else float('nan')):>6.3f}  {k}")

# gate trajectories: the SECOND readout, and the one that says whether the fixes held
print("\n---- final SMN gate per arm (G3 ~0.50 is the healthy value) ----")
for arm in ("g3_baseline", "stimulus_gatefree", "stimulus_gatelock", "row_gatelock"):
    for fold in (8, 4):
        p = f"outputs/eval/v7_t2b/sub{fold:02d}_{arm}.json"
        if os.path.exists(p):
            g = list(json.load(open(p))["checkpoints"].values())[0].get("smn_gate")
            print(f"   fold {fold}  {arm:<18} gate = {g}")
print("=" * 68)
PY

log "probe finished at $(date -Is)"
