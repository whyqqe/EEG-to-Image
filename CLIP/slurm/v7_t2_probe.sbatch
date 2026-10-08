#!/bin/bash
#SBATCH --job-name=v7-t2-probe
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --output=/project/peilab/why/CLIP/outputs/slurm/v7-t2-probe-%j.out
#SBATCH --error=/project/peilab/why/CLIP/outputs/slurm/v7-t2-probe-%j.err

# T2' WEIGHT / BASELINE-INTEGRITY PROBE -- fold 8, seed 2025, three arms.
#
# WHY THIS EXISTS. The first v7 run (T2' weight 0.5) scored WORSE than the banked G3 run on
# the SAME fold and seed: raw 22.0 vs 25.5, best deployment rung 30.0 vs 40.5. Before that
# is read as "the mechanism does not work" it must be separated from two boring causes, and
# they are separable with three runs:
#
#   arm A  g3_loso_k20.yaml            the BASELINE, re-run under the current `src/`.
#                                      If this does not reproduce ~25.5 raw, then the
#                                      additive v7 code changed the baseline path and the
#                                      whole comparison is invalid -- a bug, not a result.
#   arm B  probe_t2_w0p1.yaml          weight 0.1. Does a weaker term stop regressing?
#   arm C  probe_t2_w0p25.yaml         weight 0.25. Spans the range between B and 0.5.
#
# The arms share fold, seed, epochs, MVNN and channel set, so they differ only in
# `concept.reps_weight`. Same skip-guard discipline as the sweep: each arm is independently
# resumable.

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
MASTER="${LOG_DIR}/v7-t2-probe-${SLURM_JOB_ID}.log"
OUT_ROOT="${ROOT}/outputs/stage1/v7_probe"
EVAL_ROOT="${ROOT}/outputs/eval/v7_probe"
mkdir -p "${OUT_ROOT}" "${EVAL_ROOT}"

log() { echo "[$(date -Is)] $*" | tee -a "${MASTER}"; }
log "=== v7 T2' probe job=${SLURM_JOB_ID} host=$(hostname) date=$(date -Is) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

"${PY}" -c "import sys,torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { log "[FATAL] CUDA unavailable on $(hostname)"; exit 1; }

SUBJECT=8
SEED=2025
src=()
for o in 1 2 3 4 5 6 7 8 9 10; do [[ "${o}" -ne "${SUBJECT}" ]] && src+=("${o}"); done

run_arm() {                # run_arm <tag> <config>
  local tag="$1" cfg="$2"
  local out="${OUT_ROOT}/${tag}"
  local ev="${EVAL_ROOT}/${tag}.json"
  log "--- arm ${tag} (${cfg}) : sources ${src[*]} ---"
  if [[ -s "${out}/last.pt" ]]; then
    log "  [SKIP] train -- ${out}/last.pt exists"
  else
    "${PY}" scripts/run_stage1.py --config "${cfg}" \
      --target-subject "${SUBJECT}" --source-subjects "${src[@]}" \
      --out-dir "${out}" --seed "${SEED}" \
      --mvnn train --channel-set all63 \
      >"${LOG_DIR}/v7-probe-${tag}-${SLURM_JOB_ID}.log" 2>&1 \
      || { log "  FAILED train ${tag}"; tail -20 "${LOG_DIR}/v7-probe-${tag}-${SLURM_JOB_ID}.log" | tee -a "${MASTER}"; return 0; }
    log "     OK   train ${tag}"
  fi
  "${PY}" scripts/run_eval.py --ckpts "${out}/last.pt" \
    --target-subject "${SUBJECT}" --out "${ev}" \
    --recovery --reps --csls-k 10 --rho 0.1 --ref-top1 22.00 --ref-top5 50.00 \
    >"${LOG_DIR}/v7-probe-eval-${tag}-${SLURM_JOB_ID}.log" 2>&1 \
    || { log "  FAILED eval ${tag}"; tail -20 "${LOG_DIR}/v7-probe-eval-${tag}-${SLURM_JOB_ID}.log" | tee -a "${MASTER}"; return 0; }
  log "     OK   eval  ${tag}"
}

run_arm "g3_repro"       "${ROOT}/configs/g3_loso_k20.yaml"
run_arm "t2_w0p1"        "${ROOT}/configs/probe_t2_w0p1.yaml"
run_arm "t2_w0p25"       "${ROOT}/configs/probe_t2_w0p25.yaml"

# ------------------------------------------------------------------- verdict table
"${PY}" - <<'PY' | tee -a "${MASTER}"
import glob, json, os

BANKED = {"G3 banked (job 643149)": (25.5, 40.5, 78.0)}

rows = dict(BANKED)
for f in sorted(glob.glob("outputs/eval/v7_probe/*.json")):
    tag = os.path.basename(f)[:-5]
    ck = list(json.load(open(f))["checkpoints"].values())[0]
    r = ck["rows"]
    raw = r.get("raw cosine", {}).get("top1")
    best = max((v.get("top1"), k) for k, v in r.items()
               if isinstance(v, dict) and k.startswith("+"))
    best5 = r[best[1]].get("top5")
    rows[tag] = (raw, best[0], best5)

print("\n================ T2' weight probe -- sub-08 / seed 2025 ================")
print(f"{'arm':<26} {'raw T1':>7} {'best T1':>8} {'best T5':>8}")
for k, v in rows.items():
    print(f"{k:<26} {v[0]:>7.1f} {v[1]:>8.1f} {v[2]:>8.1f}")
print("=" * 74)
PY

log "probe finished at $(date -Is)"
