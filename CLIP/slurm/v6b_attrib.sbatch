#!/bin/bash
# =====================================================================================
# v6b ATTRIBUTION PIPELINE -- one job, three single-switch arms, chained sequentially.
#
# WHY THIS EXISTS. The first chained pipeline (`v6_chained_pipeline.sbatch`, job 642512)
# ran clean but the faithful-RAT arm regressed by 12.5pp against the recorded best on this
# fold (21.50 vs 34.00 at the +CSLS rung), and -9.00pp of that could not be charged to any
# measured difference. A big unexplained residual in a fresh arm is either a real mechanism
# or a bug, and the two are separated by flipping ONE switch at a time -- not by re-reading
# the code. Hence three arms, each identical to an already-run arm except for one key:
#
#   A1  v6-rat-nowhiten        V0 minus `episode_whiten`        -> the -9.00pp residual
#   A2  v6-rat-k20             V0 with the loss's `csls_k: 20`  -> the -3.00pp k change
#   A3  v6-multiroute-nowhiten V1 minus `episode_whiten`        -> is fusion dead, or was
#                                                                 it trained in a damaged frame?
#
# Each arm is ~5 min of H800 (measured on 642512), so the whole sweep is shorter than the
# cache build in the previous job. Both extra routes of the multi-route arm came back near
# chance (gamma 7.00, beta 3.00 against alpha 24.00 at the deployed rung), so a fusion claim
# made from that run would be made in the wrong frame.
#
# Skip-guarded per arm and per eval, same as the v6 pipeline: re-running after adding an
# arm costs only the new arm.
#
#     sbatch slurm/v6b_attrib.sbatch
# =====================================================================================
#SBATCH --job-name=v6b-attrib
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --output=/project/peilab/why/CLIP/outputs/slurm/v6b-attrib-%j.out
#SBATCH --error=/project/peilab/why/CLIP/outputs/slurm/v6b-attrib-%j.err

set -euo pipefail

ROOT="/project/peilab/why/CLIP"
PY="${ROOT}/.venv/bin/python"
TARGET_SUBJECT="${TARGET_SUBJECT:-8}"
mkdir -p "${ROOT}/outputs/slurm"

export PYTHONNOUSERSITE=1
export HF_HOME=/project/peilab/why/cache/huggingface
export HF_HUB_CACHE=/project/peilab/why/cache/huggingface/hub
export TRANSFORMERS_CACHE=/project/peilab/why/cache/huggingface
export OPENCLIP_CACHE_DIR=/project/peilab/why/cache/open_clip
export TORCH_HOME=/project/peilab/why/cache/torch
export XDG_CACHE_HOME=/project/peilab/why/cache/xdg
export PIP_CACHE_DIR=/project/peilab/why/cache/pip
export TMPDIR="/tmp/clip-${SLURM_JOB_ID:-local}"   # node-local; see AGENTS.md §2.4
export XDG_CONFIG_HOME="${ROOT}/scratch/config"
export MPLCONFIGDIR="${ROOT}/scratch/mpl"
mkdir -p "${TMPDIR}" "${XDG_CONFIG_HOME}" "${MPLCONFIGDIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-12}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"

LOG_DIR="${ROOT}/outputs/slurm"
MASTER="${LOG_DIR}/v6b-attrib-${SLURM_JOB_ID}.log"
log() { echo "[$(date -Is)] $*" | tee -a "${MASTER}"; }

log "=== v6b attribution job=${SLURM_JOB_ID} host=$(hostname) ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
co_tenants=$(squeue -w "$(hostname)" -h -o '%i %u %j' | grep -v "${SLURM_JOB_ID}" || true)
if [[ -n "${co_tenants}" ]]; then log "co-tenants:"; echo "${co_tenants}" | tee -a "${MASTER}"; fi

"${PY}" -c "import sys,torch; sys.exit(0 if torch.cuda.is_available() else 1)" \
  || { log "[FATAL] CUDA unavailable on $(hostname)"; exit 1; }

should_skip() {                    # should_skip <out> <cfg>
  local out="$1" cfg="$2" newest=""
  [[ -s "${out}" ]] || return 1
  newest="$(find "${ROOT}/src" "${ROOT}/scripts" "${cfg}" \
            "${ROOT}/slurm/v6b_attrib.sbatch" -type f -newer "${out}" 2>/dev/null \
            | head -1 || true)"
  [[ -z "${newest}" ]]
}

FAILED_STEPS=()
run_step() {                       # run_step <name> <logfile> <cmd...>
  local name="$1" lf="$2"; shift 2
  log "--- STEP ${name} : $* ---"
  local rc=0
  "$@" >"${lf}" 2>&1 || rc=$?      # NOT `if ...`: an `if` returns 0 when the cmd fails
  if [[ ${rc} -eq 0 ]]; then log "    STEP ${name} OK  (log ${lf})"; return 0; fi
  log "    STEP ${name} FAILED rc=${rc}  (log ${lf})"
  tail -n 40 "${lf}" | tee -a "${MASTER}" || true
  FAILED_STEPS+=("${name}")
  return 0
}

# --- gates first: a number produced on a code path that fails its own properties cannot
# --- be attributed to the treatment (same rule as the v6 pipeline).
step_gates() { "${PY}" scripts/smoke_test.py && "${PY}" scripts/smoke_v6.py; }
if should_skip "${LOG_DIR}/v6b-gates.ok" "${ROOT}/src/samclip/train.py"; then
  log "--- STEP gates : [SKIP] ---"
elif run_step gates "${LOG_DIR}/v6b-step1-gates-${SLURM_JOB_ID}.log" step_gates; then
  touch "${LOG_DIR}/v6b-gates.ok"
else
  log "[FATAL] gates failed; refusing to train on an unverified code path."; exit 1
fi

# =====================================================================================
# The arms. `ARMS` is the single list that drives training, eval and the summary, so an
# arm cannot be trained without being evaluated or evaluated with a stale config.
# =====================================================================================
ARMS=(
  "v6-rat-nowhiten:configs/loso_sub08_v6_rat_nowhiten.yaml:outputs/stage1/v6-rat-nowhiten"
  "v6-rat-k20:configs/loso_sub08_v6_rat_k20.yaml:outputs/stage1/v6-rat-k20"
  "v6-multiroute-nowhiten:configs/loso_sub08_v6_multiroute_nowhiten.yaml:outputs/stage1/v6-multiroute-nowhiten"
)

CKPTS=()
for spec in "${ARMS[@]}"; do
  IFS=':' read -r name cfg out <<< "${spec}"
  out_abs="${ROOT}/${out}"
  step_train() {
    "${PY}" scripts/run_stage1.py --config "${ROOT}/${cfg}" \
        --target-subject "${TARGET_SUBJECT}" --out-dir "${out_abs}" \
        --seed 2025 --mvnn train --channel-set all63
  }
  if should_skip "${out_abs}/last.pt" "${ROOT}/${cfg}"; then
    log "--- STEP ${name} : [SKIP] ${out}/last.pt is up to date ---"
  else
    run_step "${name}" "${LOG_DIR}/v6b-${name}-${SLURM_JOB_ID}.log" step_train || true
  fi
  [[ -s "${out_abs}/last.pt" ]] && CKPTS+=("${out_abs}/last.pt")
done

# --- always include the two arms from the first pipeline in the same table, so the
# --- attribution is read off ONE report and not across two JSON files by eye.
for c in "outputs/stage1/v6-rat/last.pt" "outputs/stage1/v6-multiroute/last.pt"; do
  [[ -s "${ROOT}/${c}" ]] && CKPTS+=("${ROOT}/${c}")
done

if [[ ${#CKPTS[@]} -eq 0 ]]; then log "[FATAL] nothing to evaluate."; exit 1; fi

EVAL_OUT="${ROOT}/outputs/eval/v6b-attrib-sub-$(printf '%02d' "${TARGET_SUBJECT}").json"
step_eval() {
  # `--recovery` is REQUIRED for this report and was missing when the script was written:
  # the attribution ledger's decision column is `Δrec` against the `+whiten+CSLS+recovery`
  # rung (35.50 on this fold), and without the flag that rung is absent from the JSON, so
  # every arm would print "-" in the one column the residual is measured in. `--reps` is
  # deliberately NOT passed: the v6 arms are multi-route, and a rep-cloud failure there
  # would abort the attribution the job exists to produce.
  "${PY}" scripts/run_eval.py --ckpts "${CKPTS[@]}" --target-subject "${TARGET_SUBJECT}" \
      --out "${EVAL_OUT}" --recovery --csls-k 10 --rho 0.1
}
run_step eval "${LOG_DIR}/v6b-eval-${SLURM_JOB_ID}.log" step_eval || true

log "=== summary (the attribution ledger) ==="
cat >> "${MASTER}" <<'NOTE'

  measured earlier on this fold, +CSLS rung, seed 2025:
    v5-a1-k20    (k=20, no episode)                 34.00
    v5-ep2       (k=20, episode = CSLS+recovery)    33.50
    v5-rat-csls  (k=10, no episode)                 31.00
    + recovery on v5-a1-k20 (+whiten+CSLS+recovery) 35.50   <- the number to beat

  V0 v6-rat   (k=10, full-stack episode)            21.50   <- the -9.00pp residual
  V1 multiroute (same frame, 3 routes fused)        23.50
NOTE

if [[ -s "${EVAL_OUT}" ]]; then
  "${PY}" - "${EVAL_OUT}" <<'PY' | tee -a "${MASTER}"
import json, sys
r = json.load(open(sys.argv[1]))
ref = r["reference"]["top1"]
print(f"\n{'arm':<40} {'raw':>7} {'+CSLS':>7} {'+w+CSLS':>8} {'+w+CSLS+rec':>12} {'Δrec':>7}")
print("-" * 88)
for tag, res in r["checkpoints"].items():
    rows = res["rows"]
    def g(n):
        m = rows.get(n)
        return f"{m['top1']:.2f}" if isinstance(m, dict) else "-"
    rec = rows.get("+ whiten + CSLS + recovery") or rows.get("+ whiten + CSLS")
    d = f"{rec['top1'] - ref:+.2f}" if isinstance(rec, dict) else "-"
    print(f"{tag:<40} {g('raw cosine'):>7} {g('+ CSLS'):>7} "
          f"{g('+ whiten + CSLS'):>8} {g('+ whiten + CSLS + recovery'):>12} {d:>7}")
print("-" * 88)
print(f"reference (SAMGA official, this fold): {ref:.2f} / {r['reference']['top5']:.2f}")
PY
else
  log "[WARN] no eval report at ${EVAL_OUT}"
fi

if [[ ${#FAILED_STEPS[@]} -gt 0 ]]; then
  log "FINISHED WITH FAILURES: ${FAILED_STEPS[*]}"; exit 1
fi
log "ALL STEPS OK  (master log ${MASTER})"
