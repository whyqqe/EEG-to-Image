#!/usr/bin/env bash
# =============================================================================
# submit_pipeline.sh -- submit the neural-visibility pipeline to Slurm
# =============================================================================
# Design rule for this project: ONE JOB PER LOGICAL STEP.
# An earlier version submitted five GPU jobs (one per encoder) plus four CPU gate
# jobs, which flooded the queue for work that is one step each.  Do not split an
# embarrassingly sequential workload across queue entries.
#
#                        ┌───────────────────────┐
#                        │  nv-gates  (CPU)      │  1 job, run_all_gates.sbatch
#                        │  0 self-check         │
#                        │  1 axis verification  │
#                        │  2 GATE 1 synthetic   │
#                        │  3 GATE 2 controls    │
#                        │  4 A2/A3 bounds       │
#                        └───────────┬───────────┘
#                                    │ afterok (only if every gate passed)
#                                    ▼
#                        ┌───────────────────────┐
#                        │  nv-extract-all (GPU) │  1 job, extract_all.sbatch
#                        │  5 frozen encoders    │
#                        │  sequential           │
#                        └───────────────────────┘
#
# The dependency is not just tidiness: if a gate fails, the pipeline is void and
# extracting features would burn GPU time to validate it.  Letting Slurm enforce
# the ordering removes the chance of doing that by hand.
#
# Usage:
#   ./scripts/submit_pipeline.sh gates    # submit the CPU gate job only
#   ./scripts/submit_pipeline.sh extract  # submit the GPU job, gated on gates
#   ./scripts/submit_pipeline.sh all      # both, chained  (default)
#   ./scripts/submit_pipeline.sh status   # queue + gate results
#   ./scripts/submit_pipeline.sh cancel   # cancel this project's jobs only
# =============================================================================
set -euo pipefail

ROOT=/project/peilab/why/eeg-retrieval/alignment
SLURM_D="${ROOT}/slurm"
OUT_D="${ROOT}/outputs"
LOGD="${OUT_D}/slurm"
PY=${PY:-/project/peilab/why/eeg-brainit/.venv/bin/python}
mkdir -p "${LOGD}" "${OUT_D}/pipeline"

MODE="${1:-all}"
GATE_IDF="${OUT_D}/pipeline/gate_job_id.txt"
EXTRACT_IDF="${OUT_D}/pipeline/extract_job_id.txt"

log() { printf '[submit] %s\n' "$*"; }

submit_gates() {
  local jid
  jid=$(sbatch --parsable "${SLURM_D}/run_all_gates.sbatch")
  echo "$jid" > "${GATE_IDF}"
  log "gate job (CPU, all stages) job=${jid}"
  echo "$jid"
}

submit_extract() {
  local dep="${1:-}"
  local jid
  if [[ -n "$dep" ]]; then
    jid=$(sbatch --parsable --dependency="afterok:${dep}" "${SLURM_D}/extract_all.sbatch")
    log "extract job (GPU, 5 encoders) job=${jid}  afterok:${dep}"
  else
    jid=$(sbatch --parsable "${SLURM_D}/extract_all.sbatch")
    log "extract job (GPU, 5 encoders) job=${jid}  (no dependency)"
  fi
  echo "$jid" > "${EXTRACT_IDF}"
  echo "$jid"
}

case "$MODE" in
status)
  squeue -u "$USER" -o "%.10i %.20j %.10T %.11M %.6D %R"
  echo
  [[ -f "${GATE_IDF}" ]]    && log "gate job id    : $(cat "${GATE_IDF}")"
  [[ -f "${EXTRACT_IDF}" ]] && log "extract job id : $(cat "${EXTRACT_IDF}")"
  echo
  log "gate results:"
  [[ -f "${OUT_D}/pipeline/gate_status.json" ]] \
    && cat "${OUT_D}/pipeline/gate_status.json" \
    || echo "  (not finished yet)"
  [[ -f "${OUT_D}/pipeline/extract_status.json" ]] && {
    echo; log "extraction:"; cat "${OUT_D}/pipeline/extract_status.json"; }
  exit 0
  ;;

cancel)
  # scoped to this project only -- never scancel another workload by accident
  log "cancelling nv-gates / nv-extract-all for ${USER}"
  scancel -u "$USER" -n nv-gates     2>/dev/null || true
  scancel -u "$USER" -n nv-extract-all 2>/dev/null || true
  sleep 2
  squeue -u "$USER" -o "%.10i %.20j %.10T %.11M %R"
  exit 0
  ;;

gates)
  submit_gates
  ;;

extract)
  dep=""
  [[ -f "${GATE_IDF}" ]] && dep="$(cat "${GATE_IDF}")"
  submit_extract "$dep"
  ;;

all)
  g=$(submit_gates)
  e=$(submit_extract "$g")
  echo
  log "chain: ${g} (gates) -> ${e} (extract, afterok)"
  ;;

*)
  echo "usage: $0 {all|gates|extract|status|cancel}" >&2
  exit 2
  ;;
esac

echo
squeue -u "$USER" -o "%.10i %.20j %.10T %.11M %.6D %R" | head -10
echo
log "monitor:  tail -f ${LOGD}/nv-gates-\$(cat ${GATE_IDF}).out"
log "status :  $0 status"
