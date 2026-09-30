#!/usr/bin/env bash
# =============================================================================
# Drip-feed the ten LOSO reconstruction jobs within the account's QoS cap.
#
# THE CONSTRAINT THAT FORCES THIS DESIGN
# --------------------------------------
# The account QOS allows MaxSubmitJobsPU=10 and MaxJobsPU=8, and array elements count
# against BOTH. So the whole experiment (1 clip + 9 encoder trainings + 10
# reconstructions = 20 tasks) can never be submitted in one go, and a single
# `--array=0-9` reconstruction submission cannot be queued while the encoders are still
# in flight either -- it would need all ten slots free at once.
#
# WHAT THIS SCRIPT DOES INSTEAD
# -----------------------------
# It holds the queue at the limit by submitting one reconstruction job per fold, in the
# order folds become ready, and never more than the cap. Three gates must all pass before
# a fold is submitted:
#
#   1. the shared CLIP job has terminated, so no reconstruction job can race it on the
#      shared target filenames (see slurm/samgar_clip.sbatch for why that race is silent
#      and fatal);
#   2. that fold's encoder checkpoint exists on disk -- this is the per-fold gate, and it
#      is checked directly rather than via a Slurm dependency, because Slurm cannot
#      express "wait for array element f" from outside;
#   3. there is a free submit slot right now.
#
# Fold 8's encoder already exists (job 609116), so fold 8 goes first and does not wait for
# the nine trainings -- the drip-feed starts real work immediately instead of after an
# hour of idling.
#
# It is safe to restart: submitted folds are recorded on disk and never double-submitted.
#
# Usage:
#   CLIP_JOB=609245 SAMGA_JOB=609246 nohup setsid bash scripts/recon/dispatch_loso.sh \
#       > outputs/recon/dispatch.log 2>&1 &
# =============================================================================
set -uo pipefail

RECON_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${RECON_ROOT}"

SEED="${SEED:-2025}"
CKPT_TAG="${CKPT_TAG:-checkpoint_last.pth}"
LIMIT="${LIMIT:-10}"
POLL="${POLL:-60}"
MAX_WAIT="${MAX_WAIT:-43200}"          # 12 h, then give up on stragglers

CLIP_JOB="${CLIP_JOB:-}"
SAMGA_JOB="${SAMGA_JOB:-}"

ALL_FOLDS=(8 1 2 3 4 5 6 7 9 10)       # fold 8 first: its encoder is already trained
STATE="${RECON_ROOT}/outputs/recon/.dispatch"
mkdir -p "${STATE}"

log() { printf '[%s] %s\n' "$(date -Is)" "$*"; }

# `-r` is essential: without it squeue collapses the pending array elements into a single
# line, the count comes out under the real value, the dispatcher believes it has a free
# slot, and every submit is rejected by the QOS. MaxSubmitJobsPU counts tasks.
queue_len() { squeue -u "${USER}" -r -h -t PENDING,RUNNING 2>/dev/null | wc -l | tr -d ' '; }

ckpt_for() {
  ls -1dt "${RECON_ROOT}/outputs/samga_official/inter/seed${SEED}"/*"sub-$(printf '%02d' "$1")"/"${CKPT_TAG}" \
    2>/dev/null | head -1 || true
}

clip_done() {
  [[ -f "${RECON_ROOT}/data/image_feature/clip_h14_ip_adapter/clip_h14_train.npy" \
     && -f "${RECON_ROOT}/data/image_feature/clip_h14_ip_adapter/clip_h14_test.npy" ]]
}

clip_finished() {
  # Either the arrays exist, or the producing job is no longer in the queue.
  clip_done && return 0
  [[ -z "${CLIP_JOB}" ]] && return 0
  ! squeue -h -j "${CLIP_JOB}" 2>/dev/null | grep -q . 
}

samga_alive() {
  [[ -z "${SAMGA_JOB}" ]] && return 1
  squeue -h -j "${SAMGA_JOB}" 2>/dev/null | grep -q .
}

log "dispatch start: CLIP_JOB=${CLIP_JOB:-none} SAMGA_JOB=${SAMGA_JOB:-none} LIMIT=${LIMIT}"

DEADLINE=$(( $(date +%s) + MAX_WAIT ))
PENDING_FOLDS=("${ALL_FOLDS[@]}")

while :; do
  # Log state compactly each round so the log stays readable across a 12 h wait.
  QLEN="$(queue_len)"
  FREE=$(( LIMIT - QLEN ))
  # Recompute the still-unsubmitted folds, honouring the on-disk marks. Rebuilt into a
  # fresh array rather than filtered in place: with `set -u`, expanding an empty array
  # errors on bash < 4.4, and an empty remainder is the normal terminating condition here.
  REMAIN=()
  for f in "${PENDING_FOLDS[@]}"; do
    local_mark="${STATE}/sub-$(printf '%02d' "${f}")"
    [[ -f "${local_mark}.submitted" || -f "${local_mark}.abandoned" ]] || REMAIN+=("${f}")
  done
  PENDING_FOLDS=()
  if [[ "${#REMAIN[@]}" -gt 0 ]]; then PENDING_FOLDS=("${REMAIN[@]}"); fi

  if [[ "${#PENDING_FOLDS[@]}" -eq 0 ]]; then
    log "all ten folds submitted; dispatch done"
    break
  fi

  if [[ "$(date +%s)" -gt "${DEADLINE}" ]]; then
    log "TIMEOUT after ${MAX_WAIT}s; unsubmitted folds: ${PENDING_FOLDS[*]}"
    break
  fi

  PROGRESSED=0
  for f in "${PENDING_FOLDS[@]}"; do
    [[ "${FREE}" -le 0 ]] && break

    # Gate 1: never let a reconstruction race the shared CLIP producer.
    if ! clip_finished; then
      log "waiting: CLIP job ${CLIP_JOB} still running (queue=${QLEN} free=${FREE})"
      break
    fi
    if ! clip_done; then
      log "ABORT: CLIP job finished but arrays are missing; nothing downstream can run"
      exit 1
    fi

    # Gate 2: this fold's own encoder.
    CK="$(ckpt_for "${f}")"
    if [[ -z "${CK}" ]]; then
      if ! samga_alive; then
        # Encoder job is gone and there is still no checkpoint: this fold cannot run.
        log "SKIP fold sub-$(printf '%02d' "${f}"): encoder job gone and no ${CKPT_TAG}"
        : > "${STATE}/sub-$(printf '%02d' "${f}").abandoned"
        continue
      fi
      continue          # still training; try again next round
    fi

    # Gate 3: a free submit slot.
    JID="$(sbatch --parsable \
      --job-name="samgar-f${f}" \
      --time=08:00:00 \
      --export=ALL,STAGE=all,TARGET="${f}",SEED="${SEED}" \
      slurm/samga_recon.sbatch 2>&1)" || {
        log "sbatch failed for fold ${f}: ${JID}"
        continue
      }
    log "submitted fold sub-$(printf '%02d' "${f}") -> job ${JID} (encoder: ${CK})"
    : > "${STATE}/sub-$(printf '%02d' "${f}").submitted"
    printf '%s %s\n' "${f}" "${JID}" >> "${STATE}/submitted.tsv"
    FREE=$(( FREE - 1 ))
    PROGRESSED=1
  done

  sleep "${POLL}"
done

log "dispatch exiting; submitted so far:"
cat "${STATE}/submitted.tsv" 2>/dev/null || echo "(none)"
