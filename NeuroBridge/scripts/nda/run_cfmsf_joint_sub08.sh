#!/usr/bin/env bash
# CF-MSF Stage 0 chain, sub-08, one job:
#   1) joint-train the EEG encoder against the multi-level target (arms: joint, frozen)
#   2) re-run the frozen-encoder route probe on EACH resulting encoder
#   3) summarize joint vs frozen against jobs 581546 / 581602
#
# The `frozen` arm is not a courtesy: it is the same loss on the same data for the
# same epochs with the encoder weights frozen, i.e. job 581602's setting.  Without
# it, a gain could be credited to joint training when it is only the extra steps.
set -euo pipefail

NB_ROOT="${NB_ROOT:-/project/peilab/why/NeuroBridge}"
cd "${NB_ROOT}"

PYTHON=/project/peilab/why/eeg-brainit/.venv/bin/python
[[ -x "${PYTHON}" ]] || { echo "[FATAL] missing ${PYTHON}"; exit 1; }
export HF_HOME=/project/peilab/why/cache/eeg-brainit/hf
export HF_HUB_CACHE=/project/peilab/why/cache/eeg-brainit/hf/hub
export TORCH_HOME=/project/peilab/why/cache/eeg-brainit/torch

OUT="${JOINT_OUT:-${NB_ROOT}/outputs/cfmsf_joint/sub-08}"
DEVICE="${DEVICE:-cuda:0}"
TARGET="${TARGET:-levels_mean}"
ARMS="${ARMS:-joint,frozen}"
EPOCHS="${EPOCHS:-40}"
PROBE_EPOCHS="${PROBE_EPOCHS:-80}"
mkdir -p "${OUT}"/{logs,slurm}

log() { echo "[$(date -Iseconds)] $*"; }
require() { [[ -e "$1" ]] || { echo "[FATAL] missing $1"; exit 1; }; }

require outputs/ocf/intra_enc/sub-08/checkpoint_ss_calib_best.pth
require outputs/leakfree/split.json
require scripts/nda/cfmsf_joint_train.py
require scripts/nda/cfmsf_route_probe.py
require scripts/nda/cfmsf_joint_summary.py
require scripts/nda/device_audit.py

# The GPU job re-runs the same device audit the submit script runs, so a script
# edited between submit and start still gets checked before burning GPU time.
"${PYTHON}" scripts/nda/device_audit.py \
    scripts/nda/cfmsf_joint_train.py scripts/nda/cfmsf_route_probe.py \
    scripts/nda/cfmsf_joint_summary.py \
  || { echo "[FATAL] device audit failed"; exit 1; }

# ---------------- 1. joint training ----------------
log "===== joint encoder training (target=${TARGET}, arms=${ARMS}, epochs=${EPOCHS}) ====="
"${PYTHON}" scripts/nda/cfmsf_joint_train.py \
    --out "${OUT}" --target "${TARGET}" --arms "${ARMS}" \
    --epochs "${EPOCHS}" --device "${DEVICE}" \
    2>&1 | tee "${OUT}/logs/joint_train.log"
require "${OUT}/joint_report.json"

# ---------------- 2. route probe on each new encoder ----------------
for arm in ${ARMS//,/ }; do
  require "${OUT}/${arm}/enc/sub-08/shared_r_train.npy"
  require "${OUT}/${arm}/enc/sub-08/shared_r_test.npy"
  if [[ -f "${OUT}/${arm}/probe/route_probe.json" ]]; then
    log "probe for ${arm} already present, skipping"
    continue
  fi
  log "===== route probe on encoder '${arm}' ====="
  "${PYTHON}" scripts/nda/cfmsf_route_probe.py \
      --out "${OUT}/${arm}/probe" --test-subject 8 \
      --z-root "${OUT}/${arm}/enc" --epochs "${PROBE_EPOCHS}" --device "${DEVICE}" \
      2>&1 | tee "${OUT}/logs/probe_${arm}.log"
done

# ---------------- 3. summary ----------------
log "===== summary ====="
"${PYTHON}" scripts/nda/cfmsf_joint_summary.py \
    --root "${OUT}" --out "${OUT}/summary.json" --pick lvl5+agg --tol 0.10 \
    2>&1 | tee "${OUT}/logs/summary.log"

log "===== done ====="
du -sh "${OUT}" | sed 's/^/[disk] /'
