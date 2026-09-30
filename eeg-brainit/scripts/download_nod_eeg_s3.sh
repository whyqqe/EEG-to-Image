#!/usr/bin/env bash
# Download NOD-EEG (ds005811) via OpenNeuro public S3 (resume-safe).
# Avoids full /home: keep caches on project disk.
set -euo pipefail
ROOT=/project/peilab/why/eeg-brainit
CACHE=/project/peilab/why/cache/eeg-brainit
TARGET="${ROOT}/data/nod/raw/ds005811"
LOG="${ROOT}/outputs/slurm/download_nod_eeg_s3.log"

cd "${ROOT}"
# shellcheck disable=SC1091
source scripts/activate.sh

export HOME=/home/sbaiae
if [[ "${HOME}" == *'/.cache/huggingface'* ]]; then
  export HOME="$(getent passwd "$(id -un)" | cut -d: -f6)"
fi
export XDG_CONFIG_HOME="${CACHE}/xdg-config"
export XDG_CACHE_HOME="${CACHE}/xdg"
export TMPDIR="${CACHE}/tmp"
mkdir -p "${TARGET}" "${XDG_CONFIG_HOME}" "${XDG_CACHE_HOME}" "${TMPDIR}" "$(dirname "${LOG}")"

echo "[INFO] $(date) HOME=${HOME} target=${TARGET}" | tee -a "${LOG}"
df -h /project/peilab/why | tail -1 | tee -a "${LOG}"

if ! command -v aws >/dev/null 2>&1; then
  pip install -q awscli
fi

# Phase A: one subject + metadata + events/epochs (smoke)
if [[ "${NOD_SKIP_PHASE_A:-0}" != "1" ]]; then
  echo "[INFO] $(date) phase A: metadata + sub-01" | tee -a "${LOG}"
  aws s3 sync --no-sign-request \
    s3://openneuro.org/ds005811/ "${TARGET}/" \
    --only-show-errors \
    --exclude "*" \
    --include "dataset_description.json" \
    --include "participants.tsv" \
    --include "README*" \
    --include "CHANGES*" \
    --include "sub-01/*" \
    --include "derivatives/detailed_events/sub-01*" \
    --include "derivatives/preprocessed/epochs/sub-01*" \
    2>&1 | tee -a "${LOG}"
  echo "[OK] $(date) phase A done size=$(du -sh "${TARGET}" | awk '{print $1}')" | tee -a "${LOG}"
fi

# Phase B: full dataset (resumes)
if [[ "${NOD_FULL:-1}" == "1" ]]; then
  echo "[INFO] $(date) phase B: full sync" | tee -a "${LOG}"
  aws s3 sync --no-sign-request \
    s3://openneuro.org/ds005811/ "${TARGET}/" \
    --only-show-errors \
    2>&1 | tee -a "${LOG}"
  echo "[OK] $(date) phase B done size=$(du -sh "${TARGET}" | awk '{print $1}')" | tee -a "${LOG}"
fi

ls "${TARGET}" | tee -a "${LOG}"
find "${TARGET}" -maxdepth 2 -type d | head -50 | tee -a "${LOG}"
echo "[DONE] $(date)" | tee -a "${LOG}"
