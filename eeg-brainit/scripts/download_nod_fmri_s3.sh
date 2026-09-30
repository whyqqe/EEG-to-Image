#!/usr/bin/env bash
# Selective NOD-fMRI (OpenNeuro ds004496) download for EEG→fMRI pairing.
# Pulls only ImageNet beta.dscalar.nii + label.txt + events.tsv (not full BOLD).
set -euo pipefail
ROOT=/project/peilab/why/eeg-brainit
CACHE=/project/peilab/why/cache/eeg-brainit
TARGET="${ROOT}/data/nod/raw/nod_fmri"
LOG="${ROOT}/outputs/slurm/download_nod_fmri_s3.log"
# Comma-separated BIDS ids without "sub-", default: first EEG-overlapping subject
SUBJECTS="${NOD_FMRI_SUBJECTS:-01}"

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

if ! command -v aws >/dev/null 2>&1; then
  pip install -q awscli
fi

echo "[INFO] $(date) subjects=${SUBJECTS} target=${TARGET}" | tee -a "${LOG}"

# metadata
aws s3 sync --no-sign-request \
  s3://openneuro.org/ds004496/ "${TARGET}/" \
  --exclude "*" \
  --include "dataset_description.json" \
  --include "participants.tsv" \
  --include "README*" \
  --include "CHANGES*" \
  2>&1 | tee -a "${LOG}"

IFS=',' read -ra SUBS <<< "${SUBJECTS}"
for sid in "${SUBS[@]}"; do
  sid="$(echo "${sid}" | tr -d '[:space:]')"
  [[ -z "${sid}" ]] && continue
  sub="sub-${sid}"
  echo "[INFO] $(date) sync betas+events for ${sub}" | tee -a "${LOG}"

  # ImageNet events only (tiny)
  aws s3 sync --no-sign-request \
    "s3://openneuro.org/ds004496/${sub}/" "${TARGET}/${sub}/" \
    --exclude "*" \
    --include "ses-imagenet*/func/*_events.tsv" \
    --include "ses-imagenet*/func/*_bold.json" \
    2>&1 | tee -a "${LOG}"

  # ciftify ImageNet betas + labels (skip heavy Atlas timeseries)
  # awscli globs: * does not cross '/', so match one directory level under results/
  aws s3 sync --no-sign-request \
    "s3://openneuro.org/ds004496/derivatives/ciftify/${sub}/results/" \
    "${TARGET}/derivatives/ciftify/${sub}/results/" \
    --exclude "*" \
    --include "ses-imagenet*/*_beta.dscalar.nii" \
    --include "ses-imagenet*/*_label.txt" \
    2>&1 | tee -a "${LOG}"
done

echo "[OK] $(date) size=$(du -sh "${TARGET}" 2>/dev/null | awk '{print $1}')" | tee -a "${LOG}"
find "${TARGET}/derivatives/ciftify" -name '*_beta.dscalar.nii' 2>/dev/null | wc -l | tee -a "${LOG}"
echo "[DONE] $(date)" | tee -a "${LOG}"
