#!/usr/bin/env bash
# Prepare / download THINGS-EEG2 assets.
#
# Preference order:
#   1) Shared local copy under /project/peilab/why/data
#   2) Already-downloaded eeg-vilex-neurolm OSF clone
#   3) Fresh OSF download into this experiment's data/raw (large)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_ROOT="${EEG_BRAINIT_CACHE:-/project/peilab/why/cache/eeg-brainit}"
SHARED_DATA="${SHARED_DATA:-/project/peilab/why/data}"
DEST="${1:-${ROOT}/data/raw/THINGS-EEG2}"
export PIP_CACHE_DIR="${CACHE_ROOT}/pip"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
mkdir -p "${DEST}" "${ROOT}/data/processed"

echo "[INFO] Checking shared THINGS-EEG2 under ${SHARED_DATA} ..."
ok=1
if [[ -d "${SHARED_DATA}/Preprocessed_data_250Hz" ]]; then
  echo "[OK] Found ${SHARED_DATA}/Preprocessed_data_250Hz"
else
  echo "[WARN] Missing Preprocessed_data_250Hz"
  ok=0
fi
if [[ -d "${SHARED_DATA}/images_set" ]]; then
  echo "[OK] Found ${SHARED_DATA}/images_set"
else
  echo "[WARN] Missing images_set"
  ok=0
fi

ALT="/project/peilab/why/eeg-vilex-neurolm/data/raw/THINGS-EEG2"
if [[ "${ok}" -eq 0 && -d "${ALT}" ]]; then
  echo "[INFO] Shared preprocessed missing/partial; OSF clone exists at ${ALT}"
  echo "[INFO] You can symlink or point configs at that tree without re-downloading."
fi

if [[ "${ok}" -eq 1 ]]; then
  echo "[INFO] Using shared data. Building trial-level manifest..."
  if [[ -x "${ROOT}/.venv/bin/python" ]]; then
    # shellcheck disable=SC1091
    source "${ROOT}/scripts/activate.sh"
    python -m pip install -q tqdm
    python "${ROOT}/scripts/prepare_things_eeg2.py" \
      --eeg-dir "${SHARED_DATA}/Preprocessed_data_250Hz" \
      --images-dir "${SHARED_DATA}/images_set" \
      --metadata "${SHARED_DATA}/images_set/image_metadata.npy" \
      --out-dir "${ROOT}/data/processed/things-eeg2" \
      --out "${ROOT}/data/processed/manifest.jsonl" \
      --subjects "${SUBJECTS:-1}"
  fi
  echo "[OK] Inspect data/processed/manifest.jsonl"
  exit 0
fi

if [[ "${FORCE_OSF_DOWNLOAD:-0}" != "1" ]]; then
  cat <<EOF
[INFO] Shared THINGS-EEG2 not fully available.
To download from OSF (large), re-run with:
  FORCE_OSF_DOWNLOAD=1 bash scripts/download_things_eeg2.sh

Official components:
  https://osf.io/anp5v/  (preprocessed EEG, 63ch)
  https://osf.io/y63gw/  (stimulus images)
EOF
  exit 0
fi

if [[ -x "${ROOT}/.venv/bin/python" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/scripts/activate.sh"
fi
python -m pip install --upgrade osfclient
mkdir -p "${DEST}/preprocessed" "${DEST}/images"
osf -p anp5v clone "${DEST}/preprocessed"
osf -p y63gw clone "${DEST}/images"
echo "[OK] OSF download finished at ${DEST}"
