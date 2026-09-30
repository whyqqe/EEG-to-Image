#!/usr/bin/env bash
# Robust NOD-EEG (ds005811) download to data/nod/raw/ds005811.
# Avoids full /home by forcing XDG_* and fixing a broken HOME.
set -euo pipefail
ROOT=/project/peilab/why/eeg-brainit
CACHE=/project/peilab/why/cache/eeg-brainit
TARGET="${ROOT}/data/nod/raw/ds005811"
LOG="${ROOT}/outputs/slurm/download_nod_eeg_ds005811.log"

cd "${ROOT}"
# shellcheck disable=SC1091
source scripts/activate.sh

export HOME="${HOME:-/home/$(id -un)}"
if [[ "${HOME}" == *'/.cache/huggingface'* ]]; then
  export HOME="$(getent passwd "$(id -un)" | cut -d: -f6)"
fi
export XDG_CONFIG_HOME="${CACHE}/xdg-config"
export XDG_CACHE_HOME="${CACHE}/xdg"
export TMPDIR="${CACHE}/tmp"
mkdir -p "${XDG_CONFIG_HOME}" "${XDG_CACHE_HOME}" "${TMPDIR}" "${TARGET}" \
  "$(dirname "${LOG}")"

echo "[INFO] HOME=${HOME}"
echo "[INFO] XDG_CONFIG_HOME=${XDG_CONFIG_HOME}"
echo "[INFO] target=${TARGET}"
df -h "${ROOT}" | tail -1

# Resume-friendly download (openneuro-py skips existing complete files).
# Start with participants + derivatives for a usable smoke subset if INCLUDE set.
INCLUDE_ARGS=()
if [[ "${NOD_INCLUDE_ONLY:-}" != "" ]]; then
  IFS=',' read -ra parts <<< "${NOD_INCLUDE_ONLY}"
  for p in "${parts[@]}"; do
    INCLUDE_ARGS+=(--include "${p}")
  done
fi

set -x
openneuro-py download \
  --dataset ds005811 \
  --target-dir "${TARGET}" \
  --max-concurrent-downloads 4 \
  --max-retries 8 \
  "${INCLUDE_ARGS[@]}" \
  2>&1 | tee -a "${LOG}"
set +x

echo "[OK] download finished" | tee -a "${LOG}"
du -sh "${TARGET}" | tee -a "${LOG}"
find "${TARGET}" -maxdepth 2 -type d | head -40 | tee -a "${LOG}"
