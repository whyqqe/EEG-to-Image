#!/usr/bin/env bash
# Clone Spec2VolCAMU-Net and Brain-IT into third_party/ (experiment directory only).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TP="${ROOT}/third_party"
mkdir -p "${TP}"

clone_or_update() {
  local url="$1"
  local dest="$2"
  if [[ -d "${dest}/.git" ]]; then
    echo "[INFO] Updating ${dest}"
    git -C "${dest}" pull --ff-only || true
  elif [[ -d "${dest}" ]]; then
    echo "[WARN] ${dest} exists without .git; skipping"
  else
    echo "[INFO] Cloning ${url} -> ${dest}"
    git clone --depth 1 "${url}" "${dest}"
  fi
}

clone_or_update "https://github.com/hdy6438/Spec2VolCAMU-Net.git" "${TP}/Spec2VolCAMU-Net"
clone_or_update "https://github.com/WeizmannVision/brainit-fmri.git" "${TP}/brainit-fmri"

# Optional: reuse already-downloaded trees under eeg-to-image without modifying them.
ALT_SPEC="/project/peilab/why/eeg-to-image/third_party/Spec2VolCAMU-Net"
ALT_BIT="/project/peilab/why/eeg-to-image/third_party/brainit-fmri"
if [[ ! -d "${TP}/Spec2VolCAMU-Net/.git" && -d "${ALT_SPEC}" ]]; then
  echo "[INFO] Spec2Vol clone missing; readable reference exists at ${ALT_SPEC}"
fi
if [[ ! -d "${TP}/brainit-fmri/.git" && -d "${ALT_BIT}" ]]; then
  echo "[INFO] Brain-IT clone missing; readable reference exists at ${ALT_BIT}"
fi

echo "[OK] third_party ready under ${TP}"
ls -la "${TP}"
