#!/usr/bin/env bash
# Download Brain-IT checkpoints from Hugging Face into checkpoints/brain_it.
# Spec2VolCAMU-Net has no public weights — document placeholder only.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_ROOT="${EEG_BRAINIT_CACHE:-/project/peilab/why/cache/eeg-brainit}"
export HF_HOME="${CACHE_ROOT}/hf"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export PYTHONNOUSERSITE=1

if [[ -x "${ROOT}/.venv/bin/python" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/scripts/activate.sh"
  PYTHON="${ROOT}/.venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

mkdir -p "${ROOT}/checkpoints/brain_it" "${ROOT}/checkpoints/spec2vol"

"${PYTHON}" - <<'PY'
from pathlib import Path
import shutil
from huggingface_hub import hf_hub_download

root = Path("/project/peilab/why/eeg-brainit/checkpoints/brain_it")
root.mkdir(parents=True, exist_ok=True)
files = [
    "checkpoints/combined_model.ckpt",
    "checkpoints/decoder_clipg_ext-1_save.pth",
    "checkpoints/decoder_vgg_ext-1_batch64_save.pth",
    "checkpoints/encoder_ch128.pth",
    "derived_data/v2c_128_mapping_gmm_v2.npy",
]
for f in files:
    print(f"download {f}", flush=True)
    src = hf_hub_download("RomanBeliy/Brain-IT", f)
    out = root / Path(f).name
    if (not out.exists()) or out.stat().st_size != Path(src).stat().st_size:
        shutil.copy2(src, out)
    print(f"OK {out} ({out.stat().st_size} bytes)", flush=True)

# Convert CLIP decoder pickle -> pure state_dict when possible.
import torch
import torch.nn as nn
src = root / "decoder_clipg_ext-1_save.pth"
dst = root / "decoder_clipg_state_dict.pt"
if src.exists() and not dst.exists():
    obj = torch.load(src, map_location="cpu", weights_only=False)
    if isinstance(obj, nn.Module):
        torch.save(obj.state_dict(), dst)
        print("converted", dst)
    elif isinstance(obj, dict):
        torch.save(obj, dst)
        print("saved dict", dst)
    else:
        print(f"[WARN] Unexpected checkpoint type: {type(obj)}")

readme = Path("/project/peilab/why/eeg-brainit/checkpoints/spec2vol/README.md")
readme.write_text(
    "# Spec2VolCAMU-Net weights\n\n"
    "Official Spec2VolCAMU-Net does **not** publish pretrained weights.\n"
    "Place a local encoder checkpoint at `md_tf_cae.pt` after pretraining on\n"
    "EEG-fMRI pairs, or leave the encoder randomly initialized for smoke tests.\n",
    encoding="utf-8",
)
print("NOTE: Spec2Vol weights are not public. See checkpoints/spec2vol/README.md")
PY

echo "[OK] Brain-IT checkpoints in ${ROOT}/checkpoints/brain_it"
ls -lh "${ROOT}/checkpoints/brain_it"
