#!/usr/bin/env bash
#SBATCH --job-name=erdc-w1
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --exclude=dgx-10,dgx-11
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

# ERDC W1/W3 pipeline (sub-08):
#   1) B0 fixed-scale ATM emb → SDXL+IP-Adapter
#   2) confidence-routed scale (conf)
#   3) inverted confidence control (inv_conf)
#   4) teacher CLIP ceiling
# Smoke: set ERDC_MAX_IMAGES=8 for a short debug run.

set -euo pipefail
ROOT=/project/peilab/why/eeg-brainit
module load slurm nvhpc-hpcx-cuda12/23.11
cd "${ROOT}"
source scripts/activate.sh
export PYTHONUNBUFFERED=1
export HF_HOME=/project/peilab/why/cache/eeg-brainit/hf
export HF_HUB_CACHE=/project/peilab/why/cache/eeg-brainit/hf/hub
export HUGGINGFACE_HUB_CACHE=/project/peilab/why/cache/eeg-brainit/hf/hub
export TRANSFORMERS_CACHE=/project/peilab/why/cache/eeg-brainit/hf/hub
export DIFFUSERS_CACHE=/project/peilab/why/cache/eeg-brainit/hf/hub
export OPENCLIP_CACHE_DIR=/project/peilab/why/cache/eeg-brainit/open_clip
export TORCH_HOME=/project/peilab/why/cache/eeg-brainit/torch
export TMPDIR=/project/peilab/why/cache/eeg-brainit/tmp
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export XFORMERS_DISABLED=1
mkdir -p "$TMPDIR" "$HF_HUB_CACHE/.locks" outputs/erdc outputs/slurm

python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable — exclude/requeue"
import torchvision
from diffusers import StableDiffusionXLPipeline
print(
    "GPU", torch.cuda.get_device_name(0),
    "torch", torch.__version__,
    "tv", torchvision.__version__,
    "SDXL OK",
    flush=True,
)
PY

test -f outputs/atm_bridge/sub-08_test_eeg_1024.npy
test -f outputs/atm_bridge/clip_img_test_1024.npy
test -f outputs/eval/atm_baseline/test_ViT-H-14_laion2b_s32b_b79k_features.npy
test -d /project/peilab/why/data/images_set/test_images

MAX_IMAGES="${ERDC_MAX_IMAGES:-0}"
EXTRA=()
if [[ "${MAX_IMAGES}" != "0" ]]; then
  EXTRA+=(--max-images "${MAX_IMAGES}")
fi

run_one () {
  local mode="$1"
  local src="$2"
  local out="$3"
  echo "========== ERDC ${mode} / ${src} -> ${out} =========="
  python scripts/erdc_b0_generate.py \
    --subject sub-08 \
    --embed-source "${src}" \
    --mode "${mode}" \
    --output-dir "${out}" \
    --gen-steps 30 \
    --gen-guidance 5.0 \
    --gen-size 512 \
    --base-scale 1.0 \
    --scale-min 0.35 \
    --scale-max 1.15 \
    "${EXTRA[@]}"
}

# W1 B0 + W3 confidence ablations on ATM EEG embeds
run_one fixed atm outputs/erdc/w1_b0_atm_sub08
run_one conf atm outputs/erdc/w3_conf_atm_sub08
run_one inv_conf atm outputs/erdc/w3_invconf_atm_sub08
# Ceiling with teacher CLIP image embeds
run_one teacher teacher outputs/erdc/w1_teacher_ceiling_sub08

python - <<'PY'
import json
from pathlib import Path
root = Path("outputs/erdc")
rows = []
for name in [
    "w1_b0_atm_sub08",
    "w3_conf_atm_sub08",
    "w3_invconf_atm_sub08",
    "w1_teacher_ceiling_sub08",
]:
    p = root / name / "metrics.json"
    if not p.is_file():
        rows.append(f"{name}: MISSING")
        continue
    d = json.loads(p.read_text())
    m = d.get("metrics", {})
    rows.append(
        f"{name}: PixCorr={m.get('pixcorr', float('nan')):.4f} "
        f"SSIM={m.get('ssim', float('nan')):.4f} "
        f"CLIP={m.get('clip_cosine', float('nan')):.4f}"
    )
summary = Path("outputs/erdc/w1_w3_summary.txt")
summary.write_text("\n".join(rows) + "\n", encoding="utf-8")
print(summary.read_text())
print("[OK] wrote", summary)
PY

echo "[OK] ERDC W1/W3 pipeline finished"
