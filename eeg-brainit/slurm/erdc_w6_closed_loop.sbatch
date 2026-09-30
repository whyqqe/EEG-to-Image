#!/usr/bin/env bash
#SBATCH --job-name=erdc-w6
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=96G
#SBATCH --time=18:00:00
#SBATCH --requeue
#SBATCH --exclude=dgx-10,dgx-11
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

# ERDC fastest top-venue path after conf→scale failed:
#   Stage-3 closed-loop K-select (no training).
#   1) bit_clip single B0 (stronger emb than raw ATM)
#   2) bit_clip closed-loop K=4 (brain / random / first / oracle)
#   3) atm closed-loop K=4 (fair vs W1 B0)
# Success: brain pick > first & > random on CLIP/PixCorr.

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
import torch, torchvision
assert torch.cuda.is_available(), "CUDA unavailable — exclude/requeue"
from diffusers import StableDiffusionXLPipeline
print(
    "GPU", torch.cuda.get_device_name(0),
    "torch", torch.__version__,
    "tv", torchvision.__version__,
    "SDXL OK",
    flush=True,
)
PY

test -f outputs/eval/atm_pipeline_sub08/sub-08_bit_clip_1024.npy
test -f outputs/atm_bridge/sub-08_test_eeg_1024.npy
test -d /project/peilab/why/data/images_set/test_images

MAX_IMAGES="${ERDC_MAX_IMAGES:-0}"
EXTRA=()
if [[ "${MAX_IMAGES}" != "0" ]]; then
  EXTRA+=(--max-images "${MAX_IMAGES}")
fi

echo "========== [1/3] bit_clip B0 fixed =========="
python scripts/erdc_b0_generate.py \
  --subject sub-08 \
  --embed-source bit_clip \
  --mode fixed \
  --output-dir outputs/erdc/w6_b0_bit_sub08 \
  --gen-steps 30 \
  --gen-guidance 5.0 \
  --gen-size 512 \
  --base-scale 1.0 \
  "${EXTRA[@]}"

echo "========== [2/3] bit_clip closed-loop K=4 =========="
python scripts/erdc_closed_loop.py \
  --subject sub-08 \
  --embed-source bit_clip \
  --k 4 \
  --scale-grid 0.9,1.0,1.1,1.0 \
  --seed0 42 \
  --output-dir outputs/erdc/w6_loop_bit_sub08 \
  --gen-steps 30 \
  --gen-guidance 5.0 \
  --gen-size 512 \
  "${EXTRA[@]}"

echo "========== [3/3] atm closed-loop K=4 =========="
python scripts/erdc_closed_loop.py \
  --subject sub-08 \
  --embed-source atm \
  --k 4 \
  --scale-grid 0.9,1.0,1.1,1.0 \
  --seed0 42 \
  --output-dir outputs/erdc/w6_loop_atm_sub08 \
  --gen-steps 30 \
  --gen-guidance 5.0 \
  --gen-size 512 \
  "${EXTRA[@]}"

python - <<'PY'
import json
from pathlib import Path

def clip_of(p):
    d = json.loads(Path(p).read_text())
    if "metrics" in d and "clip_cosine" in d["metrics"]:
        m = d["metrics"]
        return f"CLIP={m['clip_cosine']:.4f} Pix={m['pixcorr']:.4f}"
    # closed-loop
    lines = []
    for name in ("first", "random", "brain", "oracle"):
        blk = d["selection"][name]
        m = blk["metrics"]
        dv = blk.get("delta_vs_first", {})
        lines.append(
            f"  {name:6s} CLIP={m['clip_cosine']:.4f} Pix={m['pixcorr']:.4f} "
            f"dCLIP={dv.get('clip', 0):+.4f}"
        )
    return "\n".join(lines)

rows = []
rows.append("W1 B0 atm:     " + clip_of("outputs/erdc/w1_b0_atm_sub08/metrics.json"))
rows.append("W6 B0 bit:     " + clip_of("outputs/erdc/w6_b0_bit_sub08/metrics.json"))
rows.append("W6 loop bit:\n" + clip_of("outputs/erdc/w6_loop_bit_sub08/metrics.json"))
rows.append("W6 loop atm:\n" + clip_of("outputs/erdc/w6_loop_atm_sub08/metrics.json"))
# success flag
lb = json.loads(Path("outputs/erdc/w6_loop_bit_sub08/metrics.json").read_text())
brain = lb["selection"]["brain"]["metrics"]["clip_cosine"]
first = lb["selection"]["first"]["metrics"]["clip_cosine"]
rand = lb["selection"]["random"]["metrics"]["clip_cosine"]
ok = brain > first and brain > rand
rows.append(f"PASS_brain_gt_first_random(bit)={ok} brain={brain:.4f} first={first:.4f} rand={rand:.4f}")
out = Path("outputs/erdc/w6_closed_loop_summary.txt")
out.write_text("\n".join(rows) + "\n", encoding="utf-8")
print(out.read_text())
print("[OK]", out)
PY

echo "[OK] ERDC W6 closed-loop pipeline finished"
