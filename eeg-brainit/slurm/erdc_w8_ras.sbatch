#!/usr/bin/env bash
#SBATCH --job-name=erdc-w8
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --exclude=dgx-10,dgx-11
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

# ERDC W8: Retrieval-Augmented Structure (RAS) + closed loop
#   Aim: lift PixCorr toward official ATM / ENIGMA low-level bar
#   while keeping brain-consistency selection (ERDC differentiation).

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
mkdir -p "$TMPDIR" outputs/erdc/w8_metrics outputs/slurm

python - <<'PY'
import torch, torchvision
assert torch.cuda.is_available(), "CUDA unavailable"
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
print("GPU", torch.cuda.get_device_name(0), "img2img OK", flush=True)
PY

MAX_IMAGES="${ERDC_MAX_IMAGES:-0}"
EXTRA=()
if [[ "${MAX_IMAGES}" != "0" ]]; then EXTRA+=(--max-images "${MAX_IMAGES}"); fi

echo "========== [1] RAS bit_clip (retrieval + i2i + loop) =========="
python scripts/erdc_ras_closed_loop.py \
  --subject sub-08 \
  --embed-source bit_clip \
  --top-m 2 \
  --strengths 0.45,0.60,0.75 \
  --output-dir outputs/erdc/w8_ras_bit_sub08 \
  --gen-steps 30 --gen-guidance 5.0 --gen-size 512 \
  "${EXTRA[@]}"

echo "========== [2] RAS atm (multi-subject structure path) =========="
for SUB in sub-08 sub-01 sub-02; do
  python scripts/erdc_ras_closed_loop.py \
    --subject "${SUB}" \
    --embed-source atm \
    --top-m 2 \
    --strengths 0.45,0.60,0.75 \
    --output-dir "outputs/erdc/w8_ras_atm_${SUB}" \
    --gen-steps 30 --gen-guidance 5.0 --gen-size 512 \
    "${EXTRA[@]}"
done

echo "========== [3] Control: shuffled gallery (should hurt) =========="
python scripts/erdc_ras_closed_loop.py \
  --subject sub-08 \
  --embed-source bit_clip \
  --top-m 2 \
  --strengths 0.45,0.60,0.75 \
  --shuffle-gallery \
  --output-dir outputs/erdc/w8_ras_bit_shuffle_sub08 \
  --gen-steps 30 --gen-guidance 5.0 --gen-size 512 \
  "${EXTRA[@]}"

echo "========== [4] Full metrics =========="
full_metrics () {
  local gen="$1"; local tag="$2"
  local out="outputs/erdc/w8_metrics/${tag}.json"
  [[ -f "${out}" ]] && echo "[SKIP] ${tag}" && return 0
  [[ -d "${gen}" ]] || { echo "[WARN] miss ${gen}"; return 0; }
  python scripts/erdc_full_metrics.py --gen-dir "${gen}" --output-json "${out}" --tag "${tag}" "${EXTRA[@]}"
}

# W7 reference
full_metrics outputs/erdc/w7_loop_bit_k8_sub08/selected_brain w7_bit_k8_brain
full_metrics outputs/erdc/w7_official_flat official_atm_gen

full_metrics outputs/erdc/w8_ras_bit_sub08/selected_first   w8_ras_bit_first
full_metrics outputs/erdc/w8_ras_bit_sub08/selected_random  w8_ras_bit_random
full_metrics outputs/erdc/w8_ras_bit_sub08/selected_brain   w8_ras_bit_brain
full_metrics outputs/erdc/w8_ras_bit_sub08/selected_oracle  w8_ras_bit_oracle
full_metrics outputs/erdc/w8_ras_bit_shuffle_sub08/selected_brain w8_ras_bit_shuffle_brain

for SUB in sub-08 sub-01 sub-02; do
  full_metrics "outputs/erdc/w8_ras_atm_${SUB}/selected_brain"  "w8_ras_atm_${SUB}_brain"
  full_metrics "outputs/erdc/w8_ras_atm_${SUB}/selected_random" "w8_ras_atm_${SUB}_random"
done

echo "========== [5] Panels =========="
python scripts/erdc_qualitative_panel.py \
  --b0-dir outputs/erdc/w6_b0_bit_sub08/generated \
  --random-dir outputs/erdc/w8_ras_bit_sub08/selected_random \
  --brain-dir outputs/erdc/w8_ras_bit_sub08/selected_brain \
  --scores-npy outputs/erdc/w8_ras_bit_sub08/candidates/brain_scores.npy \
  --output-dir outputs/erdc/w8_panels_ras_bit \
  --n-panel 24

python - <<'PY'
import json
from pathlib import Path

def load(tag):
    p = Path(f"outputs/erdc/w8_metrics/{tag}.json")
    return json.loads(p.read_text()) if p.is_file() else None

def fmt(tag):
    d = load(tag)
    if not d: return f"{tag}: MISSING"
    return (f"{tag:32s} Pix={d['pixcorr']:.4f} SSIM={d['ssim']:.4f} "
            f"CLIP={d['clip_cosine']:.4f} A2={d['alexnet2']:.4f} Inc={d['inception']:.4f}")

tags = [
    "w7_bit_k8_brain", "official_atm_gen",
    "w8_ras_bit_first", "w8_ras_bit_random", "w8_ras_bit_brain", "w8_ras_bit_oracle",
    "w8_ras_bit_shuffle_brain",
    "w8_ras_atm_sub-08_brain", "w8_ras_atm_sub-08_random",
    "w8_ras_atm_sub-01_brain", "w8_ras_atm_sub-01_random",
    "w8_ras_atm_sub-02_brain", "w8_ras_atm_sub-02_random",
]
rows = [fmt(t) for t in tags]
checks = []
b, r, w7, off = [load(t) for t in ["w8_ras_bit_brain","w8_ras_bit_random","w7_bit_k8_brain","official_atm_gen"]]
if b and r:
    checks.append(f"PASS_ras_brain_gt_random_CLIP={b['clip_cosine']>r['clip_cosine']} Pix={b['pixcorr']>r['pixcorr']}")
if b and w7:
    checks.append(f"RAS_vs_W7 dCLIP={b['clip_cosine']-w7['clip_cosine']:+.4f} dPix={b['pixcorr']-w7['pixcorr']:+.4f}")
if b and off:
    checks.append(f"RAS_vs_official dCLIP={b['clip_cosine']-off['clip_cosine']:+.4f} dPix={b['pixcorr']-off['pixcorr']:+.4f}")
shuf = load("w8_ras_bit_shuffle_brain")
if b and shuf:
    checks.append(f"PASS_real_gt_shuffle_Pix={b['pixcorr']>shuf['pixcorr']} dPix={b['pixcorr']-shuf['pixcorr']:+.4f}")

out = Path("outputs/erdc/w8_ras_summary.txt")
out.write_text("\n".join(rows+[""]+checks)+"\n", encoding="utf-8")
print(out.read_text())
print("[OK]", out)
PY

echo "[OK] ERDC W8 RAS finished"
