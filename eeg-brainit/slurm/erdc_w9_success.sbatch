#!/usr/bin/env bash
#SBATCH --job-name=erdc-w9
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

# ERDC W9 — raise top-venue odds:
#   A) VALID negative controls for RAS (random / misalign)
#   B) ControlNet-canny + IP + closed loop (PixCorr push)
#   C) CN ablations (random/misalign neighbors)
#   D) Multi-subject atm CN (sub-08/01/02/04/10 spot + mean)
#   E) Full metrics + 2WC + summary

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
mkdir -p "$TMPDIR" outputs/erdc/w9_metrics outputs/slurm

python - <<'PY'
import torch, cv2
from pathlib import Path
assert torch.cuda.is_available(), "CUDA unavailable"
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
print("GPU", torch.cuda.get_device_name(0), "cv2", cv2.__version__, "CN OK", flush=True)
p = Path("/project/peilab/why/eeg-brainit/outputs/erdc/controlnet_canny_ready.txt")
assert p.is_file(), "ControlNet not downloaded"
print(p.read_text().strip(), flush=True)
PY

MAX_IMAGES="${ERDC_MAX_IMAGES:-0}"
EXTRA=()
if [[ "${MAX_IMAGES}" != "0" ]]; then EXTRA+=(--max-images "${MAX_IMAGES}"); fi

full_metrics () {
  local gen="$1"; local tag="$2"
  local out="outputs/erdc/w9_metrics/${tag}.json"
  [[ -f "${out}" ]] && echo "[SKIP] ${tag}" && return 0
  [[ -d "${gen}" ]] || { echo "[WARN] miss ${gen}"; return 0; }
  python scripts/erdc_full_metrics.py --gen-dir "${gen}" --output-json "${out}" --tag "${tag}" "${EXTRA[@]}"
}
twoway () {
  local gen="$1"; local tag="$2"
  local out="outputs/erdc/w9_metrics/${tag}_2wc.json"
  [[ -f "${out}" ]] && echo "[SKIP] 2wc ${tag}" && return 0
  [[ -d "${gen}" ]] || return 0
  python scripts/erdc_twoway_metrics.py --gen-dir "${gen}" --output-json "${out}" --tag "${tag}" "${EXTRA[@]}"
}

echo "################ A) VALID RAS negative controls ################"
# same strength grid as best lowstr RAS
for MODE in random misalign; do
  echo "----- RAS control ${MODE} -----"
  python scripts/erdc_ras_closed_loop.py \
    --subject sub-08 \
    --embed-source bit_clip \
    --neighbor-mode "${MODE}" \
    --top-m 2 \
    --strengths 0.25,0.35,0.50 \
    --output-dir "outputs/erdc/w9_ras_ctrl_${MODE}_sub08" \
    --gen-steps 30 --gen-guidance 5.0 --gen-size 512 \
    "${EXTRA[@]}"
  full_metrics "outputs/erdc/w9_ras_ctrl_${MODE}_sub08/selected_brain" "w9_ras_ctrl_${MODE}_brain"
done
# reference: existing lowstr retrieve
full_metrics outputs/erdc/w8_ras_bit_lowstr_sub08/selected_brain w8_ras_bit_lowstr_brain
full_metrics outputs/erdc/w8_ras_bit_sub08/selected_brain w8_ras_bit_brain
full_metrics outputs/erdc/w7_official_flat official_atm_gen

echo "################ B) ControlNet + IP main (bit retrieve) ################"
python scripts/erdc_struct_controlnet.py \
  --subject sub-08 \
  --embed-source bit_clip \
  --neighbor-mode retrieve \
  --top-m 1 \
  --cn-scales 0.5,0.8,1.0 \
  --ip-scales 0.8,1.0 \
  --output-dir outputs/erdc/w9_cn_bit_sub08 \
  --gen-steps 30 --gen-guidance 5.0 --gen-size 512 \
  "${EXTRA[@]}"

full_metrics outputs/erdc/w9_cn_bit_sub08/selected_first  w9_cn_bit_first
full_metrics outputs/erdc/w9_cn_bit_sub08/selected_random w9_cn_bit_random
full_metrics outputs/erdc/w9_cn_bit_sub08/selected_brain  w9_cn_bit_brain
full_metrics outputs/erdc/w9_cn_bit_sub08/selected_oracle w9_cn_bit_oracle

echo "################ C) ControlNet neighbor controls ################"
for MODE in random misalign; do
  python scripts/erdc_struct_controlnet.py \
    --subject sub-08 \
    --embed-source bit_clip \
    --neighbor-mode "${MODE}" \
    --top-m 1 \
    --cn-scales 0.5,0.8,1.0 \
    --ip-scales 1.0 \
    --no-cn-only \
    --output-dir "outputs/erdc/w9_cn_ctrl_${MODE}_sub08" \
    --gen-steps 30 --gen-guidance 5.0 --gen-size 512 \
    "${EXTRA[@]}"
  full_metrics "outputs/erdc/w9_cn_ctrl_${MODE}_sub08/selected_brain" "w9_cn_ctrl_${MODE}_brain"
done

echo "################ D) Multi-subject atm ControlNet ################"
for SUB in sub-08 sub-01 sub-02 sub-04 sub-10; do
  echo "----- CN atm ${SUB} -----"
  python scripts/erdc_struct_controlnet.py \
    --subject "${SUB}" \
    --embed-source atm \
    --neighbor-mode retrieve \
    --top-m 1 \
    --cn-scales 0.6,1.0 \
    --ip-scales 1.0 \
    --no-cn-only \
    --output-dir "outputs/erdc/w9_cn_atm_${SUB}" \
    --gen-steps 30 --gen-guidance 5.0 --gen-size 512 \
    "${EXTRA[@]}"
  full_metrics "outputs/erdc/w9_cn_atm_${SUB}/selected_brain"  "w9_cn_atm_${SUB}_brain"
  full_metrics "outputs/erdc/w9_cn_atm_${SUB}/selected_random" "w9_cn_atm_${SUB}_random"
done

echo "################ E) 2WC + panels + summary ################"
twoway outputs/erdc/w9_cn_bit_sub08/selected_brain w9_cn_bit_brain
twoway outputs/erdc/w8_ras_bit_lowstr_sub08/selected_brain w8_ras_bit_lowstr_brain
twoway outputs/erdc/w7_official_flat official_atm_gen
twoway outputs/erdc/w9_ras_ctrl_random_sub08/selected_brain w9_ras_ctrl_random_brain
twoway outputs/erdc/w9_cn_ctrl_random_sub08/selected_brain w9_cn_ctrl_random_brain

python scripts/erdc_qualitative_panel.py \
  --b0-dir outputs/erdc/w6_b0_bit_sub08/generated \
  --random-dir outputs/erdc/w9_cn_bit_sub08/selected_random \
  --brain-dir outputs/erdc/w9_cn_bit_sub08/selected_brain \
  --scores-npy outputs/erdc/w9_cn_bit_sub08/candidates/brain_scores.npy \
  --output-dir outputs/erdc/w9_panels_cn_bit \
  --n-panel 24 || true

python - <<'PY'
import json
from pathlib import Path
import numpy as np

def load(tag):
    p = Path(f"outputs/erdc/w9_metrics/{tag}.json")
    if p.is_file():
        return json.loads(p.read_text())
    # fallback overnight/w8
    for base in ["outputs/erdc/overnight_metrics", "outputs/erdc/w8_metrics", "outputs/erdc/w7_metrics"]:
        q = Path(base) / f"{tag}.json"
        if q.is_file():
            return json.loads(q.read_text())
    return None

def line(tag):
    d = load(tag)
    if not d:
        return f"{tag}: MISSING"
    tw = ""
    twp = Path(f"outputs/erdc/w9_metrics/{tag}_2wc.json")
    if twp.is_file():
        t = json.loads(twp.read_text())["twoway"]
        tw = f" | 2WC CLIP={t['clip']*100:.1f}%"
    return (f"{tag:36s} Pix={d['pixcorr']:.4f} CLIP={d['clip_cosine']:.4f} "
            f"A2={d.get('alexnet2',float('nan')):.4f} Inc={d.get('inception',float('nan')):.4f}{tw}")

tags = [
    "w8_ras_bit_brain", "w8_ras_bit_lowstr_brain", "official_atm_gen",
    "w9_ras_ctrl_random_brain", "w9_ras_ctrl_misalign_brain",
    "w9_cn_bit_first", "w9_cn_bit_random", "w9_cn_bit_brain", "w9_cn_bit_oracle",
    "w9_cn_ctrl_random_brain", "w9_cn_ctrl_misalign_brain",
    "w9_cn_atm_sub-08_brain", "w9_cn_atm_sub-01_brain", "w9_cn_atm_sub-02_brain",
    "w9_cn_atm_sub-04_brain", "w9_cn_atm_sub-10_brain",
]
rows = [line(t) for t in tags]
checks = []
low = load("w8_ras_bit_lowstr_brain")
rnd = load("w9_ras_ctrl_random_brain")
mis = load("w9_ras_ctrl_misalign_brain")
cn = load("w9_cn_bit_brain")
off = load("official_atm_gen")
cnr = load("w9_cn_ctrl_random_brain")
if low and rnd:
    checks.append(f"PASS_ras_gt_random_ctrl Pix={low['pixcorr']>rnd['pixcorr']} CLIP={low['clip_cosine']>rnd['clip_cosine']} "
                  f"dPix={low['pixcorr']-rnd['pixcorr']:+.4f} dCLIP={low['clip_cosine']-rnd['clip_cosine']:+.4f}")
if low and mis:
    checks.append(f"PASS_ras_gt_misalign Pix={low['pixcorr']>mis['pixcorr']} CLIP={low['clip_cosine']>mis['clip_cosine']}")
if cn and low:
    checks.append(f"CN_vs_RAS_lowstr dPix={cn['pixcorr']-low['pixcorr']:+.4f} dCLIP={cn['clip_cosine']-low['clip_cosine']:+.4f}")
if cn and off:
    checks.append(f"CN_vs_official dPix={cn['pixcorr']-off['pixcorr']:+.4f} dCLIP={cn['clip_cosine']-off['clip_cosine']:+.4f}")
if cn and cnr:
    checks.append(f"PASS_cn_gt_random_ctrl Pix={cn['pixcorr']>cnr['pixcorr']} CLIP={cn['clip_cosine']>cnr['clip_cosine']}")

brains = []
for sub in ["sub-08","sub-01","sub-02","sub-04","sub-10"]:
    d = load(f"w9_cn_atm_{sub}_brain"); r = load(f"w9_cn_atm_{sub}_random")
    if d and r:
        brains.append(d)
        checks.append(f"atm_{sub} brain>rand CLIP={d['clip_cosine']>r['clip_cosine']} Pix={d['pixcorr']>r['pixcorr']}")
if brains:
    checks.append(
        f"MULTI_CN_ATM_n={len(brains)} Pix={np.mean([b['pixcorr'] for b in brains]):.4f} "
        f"CLIP={np.mean([b['clip_cosine'] for b in brains]):.4f}"
    )

out = Path("outputs/erdc/w9_success_summary.txt")
out.write_text("\n".join(rows + [""] + checks) + "\n", encoding="utf-8")
print(out.read_text())
print("[OK]", out)
PY

echo "[OK] ERDC W9 long pipeline finished"
date
