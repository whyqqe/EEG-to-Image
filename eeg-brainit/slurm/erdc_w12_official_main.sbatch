#!/usr/bin/env bash
#SBATCH --job-name=erdc-w12
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

# ERDC W12 — optimal route: official ATM mixed decoder (SDXL-Turbo + low-level + prior)
# + our bit_clip embed + ERDC brain loop + fuse. W11 cancelled (SDXL-only ceiling).

set -euo pipefail
ROOT=/project/peilab/why/eeg-brainit
module load slurm nvhpc-hpcx-cuda12/23.11
cd "${ROOT}"
source scripts/activate.sh
export PYTHONUNBUFFERED=1
export HF_HOME=/project/peilab/why/cache/eeg-brainit/hf
export HF_HUB_CACHE=/project/peilab/why/cache/eeg-brainit/hf/hub
export OPENCLIP_CACHE_DIR=/project/peilab/why/cache/eeg-brainit/open_clip
export TORCH_HOME=/project/peilab/why/cache/eeg-brainit/torch
export TMPDIR=/project/peilab/why/cache/eeg-brainit/tmp
export XFORMERS_DISABLED=1
mkdir -p "$TMPDIR" outputs/erdc/w12_metrics outputs/slurm

# Allow download for turbo + latents only in this phase
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE || true
python scripts/download_atm_official_assets.py
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python - <<'PY'
import torch
assert torch.cuda.is_available()
print("GPU", torch.cuda.get_device_name(0))
PY

MAX_IMAGES="${ERDC_MAX_IMAGES:-0}"
EXTRA=()
[[ "${MAX_IMAGES}" != "0" ]] && EXTRA+=(--max-images "${MAX_IMAGES}")

full_metrics () {
  local gen="$1"; local tag="$2"
  local out="outputs/erdc/w12_metrics/${tag}.json"
  [[ -f "${out}" ]] && echo "[SKIP] ${tag}" && return 0
  [[ -d "${gen}" ]] || return 0
  python scripts/erdc_full_metrics.py --gen-dir "${gen}" --output-json "${out}" --tag "${tag}" "${EXTRA[@]}"
}

BIT_NPY="outputs/eval/atm_pipeline_sub08/sub-08_bit_clip_1024.npy"
S4_NPY="outputs/eval/atm_pipeline_sub08/sub-08_bit_clip_s4_pix_1024.npy"
if [[ -f "${S4_NPY}" ]]; then
  BIT_NPY="${S4_NPY}"
  echo "[INFO] using S4 embed ${S4_NPY}"
fi

echo "################ A) Official mixed pipeline (paper decoder stack) ################"
# A1: closest to paper — prior(eeg) + neighbor image low-level + turbo
python scripts/erdc_official_atm_pipeline.py \
  --subject sub-08 \
  --embed-source prior_atm \
  --low-level-mode neighbor_image \
  --top-m 2 \
  --strengths 0.35,0.50,0.65,0.85 \
  --use-turbo --gen-steps 4 --gen-guidance 0.0 \
  --output-dir outputs/erdc/w12_official_prior_img_sub08 \
  "${EXTRA[@]}"
full_metrics outputs/erdc/w12_official_prior_img_sub08/selected_brain w12_official_prior_brain
full_metrics outputs/erdc/w12_official_prior_img_sub08/selected_first  w12_official_prior_first

# A2: our bit_clip + same official low-level stack
python scripts/erdc_official_atm_pipeline.py \
  --subject sub-08 \
  --embed-source bit_clip \
  --bit-npy "${BIT_NPY}" \
  --low-level-mode neighbor_image \
  --top-m 2 \
  --strengths 0.35,0.50,0.65,0.85 \
  --use-turbo --gen-steps 4 --gen-guidance 0.0 \
  --output-dir outputs/erdc/w12_official_bit_img_sub08 \
  "${EXTRA[@]}"
full_metrics outputs/erdc/w12_official_bit_img_sub08/selected_brain w12_official_bit_brain

# A3: neighbor VAE latent low-level (gallery latent, no GT)
python scripts/erdc_official_atm_pipeline.py \
  --subject sub-08 \
  --embed-source prior_atm \
  --low-level-mode neighbor_latent \
  --top-m 2 \
  --strengths 0.50,0.65,0.85 \
  --use-turbo --gen-steps 4 --gen-guidance 0.0 \
  --output-dir outputs/erdc/w12_official_prior_latent_sub08 \
  "${EXTRA[@]}"
full_metrics outputs/erdc/w12_official_prior_latent_sub08/selected_brain w12_official_prior_latent_brain

# A4: raw ATM embed baseline on official stack
python scripts/erdc_official_atm_pipeline.py \
  --subject sub-08 \
  --embed-source atm \
  --low-level-mode neighbor_image \
  --top-m 2 \
  --strengths 0.50,0.65,0.85 \
  --use-turbo --gen-steps 4 --gen-guidance 0.0 \
  --output-dir outputs/erdc/w12_official_atm_img_sub08 \
  "${EXTRA[@]}"
full_metrics outputs/erdc/w12_official_atm_img_sub08/selected_brain w12_official_atm_brain

echo "################ B) Fuse on best official bank ################"
python scripts/erdc_fuse_reselect.py \
  --cand-dir outputs/erdc/w12_official_prior_img_sub08/candidates \
  --eeg-npy outputs/atm_bridge/sub-08_test_eeg_1024.npy \
  --lambda-struct 0.35 \
  --struct-mode retrieve \
  --output-dir outputs/erdc/w12_fuse_prior_img_l0p35 \
  "${EXTRA[@]}"
full_metrics outputs/erdc/w12_fuse_prior_img_l0p35/selected_fused w12_fuse_prior_fused

python scripts/erdc_fuse_reselect.py \
  --cand-dir outputs/erdc/w12_official_bit_img_sub08/candidates \
  --eeg-npy "${BIT_NPY}" \
  --lambda-struct 0.35 \
  --output-dir outputs/erdc/w12_fuse_bit_img_l0p35 \
  "${EXTRA[@]}"
full_metrics outputs/erdc/w12_fuse_bit_img_l0p35/selected_fused w12_fuse_bit_fused

echo "################ C) References ################"
full_metrics outputs/erdc/w10_fuse_lowstr_l0p35_sub08/selected_fused w10_sdxl_fuse_ref
full_metrics outputs/erdc/w7_official_flat official_atm_gen

echo "################ D) Freeze ################"
python - <<'PY'
import json
from pathlib import Path

def load(tag):
    p = Path("outputs/erdc/w12_metrics") / f"{tag}.json"
    return json.loads(p.read_text()) if p.is_file() else None

rows = []
best = None
best_tag = None
for p in sorted(Path("outputs/erdc/w12_metrics").glob("*.json")):
    if "_2wc" in p.name:
        continue
    d = json.loads(p.read_text())
    tag = p.stem
    rows.append(f"{tag:40s} Pix={d['pixcorr']:.4f} CLIP={d['clip_cosine']:.4f}")
    if best is None or d["pixcorr"] > best["pixcorr"]:
        best, best_tag = d, tag

off = load("official_atm_gen")
checks = []
if best and off:
    checks.append(
        f"BEST_W12={best_tag} Pix={best['pixcorr']:.4f} CLIP={best['clip_cosine']:.4f} "
        f"vs_official dPix={best['pixcorr']-off['pixcorr']:+.4f} PASS_PIX={best['pixcorr']>=0.15}"
    )
out = Path("outputs/erdc/w12_official_main_freeze.txt")
out.write_text("\n".join(rows) + "\n\n" + "\n".join(checks) + "\n")
print(out.read_text())
print("[OK]", out)
PY

echo "[OK] ERDC W12 finished"
date
