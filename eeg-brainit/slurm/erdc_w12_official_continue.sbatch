#!/usr/bin/env bash
#SBATCH --job-name=erdc-w12c
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

# W12 continuation — fuse + remaining official runs (A1/A2 already done in 527477)

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
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export XFORMERS_DISABLED=1
mkdir -p "$TMPDIR" outputs/erdc/w12_metrics outputs/slurm

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
[[ -f "${S4_NPY}" ]] && BIT_NPY="${S4_NPY}"

echo "################ B) Fuse on completed official banks ################"
for LAM in 0.25 0.35 0.45; do
  python scripts/erdc_fuse_reselect.py \
    --cand-dir outputs/erdc/w12_official_bit_img_sub08/candidates \
    --eeg-npy "${BIT_NPY}" \
    --lambda-struct "${LAM}" \
    --output-dir "outputs/erdc/w12_fuse_bit_img_l${LAM/./p}" \
    "${EXTRA[@]}"
  full_metrics "outputs/erdc/w12_fuse_bit_img_l${LAM/./p}/selected_fused" "w12_fuse_bit_l${LAM/./p}"
done

python scripts/erdc_fuse_reselect.py \
  --cand-dir outputs/erdc/w12_official_prior_img_sub08/candidates \
  --eeg-npy outputs/atm_bridge/sub-08_test_eeg_1024.npy \
  --lambda-struct 0.35 \
  --output-dir outputs/erdc/w12_fuse_prior_img_l0p35 \
  "${EXTRA[@]}"
full_metrics outputs/erdc/w12_fuse_prior_img_l0p35/selected_fused w12_fuse_prior_l0p35

echo "################ A3/A4 remaining official runs ################"
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
    rows.append(f"{tag:42s} Pix={d['pixcorr']:.4f} CLIP={d['clip_cosine']:.4f}")
    if best is None or d["pixcorr"] > best["pixcorr"] or (
        abs(d["pixcorr"] - best["pixcorr"]) < 1e-4 and d["clip_cosine"] > best["clip_cosine"]
    ):
        best, best_tag = d, tag

off = load("official_atm_gen")
w10 = load("w10_sdxl_fuse_ref")
checks = []
if best and off:
    checks.append(
        f"BEST_W12={best_tag} Pix={best['pixcorr']:.4f} CLIP={best['clip_cosine']:.4f} "
        f"vs_official dPix={best['pixcorr']-off['pixcorr']:+.4f} PASS_PIX={best['pixcorr']>=0.15}"
    )
if best and w10:
    checks.append(f"vs_W10_SDXL_fuse dPix={best['pixcorr']-w10['pixcorr']:+.4f} dCLIP={best['clip_cosine']-w10['clip_cosine']:+.4f}")

# inline oracle from bit metrics.json if present
mp = Path("outputs/erdc/w12_official_bit_img_sub08/metrics.json")
if mp.is_file():
    m = json.loads(mp.read_text())
    if "oracle" in m.get("selection", {}):
        o = m["selection"]["oracle"]["metrics"]
        checks.append(f"BIT_ORACLE_inline Pix={o['pixcorr']:.4f} CLIP={o['clip_cosine']:.4f}")

out = Path("outputs/erdc/w12_official_main_freeze.txt")
out.write_text("\n".join(rows) + "\n\n" + "\n".join(checks) + "\n", encoding="utf-8")
print(out.read_text())
print("[OK]", out)
PY

echo "[OK] W12 continuation finished"
date
