#!/usr/bin/env bash
#SBATCH --job-name=erdc-w14
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

# W14 — paper finalize / wrap-up:
#   A) sub-08: fuse mechanism controls (shuffle/misalign/zero) on W13 merged bank
#   B) sub-08: 2WC on best W12/W13 lines + references
#   C) Export bit_clip embeds for all 10 subjects (sub-08 S3 ckpt)
#   D) 10-subject official Turbo: bit_clip bank + fuse λ=0.15 + atm brain
#   E) Qualitative panel + paper freeze summary

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
mkdir -p "$TMPDIR" outputs/erdc/w14_metrics outputs/eval/atm_pipeline_all outputs/slurm

python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable"
print("GPU", torch.cuda.get_device_name(0), flush=True)
PY

MAX_IMAGES="${ERDC_MAX_IMAGES:-0}"
EXTRA=()
[[ "${MAX_IMAGES}" != "0" ]] && EXTRA+=(--max-images "${MAX_IMAGES}")

MET_DIR="outputs/erdc/w14_metrics"
W13_MERGED_CAND="outputs/erdc/w13_merged_turbo/candidates"
BIT_NPY_SUB08="outputs/eval/atm_pipeline_sub08/sub-08_bit_clip_1024.npy"
S3_CKPT="outputs/atm_distill_s3_sub08/checkpoints/atm_stage3_best.pt"
S1_CKPT="outputs/atm_distill_s1_sub08/checkpoints/atm_stage1_best.pt"
FUSE_LAM="0.15"

full_metrics () {
  local gen="$1"; local tag="$2"
  local out="${MET_DIR}/${tag}.json"
  [[ -f "${out}" ]] && echo "[SKIP] ${tag}" && return 0
  [[ -d "${gen}" ]] || { echo "[WARN] miss ${gen}"; return 0; }
  python scripts/erdc_full_metrics.py --gen-dir "${gen}" --output-json "${out}" --tag "${tag}" "${EXTRA[@]}"
}

twoway () {
  local gen="$1"; local tag="$2"
  local out="${MET_DIR}/${tag}_2wc.json"
  [[ -f "${out}" ]] && echo "[SKIP] 2wc ${tag}" && return 0
  [[ -d "${gen}" ]] || return 0
  python scripts/erdc_twoway_metrics.py --gen-dir "${gen}" --output-json "${out}" --tag "${tag}" "${EXTRA[@]}"
}

echo "################ A) Fuse mechanism controls (W13 merged, λ=${FUSE_LAM}) ################"
if [[ -d "${W13_MERGED_CAND}" ]]; then
  for MODE in retrieve shuffle misalign zero; do
    TAG="w14_merged_ctrl_${MODE}"
    OUT="outputs/erdc/${TAG}"
    python scripts/erdc_fuse_reselect.py \
      --cand-dir "${W13_MERGED_CAND}" \
      --eeg-npy "${BIT_NPY_SUB08}" \
      --lambda-struct "${FUSE_LAM}" \
      --struct-mode "${MODE}" \
      --output-dir "${OUT}" \
      "${EXTRA[@]}"
    full_metrics "${OUT}/selected_fused" "${TAG}"
  done
else
  echo "[WARN] missing ${W13_MERGED_CAND}; run W13 merge first"
fi

echo "################ B) 2WC on sub-08 best lines ################"
full_metrics outputs/erdc/w7_official_flat official_atm_gen
full_metrics outputs/erdc/w13_merged_fuse_l0p15/selected_brain w13_merged_brain
twoway outputs/erdc/w7_official_flat official_atm_gen
twoway outputs/erdc/w12_official_atm_img_sub08/selected_brain w12_official_atm_brain
twoway outputs/erdc/w13_merged_fuse_l0p15/selected_fused w13_merged_fuse_l0p15
twoway outputs/erdc/w13_merged_fuse_l0p15/selected_brain w13_merged_brain
twoway outputs/erdc/w13_atm_fuse_l0p15/selected_fused w13_atm_fuse_l0p15
twoway outputs/erdc/w12_fuse_bit_img_l0p25/selected_fused w12_fuse_bit_l0p25
twoway outputs/erdc/w10_fuse_lowstr_l0p35_sub08/selected_fused w10_sdxl_fuse_ref
for MODE in retrieve shuffle misalign zero; do
  twoway "outputs/erdc/w14_merged_ctrl_${MODE}/selected_fused" "w14_merged_ctrl_${MODE}"
done

echo "################ C) Export bit_clip embeds (10 subjects, sub-08 S3) ################"
for SUB in sub-01 sub-02 sub-03 sub-04 sub-05 sub-06 sub-07 sub-08 sub-09 sub-10; do
  OUT_NPY="outputs/eval/atm_pipeline_all/${SUB}_bit_clip_1024.npy"
  if [[ -f "${OUT_NPY}" ]]; then
    echo "[SKIP] embed ${SUB}"
    continue
  fi
  echo "----- export bit_clip ${SUB} -----"
  python scripts/eval_atm_pipeline.py \
    --subject "${SUB}" \
    --prior-ckpt "checkpoints/atm_diffusion_prior/${SUB}/diffusion_prior.pt" \
    --s1-ckpt "${S1_CKPT}" \
    --s3-ckpt "${S3_CKPT}" \
    --output-dir outputs/eval/atm_pipeline_all \
    --skip-generate \
    --gen-sources bit_clip \
    "${EXTRA[@]}"
done

echo "################ D) 10-subject Turbo official + fuse ################"
for SUB in sub-01 sub-02 sub-03 sub-04 sub-05 sub-06 sub-07 sub-08 sub-09 sub-10; do
  BIT_NPY="outputs/eval/atm_pipeline_all/${SUB}_bit_clip_1024.npy"
  if [[ ! -f "${BIT_NPY}" ]]; then
    echo "[WARN] missing ${BIT_NPY}; skip ${SUB}"
    continue
  fi

  BIT_OUT="outputs/erdc/w14_turbo_bit_${SUB}"
  if [[ ! -f "${BIT_OUT}/metrics.json" ]]; then
    echo "----- Turbo bit_clip ${SUB} -----"
    python scripts/erdc_official_atm_pipeline.py \
      --subject "${SUB}" \
      --embed-source bit_clip \
      --bit-npy "${BIT_NPY}" \
      --low-level-mode neighbor_image \
      --top-m 2 \
      --strengths 0.35,0.50,0.65,0.85 \
      --use-turbo --gen-steps 4 --gen-guidance 0.0 \
      --output-dir "${BIT_OUT}" \
      "${EXTRA[@]}"
  else
    echo "[SKIP] gen ${BIT_OUT}"
  fi
  full_metrics "${BIT_OUT}/selected_brain" "w14_turbo_bit_${SUB}_brain"

  FUSE_OUT="outputs/erdc/w14_fuse_bit_${SUB}_l${FUSE_LAM/./p}"
  if [[ ! -f "${FUSE_OUT}/metrics.json" ]]; then
    echo "----- fuse bit ${SUB} λ=${FUSE_LAM} -----"
    python scripts/erdc_fuse_reselect.py \
      --cand-dir "${BIT_OUT}/candidates" \
      --eeg-npy "${BIT_NPY}" \
      --lambda-struct "${FUSE_LAM}" \
      --output-dir "${FUSE_OUT}" \
      "${EXTRA[@]}"
  fi
  full_metrics "${FUSE_OUT}/selected_fused" "w14_fuse_bit_${SUB}_l${FUSE_LAM/./p}"

  ATM_OUT="outputs/erdc/w14_turbo_atm_${SUB}"
  if [[ ! -f "${ATM_OUT}/metrics.json" ]]; then
    echo "----- Turbo atm ${SUB} -----"
    python scripts/erdc_official_atm_pipeline.py \
      --subject "${SUB}" \
      --embed-source atm \
      --low-level-mode neighbor_image \
      --top-m 2 \
      --strengths 0.50,0.65,0.85 \
      --use-turbo --gen-steps 4 --gen-guidance 0.0 \
      --output-dir "${ATM_OUT}" \
      "${EXTRA[@]}"
  else
    echo "[SKIP] gen ${ATM_OUT}"
  fi
  full_metrics "${ATM_OUT}/selected_brain" "w14_turbo_atm_${SUB}_brain"

  twoway "${FUSE_OUT}/selected_fused" "w14_fuse_bit_${SUB}_l${FUSE_LAM/./p}"
done

echo "################ E) Qualitative panel (sub-08) ################"
python scripts/erdc_qualitative_panel.py \
  --b0-dir outputs/erdc/w7_official_flat \
  --random-dir outputs/erdc/w12_official_bit_img_sub08/selected_random \
  --brain-dir outputs/erdc/w13_merged_fuse_l0p15/selected_fused \
  --scores-npy outputs/erdc/w13_merged_fuse_l0p15/fused_scores.npy \
  --output-dir outputs/erdc/w14_panels_main \
  --n-panel 24 || true

echo "################ F) Paper freeze ################"
python - <<'PY'
import json
from pathlib import Path
import numpy as np

OFF_TAG = "official_atm_gen"
MET = Path("outputs/erdc/w14_metrics")
W12 = Path("outputs/erdc/w12_metrics")
W13 = Path("outputs/erdc/w13_metrics")


def load_tag(tag: str):
    for base in [MET, W13, W12, Path("outputs/erdc/w10_metrics"), Path("outputs/erdc/w7_metrics")]:
        p = base / f"{tag}.json"
        if p.is_file():
            return json.loads(p.read_text()), p
    return None, None


def load_2wc(tag: str):
    p = MET / f"{tag}_2wc.json"
    if p.is_file():
        return json.loads(p.read_text())["twoway"]
    p = Path("outputs/erdc/w10_metrics") / f"{tag}_2wc.json"
    if p.is_file():
        return json.loads(p.read_text())["twoway"]
    return None


def line(tag: str):
    d, _ = load_tag(tag)
    if not d:
        return f"{tag}: MISSING"
    tw = load_2wc(tag)
    tw_s = f" | 2WC={tw['clip']*100:.1f}%" if tw else ""
    return f"{tag:46s} Pix={d['pixcorr']:.4f} CLIP={d['clip_cosine']:.4f}{tw_s}"


rows = ["=== sub-08 main lines ==="]
for tag in [
    OFF_TAG,
    "w12_official_atm_brain",
    "w13_merged_fuse_l0p15",
    "w13_merged_brain",
    "w13_atm_fuse_l0p15",
    "w12_fuse_bit_l0p25",
    "w10_sdxl_fuse_ref",
]:
    rows.append(line(tag))

rows.append("")
rows.append("=== fuse mechanism (merged bank λ=0.15) ===")
for mode in ["retrieve", "shuffle", "misalign", "zero"]:
    rows.append(line(f"w14_merged_ctrl_{mode}"))

off, _ = load_tag(OFF_TAG)
ctrl_ret, _ = load_tag("w14_merged_ctrl_retrieve")
ctrl_shuf, _ = load_tag("w14_merged_ctrl_shuffle")
main, _ = load_tag("w13_merged_fuse_l0p15")

checks = []
if off and main:
    checks.append(
        f"MAIN_vs_official dPix={main['pixcorr']-off['pixcorr']:+.4f} "
        f"dCLIP={main['clip_cosine']-off['clip_cosine']:+.4f} "
        f"PASS_BOTH={main['pixcorr']>=off['pixcorr'] and main['clip_cosine']>=off['clip_cosine']}"
    )
if ctrl_ret and ctrl_shuf:
    checks.append(
        f"MECH_shuffle_vs_retrieve dPix={ctrl_shuf['pixcorr']-ctrl_ret['pixcorr']:+.4f} "
        f"dCLIP={ctrl_shuf['clip_cosine']-ctrl_ret['clip_cosine']:+.4f}"
    )

# 10-subject fuse summary
fuse_rows, fuse_pix, fuse_clip = [], [], []
atm_rows, atm_pix, atm_clip = [], [], []
for i in range(1, 11):
    sub = f"sub-{i:02d}"
    ft = f"w14_fuse_bit_{sub}_l0p15"
    at = f"w14_turbo_atm_{sub}_brain"
    fd, _ = load_tag(ft)
    ad, _ = load_tag(at)
    if fd:
        fuse_pix.append(fd["pixcorr"])
        fuse_clip.append(fd["clip_cosine"])
        fuse_rows.append(f"{sub} fuse: Pix={fd['pixcorr']:.4f} CLIP={fd['clip_cosine']:.4f}")
    if ad:
        atm_pix.append(ad["pixcorr"])
        atm_clip.append(ad["clip_cosine"])
        atm_rows.append(f"{sub} atm_brain: Pix={ad['pixcorr']:.4f} CLIP={ad['clip_cosine']:.4f}")

if fuse_rows:
    rows.append("")
    rows.append("=== 10-subject Turbo bit fuse λ=0.15 ===")
    rows.extend(fuse_rows)
    checks.append(
        f"MULTI_FUSE_n={len(fuse_pix)} "
        f"Pix_mean={np.mean(fuse_pix):.4f} CLIP_mean={np.mean(fuse_clip):.4f}"
    )
    if off:
        checks.append(
            f"MULTI_FUSE_vs_official Pix_pass={sum(p>=off['pixcorr'] for p in fuse_pix)}/10 "
            f"CLIP_pass={sum(c>=off['clip_cosine'] for c in fuse_clip)}/10"
        )

if atm_rows:
    rows.append("")
    rows.append("=== 10-subject Turbo atm brain ===")
    rows.extend(atm_rows)
    checks.append(
        f"MULTI_ATM_n={len(atm_pix)} "
        f"Pix_mean={np.mean(atm_pix):.4f} CLIP_mean={np.mean(atm_clip):.4f}"
    )
    if off:
        checks.append(
            f"MULTI_ATM_vs_official Pix_pass={sum(p>=off['pixcorr'] for p in atm_pix)}/10 "
            f"CLIP_pass={sum(c>=off['clip_cosine'] for c in atm_clip)}/10"
        )

# 2WC summary
tw_rows = []
for tag in ["official_atm_gen", "w13_merged_fuse_l0p15", "w13_atm_fuse_l0p15", "w10_sdxl_fuse_ref"]:
    tw = load_2wc(tag)
    if tw:
        tw_rows.append(f"{tag:46s} 2WC_CLIP={tw['clip']*100:.1f}% Alex2={tw['alex2']*100:.1f}%")
if tw_rows:
    rows.append("")
    rows.append("=== 2WC (CLIP) ===")
    rows.extend(tw_rows)

out = Path("outputs/erdc/w14_paper_finalize_freeze.txt")
out.write_text("\n".join(rows) + "\n\n" + "\n".join(checks) + "\n", encoding="utf-8")
print(out.read_text())
print("[OK]", out)
PY

echo "[OK] W14 paper finalize finished"
date
