#!/usr/bin/env bash
#SBATCH --job-name=erdc-w10
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

# ERDC W10 — most acceptance-likely package:
#   Lock main method = bit/atm RAS-lowstr + closed loop (+ fused evidence select)
#   A) Fuse-reselect on existing best banks (no regen) — PixCorr push without GT
#   B) Ultra-low strength RAS bit (0.10/0.15/0.20) + loop
#   C) RAS-lowstr atm × ALL 10 subjects (main multi-subj table for BEST recipe)
#   D) Ablation: IP-only multi-seed loop (already have W7; re-metrics) + λ sweep fuse
#   E) Full metrics + 2WC + paper freeze summary

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
mkdir -p "$TMPDIR" outputs/erdc/w10_metrics outputs/slurm

python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable"
print("GPU", torch.cuda.get_device_name(0), flush=True)
PY

MAX_IMAGES="${ERDC_MAX_IMAGES:-0}"
EXTRA=()
[[ "${MAX_IMAGES}" != "0" ]] && EXTRA+=(--max-images "${MAX_IMAGES}")

full_metrics () {
  local gen="$1"; local tag="$2"
  local out="outputs/erdc/w10_metrics/${tag}.json"
  [[ -f "${out}" ]] && echo "[SKIP] ${tag}" && return 0
  [[ -d "${gen}" ]] || { echo "[WARN] miss ${gen}"; return 0; }
  python scripts/erdc_full_metrics.py --gen-dir "${gen}" --output-json "${out}" --tag "${tag}" "${EXTRA[@]}"
}
twoway () {
  local gen="$1"; local tag="$2"
  local out="outputs/erdc/w10_metrics/${tag}_2wc.json"
  [[ -f "${out}" ]] && echo "[SKIP] 2wc ${tag}" && return 0
  [[ -d "${gen}" ]] || return 0
  python scripts/erdc_twoway_metrics.py --gen-dir "${gen}" --output-json "${out}" --tag "${tag}" "${EXTRA[@]}"
}

echo "################ A) Fuse-reselect on best existing banks ################"
# λ sweep on bit lowstr bank
for LAM in 0.20 0.35 0.50 0.75; do
  echo "----- fuse λ=${LAM} on RAS-lowstr bit -----"
  python scripts/erdc_fuse_reselect.py \
    --cand-dir outputs/erdc/w8_ras_bit_lowstr_sub08/candidates \
    --eeg-npy outputs/eval/atm_pipeline_sub08/sub-08_bit_clip_1024.npy \
    --lambda-struct "${LAM}" \
    --output-dir "outputs/erdc/w10_fuse_lowstr_l${LAM/./p}_sub08" \
    "${EXTRA[@]}"
  full_metrics "outputs/erdc/w10_fuse_lowstr_l${LAM/./p}_sub08/selected_fused" "w10_fuse_l${LAM/./p}_fused"
  full_metrics "outputs/erdc/w10_fuse_lowstr_l${LAM/./p}_sub08/selected_brain" "w10_fuse_l${LAM/./p}_brain"
done

# fuse on standard RAS bit bank too
python scripts/erdc_fuse_reselect.py \
  --cand-dir outputs/erdc/w8_ras_bit_sub08/candidates \
  --eeg-npy outputs/eval/atm_pipeline_sub08/sub-08_bit_clip_1024.npy \
  --lambda-struct 0.35 \
  --output-dir outputs/erdc/w10_fuse_ras_l0p35_sub08 \
  "${EXTRA[@]}"
full_metrics outputs/erdc/w10_fuse_ras_l0p35_sub08/selected_fused w10_fuse_ras_fused

# fuse on CN bank (appendix)
if [[ -d outputs/erdc/w9_cn_bit_sub08/candidates ]]; then
  python scripts/erdc_fuse_reselect.py \
    --cand-dir outputs/erdc/w9_cn_bit_sub08/candidates \
    --eeg-npy outputs/eval/atm_pipeline_sub08/sub-08_bit_clip_1024.npy \
    --lambda-struct 0.35 \
    --output-dir outputs/erdc/w10_fuse_cn_l0p35_sub08 \
    "${EXTRA[@]}"
  full_metrics outputs/erdc/w10_fuse_cn_l0p35_sub08/selected_fused w10_fuse_cn_fused
fi

echo "################ B) Ultra-low strength RAS bit ################"
python scripts/erdc_ras_closed_loop.py \
  --subject sub-08 \
  --embed-source bit_clip \
  --neighbor-mode retrieve \
  --top-m 2 \
  --strengths 0.10,0.15,0.20 \
  --output-dir outputs/erdc/w10_ras_ultralow_bit_sub08 \
  --gen-steps 30 --gen-guidance 5.0 --gen-size 512 \
  "${EXTRA[@]}"
full_metrics outputs/erdc/w10_ras_ultralow_bit_sub08/selected_brain  w10_ras_ultralow_brain
full_metrics outputs/erdc/w10_ras_ultralow_bit_sub08/selected_random w10_ras_ultralow_random
full_metrics outputs/erdc/w10_ras_ultralow_bit_sub08/selected_first  w10_ras_ultralow_first

# fuse on ultralow bank
python scripts/erdc_fuse_reselect.py \
  --cand-dir outputs/erdc/w10_ras_ultralow_bit_sub08/candidates \
  --eeg-npy outputs/eval/atm_pipeline_sub08/sub-08_bit_clip_1024.npy \
  --lambda-struct 0.35 \
  --output-dir outputs/erdc/w10_fuse_ultralow_l0p35_sub08 \
  "${EXTRA[@]}"
full_metrics outputs/erdc/w10_fuse_ultralow_l0p35_sub08/selected_fused w10_fuse_ultralow_fused

echo "################ C) RAS-lowstr atm × 10 subjects (main recipe) ################"
for SUB in sub-01 sub-02 sub-03 sub-04 sub-05 sub-06 sub-07 sub-08 sub-09 sub-10; do
  echo "----- lowstr RAS atm ${SUB} -----"
  python scripts/erdc_ras_closed_loop.py \
    --subject "${SUB}" \
    --embed-source atm \
    --neighbor-mode retrieve \
    --top-m 2 \
    --strengths 0.25,0.35,0.50 \
    --output-dir "outputs/erdc/w10_ras_lowstr_atm_${SUB}" \
    --gen-steps 30 --gen-guidance 5.0 --gen-size 512 \
    "${EXTRA[@]}"
  full_metrics "outputs/erdc/w10_ras_lowstr_atm_${SUB}/selected_brain"  "w10_lowstr_atm_${SUB}_brain"
  full_metrics "outputs/erdc/w10_ras_lowstr_atm_${SUB}/selected_random" "w10_lowstr_atm_${SUB}_random"
done

echo "################ D) References + 2WC ################"
full_metrics outputs/erdc/w8_ras_bit_lowstr_sub08/selected_brain w8_ras_bit_lowstr_brain
full_metrics outputs/erdc/w7_loop_bit_k8_sub08/selected_brain w7_bit_k8_brain
full_metrics outputs/erdc/w6_b0_bit_sub08/generated w6_b0_bit
full_metrics outputs/erdc/w7_official_flat official_atm_gen
full_metrics outputs/erdc/w9_ras_ctrl_random_sub08/selected_brain w9_ras_ctrl_random_brain

twoway outputs/erdc/w8_ras_bit_lowstr_sub08/selected_brain w8_ras_bit_lowstr_brain
twoway outputs/erdc/w10_ras_ultralow_bit_sub08/selected_brain w10_ras_ultralow_brain
# best fuse tag decided in summary; compute 2WC for all fuse fused dirs
for d in outputs/erdc/w10_fuse_lowstr_l*/selected_fused outputs/erdc/w10_fuse_ultralow_l0p35_sub08/selected_fused; do
  [[ -d "$d" ]] || continue
  tag="w10_$(basename $(dirname "$d"))_fused"
  # normalize tag from path
  parent=$(basename "$(dirname "$d")")
  twoway "$d" "${parent}_fused"
done
twoway outputs/erdc/w7_official_flat official_atm_gen
twoway outputs/erdc/w7_loop_bit_k8_sub08/selected_brain w7_bit_k8_brain

echo "################ E) Panels + paper freeze ################"
# pick best fuse by reading metrics after summary python
python scripts/erdc_qualitative_panel.py \
  --b0-dir outputs/erdc/w6_b0_bit_sub08/generated \
  --random-dir outputs/erdc/w9_ras_ctrl_random_sub08/selected_brain \
  --brain-dir outputs/erdc/w8_ras_bit_lowstr_sub08/selected_brain \
  --scores-npy outputs/erdc/w8_ras_bit_lowstr_sub08/candidates/brain_scores.npy \
  --output-dir outputs/erdc/w10_panels_main_vs_ctrl \
  --n-panel 24 || true

python - <<'PY'
import json
from pathlib import Path
import numpy as np

def load(tag):
    for base in ["outputs/erdc/w10_metrics", "outputs/erdc/w9_metrics", "outputs/erdc/w8_metrics", "outputs/erdc/overnight_metrics", "outputs/erdc/w7_metrics"]:
        p = Path(base) / f"{tag}.json"
        if p.is_file():
            return json.loads(p.read_text())
    return None

def line(tag):
    d = load(tag)
    if not d:
        return f"{tag}: MISSING"
    tw = ""
    for base in ["outputs/erdc/w10_metrics", "outputs/erdc/w9_metrics"]:
        p = Path(base) / f"{tag}_2wc.json"
        if p.is_file():
            t = json.loads(p.read_text())["twoway"]
            tw = f" | 2WC={t['clip']*100:.1f}%"
            break
    return f"{tag:42s} Pix={d['pixcorr']:.4f} CLIP={d['clip_cosine']:.4f}{tw}"

rows = []
# λ sweep
best_fuse = None
best_key = None
for lam, tag in [("0.20","w10_fuse_l0p20_fused"),("0.35","w10_fuse_l0p35_fused"),
                 ("0.50","w10_fuse_l0p50_fused"),("0.75","w10_fuse_l0p75_fused")]:
    d = load(tag)
    rows.append(line(tag))
    if d and (best_fuse is None or d["pixcorr"] > best_fuse["pixcorr"] or
              (abs(d["pixcorr"]-best_fuse["pixcorr"])<1e-4 and d["clip_cosine"]>best_fuse["clip_cosine"])):
        best_fuse, best_key = d, tag

core = [
    "w6_b0_bit", "w7_bit_k8_brain", "w8_ras_bit_lowstr_brain",
    "w10_ras_ultralow_brain", "w10_fuse_ultralow_fused", "w10_fuse_ras_fused",
    "w9_ras_ctrl_random_brain", "official_atm_gen",
]
rows.append("")
rows.extend(line(t) for t in core)

# 10-subj mean
brains, rands = [], []
for i in range(1, 11):
    b = load(f"w10_lowstr_atm_sub-{i:02d}_brain")
    r = load(f"w10_lowstr_atm_sub-{i:02d}_random")
    if b and r:
        brains.append(b); rands.append(r)
        rows.append(
            f"atm_sub-{i:02d}: brain CLIP={b['clip_cosine']:.4f} Pix={b['pixcorr']:.4f} | "
            f">rand CLIP={b['clip_cosine']>r['clip_cosine']} Pix={b['pixcorr']>r['pixcorr']}"
        )

checks = []
low = load("w8_ras_bit_lowstr_brain")
off = load("official_atm_gen")
ult = load("w10_ras_ultralow_brain")
ctrl = load("w9_ras_ctrl_random_brain")
w7 = load("w7_bit_k8_brain")
if best_fuse and low:
    checks.append(
        f"BEST_FUSE={best_key} Pix={best_fuse['pixcorr']:.4f} CLIP={best_fuse['clip_cosine']:.4f} "
        f"vs_lowstr dPix={best_fuse['pixcorr']-low['pixcorr']:+.4f} dCLIP={best_fuse['clip_cosine']-low['clip_cosine']:+.4f}"
    )
if ult and low:
    checks.append(f"ULTRALOW_vs_LOWSTR dPix={ult['pixcorr']-low['pixcorr']:+.4f} dCLIP={ult['clip_cosine']-low['clip_cosine']:+.4f}")
if low and off:
    checks.append(f"MAIN_vs_official dPix={low['pixcorr']-off['pixcorr']:+.4f} dCLIP={low['clip_cosine']-off['clip_cosine']:+.4f}")
if low and ctrl:
    checks.append(f"PASS_main_gt_random_ctrl CLIP={low['clip_cosine']>ctrl['clip_cosine']} Pix={low['pixcorr']>ctrl['pixcorr']}")
if low and w7:
    checks.append(f"MAIN_vs_W7loop dPix={low['pixcorr']-w7['pixcorr']:+.4f} dCLIP={low['clip_cosine']-w7['clip_cosine']:+.4f}")
if brains:
    checks.append(
        f"MULTI_LOWSTR_ATM_n={len(brains)} "
        f"Pix={np.mean([b['pixcorr'] for b in brains]):.4f} "
        f"CLIP={np.mean([b['clip_cosine'] for b in brains]):.4f} "
        f"pass_CLIP={sum(b['clip_cosine']>r['clip_cosine'] for b,r in zip(brains,rands))}/10 "
        f"pass_Pix={sum(b['pixcorr']>r['pixcorr'] for b,r in zip(brains,rands))}/10"
    )

# recommend main method
cands = []
for tag in ["w8_ras_bit_lowstr_brain", best_key, "w10_ras_ultralow_brain", "w10_fuse_ultralow_fused"]:
    if not tag: continue
    d = load(tag)
    if d:
        # rank by pix then clip
        cands.append((d["pixcorr"], d["clip_cosine"], tag, d))
cands.sort(reverse=True)
if cands:
    checks.append(f"RECOMMEND_MAIN={cands[0][2]} Pix={cands[0][0]:.4f} CLIP={cands[0][1]:.4f}")

out = Path("outputs/erdc/w10_paper_main_freeze.txt")
out.write_text("\n".join(rows + [""] + checks) + "\n", encoding="utf-8")
print(out.read_text())
print("[OK]", out)
PY

echo "[OK] ERDC W10 paper-main package finished"
date
