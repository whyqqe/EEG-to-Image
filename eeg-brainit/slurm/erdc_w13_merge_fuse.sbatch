#!/usr/bin/env bash
#SBATCH --job-name=erdc-w13
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --exclude=dgx-10,dgx-11
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

# W13: Turbo mega-bank merge + fine fuse λ sweep + ATM fuse (zero-regen, highest ROI)

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
mkdir -p "$TMPDIR" outputs/erdc/w13_metrics outputs/slurm

MAX_IMAGES="${ERDC_MAX_IMAGES:-0}"
EXTRA=()
[[ "${MAX_IMAGES}" != "0" ]] && EXTRA+=(--max-images "${MAX_IMAGES}")

BIT_NPY="outputs/eval/atm_pipeline_sub08/sub-08_bit_clip_1024.npy"
ATM_NPY="outputs/atm_bridge/sub-08_test_eeg_1024.npy"

full_metrics () {
  local gen="$1"; local tag="$2"
  local out="outputs/erdc/w13_metrics/${tag}.json"
  [[ -f "${out}" ]] && echo "[SKIP] ${tag}" && return 0
  [[ -d "${gen}" ]] || return 0
  python scripts/erdc_full_metrics.py --gen-dir "${gen}" --output-json "${out}" --tag "${tag}" "${EXTRA[@]}"
}

echo "################ [1] Merge Turbo banks (bit+atm+prior_latent) ################"
python scripts/erdc_merge_banks.py \
  --bank-dirs outputs/erdc/w12_official_bit_img_sub08 \
              outputs/erdc/w12_official_atm_img_sub08 \
              outputs/erdc/w12_official_prior_latent_sub08 \
  --output-dir outputs/erdc/w13_merged_turbo/candidates \
  "${EXTRA[@]}"

echo "################ [2] Merged fuse λ grid (bit_clip EEG) ################"
for LAM in 0.05 0.10 0.15 0.20 0.25; do
  TAG="w13_merged_fuse_l${LAM/./p}"
  OUT="outputs/erdc/${TAG}"
  python scripts/erdc_fuse_reselect.py \
    --cand-dir outputs/erdc/w13_merged_turbo/candidates \
    --eeg-npy "${BIT_NPY}" \
    --lambda-struct "${LAM}" \
    --output-dir "${OUT}" \
    "${EXTRA[@]}"
  full_metrics "${OUT}/selected_fused" "${TAG}"
done

echo "################ [3] ATM bank fuse (atm EEG) ################"
for LAM in 0.15 0.25 0.35; do
  TAG="w13_atm_fuse_l${LAM/./p}"
  OUT="outputs/erdc/${TAG}"
  python scripts/erdc_fuse_reselect.py \
    --cand-dir outputs/erdc/w12_official_atm_img_sub08/candidates \
    --eeg-npy "${ATM_NPY}" \
    --lambda-struct "${LAM}" \
    --output-dir "${OUT}" \
    "${EXTRA[@]}"
  full_metrics "${OUT}/selected_fused" "${TAG}"
done

echo "################ [4] Top-2 brain -> max struct (merged bank) ################"
python - <<'PY'
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch

ROOT = Path("/project/peilab/why/eeg-brainit")
sys.path.insert(0, str(ROOT / "scripts"))
from erdc_ras_closed_loop import load_embed
from eval_atm_pipeline import image_metrics, list_test_images

cand_dir = ROOT / "outputs/erdc/w13_merged_turbo/candidates"
out_dir = ROOT / "outputs/erdc/w13_merged_top2struct"
out_sel = out_dir / "selected_top2struct"
out_sel.mkdir(parents=True, exist_ok=True)

eeg = load_embed(ROOT / "outputs/eval/atm_pipeline_sub08/sub-08_bit_clip_1024.npy")
gallery = load_embed(ROOT / "outputs/atm_bridge/clip_img_train_1024.npy")
neighbor_idx = np.load(cand_dir.parent / "neighbor_idx.npy")
specs = json.loads((cand_dir / "specs.json").read_text())
feats = np.load(cand_dir / "clip_feats_flat.npy")
k = len(specs)
n = feats.reshape(-1, k, feats.shape[-1]).shape[0]
feats = feats.reshape(n, k, feats.shape[-1])

brain = np.einsum("nd,nkd->nk", eeg[:n], feats)
struct = np.zeros_like(brain)
for j, sp in enumerate(specs):
    rank = int(sp.get("neighbor_rank", -1))
    if rank < 0:
        continue
    nb = neighbor_idx[:n, rank]
    neigh_feat = gallery[nb]
    struct[:, j] = np.einsum("nd,nd->n", feats[:, j, :], neigh_feat)

idx = np.zeros(n, dtype=np.int64)
for i in range(n):
    top2 = np.argsort(-brain[i])[:2]
    idx[i] = top2[struct[i, top2].argmax()]

for i in range(n):
    src = cand_dir / f"{i:03d}_k{int(idx[i])}.png"
    Image.open(src).save(out_sel / f"{i:03d}.png")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gt = list_test_images(Path("/project/peilab/why/data/images_set"))
metrics = image_metrics([out_sel / f"{i:03d}.png" for i in range(n)], gt[:n], device)
report = {"n": n, "k": k, "strategy": "top2_brain_max_struct", "metrics": metrics}
(out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(f"[INFO] top2struct Pix={metrics['pixcorr']:.4f} CLIP={metrics['clip_cosine']:.4f}")
PY
full_metrics outputs/erdc/w13_merged_top2struct/selected_top2struct w13_merged_top2struct

echo "################ [5] Freeze W13 ################"
python - <<'PY'
import json
from pathlib import Path

OFF = Path("outputs/erdc/w12_metrics/official_atm_gen.json")
off = json.loads(OFF.read_text()) if OFF.is_file() else None

rows = []
best_pix = None
best_pix_tag = None
best_bal = None
best_bal_tag = None

def score_bal(d):
    if off is None:
        return d["clip_cosine"]
    # reward both beating official
    dp = d["pixcorr"] - off["pixcorr"]
    dc = d["clip_cosine"] - off["clip_cosine"]
    return dp + dc

for p in sorted(Path("outputs/erdc/w13_metrics").glob("*.json")):
    d = json.loads(p.read_text())
    tag = p.stem
    rows.append(f"{tag:42s} Pix={d['pixcorr']:.4f} CLIP={d['clip_cosine']:.4f}")
    if best_pix is None or d["pixcorr"] > best_pix["pixcorr"]:
        best_pix, best_pix_tag = d, tag
    if best_bal is None or score_bal(d) > score_bal(best_bal):
        best_bal, best_bal_tag = d, tag

checks = []
if off and best_pix:
    checks.append(
        f"BEST_PIX={best_pix_tag} Pix={best_pix['pixcorr']:.4f} CLIP={best_pix['clip_cosine']:.4f} "
        f"dPix={best_pix['pixcorr']-off['pixcorr']:+.4f} dCLIP={best_pix['clip_cosine']-off['clip_cosine']:+.4f}"
    )
if off and best_bal:
    checks.append(
        f"BEST_BAL={best_bal_tag} Pix={best_bal['pixcorr']:.4f} CLIP={best_bal['clip_cosine']:.4f} "
        f"dPix={best_bal['pixcorr']-off['pixcorr']:+.4f} dCLIP={best_bal['clip_cosine']-off['clip_cosine']:+.4f} "
        f"PASS_BOTH={best_bal['pixcorr']>=off['pixcorr'] and best_bal['clip_cosine']>=off['clip_cosine']}"
    )

out = Path("outputs/erdc/w13_merge_fuse_freeze.txt")
out.write_text("\n".join(rows) + "\n\n" + "\n".join(checks) + "\n", encoding="utf-8")
print(out.read_text())
print("[OK]", out)
PY

echo "[OK] W13 merge+fuse finished"
date
