#!/bin/bash
#SBATCH --job-name=nw4_offtw
#SBATCH --partition=normal
#SBATCH --account=peilab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --exclude=dgx-09,dgx-11,dgx-17,dgx-30
#SBATCH --time=02:00:00
#SBATCH --output=/project/peilab/why/NeuroBridge/outputs/slurm/nw4_offtw_%j.out
#SBATCH --error=/project/peilab/why/NeuroBridge/outputs/slurm/nw4_offtw_%j.err
#
# Re-score the ten-subject generations under the OFFICIAL 2-way protocol.
#
# The comparison that was reported used cosine similarity for 2-way identification
# (eval_official_seven_dir.py -> erdc_twoway_metrics, whose encode_bundle L2-normalises
# every feature, so `t @ g.T` is cosine), while every published bar it was compared
# against -- and our own HCMA prior -- used Pearson (eval_standard7.py: "PEARSON
# correlation (not cosine) per official ATM notebook").
#
# Cosine keeps the component shared by all 200 images of the dataset; because both
# sides are unit-norm, that shared component consumes most of the similarity range and
# squeezes the margin by which the true pair must win.  Pearson removes it.  The tell is
# in the pattern: HCMA 0.9015 vs ours 0.6867 on Inception, yet 0.8599 vs 0.8589 on
# AlexNet-5 -- an image-quality gap cannot move Inception by 0.21 and leave AlexNet-5
# untouched, but a change of similarity statistic hits exactly the models with the
# largest shared component.
#
# This job re-encodes the generated images with the SAME backbone code and reports both
# statistics, so the cosine column must reproduce the recorded numbers (proving the
# re-encode is faithful) and the Pearson column is what belongs in the paper's table.
# PixCorr / SSIM / FID are statistic-independent and are carried over from the original
# evals rather than recomputed.
set -uo pipefail
cd /project/peilab/why/NeuroBridge

export PYTHON="${PYTHON:-/project/peilab/why/eeg-brainit/.venv/bin/python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export HF_HOME="${HF_HOME:-/project/peilab/why/cache/eeg-brainit/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-/project/peilab/why/cache/eeg-brainit/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/project/peilab/why/cache/xdg}"
export OPENCLIP_CACHE_DIR="${OPENCLIP_CACHE_DIR:-${HF_HOME}/open_clip}"
export HOME="${XDG_CACHE_HOME}"

mkdir -p outputs/slurm

"${PYTHON}" - <<'PY' || { echo "[FATAL] no usable CUDA device"; exit 1; }
import sys, torch
if not torch.cuda.is_available():
    print("[FATAL] CUDA unavailable"); sys.exit(1)
print(f"[gpu] {torch.cuda.get_device_name(0)} torch={torch.__version__} cuda={torch.version.cuda}")
PY

# all four arms: a_hi is the headline, a_cal/a_raw give the calibration ablation under
# the official protocol, a_ret1cal is kept so its failure is documented in the same units
"${PYTHON}" scripts/nda/nw4_official_twoway.py \
  --gen-root outputs/nw4_10s/arms \
  --arms a_hi,a_cal,a_raw,a_ret1cal \
  --subjects 1,2,3,4,5,6,7,8,9,10 \
  --out outputs/nw4_10s/official_twoway.json \
  --device cuda:0

echo
echo "########## CORRECTED HEADLINE TABLE ##########"
"${PYTHON}" - <<'PY'
import json, numpy as np
from pathlib import Path
p = Path("outputs/nw4_10s/official_twoway.json")
if not p.is_file():
    print("no official_twoway.json produced"); raise SystemExit(0)
d = json.loads(p.read_text())

BARS = {
    "ATM":            {"pixcorr": 0.160, "ssim": 0.345, "alex2": 0.776, "alex5": 0.866, "inception": 0.734, "clip": 0.786, "scope": "sub-08"},
    "CogCap":         {"pixcorr": 0.150, "ssim": 0.347, "alex2": 0.754, "alex5": 0.623, "inception": 0.669, "clip": 0.715, "scope": "10subj"},
    "CogCapPro":      {"pixcorr": 0.163, "ssim": 0.398, "inception": 0.779, "clip": 0.830, "scope": "paper"},
    "D2-FOSA":        {"pixcorr": 0.193, "ssim": 0.350, "fid": 146.33, "scope": "paper"},
    "MB2C":           {"pixcorr": 0.188, "ssim": 0.333, "fid": 163.94, "scope": "paper"},
    "our HCMA prev":  {"pixcorr": 0.0886, "ssim": 0.2262, "alex2": 0.7170, "alex5": 0.8599,
                       "inception": 0.9015, "clip": 0.9230, "fid_mean": 148.74, "fid": 129.47, "scope": "10subj pooled"},
}
cols = ["inception", "clip", "alex2", "alex5", "pixcorr", "ssim", "fid"]
print()
print(f"{'run':<22}{'protocol':<10}" + "".join(f"{c:>10}" for c in cols))
print("-" * (32 + 10 * len(cols)))
for arm in d["arms"]:
    m = d["arms"][arm]["mean"]
    def g(c):
        if c in m and isinstance(m[c], dict):
            return m[c].get("pearson", m[c].get("value"))
        return None
    vals = {"inception": g("inception"), "clip": g("clip"), "alex2": g("alex2"),
            "alex5": g("alex5"), "pixcorr": g("pixcorr"), "ssim": g("ssim"),
            "fid": (m["fid"]["value"] if "fid" in m and isinstance(m["fid"], dict) else None)}
    print(f"{arm:<22}{'Pearson':<10}" + "".join(
        (f"{vals[c]:>10.4f}" if vals[c] is not None else f"{'--':>10}") for c in cols))
print("-" * (32 + 10 * len(cols)))
for name, bar in BARS.items():
    print(f"{name:<22}{bar['scope']:<10}" + "".join(
        (f"{bar[c]:>10.4f}" if c in bar else f"{'--':>10}") for c in cols))

print()
print("verdict per arm, vs the union of published bars (higher better; FID lower better):")
best_pub = {}
for c in cols:
    v = [b[c] for b in BARS.values() if c in b]
    if v:
        best_pub[c] = min(v) if c == "fid" else max(v)
for arm in d["arms"]:
    m = d["arms"][arm]["mean"]
    def g(c):
        if c in m and isinstance(m[c], dict):
            return m[c].get("pearson", m[c].get("value"))
        return None
    wins, loss = [], []
    for c, bp in best_pub.items():
        v = g(c)
        if v is None:
            continue
        ok = v < bp if c == "fid" else v > bp
        (wins if ok else loss).append(f"{c} {v:.4f} vs {bp:.4f}")
    print(f"  {arm:<12} SOTA on: {wins if wins else 'none'}")
    if loss:
        print(f"  {'':<12} below:   {loss}")

print()
print("calibration ablation, official protocol (the effect that was verified at n=10):")
for c in ("inception", "clip", "alex2", "alex5"):
    try:
        a = d["arms"]["a_cal"]["mean"][c]["pearson"]; b = d["arms"]["a_raw"]["mean"][c]["pearson"]
        print(f"  {c:<10} a_raw {b:.4f} -> a_cal {a:.4f}  ({a-b:+.4f})")
    except KeyError:
        pass
try:
    a = d["arms"]["a_hi"]["mean"]["inception"]["pearson"]; b = d["arms"]["a_cal"]["mean"]["inception"]["pearson"]
    print(f"  {'realism':<10} a_cal {b:.4f} -> a_hi  {a:.4f}  ({a-b:+.4f})  [a_hi - a_cal]")
except KeyError:
    pass
PY
echo "[nw4-offtw] done @ $(date -Iseconds)"
