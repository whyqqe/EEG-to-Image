#!/usr/bin/env bash
#SBATCH --job-name=erdc-w15-sum
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --dependency=singleton
#SBATCH --output=outputs/slurm/%x-%j.out
#SBATCH --error=outputs/slurm/%x-%j.err

# Aggregate LOSO metrics after array completes (submit manually or with dependency)

set -euo pipefail
cd /project/peilab/why/eeg-brainit
python - <<'PY'
import json
from pathlib import Path
import numpy as np

rows = ["# W15 LOSO 汇总", "", "| 被试 | Pix | CLIP | FID |", "|------|-----|------|-----|"]
pix, clip, fid = [], [], []
for i in range(1, 11):
    sub = f"sub-{i:02d}"
    tag = f"w15_loso_fuse_{sub}"
    m = Path(f"outputs/erdc/w15_loso_metrics/{tag}.json")
    f = Path(f"outputs/erdc/w15_loso_metrics/{tag}_fid.json")
    if not m.is_file():
        rows.append(f"| {sub} | — | — | — |")
        continue
    d = json.loads(m.read_text())
    fd = json.loads(f.read_text()) if f.is_file() else None
    pix.append(d["pixcorr"])
    clip.append(d["clip_cosine"])
    if fd:
        fid.append(fd["fid"])
    fs = f"{fd['fid']:.1f}" if fd else "—"
    rows.append(f"| {sub} | {d['pixcorr']:.4f} | {d['clip_cosine']:.4f} | {fs} |")

if pix:
    rows.append("")
    rows.append(f"**均值**: Pix={np.mean(pix):.4f} CLIP={np.mean(clip):.4f}")
    if fid:
        rows.append(f"**FID 均值**: {np.mean(fid):.1f}")

out = Path("outputs/erdc/w15_loso_summary.md")
out.write_text("\n".join(rows) + "\n", encoding="utf-8")
print(out.read_text())
PY
