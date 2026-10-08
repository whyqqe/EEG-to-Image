"""Seed ensembling for a fold: pool the 3 seeds' SCORE MATRICES, then argmax.

Pooling the reported top-1 scalars is NOT the same thing and understates the effect -- the
whole point of ensembling is that the three seeds disagree on DIFFERENT concepts, and only the
score matrix carries where they disagree. Deployment would average the matrices; so does this.

Usage:  python scratch/ensemble.py <scores_dir> [row]
"""
import glob
import sys
from pathlib import Path

import numpy as np


def top1(scores):
    return float(np.mean(scores.argmax(1) == np.arange(scores.shape[0])) * 100)


def main():
    d = sys.argv[1]
    row = sys.argv[2] if len(sys.argv) > 2 else "fused::+ T1(CSLS + recovery) + T2 reps"
    files = sorted(glob.glob(d + "/sub*_seed*.npz"))
    if not files:
        raise SystemExit(f"no score files in {d}")
    by_fold = {}
    for f in files:
        fold = Path(f).stem.split("_seed")[0]
        by_fold.setdefault(fold, []).append(f)

    keys = [k for k in np.load(files[0]).files if k.startswith("row::") or k.startswith("fused::")]
    print(f"dir={d}   folds={len(by_fold)}   rows={len(keys)}")
    for target in ([row] if "::" in row else keys):
        singles, ens, seg = [], [], {}
        for fold, fs in sorted(by_fold.items()):
            if len(fs) < 2:
                continue
            mats = [np.load(f)[target] for f in fs]
            for m in mats:
                singles.append(top1(m))
                seg.setdefault(fold, []).append(top1(m))
            ens.append(top1(np.mean(mats, 0)))
        if not ens:
            continue
        # per-fold paired delta, so the t statistic is over folds (the unit that varies)
        deltas = [np.mean(seg[f]) - e for f, e in zip(sorted(seg), ens)]
        deltas = [-x for x in deltas]  # ens - single
        t = np.mean(deltas) / (np.std(deltas, ddof=1) / np.sqrt(len(deltas))) if len(deltas) > 1 else float("nan")
        print("\n%s" % target)
        print("   single-seed mean (pooled runs) = %.2f  (n=%d)" % (np.mean(singles), len(singles)))
        print("   seed-ensembled mean (per fold) = %.2f  (n=%d)" % (np.mean(ens), len(ens)))
        print("   gain = %+.2f pp   (per-fold t=%.2f, pos %d / neg %d)"
              % (np.mean(ens) - np.mean(singles), t,
                 sum(1 for x in deltas if x > 0), sum(1 for x in deltas if x < 0)))


if __name__ == "__main__":
    main()
