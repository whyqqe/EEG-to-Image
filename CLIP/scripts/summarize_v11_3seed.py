#!/usr/bin/env python
"""v11 structural-fusion, 3-seed summary vs the shippped single-mean metric.

WHY THIS EXISTS SEPARATELY FROM THE READOUT INSIDE `slurm/v11_struct_fuse.sbatch`. That readout
pairs per SUBJECT and reports Top-1 only. E1's job is different in both respects (see
`docs/eeg2image_v12_core_claim.md`): it must make the SOTA claim significant, so it needs (a) the
full 10-fold x 3-seed = 30 paired samples, because the single-seed +0.85pp was t=1.72 and the
between-fold spread is +-11pp, and (b) Top-5 as well as Top-1, because the SCORE comparison is a
TIE on Top-5 and a win on Top-1.

PAIRED, NOT A DIFFERENCE OF MEANS. Every cell is evaluated from the SAME checkpoint as its fuse=0
twin, so the fold is held fixed and the between-fold spread cancels -- the same discipline §7.5
records for the recovery rung (+3.62 +- 1.72 over 30 runs, where the spread dominated the effect).
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

# Published cells this result is positioned against (200-way, LOSO, THINGS-EEG2).
REFERENCE = {
    "SCORE (published)": (53.23, 83.55),
    "SVTL (transductive)": (48.1, 77.1),
    "SAMGA (our re-measure)": (26.22, 57.98),
}


def _load(glob_pat: str):
    cells: dict[str, dict[str, dict]] = {}
    for f in sorted(glob.glob(glob_pat)):
        key = os.path.basename(f)[:-5]                      # subNN_seedYYYY
        ck = list(json.load(open(f))["checkpoints"].values())[0]["rows"]
        cells[key] = ck
    return cells


def _col(cells, tag, metric):
    keys = sorted(cells)
    v = [cells[k][tag][metric] for k in keys if tag in cells[k]]
    return np.array(v, float), [k for k in keys if tag in cells[k]]


def paired(cells, name, base, metric):
    a, ka = _col(cells, name, metric)
    b, kb = _col(cells, base, metric)
    if a.size == 0 or b.size == 0 or ka != kb:
        return None
    d = a - b
    sd = d.std(ddof=1) if d.size > 1 else 0.0
    t = d.mean() / (sd / np.sqrt(d.size)) if sd > 0 else float("nan")
    return dict(mean=a.mean(), base=b.mean(), delta=d.mean(), sd=sd, t=t,
                pos=int((d > 0).sum()), n=int(d.size))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="outputs/eval/v11_fuse/sub*_seed*.json")
    ap.add_argument("--r", type=int, default=80)
    args = ap.parse_args()

    cells = _load(args.glob)
    if not cells:
        raise SystemExit(f"[v11-3seed] no cells matched {args.glob}")
    seeds = sorted({k.split("seed")[-1] for k in cells})
    subs = sorted({k.split("_")[0] for k in cells})
    print(f"[v11-3seed] {len(cells)} cells = {len(subs)} folds x seeds {seeds}")

    base = f"+ T2 R={args.r},a=0.75,t=0.03,fuse=0"
    print(f"\nshipped single-mean metric (fuse=0): "
          f"Top-1 {_col(cells, base, 'top1')[0].mean():.2f}  "
          f"Top-5 {_col(cells, base, 'top5')[0].mean():.2f}")

    for metric, name in (("top1", "Top-1"), ("top5", "Top-5")):
        print(f"\n=== {name} (paired vs fuse=0, n={len(cells)}) ===")
        for B in (2, 4, 8, 16):
            for suffix, lab in (("", "fused"), (" (structural-off)", "structural-OFF")):
                tag = f"+ T2 R={args.r},a=0.75,t=0.03,fuse={B}{suffix}"
                r = paired(cells, tag, base + (" (structural-off)" if suffix else ""), metric)
                if r is None:
                    continue
                print(f"  fuse={str(B):<3s} {lab:<15s} {r['mean']:6.2f} vs {r['base']:6.2f}  "
                      f"delta {r['delta']:+6.2f}pp  t={r['t']:5.2f}  {r['pos']:>2}/{r['n']} pos")

    # the headline cell and its pre-registered verdict
    best, best_d = None, -1e9
    for B in (2, 4, 8, 16):
        r = paired(cells, f"+ T2 R={args.r},a=0.75,t=0.03,fuse={B}", base, "top1")
        if r and r["delta"] > best_d:
            best, best_d = (B, r), r["delta"]
    print("\n[v11-3seed] PRE-REGISTERED VERDICT (Top-1, n=%d)" % len(cells))
    if best is None:
        print("  (no fused cells)")
    else:
        B, r = best
        passed = r["delta"] > 0 and r["t"] > 2.0 and r["pos"] >= (r["n"] + 1) // 2
        print(f"  best fuse={B}: {r['delta']:+.2f}pp, t={r['t']:.2f}, {r['pos']}/{r['n']} folds "
              f"positive -> {'PASS (t>2, majority positive)' if passed else 'NOT PASSED'}")
        t5 = paired(cells, f"+ T2 R={args.r},a=0.75,t=0.03,fuse={B}", base, "top5")
        if t5:
            print(f"  same cell Top-5: {t5['delta']:+.2f}pp, t={t5['t']:.2f}, "
                  f"{t5['pos']}/{t5['n']} folds positive")

    print("\n[v11-3seed] against published cells (Top-1 / Top-5):")
    for name, (a, b) in REFERENCE.items():
        print(f"  {name:<26s} {a:6.2f} / {b:6.2f}")


if __name__ == "__main__":
    main()
