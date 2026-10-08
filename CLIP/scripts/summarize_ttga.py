"""Summarise the TTGA sweep: per checkpoint, per fold, paired against the shipped cell.

Reads the per-fold probe JSONs and reports, for each checkpoint:

  * the REYNOLDS RESIDUAL curve over K -- E_M ||z(Mx) - z(x)||, the distance the encoder
    still has to travel into the invariant subspace. The theory predicts it is smaller for
    an encoder trained WITH the group augmentation than for one trained without.
  * the paired dTop1 / dTop5 against `gamma = 0` (the shipped cell, reproduced bit-for-bit),
    for every (K, gamma) cell, with a t-statistic over folds.

Two predictions are on record and both are meant to be able to fail:
  P1  residual(sqa) < residual(v8) and residual(sqa) < residual(sqa_noaug).
  P2  dTop1 at gamma=1 is POSITIVE for sqa and NEGATIVE for v8/sqa_noaug (the crossover).

Usage:
    python scripts/summarize_ttga.py --glob 'outputs/probe/ttga/*/sub*.json' \
        --out outputs/ttga_summary.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import re
from pathlib import Path

FOLD_RE = re.compile(r"(sub\d{2})")
N_FOLDS = 10


def _paired(deltas: list[float]) -> dict:
    n = len(deltas)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "sd": float("nan"),
                "t": float("nan"), "n_pos": 0}
    mean = sum(deltas) / n
    sd = math.sqrt(sum((d - mean) ** 2 for d in deltas) / (n - 1)) if n > 1 else float("nan")
    se = sd / math.sqrt(n)
    return {"n": n, "mean": mean, "sd": sd,
            "t": mean / se if se and se > 0 else float("nan"),
            "n_pos": sum(1 for d in deltas if d > 0)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="outputs/probe/ttga/*/sub*.json")
    ap.add_argument("--out", default="outputs/ttga_summary.json")
    args = ap.parse_args()

    # data[ckpt][fold] = {"resid": {K: v}, "cells": {(K, gamma): (t1, t5), "base": (t1,t5)}
    data: dict[str, dict[str, dict]] = {}
    for path in sorted(glob.glob(args.glob)):
        ckpt = Path(path).parent.name
        m = FOLD_RE.search(Path(path).name)
        if not m:
            continue
        fold = int(m.group(1)[3:])
        try:
            d = json.load(open(path))
        except Exception:
            continue
        for _ck, rep in d.get("reports", {}).items():
            rows = rep.get("rows", {})
            base = None
            cells = {}
            for name, rv in rows.items():
                if rv.get("gamma", None) == 0.0 or (rv.get("K") == 1 and "gamma" not in rv):
                    pass
                if "gamma" in rv:
                    cells[(int(rv["K"]), float(rv["gamma"]))] = (rv["top1"], rv["top5"])
                else:
                    base = (rv["top1"], rv["top5"])
            # the plain CELL row (no `gamma` key) is the shipped cell
            data.setdefault(ckpt, {})[fold] = {
                "base": base, "cells": cells,
                "resid": {int(k): float(v) for k, v in rep.get("reynolds_residual", {}).items()},
                "mix_check": rep.get("mixing_check"),
            }

    summary: dict = {"checkpoints": {}}
    for ckpt, folds in sorted(data.items()):
        complete = sorted(f for f in folds if folds[f]["base"] is not None)
        print(f"\n== {ckpt}  ({len(complete)}/{N_FOLDS} folds) ==")
        if len(complete) < N_FOLDS:
            print(f"   !! GRID INCOMPLETE: {len(complete)}/{N_FOLDS} -- partial means below")
        # residual curve
        allK = sorted({k for f in complete for k in folds[f]["resid"]})
        resid = {k: (sum(folds[f]["resid"].get(k, float("nan")) for f in complete)
                     / len(complete)) for k in allK}
        print("   Reynolds residual: " +
              "  ".join(f"K={k}:{resid[k]:.4f}" for k in allK))
        base_mean = sum(folds[f]["base"][0] for f in complete) / len(complete)
        print(f"   shipped (g=0) Top-1 mean: {base_mean:.2f}")

        cells = {}
        keys = sorted({key for f in complete for key in folds[f]["cells"]})
        for key in keys:
            K, g = key
            d1 = [folds[f]["cells"][key][0] - folds[f]["base"][0]
                  for f in complete if key in folds[f]["cells"]]
            d5 = [folds[f]["cells"][key][1] - folds[f]["base"][1]
                  for f in complete if key in folds[f]["cells"]]
            p1, p5 = _paired(d1), _paired(d5)
            t1m = sum(folds[f]["cells"][key][0] for f in complete
                      if key in folds[f]["cells"]) / len(d1) if d1 else float("nan")
            cells[f"K={K},g={g:g}"] = {"top1_mean": t1m, "dtop1": p1, "dtop5": p5}
            print(f"   K={K:>2}, g={g:<4g}  Top-1 {t1m:6.2f}  dTop1 {p1['mean']:+6.2f}pp "
                  f"(sd {p1['sd']:5.2f}, t {p1['t']:+5.2f}, {p1['n_pos']}/{p1['n']})  "
                  f"dTop5 {p5['mean']:+6.2f}pp (t {p5['t']:+5.2f})")
        summary["checkpoints"][ckpt] = {
            "n_folds": len(complete), "grid_complete": len(complete) == N_FOLDS,
            "shipped_top1_mean": base_mean, "reynolds_residual": resid, "cells": cells}

    # P1 / P2 read-out, so the verdict is mechanical rather than narrated
    cps = summary["checkpoints"]
    print("\n== predictions ==")
    if "sqa" in cps and "v8" in cps:
        r_sqa, r_v8 = cps["sqa"]["reynolds_residual"], cps["v8"]["reynolds_residual"]
        common = sorted(k for k in set(r_sqa) & set(r_v8) if k > 1)   # K=1 is 0 by definition
        ok = all(r_sqa[k] < r_v8[k] for k in common) if common else False
        print(f"P1 residual(sqa) < residual(v8): {'HOLDS' if ok else 'FAILS'}  " +
              "  ".join(f"K={k}:{r_sqa[k]:.4f} vs {r_v8[k]:.4f}" for k in common))
    for name in ("sqa", "v8", "sqa_noaug"):
        c = cps.get(name)
        if not c:
            continue
        g1 = [v for k, v in c["cells"].items() if k.endswith("g=1")]
        if g1:
            v = g1[-1]
            print(f"P2 dTop1(g=1) {name:9s}: {v['dtop1']['mean']:+.2f}pp "
                  f"(t {v['dtop1']['t']:+.2f}, {v['dtop1']['n_pos']}/{v['dtop1']['n']}) "
                  f"-> {'POSITIVE' if v['dtop1']['mean'] > 0 else 'NEGATIVE'}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
