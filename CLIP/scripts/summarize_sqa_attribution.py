"""SQA attribution: which subgroup of GL(C) carries the gain?

Reads the 10-fold deployed cell (`+ T2 R=80,a=0.75,t=0.03,fuse=16`) for a panel of mixing
arms and reports each one PAIRED against a common baseline (the augmentation-off twin).
Every arm is trained on the same folds with the same seed and the same code; only
`augment.mixing_mode` differs, and every mode is rescaled to the same ||M - I||_F, so the
perturbation the encoder sees is matched by construction. The interpretation is fixed in
advance:

    diag  (diagonal subgroup,  no channel mixing) reproduces the gain  ->  the win is
          generic per-channel perturbation, i.e. what `gain` already does; the group claim
          is NOT what earned it.
    diag ~= 0 while dense/orth carry it                                    ->  the win is the
          OFF-DIAGONAL channel recombination, which is the group action proper.
    orth ~= dense (both mix)                                               ->  the win is the
          mixing, not an amplitude/scale effect (orth is norm-preserving).

Usage:
    python scripts/summarize_sqa_attribution.py \
        --arms 'dense@outputs/eval/sqa_attr/dense,diag@...,orth@...' \
        --baseline sqa_noaug --baseline-glob 'outputs/eval/sqa_noaug/*.json' \
        --out outputs/sqa_attribution.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import re
from pathlib import Path

ROW = "+ T2 R=80,a=0.75,t=0.03,fuse=16"
SEED_RE = re.compile(r"sub(\d{2})(?:_k20)?_seed(\d+)")


def _cell(path: str) -> tuple[float, float] | None:
    try:
        d = json.load(open(path))
    except Exception:
        return None
    for _ck, cv in d.get("checkpoints", {}).items():
        for row, rv in cv.get("rows", {}).items():
            if row == ROW:
                return float(rv["top1"]), float(rv["top5"])
    return None


def _by_fold(pattern: str) -> dict[int, tuple[float, float]]:
    out: dict[int, tuple[float, float]] = {}
    for path in sorted(glob.glob(pattern)):
        m = SEED_RE.search(Path(path).name)
        if not m:
            continue
        cell = _cell(path)
        if cell is not None:
            out[int(m.group(1))] = cell
    return out


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
    ap.add_argument("--arms", required=True,
                    help="comma list of name@glob")
    ap.add_argument("--baseline", default="noaug")
    ap.add_argument("--baseline-glob", default="outputs/eval/sqa_noaug/*.json")
    ap.add_argument("--out", default="outputs/sqa_attribution.json")
    args = ap.parse_args()

    base = _by_fold(args.baseline_glob)
    arms = {}
    for spec in args.arms.split(","):
        name, pat = spec.split("@", 1)
        arms[name.strip()] = _by_fold(pat.strip())

    arm_folds: set[int] = set()
    for a in arms.values():
        arm_folds |= set(a)
    folds = sorted(set(base) & arm_folds)

    print(f"cell: {ROW}")
    header = f"{'fold':>4} {'base t1/t5':>15}" + "".join(f"{n:>16}" for n in arms)
    print(header)
    for f in folds:
        b = base[f]
        line = f"{f:>4} {b[0]:>7.2f}/{b[1]:<7.2f}"
        for n in arms:
            line += (f"{arms[n][f][0]:>7.2f}/{arms[n][f][1]:<7.2f}"
                     if f in arms[n] else f"{'--':>16}")
        print(line)

    summary = {"cell": ROW, "baseline": args.baseline,
               "baseline_top1_mean": (sum(base[f][0] for f in folds) / len(folds)) if folds else None,
               "arms": {}}
    print()
    if len(folds) < 10:
        print(f"!! GRID INCOMPLETE: {len(folds)}/10 folds present. Partial means -- do not "
              f"quote against a 10-fold baseline.")
    for n, tbl in arms.items():
        common = [f for f in folds if f in tbl]
        d1 = [tbl[f][0] - base[f][0] for f in common]
        d5 = [tbl[f][1] - base[f][1] for f in common]
        p1, p5 = _paired(d1), _paired(d5)
        done = [f for f in common]
        if not done:
            print(f"{n:>6}: no folds yet")
            summary["arms"][n] = {"n_folds": 0, "top1_mean": None, "top5_mean": None,
                                  "paired_dtop1": p1, "paired_dtop5": p5}
            continue
        summary["arms"][n] = {
            "n_folds": len(common),
            "top1_mean": sum(tbl[f][0] for f in common) / len(common),
            "top5_mean": sum(tbl[f][1] for f in common) / len(common),
            "paired_dtop1": p1, "paired_dtop5": p5}
        mean1 = summary["arms"][n]["top1_mean"]
        print(f"{n:>6}: Top-1 {mean1:.2f}  dTop1 {p1['mean']:+.2f}pp (sd {p1['sd']:.2f}, "
              f"t {p1['t']:+.2f}, {p1['n_pos']}/{p1['n']})  "
              f"dTop5 {p5['mean']:+.2f}pp (t {p5['t']:+.2f}, {p5['n_pos']}/{p5['n']})")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
