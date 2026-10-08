"""Summarise the TQA 10-fold run: Arm A (v8 + group augmentation) and Arm B (+ GQF).

Reads the deployed cell `+ T2 R=80,a=0.75,t=0.03,fuse=16` from every fold and reports, for
each arm, the 10-fold mean and the PAIRED delta against the banked `v8` encoder
(`outputs/eval/v11_fuse/*_seed2025.json`, the 55.10 cell). It also pairs Arm B against
Arm A, which is the GQF-only comparison, and prints the SCORE reference so the SOTA claim
is read off directly rather than narrated.

Usage:
    python scripts/summarize_tqa.py \
        --arms 'tqa_v8@outputs/eval/tqa_v8/*.json,tqa_v8_gqf@outputs/eval/tqa_v8_gqf/*.json' \
        --baseline v8 --baseline-glob 'outputs/eval/v11_fuse/*_seed2025.json' \
        --out outputs/tqa_summary.json
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
SCORE_TOP1, SCORE_TOP5 = 53.23, 83.55
N_FOLDS = 10


def _cell(path: str):
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


def _line(name: str, tbl: dict, base: dict, folds: list[int]) -> dict:
    common = [f for f in folds if f in tbl]
    if not common:
        print(f"{name:>12}: no folds")
        return {}
    t1 = [tbl[f][0] for f in common]
    t5 = [tbl[f][1] for f in common]
    m1, m5 = sum(t1) / len(t1), sum(t5) / len(t5)
    p1 = _paired([tbl[f][0] - base[f][0] for f in common])
    p5 = _paired([tbl[f][1] - base[f][1] for f in common])
    print(f"{name:>12}: Top-1 {m1:6.2f} / Top-5 {m5:6.2f}   "
          f"dTop1 {p1['mean']:+6.2f}pp (sd {p1['sd']:5.2f}, t {p1['t']:+5.2f}, "
          f"{p1['n_pos']}/{p1['n']})   dTop5 {p5['mean']:+6.2f}pp (t {p5['t']:+5.2f})")
    return {"top1_mean": m1, "top5_mean": m5, "paired_dtop1": p1, "paired_dtop5": p5}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True)
    ap.add_argument("--baseline", default="v8")
    ap.add_argument("--baseline-glob", default="outputs/eval/v11_fuse/*_seed2025.json")
    ap.add_argument("--out", default="outputs/tqa_summary.json")
    args = ap.parse_args()

    base = _by_fold(args.baseline_glob)
    arms = {}
    for spec in args.arms.split(","):
        name, pat = spec.split("@", 1)
        arms[name.strip()] = _by_fold(pat.strip())

    all_folds: set[int] = set()
    for a in arms.values():
        all_folds |= set(a)
    folds = sorted(all_folds)

    print(f"cell: {ROW}")
    print(f"baseline `{args.baseline}`: {len(base)} folds; "
          f"mean Top-1 {sum(base[f][0] for f in base)/len(base):.2f}"
          if base else "baseline: none")
    print(f"reference: SCORE published {SCORE_TOP1} / {SCORE_TOP5}\n")

    if len(folds) < N_FOLDS:
        print(f"!! GRID INCOMPLETE: {len(folds)}/10 folds present -- do not quote partial means\n")

    summary = {"cell": ROW, "baseline": args.baseline,
               "score_reference": {"top1": SCORE_TOP1, "top5": SCORE_TOP5},
               "baseline_mean_top1": (sum(base[f][0] for f in base) / len(base)) if base else None,
               "arms": {}}
    for name, tbl in arms.items():
        summary["arms"][name] = _line(name, tbl, base if base else {f: (0.0, 0.0) for f in folds},
                                      folds)

    # the GQF-only paired comparison, Arm B vs Arm A
    if "tqa_v8" in arms and "tqa_v8_gqf" in arms:
        a, b = arms["tqa_v8"], arms["tqa_v8_gqf"]
        common = [f for f in folds if f in a and f in b]
        if common:
            p1 = _paired([b[f][0] - a[f][0] for f in common])
            p5 = _paired([b[f][1] - a[f][1] for f in common])
            print(f"\nGQF only (tqa_v8_gqf - tqa_v8): dTop1 {p1['mean']:+.2f}pp "
                  f"(sd {p1['sd']:.2f}, t {p1['t']:+.2f}, {p1['n_pos']}/{p1['n']})  "
                  f"dTop5 {p5['mean']:+.2f}pp (t {p5['t']:+.2f})")
            summary["gqf_only_vs_arm_a"] = {"paired_dtop1": p1, "paired_dtop5": p5,
                                            "n_folds": len(common)}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
