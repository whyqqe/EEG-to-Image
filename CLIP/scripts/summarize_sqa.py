"""Summarise the SQA group-action twin: 10 folds, two arms, PAIRED.

The whole point of the twin (`configs/sqa_loso_k20.yaml` vs `..._noaug.yaml`) is that the
two runs of a fold differ in ONE boolean, so the comparison is paired and the fold-to-fold
spread cancels. This script reads the deployed cell (`+ T2 R=80,a=0.75,t=0.03,fuse=16`, the
one that holds 54.83/83.45 for the current best recipe) from both arms, reports the per-fold
delta, and refuses to print a mean over fewer than the full grid -- a partial grid that
silently averaged over the wins it happened to collect is exactly the "grid incomplete" trap
`summarize_g3.py` was written to close.

Usage:
    python scripts/summarize_sqa.py --glob 'outputs/eval/sqa/*.json' \
        --out outputs/sqa_summary.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import re
from pathlib import Path

ROW = "+ T2 R=80,a=0.75,t=0.03,fuse=16"   # the deployed best cell (see slurm/v11_struct_fuse.sbatch)
ARM_RE = re.compile(r"(sub\d{2})(?:_k20)?_seed(\d+)")
N_FOLDS = 10


def _arm_of(path: str) -> str:
    p = str(path)
    return "noaug" if "sqa_noaug" in p else "sqa"


def _load(path: str) -> dict | None:
    try:
        d = json.load(open(path))
    except Exception:
        return None
    for _ck, cv in d.get("checkpoints", {}).items():
        for row, rv in cv.get("rows", {}).items():
            if row == ROW:
                return {"fold": d.get("target_subject"),
                        "top1": rv.get("top1"), "top5": rv.get("top5")}
    return None


def _paired(deltas: list[float]) -> dict:
    n = len(deltas)
    if n == 0:
        return {"n": 0, "mean_delta": float("nan"), "sd": float("nan"),
                "t": float("nan"), "n_positive": 0}
    mean = sum(deltas) / n
    if n > 1:
        var = sum((d - mean) ** 2 for d in deltas) / (n - 1)
        sd = math.sqrt(var)
        se = sd / math.sqrt(n)
        t = mean / se if se > 0 else float("nan")
    else:
        sd = se = t = float("nan")
    return {"n": n, "mean_delta": mean, "sd": sd, "t": t,
            "n_positive": sum(1 for d in deltas if d > 0)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="outputs/eval/sqa/*.json")
    ap.add_argument("--out", default="outputs/sqa_summary.json")
    args = ap.parse_args()

    rows: dict[int, dict] = {}
    for path in sorted(glob.glob(args.glob)):
        m = ARM_RE.search(Path(path).name)
        if not m:
            continue
        fold, _seed = int(m.group(1)[3:]), int(m.group(2))
        rec = _load(path)
        if rec is None:
            continue
        rows.setdefault(fold, {})[_arm_of(path)] = rec

    complete = [f for f in sorted(rows) if {"sqa", "noaug"} <= set(rows[f])]
    deltas = [rows[f]["sqa"]["top1"] - rows[f]["noaug"]["top1"] for f in complete]
    top5_d = [rows[f]["sqa"]["top5"] - rows[f]["noaug"]["top5"] for f in complete]

    print(f"cell: {ROW}")
    print(f"{'fold':>4} {'noaug t1/t5':>16} {'sqa t1/t5':>16} {'dTop1':>7} {'dTop5':>7}")
    for f in complete:
        a, b = rows[f]["noaug"], rows[f]["sqa"]
        print(f"{f:>4} {a['top1']:>7.2f}/{a['top5']:<7.2f} {b['top1']:>7.2f}/{b['top5']:<7.2f} "
              f"{b['top1']-a['top1']:>+7.2f} {b['top5']-a['top5']:>+7.2f}")

    summary = {"cell": ROW, "n_folds_present": len(rows),
               "grid_complete": len(complete) == N_FOLDS, "paired_top1": _paired(deltas)}
    if top5_d:
        summary["paired_top5"] = _paired(top5_d)
    # base means, so the absolute number next to the delta is auditable
    if complete:
        summary["noaug_top1_mean"] = sum(rows[f]["noaug"]["top1"] for f in complete) / len(complete)
        summary["sqa_top1_mean"] = sum(rows[f]["sqa"]["top1"] for f in complete) / len(complete)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(args.out, "w"), indent=2)

    print()
    if not summary["grid_complete"]:
        print(f"!! GRID INCOMPLETE: {len(complete)}/{N_FOLDS} folds have both arms. "
              f"Means below are PARTIAL -- do not quote them against a 10-fold baseline.")
    p = summary["paired_top1"]
    if p["n"] == 0:
        print("paired dTop1: no fold has both arms yet (grid empty).")
    else:
        print(f"paired dTop1 (sqa - noaug): mean {p['mean_delta']:+.2f}pp  sd {p['sd']:.2f}  "
              f"t {p['t']:+.2f}  {p['n_positive']}/{p['n']} folds positive")
    if top5_d:
        p5 = summary["paired_top5"]
        print(f"paired dTop5 (sqa - noaug): mean {p5['mean_delta']:+.2f}pp  sd {p5['sd']:.2f}  "
              f"t {p5['t']:+.2f}  {p5['n_positive']}/{p5['n']} folds positive")
    if summary.get("noaug_top1_mean") is not None:
        print(f"absolute Top-1: noaug {summary['noaug_top1_mean']:.2f}  "
              f"sqa {summary['sqa_top1_mean']:.2f}   (SCORE 53.23, current best recipe 54.83)")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
