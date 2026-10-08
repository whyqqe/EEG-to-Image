#!/usr/bin/env python
"""summarize_s3r -- paired, per-fold, per-seed comparison of a deployment arm against G3.

WHY A PAIRED SUMMARY AND NOT TWO MEANS. The claim being tested is "swap the recovery
operator and the headline rises", and the arms are evaluated on the SAME 30 checkpoints. The
unpaired difference of two means throws away that pairing and, on this grid, is the weaker
test: recovery gain is known to vary several pp across folds (+3.62 +- 1.72 over 30 runs), so
the between-fold spread dominates a +3pp effect. The paired per-run delta cancels the fold
effect entirely, which is why it is the number this script prints first -- the same discipline
`docs/eeg2image_v7_architecture.md` §7.5 records after a best-rung-vs-fixed-rung comparison
manufactured a +6.5pp illusion.

It also reports `+/-` counts and a one-sample t statistic, because "mean +3.2" and "28 of 30
runs improved" carry different evidential weight and a sign count is what exposes a mean that
is carried by two outlier folds.

Reads `run_eval` reports only (`outputs/eval/.../*.json`), so it cannot disagree with the
artefacts that were written.
"""
from __future__ import annotations

import argparse
import json
import glob
import statistics as st
from pathlib import Path

import numpy as np

HEAD = "+ T1(CSLS + recovery) + T2 reps"
T2 = "+ T2 reps"
T1 = "+ CSLS + recovery"


def _rows(path: Path) -> dict:
    """`{row_name: report}` for one eval report, tolerating either report layout.

    Two layouts are on disk and both have to be read. The multi-route report puts `rows` at
    the top level; the single-route one nests it under `checkpoints/<tag>/rows` because that
    is where a per-checkpoint ladder belongs. Rather than guessing, this walks to the first
    dict that actually has a `rows` key -- a summary that silently read zero rows from half
    the grid would report a smaller n and look like a collection problem instead of a parser
    problem.
    """
    def find(o):
        if isinstance(o, dict):
            if isinstance(o.get("rows"), dict):
                return o["rows"]
            for v in o.values():
                got = find(v)
                if got is not None:
                    return got
        return None

    rows = find(json.loads(path.read_text()))
    if rows is None:
        raise KeyError(f"no ladder rows in {path}")
    return rows


def _fold_key(path: Path) -> str:
    stem = path.stem                      # sub08_seed2025
    sub, _, seed = stem.partition("_")
    return f"{sub}_{seed}"


def _tuple(path: Path) -> str:
    """`sub08_seed2025` -> key both arms can be joined on, independent of directory name."""
    return _fold_key(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="outputs/eval/g3",
                    help="directory (or glob) of the reference arm, e.g. the banked G3 grid")
    ap.add_argument("--ours", required=True,
                    help="directory or glob of the arm under test; a glob matches each tau")
    ap.add_argument("--out", default="outputs/s3r_summary.json")
    ap.add_argument("--rows", nargs="*", default=[HEAD, T2, T1])
    args = ap.parse_args()

    base = {}
    for p in sorted(glob.glob(str(Path(args.base) / "sub*_seed*.json"))):
        base[_tuple(Path(p))] = (_rows(Path(p)), p)

    arms: dict[str, dict] = {}
    for p in sorted(glob.glob(str(Path(args.ours) / "sub*_seed*.json"))):
        arm = Path(p).parent.name
        arms.setdefault(arm, {})[_tuple(Path(p))] = _rows(Path(p))

    if not base:
        raise SystemExit(f"no reference reports under {args.base}")
    if not arms:
        raise SystemExit(f"no reports under {args.ours}")

    out: dict = {"base": args.base, "n_base": len(base), "arms": {}}
    print(f"reference: {args.base}  ({len(base)} reports)")
    print("=" * 96)
    for arm, folds in sorted(arms.items()):
        shared = sorted(set(folds) & set(base))
        print(f"\n### {arm}   ({len(shared)} runs joined with the reference)")
        if not shared:
            print("  nothing joined; skipping")
            continue
        arm_out: dict = {"n": len(shared), "rows": {}}
        for row in args.rows:
            d, a, miss = [], [], 0
            for k in shared:
                if row not in folds[k] or row not in base[k][0]:
                    miss += 1
                    continue
                d.append(folds[k][row]["top1"] - base[k][0][row]["top1"])
                a.append(folds[k][row]["top1"])
            if miss:
                print(f"  [warn] {row}: {miss} runs missing the row and were dropped")
            if not d:
                continue
            m = st.mean(d)
            sd = st.pstdev(d) if len(d) > 1 else 0.0
            se = sd / np.sqrt(len(d)) if sd > 0 else float("nan")
            t = m / se if se and np.isfinite(se) and se > 0 else float("nan")
            pos, neg = sum(1 for x in d if x > 0), sum(1 for x in d if x < 0)
            arm_out["rows"][row] = {
                "n": len(d), "mean_pct": st.mean(a), "delta_mean": m, "delta_sd": sd,
                "delta_t": t, "n_pos": pos, "n_neg": neg,
            }
            print(f"  {row:<44} top1={st.mean(a):5.2f}  "
                  f"delta={m:+5.2f}+-{sd:4.2f}  {pos:+d}/{neg:+d}  t={t:5.2f}")
            # per-fold means over seeds, so a fold carried by one seed is visible
            by_fold: dict[str, list[float]] = {}
            for k, x in zip((k for k in shared if row in folds[k] and row in base[k][0]), d):
                by_fold.setdefault(k.split("_")[0], []).append(x)
            per = {f: round(st.mean(v), 2) for f, v in sorted(by_fold.items())}
            arm_out["rows"][row]["per_fold_delta"] = per
            improving = sum(1 for v in per.values() if v > 0)
            print(f"      per-fold ({improving}/{len(per)} improving): {per}")
        out["arms"][arm] = arm_out

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {args.out}")
    print("reference points: G3 banked headline 45.53 (10-fold x 3-seed mean); "
          "SCORE 53.23.")


if __name__ == "__main__":
    main()
