#!/usr/bin/env python
"""Aggregate `outputs/eval/*.json` into an arm x seed mean/std table.

Why a script and not a hand-read table: a single fold is 200 queries, so one trial is
0.5pp of Top-1, and the entire v3 -> v3.1 difference (14.00 -> 14.50 raw) is one trial.
Reading several of those off separate files and comparing them by eye is how a 1-trial
difference gets reported as an improvement. This prints the per-seed values *and* the
spread, so the uncertainty is on screen next to the mean, and it prints the PAIRED
per-seed delta against a reference arm rather than two independent means -- the seeds
share the fold, the test set and the sampler, so pairing removes most of the variance
that a difference of means leaves in.

Run:
  python scripts/summarize_arms.py --glob 'v32-*-sub-08*' --ref base
  python scripts/summarize_arms.py --glob '*-sub-08*'          # everything on disk
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from samclip import config  # noqa: E402

# The report's row names, in the order they should be read (raw -> calibrated). The
# `centre` rungs were added to `scripts/run_eval.py` with the SMN redesign; a row name
# missing from this list is silently dropped from the table, so it has to track the
# evaluator's ladder rather than being a historical copy of it.
ROW_ORDER = ["raw cosine", "+ CSLS", "+ centre", "+ centre + CSLS",
             "+ SAW whiten", "+ whiten + CSLS"]


def arm_of(stem: str) -> str:
    """`v32-aug-sub-08_seed2026` -> `aug`. Falls back to the whole stem."""
    m = re.match(r"^(?:v[\d.]+-)?([a-z0-9_]+?)-sub-?\d+$", stem.split("_seed")[0])
    return m.group(1) if m else stem.split("_seed")[0]


def seed_of(stem: str) -> int | None:
    m = re.search(r"_seed(\d+)$", stem)
    return int(m.group(1)) if m else None


def load_row(path: Path) -> tuple[str, dict] | None:
    try:
        d = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        print(f"  [skip] {path.name}: {e}")
        return None
    ckpts = d.get("checkpoints") or {}
    if not ckpts:
        print(f"  [skip] {path.name}: no checkpoints")
        return None
    name, c = next(iter(ckpts.items()))
    rows = {k: v.get("top1") for k, v in (c.get("rows") or {}).items()}
    if not rows:
        print(f"  [skip] {path.name}: no rows")
        return None
    return name, {"epoch": c.get("epoch"), "rows": rows,
                  "fusion": c.get("target_fusion"), "file": path.name}


def mean_std(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(xs) / n
    if n == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (n - 1)      # sample std (seeds are a sample)
    return m, var ** 0.5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="*-sub-08*",
                    help="filename glob inside outputs/eval (without .json)")
    ap.add_argument("--ref", default=None,
                    help="arm name to pair against (e.g. `base`). Prints a paired "
                         "per-seed delta table instead of only two independent means.")
    args = ap.parse_args()

    evaldir = config.OUTPUTS / "eval"
    files = sorted(Path(p) for p in glob.glob(str(evaldir / f"{args.glob}.json")))
    if not files:
        raise SystemExit(f"no eval reports match {evaldir}/{args.glob}.json")

    arms: dict[str, dict[int | None, dict]] = defaultdict(dict)
    for f in files:
        if f.stem.endswith("_ranks") or f.stem.endswith(".ranks"):
            # An older eval wrote ranks to a SIBLING file; it is the same run, and its
            # stem has no `_seedNNN` suffix, so folding it in would add a phantom
            # seed=None column to the arm it duplicates.
            continue
        got = load_row(f)
        if got:
            _, rec = got
            arms[arm_of(f.stem)][seed_of(f.stem)] = rec

    print(f"[summary] {len(files)} report(s) -> {len(arms)} arm(s): "
          f"{', '.join(sorted(arms))}")
    rows_seen = [r for r in ROW_ORDER
                 if any(r in rec["rows"] for a in arms.values() for rec in a.values())]
    max_seeds = max(len(v) for v in arms.values())

    # ---- per-arm mean +/- std over seeds -------------------------------------------
    print(f"\n{'arm':<12} {'seeds':>5}  " + "".join(f"{r:>18}" for r in rows_seen))
    for a in sorted(arms):
        per_seed = arms[a]
        vals = {r: [per_seed[s]["rows"].get(r) for s in sorted(per_seed, key=lambda x: (x is None, x))]
                for r in rows_seen}
        cells = []
        for r in rows_seen:
            xs = [float(x) for x in vals[r] if x is not None]
            m, sd = mean_std(xs)
            cells.append(f"{m:>10.2f}±{sd:<5.2f}" if len(xs) > 1 else f"{m:>18.2f}")
        print(f"{a:<12} {len(per_seed):>5}  " + "".join(cells))
        if len(per_seed) < max_seeds:
            print(f"{'':<12} {'':>5}  (only {len(per_seed)} seed(s); std needs >=2)")

    # ---- paired comparison against the reference arm -------------------------------
    if not args.ref:
        return
    if args.ref not in arms:
        raise SystemExit(f"--ref {args.ref!r} not among {sorted(arms)}")
    ref = arms[args.ref]
    print(f"\n[paired vs `{args.ref}`] per-seed delta in Top-1 (same fold, same test set)")
    for a in sorted(arms):
        if a == args.ref:
            continue
        shared = sorted(set(ref) & set(arms[a]), key=lambda x: (x is None, x))
        if not shared:
            print(f"\n  {a}: no shared seeds with `{args.ref}` -- cannot pair")
            continue
        print(f"\n  {a}:")
        for r in rows_seen:
            deltas = []
            for s in shared:
                x, y = ref[s]["rows"].get(r), arms[a][s]["rows"].get(r)
                if x is not None and y is not None:
                    deltas.append(float(y) - float(x))
            if not deltas:
                continue
            m, sd = mean_std(deltas)
            wins = sum(d > 0 for d in deltas)
            # A delta smaller than one trial is not readable at n=200: 1/200 = 0.5pp.
            trials = m / (100.0 / 200)
            line = (f"    {r:<16} {m:>+7.2f}pp ({trials:>+5.1f} trials, sd {sd:.2f}, "
                    f"{wins}/{len(deltas)} seeds up)")
            if len(deltas) >= 2:
                try:
                    from scipy import stats
                    _, p = stats.ttest_rel([arms[a][s]["rows"][r] for s in shared],
                                           [ref[s]["rows"][r] for s in shared])
                    # Say the honestly-weaker thing at small n: p is printed, and the
                    # `n` that WOULD be needed is printed next to it, because "p=0.2,
                    # n=3" invites the reader to treat a direction as a result. The
                    # required n is the paired-sample size that reaches 80% power at the
                    # observed effect and variance: n ~= 7.85 * (sd/|m|)^2.
                    need = 7.85 * (sd / abs(m)) ** 2 if m else float("inf")
                    line += f"  p={p:.3f}"
                    if need > len(deltas):
                        line += f"  (needs n~{min(need, 999):.0f} for 80% power)"
                except ImportError:
                    pass
            print(line + f"  [{', '.join(f'{d:+.1f}' for d in deltas)}]")

    # ---- what this n can resolve at all --------------------------------------------
    print("\n[noise floor] 1 trial = 1/200 = 0.50pp, so 2 trials = 1pp; the smallest "
          "effect this design can call is ~2pp at 3 seeds")
    print("            same-seed reruns of one config measured 0.54pp apart on the "
          "Stage-2 mean (up to 3pp per epoch),")
    print("            so differences under ~2pp in this table are NOT results.")


if __name__ == "__main__":
    main()
