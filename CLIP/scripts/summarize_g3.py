#!/usr/bin/env python
"""Summarise the G3 sweep: 10-fold LOSO x 3 seeds -> one number comparable to the literature.

    python scripts/summarize_g3.py                       # reads outputs/eval/g3/*.json
    python scripts/summarize_g3.py --glob 'outputs/eval/g3/*.json' --out outputs/g3_summary.json

WHY THIS EXISTS RATHER THAN EYEBALLING 30 JSONs.

A single fold's Top-1 is not comparable to a number the literature reports as a 10-fold
mean, and this project has already spent a round chasing exactly that confusion (sub-08's
22.00/50.00 SAMGA cell vs the published 34.4, and then again the 26.22/57.98). The report
that goes on the record has to be built from the fold SET, with the spread shown, and it
has to SAY when the set is incomplete -- a mean over 7 of 30 runs is not a weaker version
of a mean over 30, it is a different quantity, and averaging silently is how an interrupted
sweep becomes a headline.

The comparison row is the final-epoch, 200-way, 63ch inter-subject protocol, which is what
`--ckpts last.pt` produces and what 26.22 / 53.23 are reported under. Best-epoch numbers
(34.4 / 64.8) are a different selection rule and are deliberately NOT the reference here.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
from pathlib import Path

# The rows to track, in reporting order. They are the rungs that carry the argument:
# the raw floor, CSLS alone, the two recovery compositions (what SCORE does and what we
# used to do), the repetition cloud alone, and the two fusions. A row missing from every
# file is reported as missing rather than dropped, so a `--reps` that silently did nothing
# is visible in the table instead of in the run that was never re-checked.
ROWS = [
    "raw cosine",
    "+ CSLS",
    "+ CSLS + recovery",
    "+ whiten + CSLS + recovery",
    "+ T2 reps",
    "+ T1(CSLS + recovery) + T2 reps",
    "+ T1(whiten + CSLS + recovery) + T2 reps",
]

# The protocol's reference cells. Both are final-epoch and both are 10-fold means, which is
# the only reason they may sit in the same table as our mean.
REFERENCE = [
    ("SAMGA encoder, re-measured by SCORE", 26.22, 57.98),
    ("SCORE (2026)", 53.23, 83.55),
]

FOLD_SEED = re.compile(r"sub(\d+)_seed(\d+)\.json$")


def load_runs(pattern: str) -> dict[tuple[int, int], dict]:
    """`{(fold, seed): report}`; the fold and seed come from the FILENAME.

    From the filename and not from a field inside the file on purpose: the eval report has
    no seed field (the checkpoint does), and a filename that disagrees with its content is
    something this script should not paper over with a fallback.
    """
    runs: dict[tuple[int, int], dict] = {}
    for p in sorted(Path().glob(pattern)):
        m = FOLD_SEED.search(p.name)
        if not m:
            print(f"[summarize] IGNORING {p.name}: not a subNN_seedYYYY.json name")
            continue
        runs[(int(m.group(1)), int(m.group(2)))] = json.load(p.open())
    return runs


def row_of(report: dict, name: str) -> dict | None:
    """The single checkpoint's row `name`, or None. One checkpoint per G3 run by design."""
    for res in report.get("checkpoints", {}).values():
        if name in (res.get("rows") or {}):
            return res["rows"][name]
    return None


def fmt(vals: list[float]) -> str:
    if not vals:
        return f"{'--':>16}"
    if len(vals) == 1:
        return f"{vals[0]:>12.2f} (n=1)"
    return f"{st.mean(vals):>8.2f} +- {st.pstdev(vals):<5.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="outputs/eval/g3/sub*_seed*.json",
                    help="shell-style glob, relative to the project root")
    ap.add_argument("--stage1-glob", default="outputs/stage1/g3/sub*_k20_seed*/result.json",
                    help="per-run `result.json`, for the best-epoch diagnostic cell")
    ap.add_argument("--out", default=None, help="also write the summary as JSON")
    args = ap.parse_args()

    runs = load_runs(args.glob)
    if not runs:
        raise SystemExit(f"no runs matched {args.glob!r} -- nothing to summarise")

    folds = sorted({f for f, _ in runs})
    seeds = sorted({s for _, s in runs})
    expected = {(f, s) for f in range(1, 11) for s in (2025, 2026, 2027)}
    missing = sorted(expected - set(runs))
    extra = sorted(set(runs) - expected)

    print("=" * 100)
    print("G3 | 10-fold inter-subject LOSO x 3 seeds | final-epoch, 200-way, 63ch | "
          "configs/g3_loso_k20.yaml")
    print(f"runs found: {len(runs)}/{len(expected)}   folds {folds[0]}-{folds[-1]}   "
          f"seeds {seeds}")
    if missing:
        # LOUD, and before any number: see the module docstring.
        print(f"INCOMPLETE: {len(missing)} expected run(s) absent -> "
              f"{', '.join(f'sub{f:02d}/seed{s}' for f, s in missing)}")
        print("            the means below are over what landed, NOT over the protocol.")
    if extra:
        print(f"UNEXPECTED: runs outside the 10x3 grid: "
              f"{', '.join(f'sub{f:02d}/seed{s}' for f, s in extra)}")
    print("=" * 100)

    # ---- per-row, averaged over folds ------------------------------------------------
    # The fold is the unit of the protocol (53.23 +- 1.62 is a spread over FOLDS), so a row
    # is first averaged within each fold over its seeds, then across folds. Pooling all 30
    # runs instead would understate the spread, because three seeds of one fold are not
    # three independent folds.
    print(f"\n{'row':<38} {'Top-1 (fold-mean +- sd)':>24} {'Top-5 (fold-mean +- sd)':>24}"
          f" {'runs':>6}")
    print("-" * 100)
    summary: dict = {"n_runs": len(runs), "missing": missing, "rows": {}}
    for name in ROWS:
        per_fold_t1, per_fold_t5 = [], []
        n = 0
        for f in folds:
            t1 = [row_of(runs[(f, s)], name) for s in seeds if (f, s) in runs]
            t1 = [x["top1"] for x in t1 if x]
            t5 = [row_of(runs[(f, s)], name) for s in seeds if (f, s) in runs]
            t5 = [x["top5"] for x in t5 if x]
            if t1:
                per_fold_t1.append(st.mean(t1))
                per_fold_t5.append(st.mean(t5))
                n += len(t1)
        print(f"{name:<38} {fmt(per_fold_t1):>24} {fmt(per_fold_t5):>24} {n:>6}")
        summary["rows"][name] = {
            "top1": st.mean(per_fold_t1) if per_fold_t1 else None,
            "top1_std_across_folds": st.pstdev(per_fold_t1) if len(per_fold_t1) > 1 else None,
            "top5": st.mean(per_fold_t5) if per_fold_t5 else None,
            "n_runs": n,
        }
    print("-" * 100)

    # ---- per fold, at the strongest available T2-fused rung ---------------------------
    # Reported per fold because the fold-level spread is the thing a 10-fold claim rests on:
    # a mean that only clears SCORE because one fold is a huge outlier is a different result
    # from a mean that clears it on 8 of 10.
    #
    # WHICH ROW IS THE HEADLINE is the best Top-1 among `ROWS` -- a list fixed at the top of
    # this file BEFORE the sweep ran, so "best of 7 pre-registered rungs" is not a search
    # over variants, and the ranking below is printed for every row anyway. The best row is
    # NOT assumed to be the longest-named one: on sub-08 the fusions differ by 2 points
    # (41.50 vs 39.50) and picking by position would have reported the weaker one.
    scored = [(v["top1"], i, r) for i, r in enumerate(ROWS)
              if (v := summary["rows"].get(r, {})).get("top1") is not None]
    if not scored:
        raise SystemExit("no tracked row was present in any run -- nothing to headline")
    scored.sort(key=lambda t: (-t[0], t[1]))
    headline = scored[0][2]
    print(f"\nper fold, mean over seeds -- headline row: {headline}")
    print(f"{'fold':>5} {'Top-1':>8} {'Top-5':>8}   seeds")
    print("-" * 100)
    fold_rows = {}
    for f in folds:
        t1 = [row_of(runs[(f, s)], headline) for s in seeds if (f, s) in runs]
        t1 = [x for x in t1 if x]
        if not t1:
            print(f"{f:>5} {'--':>8} {'--':>8}")
            continue
        print(f"{f:>5} {st.mean([x['top1'] for x in t1]):>8.2f} "
              f"{st.mean([x['top5'] for x in t1]):>8.2f}   "
              f"{','.join(str(s) for s in seeds if (f, s) in runs)}")
        fold_rows[f] = {"top1": st.mean([x["top1"] for x in t1]),
                        "top5": st.mean([x["top5"] for x in t1])}
    summary["headline_row"] = headline
    summary["per_fold"] = fold_rows
    print("-" * 100)

    # ---- best-epoch RAW cell: reported, and labelled as the test-selection it is ------
    # The G3 gate in the v6 doc asks for "final + best 同报" so the sweep is comparable to
    # the SAMGA 34.4/64.8 cell, which is a BEST-EPOCH number. Two things make this a
    # diagnostic and not a claim:
    #   * `best_seen` is `max over epochs of the TEST top-1` (`train.py`) -- the protocol
    #     deliberately has no validation split (`concept_split(n_val=0)`), so the selection
    #     signal IS the test set. It is an oracle cell, exactly like SAMGA's 34.4 ("测试集
    #     选点"), and the same number cannot be produced at deployment.
    #   * it is the RAW-cosine cell only: `best.pt` is not written, so there is no
    #     checkpoint to run the calibrated ladder on. Reporting it beside the final-epoch
    #     headline without this label would read as "+11pp from nothing".
    best_runs: dict[tuple[int, int], float] = {}
    last_runs: dict[tuple[int, int], float] = {}
    for p in sorted(Path().glob(args.stage1_glob)):
        m = re.search(r"sub(\d+)_k20_seed(\d+)/result\.json$", str(p))
        if not m:
            continue
        d = json.load(p.open())
        best_runs[(int(m.group(1)), int(m.group(2)))] = float(d["best_seen"])
        last_runs[(int(m.group(1)), int(m.group(2)))] = float(d["last"]["top1"])
    if best_runs:
        per_fold_best = [st.mean([v for (f2, _), v in best_runs.items() if f2 == f])
                         for f in folds if any(f2 == f for f2, _ in best_runs)]
        per_fold_last = [st.mean([v for (f2, _), v in last_runs.items() if f2 == f])
                         for f in folds if any(f2 == f for f2, _ in last_runs)]
        print(f"\nbest-epoch diagnostic cell (RAW cosine, TEST-SELECTED -- see the comment in "
              f"summarize_g3.py)")
        print(f"  last-epoch raw cosine (deployable)  {fmt(per_fold_last)}   "
              f"({len(last_runs)} runs)")
        print(f"  best-seen  raw cosine (ORACLE)      {fmt(per_fold_best)}   "
              f"({len(best_runs)} runs)")
        print(f"  the gap is what test-set selection is worth on this representation; it is NOT "
              f"available at deployment and NOT comparable to the calibrated headline row.")
        summary["best_epoch_raw_cosine"] = {
            "last_epoch_mean": st.mean(per_fold_last) if per_fold_last else None,
            "best_seen_mean": st.mean(per_fold_best) if per_fold_best else None,
            "n_runs": len(best_runs),
        }
    else:
        print(f"\nbest-epoch cell: no `result.json` matched {args.stage1_glob!r}")

    # ---- the comparison ---------------------------------------------------------------
    print(f"\nreference cells (SAME protocol: inter-subject LOSO, 200-way, 63ch, final epoch)")
    for label, t1, t5 in REFERENCE:
        print(f"  {label:<44} {t1:>8.2f} {t5:>8.2f}")
    ours = summary["rows"].get(headline, {})
    if ours.get("top1") is not None:
        sd = ours.get("top1_std_across_folds")
        sd_str = f"(@ {sd:.2f} across folds)" if sd is not None else \
            "(spread unavailable: fewer than 2 folds present)"
        print(f"  {'ours (this sweep): ' + headline:<44} {ours['top1']:>8.2f} "
              f"{ours['top5']:>8.2f}   {sd_str}")
        for label, t1, t5 in REFERENCE:
            print(f"      vs {label:<40} {ours['top1'] - t1:>+8.2f} Top-1  "
                  f"{ours['top5'] - t5:>+8.2f} Top-5")
    print("=" * 100)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, indent=2, default=str))
        print(f"[summarize] wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
