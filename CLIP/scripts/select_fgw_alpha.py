#!/usr/bin/env python
"""select_fgw_alpha -- pick `alpha` by LEAVE-ONE-SUBJECT-OUT, then report the full grid.

WHY THIS EXISTS RATHER THAN ARGMAX ON THE POOL. The FGW weight has a measured interior optimum
and a catastrophic alpha=1 collapse, so it is a real hyperparameter and choosing it on the folds
it is then reported on would be selection on the test set -- the same class of error as the
best-rung-vs-fixed-rung comparison that manufactured a +6.5pp illusion in
`docs/eeg2image_v7_architecture.md` §7.5. This selects alpha for the held-out subject using ONLY
the other nine subjects' runs, so the reported number is nested-LOSO honest, and it *also*
prints the full grid so the shape (which is the robust evidence, per `eeg2image_v9` §5.1) stays
visible rather than being hidden behind a single chosen value.

`--mode` chooses the selection criterion:
  `accuracy`  the held-out-mean top-1 over the other nine subjects. Uses labels of the OTHER
              folds only. This is the default and the one that matches how the operator will be
              chosen in practice.
  `plan_acc`  the plan's mass on the true diagonal, averaged over the other nine. Still uses
              labels, but far less (it never touches ranking), so it is reported as a
              robustness check on the selection rather than as a label-free criterion -- there
              is no genuinely label-free option here, and pretending otherwise would be the
              dishonest part.

Writes `outputs/fgw_alpha.json` with `selected_alpha` plus the full grid, which the pipeline's
training stage reads.
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics as st
from pathlib import Path

import numpy as np

HEAD = "+ T1(CSLS + recovery) + T2 reps"


def rows(path: Path) -> dict:
    def find(o):
        if isinstance(o, dict):
            if isinstance(o.get("rows"), dict):
                return o["rows"]
            for v in o.values():
                g = find(v)
                if g is not None:
                    return g
        return None
    r = find(json.loads(path.read_text()))
    if r is None:
        raise KeyError(f"no ladder rows in {path}")
    return r


def load_grid(root: str, alphas: list[str]) -> dict[str, dict[str, dict]]:
    """`{alpha: {fold_key: rows}}`. A fold missing from an alpha is dropped from that alpha."""
    grid: dict[str, dict] = {}
    for a in alphas:
        d = Path(f"{root}_a{a}")
        if not d.is_dir():
            print(f"[alpha] WARNING: no reports under {d}")
            continue
        got = {}
        for p in sorted(glob.glob(str(d / "sub*_seed*.json"))):
            got[Path(p).stem] = rows(Path(p))
        grid[a] = got
        print(f"[alpha] alpha={a}: {len(got)} reports from {d}")
    return grid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", default="outputs/eval/fgw",
                    help="prefix; the per-alpha dirs are `<prefix>_a<alpha>`")
    ap.add_argument("--alphas", nargs="*", default=["0.1", "0.25", "0.5"])
    ap.add_argument("--row", default=HEAD)
    ap.add_argument("--mode", choices=["accuracy", "plan_acc"], default="accuracy")
    ap.add_argument("--out", default="outputs/fgw_alpha.json")
    args = ap.parse_args()

    grid = load_grid(args.eval_root, args.alphas)
    if not grid:
        raise SystemExit(f"no alpha grids under {args.eval_root}_a*")

    shared = set.intersection(*(set(v) for v in grid.values()))
    if not shared:
        raise SystemExit("the alpha grids share no folds; cannot compare")
    shared = sorted(shared)
    print(f"[alpha] {len(shared)} folds present in EVERY alpha: {len(shared)} runs")

    def metric(alpha: str, key: str) -> float:
        r = grid[alpha][key]
        if args.mode == "accuracy":
            return float(r[args.row]["top1"])
        # `plan_acc` sits in a DIFFERENT place depending on the row: `run_eval` flattens the
        # soft operator's diagnostics onto the `+ T2 reps` row (it drops only array-valued
        # keys), while the T1 row keeps them nested under `recovery_diag`. Reading one and not
        # the other would silently produce a grid of NaNs for half the rows.
        diag = r[args.row].get("diag", {}) or {}
        v = diag.get("plan_acc")
        if v is None:
            v = (diag.get("recovery_diag") or {}).get("plan_acc")
        return float(v) if v is not None else float("nan")

    # ------------------------------------------------------------------ the grid
    print("\nFULL GRID (mean over all shared runs) -- the shape is the robust evidence")
    print(f"{'alpha':>8}{'mean':>9}{'sd':>8}{'n':>6}")
    for a in args.alphas:
        if a not in grid:
            continue
        v = [metric(a, k) for k in shared]
        print(f"{a:>8}{st.mean(v):>9.2f}{st.pstdev(v):>8.2f}{len(v):>6}")

    # ------------------------------------------------- leave-one-SUBJECT-out selection
    subjects = sorted({k.split("_")[0] for k in shared})
    print(f"\nLEAVE-ONE-SUBJECT-OUT SELECTION over {len(subjects)} subjects "
          f"(criterion: {args.mode})")
    print(f"{'held-out':>10}{'selected alpha':>16}{'held-out value':>17}")
    chosen_runs: list[float] = []
    counts: dict[str, int] = {}
    for s in subjects:
        others = [k for k in shared if not k.startswith(s + "_")]
        mine = [k for k in shared if k.startswith(s + "_")]
        if not others or not mine:
            continue
        scores = {a: st.mean([metric(a, k) for k in others])
                  for a in args.alphas if a in grid}
        best = max(scores, key=scores.get)
        counts[best] = counts.get(best, 0) + 1
        held = st.mean([metric(best, k) for k in mine])
        chosen_runs.extend(metric(best, k) for k in mine)
        print(f"{s:>10}{best:>16}{held:>17.2f}   (other-subject means: "
              f"{', '.join(f'{a}:{scores[a]:.2f}' for a in sorted(scores))})")

    # the honest headline: every run scored by the alpha its subject's held-out peers chose
    fixed = {a: [metric(a, k) for k in shared] for a in args.alphas if a in grid}
    print("\n" + "=" * 76)
    print("NESTED-LOSO HEADLINE (each run scored by its subject's held-out-selected alpha)")
    print(f"  nested-LOSO mean        {st.mean(chosen_runs):.2f}  "
          f"(n={len(chosen_runs)} runs)")
    print(f"  best fixed alpha        "
          f"{max(st.mean(v) for v in fixed.values()):.2f}  <- for reference; this one IS "
          f"selected on the reported runs")
    print(f"  alpha chosen per-subject: {counts}")
    print("=" * 76)
    overall_best = max(fixed, key=lambda a: st.mean(fixed[a]))
    out = {
        "mode": args.mode, "row": args.row, "alphas": args.alphas,
        "n_runs": len(shared),
        "grid_mean": {a: st.mean(v) for a, v in fixed.items()},
        "grid_sd": {a: st.pstdev(v) for a, v in fixed.items()},
        "best_fixed_alpha": overall_best,
        "best_fixed_mean": st.mean(fixed[overall_best]),
        "nested_loso_mean": st.mean(chosen_runs),
        "per_subject_choice": counts,
        # What the pipeline passes to `--soft-plan-alpha` and `--recovery-alpha`. The nested-LOSO
        # choice is not a single number by construction, so the shipped value is the mode of the
        # per-subject choices -- the alpha the held-out peeking would have picked most often.
        "selected_alpha": float(max(counts, key=counts.get)),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    print(f"[alpha] selected_alpha = {out['selected_alpha']}  (mode of the per-subject choices)")
    print(f"[alpha] wrote {args.out}")


if __name__ == "__main__":
    main()
