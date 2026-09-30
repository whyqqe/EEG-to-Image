#!/usr/bin/env python
"""Read the Stage A diagnostic sweep and say what it found.

Stage A runs six arms that differ from the diagnosed run in exactly one setting each
(see `scripts/run_diag_sweep.sh`). This turns their `result.json` files into the three
comparisons that decide what to do next:

  1. Where does the holdout peak, and is a higher peak available?
     A setting that raises the peak is a FIX.
  2. Where does the 200-way TEST set peak, and does it agree with the holdout?
     Disagreement means the selection rule is broken and nothing else matters yet.
     This measurement did not exist before `--test-every`: the diagnosed run kept only
     its best checkpoint, so its val curve could not be checked against test at all.
  3. How much capacity is going into memorisation at the peak?
     `fit - val` at the peak epoch, under an identical protocol. The diagnosed run
     read 86.73 vs 33.67, i.e. 53 points. A change that lowers the peak while leaving
     this gap alone is not a fix.

Deliberately NOT reported as a headline: the test score of the best test epoch. That
would be reading the test set for model selection, which is the bias the holdout
exists to prevent. The test curve is printed as a curve and its peak location is used
to judge the holdout, never to pick an arm.

Usage
    python scripts/analyze_diag_sweep.py [--subject 8] [--ref <result.json>]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Order matters: the control first, then the arms roughly by how strong a candidate
# each one was before the sweep ran.
ARMS = ["ctl60", "lr1e4", "soft", "frz8", "noaug", "b512"]


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:  # a half-written file means the arm is still running
        print(f"  (skipping {path.name}: {exc})")
        return None


def summarise(d: dict) -> dict:
    """Everything the comparison needs, from one arm's result.json."""
    h = d.get("history") or []
    if not h:
        return {}

    val_curve = [(r["epoch"], r.get("val_top1")) for r in h if r.get("val_top1") is not None]
    v_ep, v_val = max(val_curve, key=lambda t: t[1])

    diag = [(r["epoch"], r["test_top1_diag"]) for r in h
            if r.get("test_top1_diag") is not None]
    if diag:
        t_ep, t_val = max(diag, key=lambda t: t[1])
    else:
        t_ep = t_val = None

    # The fit diagnostic is stored as a block; it is measured at the val-selected
    # checkpoint, i.e. at the peak, which is what makes it comparable across arms.
    fit = d.get("fit_diagnostic") or {}
    fit_top1 = fit.get("top1", fit.get("fit_top1"))

    best = d.get("best_val") or {}
    test = d.get("test") or {}

    return {
        "val_peak_ep": v_ep, "val_peak": v_val,
        "test_peak_ep": t_ep, "test_peak": t_val,
        "val_final": val_curve[-1][1] if val_curve else None,
        "test_final_diag": diag[-1][1] if diag else None,
        "test_at_val_selected": test.get("top1"),
        "fit_top1": fit_top1,
        "memorisation_gap": (fit_top1 - v_val) if fit_top1 is not None else None,
        "epochs_run": d.get("epochs_run"),
        "trainable": d.get("trainable_params"),
        "curve": diag,
        "val_curve": val_curve,
    }


def fmt(v, w=6, p=2):
    return f"{v:>{w}.{p}f}" if isinstance(v, (int, float)) else f"{'-':>{w}}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, default=8)
    ap.add_argument("--ref", type=Path, default=None,
                    help="the run being diagnosed, shown as a reference row. Defaults "
                         "to outputs/sub<NN>/patch_dual_result.json.")
    ap.add_argument("--out", type=Path, default=None,
                    help="write the comparison as JSON here")
    a = ap.parse_args()

    out_dir = ROOT / f"outputs/sub{a.subject:02d}"
    ref_path = a.ref or (out_dir / "patch_dual_result.json")

    rows: dict[str, dict] = {}
    for arm in ARMS:
        d = load(out_dir / f"diag_{arm}_result.json")
        if d:
            rows[arm] = summarise(d)

    ref = load(ref_path)
    if ref:
        rows = {"[ref] prev@100ep": summarise(ref), **rows}

    if not rows:
        print(f"nothing to read yet under {out_dir}/diag_*/ -- arms still running?")
        return

    print(f"\nStage A diagnostic sweep -- sub-{a.subject:02d}")
    print("=" * 108)
    print("  every arm: 60-epoch cosine, semantic tower only (`--struct-backbone \"\"`), "
          "so `sel` is exactly val_top1")
    print("  the ref row is the diagnosed run: 100-epoch cosine, dual tower\n")
    hdr = (f"{'arm':<18}{'val pk':>7}{'ep':>5}{'test pk':>9}{'ep':>5}"
           f"{'test@val sel':>14}{'fit':>8}{'gap':>8}{'val end':>9}{'ep run':>8}")
    print(hdr)
    print("-" * len(hdr))
    for name, r in rows.items():
        if not r:
            continue
        print(f"{name:<18}{fmt(r['val_peak'])}{r['val_peak_ep']:>5}"
              f"{fmt(r['test_peak'], 9)}{(r['test_peak_ep'] or 0):>5}"
              f"{fmt(r['test_at_val_selected'], 14)}{fmt(r['fit_top1'], 8)}"
              f"{fmt(r['memorisation_gap'], 8)}{fmt(r['val_final'], 9)}"
              f"{(r['epochs_run'] or 0):>8}")

    # ---- verdicts, one per arm, against the control -------------------------
    ctl = rows.get("ctl60")
    if ctl and ctl.get("val_peak") is not None:
        print(f"\nAgainst the control `ctl60` (val peak {ctl['val_peak']:.2f} "
              f"@ ep {ctl['val_peak_ep']}):")
        for name, r in rows.items():
            if not r or name == "ctl60" or r.get("val_peak") is None:
                continue
            dv = r["val_peak"] - ctl["val_peak"]
            dep = r["val_peak_ep"] - ctl["val_peak_ep"]
            dgap = (r["memorisation_gap"] - ctl["memorisation_gap"]
                    if r["memorisation_gap"] is not None
                    and ctl["memorisation_gap"] is not None else None)
            if dv >= 2.0:
                what = "FIX      (peak clearly higher)"
            elif dv <= -2.0:
                what = "REGRESSION"
            elif dep >= 5:
                what = "DELAY    (peak later, same height)"
            else:
                what = "no effect"
            extra = f", gap {dgap:+.1f}" if dgap is not None else ""
            print(f"  {name:<10} {dv:+6.2f} val, {dep:+3d} ep on the peak{extra}   {what}")

    # ---- is the holdout a valid proxy for test? -----------------------------
    print("\nIs the 150-concept holdout a faithful proxy for the 200-way test set?")
    for name, r in rows.items():
        if not r or r.get("test_peak_ep") is None:
            continue
        lag = r["test_peak_ep"] - r["val_peak_ep"]
        tol = 3  # +-3 epochs, since the curve is sampled every 5
        same = abs(lag) <= tol
        print(f"  {name:<18} val peaks ep {r['val_peak_ep']:>3} "
              f"({r['val_peak']:.2f})   test peaks ep {r['test_peak_ep']:>3} "
              f"({r['test_peak']:.2f})   lag {lag:+3d}   "
              + ("AGREES -- the selection rule is sound; post-peak epochs are wasted "
                 "time, not lost accuracy" if same else
                 "DISAGREES -- fix the selection rule before the recipe"))
    print("\n  A `DISAGREES` row is the more consequential finding of the two: it would "
          "mean the checkpoint\n  being shipped is not the best one, and every recipe "
          "comparison made so far would inherit that.")
    print("  Note the two are on different scales by construction (holdout EEG averages "
          "4 repetitions,\n  test 80), so only the PEAK LOCATIONS are compared, never "
          "the values.")

    if a.out:
        a.out.write_text(json.dumps(rows, indent=2, default=float))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
