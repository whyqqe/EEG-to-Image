#!/usr/bin/env python
"""Read Stage 0 (anchor) and Stage 1 (localise) and say what they found.

Two independent measurements, one report, because the conclusion needs both:

Stage 0 -- `outputs/anchor/<arm>/*/train.log`, written by SAMGA's own code.
    Reports the two quantities its published protocol conflates:
      (a) best-test-epoch score   what `checkpoint_test_best.pth` is, i.e. the max of
                                  ~60 evaluations of the 200-way test set
      (b) final-epoch score       what the same run gives with no test access at all
    (a) - (b) is the SELECTION OPTIMISM of the reference protocol, and it is a floor on
    how much of SAMGA's published sub-08 number is protocol rather than capability.
    This matters because our own pipeline selects on a held-out concept split, so we
    have been comparing a held-out number against a maximum. If the gap is large, the
    honest target is below 94.8 and every "points behind SOTA" figure this project has
    quoted is inflated by that much.

Stage 1 -- `outputs/sub<NN>/loc_<arm>/loc_<arm>_result.json`, written by our pipeline.
    Reports, per arm, the validation peak on the raw and EMA curves, where the test set
    peaks under each, what the shipped (val-selected) checkpoint actually scores, and
    the fit-vs-holdout gap. The new columns exist because the 6-arm sweep could not
    answer its own question: it kept only the best checkpoint, so "validation peaked at
    epoch 16" and "the test set peaked at epoch 16" were indistinguishable, and the two
    are not the same claim (the holdout averages 4 EEG repetitions, the test set 80).

Resolution is printed with every comparison and is not optional: the 200-way test set
gives one trial per concept over 200 concepts, so 1 sigma is ~3.5 points at a score near
50 and the smallest resolvable difference is ~7. Two arms closer than that are tied, and
the report says so rather than ranking them.

Usage
    python scripts/analyze_localize.py [--subject 8]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

LOC_ARMS = ["base_ema", "ch17", "mmd", "samgaish"]
ANCHOR_ARMS = ["samga_rn50", "samga_clip5"]

# SAMGA's per-epoch line, e.g.:
#   top5 acc 78.50%	top1 acc 55.00%	Test Loss: 1.2345
ANCHOR_LINE = re.compile(r"top5 acc\s+([\d.]+)%.*?top1 acc\s+([\d.]+)%", re.S)


def wilson(pct: float, n: int) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 100.0)
    p, z = pct / 100.0, 1.959963985
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * max(0.0, c - h), 100 * min(1.0, c + h))


def min_diff(p1: float, p2: float, n: int = 200) -> float:
    """95% threshold on the difference of two Top-1 percentages, in points."""
    q1, q2 = p1 / 100, p2 / 100
    se = math.sqrt(q1 * (1 - q1) / n + q2 * (1 - q2) / n)
    return 1.959963985 * se * 100


# ---------------------------------------------------------------- stage 0
def read_anchor(arm: str) -> dict | None:
    """Per-epoch test Top-1 from SAMGA's own log, plus the optimism it implies."""
    logs = sorted(glob.glob(str(ROOT / "outputs" / "anchor" / arm / "*" / "train.log")))
    if not logs:
        return None
    text = Path(logs[-1]).read_text(errors="replace")
    top1 = [float(m.group(2)) for m in ANCHOR_LINE.finditer(text)]
    if not top1:
        return None
    best_ep = max(range(len(top1)), key=lambda i: top1[i])
    # SAMGA's own result.csv, when the run finished, for the number it would publish.
    csv = ""
    for c in sorted(glob.glob(str(ROOT / "outputs" / "anchor" / arm / "*" / "result.csv"))):
        csv = c
    published = None
    if csv:
        try:
            head = Path(csv).read_text().splitlines()
            if len(head) >= 2:
                published = float(dict(zip(head[0].split(","),
                                          head[1].split(",")))["best top1 acc"])
        except Exception:
            published = None
    return {
        "n_epochs": len(top1), "curve": top1,
        "best_epoch": best_ep + 1, "best": top1[best_ep],
        "final": top1[-1],
        "optimism": top1[best_ep] - top1[-1],
        "published_csv": published,
        "log": logs[-1],
    }


# ---------------------------------------------------------------- stage 1
def read_loc(subj: int, arm: str) -> dict | None:
    p = ROOT / "outputs" / f"sub{subj:02d}" / f"loc_{arm}_result.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text())
    except Exception:
        return None
    h = d.get("history") or []
    if not h:
        return None

    def peak(key):
        pts = [(r["epoch"], r[key]) for r in h if r.get(key) is not None]
        return max(pts, key=lambda t: t[1]) if pts else (None, None)

    v_ep, v = peak("val_top1")
    e_ep, e = peak("val_top1_ema")
    t_ep, t = peak("test_top1_diag")
    fit = (d.get("fit_diagnostic") or {}).get("top1")
    test = d.get("test") or {}
    best = d.get("best_val") or {}
    diag = [(r["epoch"], r["test_top1_diag"]) for r in h
            if r.get("test_top1_diag") is not None]
    return {
        "val_peak": v, "val_peak_ep": v_ep,
        "ema_peak": e, "ema_peak_ep": e_ep,
        "test_peak": t, "test_peak_ep": t_ep,
        "test_at_sel": test.get("top1"),
        "sel_is_ema": bool(best.get("is_ema")),
        "sel_epoch": best.get("epoch"),
        "ci95": test.get("ci95"),
        "min_detect": test.get("min_detectable_diff"),
        "fit": fit, "gap": (fit - v) if fit is not None and v is not None else None,
        "diag": diag, "val_curve": [(r["epoch"], r.get("val_top1")) for r in h],
        "epochs_run": d.get("epochs_run"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, default=8)
    a = ap.parse_args()

    # ---------------- stage 0
    print("=" * 100)
    print("STAGE 0 -- SAMGA's own code, unmodified, on this data")
    print("=" * 100)
    anchors = {arm: read_anchor(arm) for arm in ANCHOR_ARMS}
    if not any(anchors.values()):
        print("  no anchor logs yet (outputs/anchor/<arm>/*/train.log)")
    for arm, r in anchors.items():
        if not r:
            print(f"  {arm:<14} (not finished / no log)")
            continue
        print(f"  {arm:<14} best ep {r['best_epoch']:>3} = {r['best']:>6.2f}   "
              f"final ({r['n_epochs']} ep) = {r['final']:>6.2f}   "
              f"SELECTION OPTIMISM = {r['optimism']:>+6.2f} points")
    if all(anchors.values()):
        hi = max(r["best"] for r in anchors.values())
        lo = min(r["final"] for r in anchors.values())
        print(f"\n  The published-style number would be {hi:.1f} (max over epochs); "
              f"the honest fixed-schedule number is {lo:.1f}.")
        print(f"  So the reference this project has been measuring against is a MAX, and "
              f"the gap between it\n  and a held-out score is ~{hi - lo:.1f} points of "
              f"protocol before any capability difference.")

    # ---------------- stage 1
    print()
    print("=" * 100)
    print(f"STAGE 1 -- our framework, sub-{a.subject:02d}, all arms select on EMA")
    print("=" * 100)
    locs = {arm: read_loc(a.subject, arm) for arm in LOC_ARMS}
    if not any(locs.values()):
        print("  no localise results yet (outputs/sub<NN>/loc_<arm>/loc_<arm>_result.json)")
        return

    hdr = (f"{'arm':<11}{'val pk':>8}{'ep':>4}{'ema pk':>9}{'ep':>4}"
           f"{'test pk':>9}{'ep':>4}{'test@sel':>10}{'sel ep':>7}{'fit':>7}{'gap':>7}")
    print(hdr)
    print("-" * len(hdr))
    for arm, r in locs.items():
        if not r:
            print(f"{arm:<11} (not finished)")
            continue
        f = lambda x, w=8: (f"{x:>{w}.2f}" if isinstance(x, (int, float)) else f"{'-':>{w}}")
        print(f"{arm:<11}{f(r['val_peak'])}{(r['val_peak_ep'] or 0):>4}"
              f"{f(r['ema_peak'], 9)}{(r['ema_peak_ep'] or 0):>4}"
              f"{f(r['test_peak'], 9)}{(r['test_peak_ep'] or 0):>4}"
              f"{f(r['test_at_sel'], 10)}{(r['sel_epoch'] or 0):>7}"
              f"{f(r['fit'], 7)}{f(r['gap'], 7)}")

    # Does the holdout track test, and does EMA selection capture the test peak?
    print("\nDoes the val holdout track the test set? (compare PEAK LOCATIONS only -- the")
    print("two are on different scales by construction: holdout EEG averages 4 repetitions,")
    print("test 80, so the test query is ~4.5x cleaner and an easier task.)")
    for arm, r in locs.items():
        if not r or r["test_peak_ep"] is None or r["val_peak_ep"] is None:
            continue
        lv, le = r["test_peak_ep"] - r["val_peak_ep"], r["test_peak_ep"] - (r["ema_peak_ep"] or 0)
        cost = (r["test_peak"] - r["test_at_sel"]) if r["test_at_sel"] is not None else None
        print(f"  {arm:<11} raw val peaks ep {r['val_peak_ep']:>3} (lag {lv:+3d})   "
              f"EMA peaks ep {(r['ema_peak_ep'] or 0):>3} (lag {le:+3d})   "
              f"shipped ckpt gives up "
              f"{('%.2f' % cost) if cost is not None else '-'} points vs the test peak")

    # Resolution-first arm comparison.
    print("\nArm comparison. A gap smaller than the printed threshold is a TIE.")
    done = {k: v for k, v in locs.items() if v and v["test_at_sel"] is not None}
    if len(done) >= 2:
        ref_arm, ref = max(done.items(), key=lambda kv: kv[1]["test_at_sel"])
        for arm, r in done.items():
            if arm == ref_arm:
                continue
            d = ref["test_at_sel"] - r["test_at_sel"]
            thr = min_diff(ref["test_at_sel"], r["test_at_sel"])
            print(f"  {ref_arm:<11} {ref['test_at_sel']:>6.2f}  vs  {arm:<11} "
                  f"{r['test_at_sel']:>6.2f}   diff {d:>5.2f}   "
                  f"threshold {thr:>4.2f}   "
                  + ("DIFFERENT" if d > thr else "TIE (not evidence)"))
    print("\n  Reading note: `test pk` is a maximum over the sampled epochs and is shown")
    print("  for the holdout-vs-test comparison only. It is NOT the arm's score -- the")
    print("  shipped checkpoint is the val-selected one, reported as `test@sel`.")


if __name__ == "__main__":
    main()
