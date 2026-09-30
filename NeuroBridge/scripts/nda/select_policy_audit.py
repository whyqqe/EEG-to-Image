#!/usr/bin/env python3
"""Audit a training log: which checkpoint-selection criterion would you have used,
and what would it have cost?

WHY THIS EXISTS
---------------
Replacing a leaky criterion (best on the TEST set) with a leak-free one is only
half the job: the leak-free criterion must actually TRACK the quantity you care
about. Measured on the clean sub-08 root encoder (50 epochs, 830-concept held-in
validation, 200-way test):

    criterion                   epoch chosen   test Top-1
    val loss  minimum               10          57.00%   <- trap
    val top1  maximum               18          74.50%   <- oracle-equal
    test loss minimum (LEAKY)       27          72.00%
    last epoch                      50          69.50%
    mean of last 5 epochs             -          69.80%

    corr(val top1, test top1) = +0.961
    corr(val loss, test top1) = -0.310   <- anti-correlated

So a naive "use held-in val loss" fix would have reported 57% instead of 74.5%.
This script makes that comparison explicit and reproducible for any run.

Usage:
  python select_policy_audit.py --log <train.log> --out <audit.json>
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def parse_log(txt: str) -> tuple[list, list]:
    val = [
        (float(a), float(b), float(c))
        for a, b, c in re.findall(
            r"\[val\] top1 ([\d.]+)%  top5 ([\d.]+)%  loss ([\d.]+)  \(selection\)", txt
        )
    ]
    tst = [
        (float(b), float(a), float(c))
        for a, b, c in re.findall(
            r"top5 acc ([\d.]+)%\ttop1 acc ([\d.]+)%\tTest Loss: ([\d.]+)", txt
        )
    ]
    return val, tst


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="root_encoder")
    a = ap.parse_args()

    txt = Path(a.log).read_text(errors="ignore")
    val, tst = parse_log(txt)
    if not val or not tst:
        # no validation split was used -> nothing leak-free to audit
        out = {"label": a.label, "has_val_split": False,
               "note": "no [val] lines found: selection was likely done on the test set"}
        Path(a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"[WARN] {a.label}: no validation lines found -> selection was not leak-free")
        return

    n = min(len(val), len(tst))
    val, tst = val[:n], tst[:n]
    vt1 = np.array([v[0] for v in val])
    vl = np.array([v[2] for v in val])
    tt1 = np.array([t[0] for t in tst])
    tl = np.array([t[2] for t in tst])

    def at(i):
        return {"epoch": int(i) + 1, "test_top1": round(float(tt1[i]), 2)}

    policies = {
        "val_loss_min": at(int(vl.argmin())),
        "val_top1_max": at(int(vt1.argmax())),
        "test_loss_min_LEAKY": at(int(tl.argmin())),
        "test_top1_max_ORACLE": at(int(tt1.argmax())),
        "last_epoch": at(n - 1),
        "mean_last5_no_selection": {"epoch": None, "test_top1": round(float(tt1[-5:].mean()), 2)},
    }
    out = {
        "label": a.label,
        "has_val_split": True,
        "n_epochs": n,
        "correlations": {
            "val_top1_vs_test_top1": {
                "pearson": round(float(np.corrcoef(vt1, tt1)[0, 1]), 3),
                "spearman": round(spearman(vt1, tt1), 3),
            },
            "val_loss_vs_test_top1": {
                "pearson": round(float(np.corrcoef(vl, tt1)[0, 1]), 3),
                "spearman": round(spearman(vl, tt1), 3),
            },
        },
        "policies": policies,
        "verdict": {
            "leak_free_best": "val_top1_max",
            "cost_of_naive_val_loss": round(
                policies["val_top1_max"]["test_top1"] - policies["val_loss_min"]["test_top1"], 2
            ),
            "gain_over_last_epoch": round(
                policies["val_top1_max"]["test_top1"] - policies["last_epoch"]["test_top1"], 2
            ),
        },
    }
    Path(a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"\n=== selection-policy audit: {a.label} ({n} epochs) ===")
    print(f"{'policy':<26}{'epoch':>7}{'test Top-1':>12}")
    for k, v in policies.items():
        e = "-" if v["epoch"] is None else str(v["epoch"])
        print(f"{k:<26}{e:>7}{v['test_top1']:>11.2f}%")
    c = out["correlations"]
    print(f"\ncorr(val top1, test top1) = {c['val_top1_vs_test_top1']['pearson']:+.3f}")
    print(f"corr(val loss, test top1) = {c['val_loss_vs_test_top1']['pearson']:+.3f}")
    print(f"\nverdict: use {out['verdict']['leak_free_best']}; "
          f"naive val-loss would cost {out['verdict']['cost_of_naive_val_loss']:+.2f}pp; "
          f"leak-free selection beats the last epoch by {out['verdict']['gain_over_last_epoch']:+.2f}pp")
    print(f"[OK] -> {a.out}")


if __name__ == "__main__":
    main()
