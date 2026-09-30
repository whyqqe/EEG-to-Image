#!/usr/bin/env python3
"""Calibrate an EEG-predicted conditioning vector onto the manifold of REAL
CLIP image embeddings, then hand it to IP-Adapter.

WHY -- the measurement that motivates this
------------------------------------------
`c_self` = mean cosine between each vector and the mean direction of its own set.
Measured on sub-08 (2026-09-11), all conditions live off the value that real CLIP
image embeddings have:

    vector set                                   c_self   cos-to-true-image   200-way 2way
    REAL CLIP image embeddings (train)           0.6158        1.000              --
    REAL CLIP image embeddings (test)            0.6276        1.000              --
    HCMA blend_nda_cfm_f_a40 (what 9/10 used)     ~0.55        0.5451            0.755
    g2f  ip_mem                                  0.7994        0.580             0.761
    g2f  ip_fused                                0.4387        0.301             0.955
    G3F  ip_fused                                0.4221        0.335             0.892

Two facts follow:

  1. NEITHER of our proxies predicts image quality. `g2f ip_mem` and HCMA's
     condition have the SAME 2-way identification (0.761 vs 0.755) yet differ by
     +6.3pp CLIP and -7.8 FID in generation -- and the MORE collapsed one
     (c_self 0.799) won. So "collapsed is bad" is false and "higher 2-way is
     better" is unsupported at image level.
  2. What does differ sharply across those rows is c_self, and every one of our
     conditions departs from the real value: `ip_mem` too concentrated (0.799),
     `ip_fused` too diffuse (0.406-0.439).

IP-Adapter was trained on REAL CLIP image embeddings, so our condition is an
out-of-distribution input to it, and c_self is the simplest statistic measuring
how far out. This script maps each condition row along the great circle through
its own mean so that its concentration matches the corresponding QUANTILE of the
real embedding distribution, and nothing else.

    y = normalize((1 - a) * x + a * m)
    a > 0 moves toward the mean (more concentrated), a < 0 away from it

Both directions are needed: matching only the mean leaves rows that are already
more concentrated than the target untouched and skews the result upward (that was
the first implementation's bug).

The reference distribution comes from TRAIN image embeddings: subject
independent, test free, label free. No model, no test data, no labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def l2n(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8, None)


def cos_to_mean(x: np.ndarray, m: np.ndarray) -> np.ndarray:
    return np.clip(x @ m, -1.0, 1.0)


def alpha_for_target(c: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """Per-row a with cos(normalize((1-a)x + a m), m) == tgt[i].

    cos(y,m) = ((1-a)c + a) / sqrt((1-a)^2 + a^2 + 2a(1-a)c), strictly increasing
    in a on (-inf, 1), so a bisection on a bracket containing the solution is
    exact to machine precision and cannot diverge the way a closed form can when
    c -> 1. The bracket starts at [-4, 1] and is widened if any row falls outside,
    which happens when c is very close to 1 and the target demands a lower
    concentration than the row already has.
    """
    def cosy(a: np.ndarray) -> np.ndarray:
        num = (1 - a) * c + a
        den = np.sqrt(np.clip((1 - a) ** 2 + a**2 + 2 * a * (1 - a) * c, 1e-12, None))
        return num / den

    lo = np.full_like(c, -4.0)
    hi = np.full_like(c, 1.0 - 1e-7)
    for _ in range(12):                      # widen until the root is bracketed
        if np.all(cosy(lo) <= tgt) and np.all(cosy(hi) >= tgt):
            break
        lo = np.where(cosy(lo) > tgt, lo * 4.0, lo)
        hi = np.where(cosy(hi) < tgt, 1.0 - (1.0 - hi) / 4.0, hi)
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        need_bigger = cosy(mid) < tgt
        lo = np.where(need_bigger, mid, lo)
        hi = np.where(need_bigger, hi, mid)
    return 0.5 * (lo + hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-npy", type=str, required=True)
    ap.add_argument("--out-npy", type=str, required=True)
    ap.add_argument("--ref-npy", type=str, required=True,
                    help="TRAIN image embeddings defining the target distribution")
    ap.add_argument("--mode", type=str, default="identity",
                    choices=["identity", "quantile"],
                    help="identity = passthrough (the control row)")
    ap.add_argument("--report-json", type=str, default="")
    args = ap.parse_args()

    x = l2n(np.load(args.in_npy).astype(np.float32))
    m = l2n(x.mean(0, keepdims=True))[0]
    c = cos_to_mean(x, m)

    if args.mode == "identity":
        y, a, stats = x, np.zeros(len(x), dtype=np.float32), {}
    else:
        ref = l2n(np.load(args.ref_npy).astype(np.float32))
        mr = l2n(ref.mean(0, keepdims=True))[0]
        c_ref = np.sort(cos_to_mean(ref, mr))
        # quantile match: this row's rank in our own concentration ranking is
        # mapped onto the same quantile of the real distribution
        order = np.argsort(c)
        q = (np.arange(len(c)) + 0.5) / len(c)
        tgt = np.empty_like(c)
        tgt[order] = np.quantile(c_ref, q)
        a = alpha_for_target(c, tgt).astype(np.float32)
        y = l2n((1 - a)[:, None] * x + a[:, None] * m[None, :])
        stats = {
            "c_self_before": float((x @ m).mean()),
            "c_self_after": float((l2n(y) @ m).mean()),
            "c_ref_mean": float(c_ref.mean()),
            "c_ref_std": float(c_ref.std()),
            "alpha_mean": float(a.mean()),
            "alpha_p5": float(np.percentile(a, 5)),
            "alpha_p95": float(np.percentile(a, 95)),
            "frac_moved_away": float((a < 0).mean()),
            "cos_drift_vs_own_mean": float(np.abs(cos_to_mean(l2n(y), m) - tgt).max()),
        }
        # the transform is along the great circle, so the true-image cosine must
        # change: report it, because "calibration" must not silently destroy the
        # little alignment signal the condition had
    y = l2n(y).astype(np.float32)
    np.save(args.out_npy, y)
    rep = {"in": args.in_npy, "out": args.out_npy, "mode": args.mode, **stats,
           "norm_drift": float(np.abs(np.linalg.norm(y, axis=1) - 1).max())}
    print("[cal] " + json.dumps(rep))
    if args.report_json:
        Path(args.report_json).write_text(json.dumps(rep, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
