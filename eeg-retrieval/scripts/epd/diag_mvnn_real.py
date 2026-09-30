#!/usr/bin/env python
"""MVNN on real EEG: what it removes, reported per subject.

Synthetic fixtures prove the linear algebra; only real data shows whether the
channel imbalance MVNN targets is actually present. It is large -- the quietest and
loudest electrodes on this montage differ by two orders of magnitude -- and an
encoder that receives that array unwhitened will spend its first layer on the
electrode with the best impedance rather than on the signal.

Run on a compute node (reads the 4 GiB train files):
    python scripts/epd/diag_mvnn_real.py --subjects 1 2 8 --splits train test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from epd import config                                    # noqa: E402
from epd.data import _raw_blocks, mvnn_whitener            # noqa: E402
from epd.mvnn import apply                                 # noqa: E402


def pooled(r: np.ndarray) -> np.ndarray:
    """Time-averaged sample covariance of (n, C, T) residuals."""
    C, T = r.shape[1], r.shape[2]
    s = np.zeros((C, C))
    for t in range(T):
        xt = r[:, :, t]
        s += xt.T @ xt / len(xt)
    return s / T


def residuals(blocks: np.ndarray) -> np.ndarray:
    r = blocks.astype(np.float64)
    r -= r.mean(axis=1, keepdims=True)          # demean over REPETITIONS (axis 1)
    return r.reshape(-1, r.shape[2], r.shape[3])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 8])
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--out", default=str(config.OUTPUTS / "diag" / "mvnn_real.json"))
    args = ap.parse_args()

    report: dict[str, dict] = {}
    for s in args.subjects:
        for split in args.splits:
            t0 = time.time()
            wh = mvnn_whitener(s, split, None, verbose=False)
            r = residuals(_raw_blocks(s, split))
            before = pooled(r)
            d0 = np.diag(before)
            corr0 = before / np.sqrt(np.outer(d0, d0))
            y = apply(r, wh).astype(np.float64)
            after = pooled(y)
            da = np.diag(after)

            # How much of each channel's variance survives whitening. A whitener that
            # is even-handed leaves this flat at 1.0; the naive covariance-space
            # shrinkage would leave the quiet channels far below it.
            ratio = d0 * da
            row = {
                "n_rep": int(wh.n_rep), "n_cond": int(wh.n_cond),
                "lam": wh.lam, "cond": wh.cond,
                "ch_var_min": float(d0.min()), "ch_var_max": float(d0.max()),
                "ch_var_spread": float(d0.max() / d0.min()),
                "max_abs_corr_before": float(np.abs(corr0 - np.eye(len(d0))).max()),
                "whitened_var_min": float(da.min()), "whitened_var_max": float(da.max()),
                "max_abs_resid_after": float(np.abs(after - np.eye(len(da))).max()),
                "diag_preserved": bool(np.allclose(np.diag(wh.sigma), d0, rtol=1e-6)),
                "quietest_channel_ratio": float(ratio.min()),
                "loudest_channel_ratio": float(ratio.max()),
                "seconds": round(time.time() - t0, 1),
            }
            report[f"sub-{s:02d}_{split}"] = row
            print(f"sub-{s:02d} {split:5s}  var spread {row['ch_var_spread']:7.1f}x  "
                  f"max|corr| {row['max_abs_corr_before']:.3f} -> "
                  f"{row['max_abs_resid_after']:.3f}  "
                  f"whitened var [{da.min():.3f},{da.max():.3f}]  "
                  f"lam {wh.lam:.4f} cond {wh.cond:.0f}  {row['seconds']}s", flush=True)
            del r, y, before, after

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
