#!/usr/bin/env python3
"""Quantile-match a condition's concentration onto the TRAIN concept bank.

WHY THIS EXISTS.  The IP-Adapter reads a condition's DIRECTION, but the diffusion
prior's response also depends on its CONCENTRATION.  Two conditions that differ
only in how tightly they cluster produce different images, so an uncalibrated row
is partly a measurement of the read-out's own shrinkage rather than of the EEG.
`ocf_train.calibrate_quantile` is the label-free fix already used by this project:
move each row's `c_self = cos(x, mean)` onto the quantiles of the TRAIN concept
bank's own `c_self` distribution.  It is reused verbatim, not reimplemented.

WHY IT IS A SEPARATE SCRIPT.  The calibration is applied AFTER validation and only
to what is handed to the generator.  Keeping it out of the training script means
the exported `ip_fused_*` arrays stay exactly what the model predicted, so a
difference between two generation rows cannot come from a calibration step that
ran in one arm and not the other.

LEAK-FREE by construction: the reference statistic is the 1654-concept TRAIN
gallery.  The test set is not involved, and the operation is a per-row monotone
rescale, so it cannot introduce information from any other row.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

NB_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ocf_train import calibrate_quantile                            # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", type=str, required=True)
    ap.add_argument("--out", dest="dst", type=str, required=True)
    ap.add_argument("--ref", type=str, required=True,
                    help="TRAIN concept-bank embeddings; the calibration statistic")
    ap.add_argument("--tag", type=str, default="cond")
    ap.add_argument("--report", type=str, default="")
    ap.add_argument("--one-sided", action="store_true",
                    help="reproduce the OLD behaviour, which could only increase "
                         "concentration and was a no-op on any row already more "
                         "concentrated than the reference. Kept so the fix can be "
                         "ablated; do not use it for a reported result.")
    args = ap.parse_args()

    ref_p = Path(args.ref)
    if not ref_p.is_file():
        raise SystemExit(f"[FATAL] reference bank missing: {ref_p}. Calibrating against "
                         f"the test set instead is the contamination this avoids.")
    x = np.load(args.src).astype(np.float32)
    ref = np.load(ref_p).astype(np.float32)
    y, rep = calibrate_quantile(x, ref, two_sided=not args.one_sided)
    Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.dst, y.astype(np.float32))
    rep.update({"tag": args.tag, "src": str(args.src), "dst": str(args.dst),
                "reference": str(ref_p), "n": int(len(x))})
    out = Path(args.report) if args.report else Path(args.dst).with_suffix(".json")
    out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"[calib] {args.tag}: c_self {rep['c_self_ours_before']:.4f} -> "
          f"{rep['c_self_ours_after']:.4f} (bank {rep['c_self_ref']:.4f}), "
          f"ratio {rep['ratio_before']:.3f} -> {rep['ratio_after']:.3f}, "
          f"alpha_mean {rep['alpha_mean']:.3f}, zero_frac {rep['alpha_zero_frac']:.3f}")
    # MOVE-COUNT, PRINTED LOUDLY.  The one-sided version of this function was a
    # total no-op on sub-08 (all 200 rows already more concentrated than the bank
    # reference) and said so only through `ratio_before == ratio_after` in a log
    # line that nobody was comparing.  A calibration stage that moves nothing must
    # be as visible as one that moves everything.
    print(f"[calib] {args.tag}: rows raised {rep['n_raised']} / lowered "
          f"{rep['n_lowered']} / no-op {rep['n_noop']} / unreachable "
          f"{rep['n_blocked']}   (exactness mean|achieved-target| "
          f"{rep['achieved_vs_target_mae']:.2e}, two_sided={rep['two_sided']})")
    if rep["n_noop"] == rep["n_rows"]:
        print(f"[calib] [FAIL] {args.tag}: this calibration changed NOTHING "
              f"({rep['n_rows']} of {rep['n_rows']} rows no-op). The condition handed "
              f"to the generator is the uncalibrated one, so any claim that the "
              f"pipeline calibrates concentration is false for this run.")
    elif rep["n_noop"] > 0:
        print(f"[calib] [warn] {args.tag}: {rep['n_noop']} rows left as-is.")


if __name__ == "__main__":
    main()
