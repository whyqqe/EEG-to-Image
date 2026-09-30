#!/usr/bin/env python3
"""Summarize SAMGA-R reconstruction metrics for one held-out subject.

WHY THIS IS A SEPARATE FILE RATHER THAN AN INLINE HEREDOC
--------------------------------------------------------
This merge logic is subtle enough that it deserves to be runnable and diffable on its own.
The four metric files written by the pipeline share key names but NOT value types, and one
of them is written later in alphabetical order than the others:

  samgar_<tgt>_<mode>.json            scalars: pixcorr, ssim, clip_cosine, alexnet2/5, ...
  samgar_<tgt>_<mode>_2wc.json        twoway: {clip, alex2, alex5, inception}
  samgar_<tgt>_<mode>_fid.json        fid
  samgar_<tgt>_<mode>_bootstrap.json  pixcorr/ssim/clip_cosine as {mean, ci95_lo, ci95_hi}

Sorted alphabetically, `..._bootstrap.json` lands last. A plain dict update over the four
files therefore replaces the three scalars with dicts, and the first version of this
summary did exactly that. The visible symptoms were cosmetic-looking but hid real
information: the table cells printed raw dictionaries, and the head-vs-identity delta lines
disappeared entirely because the `isinstance(value, float)` guard silently failed. Nothing
crashed, so it read like a correct report of "no comparison available".

So the merge is explicit: scalars come from the main file, confidence intervals come from
the bootstrap file, and both are reported.

Usage:
  summarize_metrics.py <metrics_dir> <sub-08|...>
"""

from __future__ import annotations

import glob
import json
import os
import sys

SCALAR_KEYS = ("pixcorr", "ssim", "clip_cosine", "alexnet2", "alexnet5", "inception",
               "effnet_b1", "fid")
CI_KEYS = ("pixcorr", "ssim", "clip_cosine")
TWO_WAY_KEYS = ("clip", "alex2", "alex5", "inception")
# Every metric here is higher-is-better except FID, which is a distance. Mixing them in one
# delta column without saying so invites reading a negative FID delta as a regression.
LOWER_IS_BETTER = {"fid"}


def load_rows(met_dir: str, tgt: str):
    scalars: dict[str, dict[str, float]] = {}
    cis: dict[str, dict[str, dict]] = {}
    twoway: dict[str, dict[str, float]] = {}

    pattern = os.path.join(met_dir, f"samgar_{tgt}_*.json")
    for path in sorted(glob.glob(pattern)):
        try:
            with open(path) as fh:
                d = json.load(fh)
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        mode = "identity" if "_identity" in os.path.basename(path) else "head"
        is_boot = "bootstrap" in os.path.basename(path)

        for k in SCALAR_KEYS:
            v = d.get(k)
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                # setdefault: the first writer wins, and the main file sorts first, so a
                # later duplicate scalar can never shadow the primary measurement.
                scalars.setdefault(mode, {}).setdefault(k, float(v))
            elif is_boot and isinstance(v, dict) and "mean" in v:
                cis.setdefault(mode, {})[k] = v

        tw = d.get("twoway")
        if isinstance(tw, dict):
            for k in TWO_WAY_KEYS:
                if isinstance(tw.get(k), (int, float)):
                    twoway.setdefault(mode, {}).setdefault(k, float(tw[k]))

    return scalars, cis, twoway


def ci_note(cis, mode_a, mode_b, key):
    """Return whether the two 95% CIs overlap, as 'yes'/'no'/''.

    Overlapping intervals are the honest default verdict for a 200-trial comparison: a
    delta with overlapping CIs is not yet separable from resampling noise, and this pipeline
    reports one fold, so there is no fold-level variance to appeal to either.
    """
    ca = cis.get(mode_a, {}).get(key)
    cb = cis.get(mode_b, {}).get(key)
    if not (ca and cb):
        return ""
    disjoint = ca["ci95_hi"] < cb["ci95_lo"] or cb["ci95_hi"] < ca["ci95_lo"]
    return "no" if disjoint else "yes"


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    met_dir, tgt = sys.argv[1], sys.argv[2]

    scalars, cis, twoway = load_rows(met_dir, tgt)
    if not scalars:
        print(f"  no metrics found under {met_dir} matching samgar_{tgt}_*.json")
        return 1

    for mode in ("head", "identity"):
        if mode not in scalars:
            continue
        s = scalars[mode]
        g = lambda k: s.get(k, float("nan"))
        print(f"  [{mode}]  (gallery: selected_brain, n=200)")
        print(f"    low-level    pixcorr={g('pixcorr'):.4f}   ssim={g('ssim'):.4f}")
        for k in CI_KEYS:
            c = cis.get(mode, {}).get(k)
            if c:
                print(f"                 {k:<10} 95% CI [{c['ci95_lo']:.4f}, {c['ci95_hi']:.4f}]")
        print(f"    high-level   clip={g('clip_cosine'):.4f}   alexnet2={g('alexnet2'):.4f}"
              f"   alexnet5={g('alexnet5'):.4f}   inception={g('inception'):.4f}")
        print(f"    distribution effnet_b1={g('effnet_b1'):.4f}   FID={g('fid'):.2f}")
        if mode in twoway:
            t = twoway[mode]
            print("    2-way (200-way, 199 distractors)   "
                  + "   ".join(f"{k}={t.get(k, float('nan')):.4f}" for k in TWO_WAY_KEYS))
        print()

    if "head" in scalars and "identity" in scalars:
        print("  head vs identity  (does the generation head earn its place?)")
        print(f"    {'metric':<12}{'head':>10}{'identity':>10}{'delta':>10}    CI overlap")
        for k in SCALAR_KEYS:
            a, b = scalars["head"].get(k), scalars["identity"].get(k)
            if a is None or b is None:
                continue
            width = 10 if k == "fid" else 10
            prec = 2 if k == "fid" else 4
            direction = " (lower better)" if k in LOWER_IS_BETTER else ""
            print(f"    {k:<12}{a:>{width}.{prec}f}{b:>{width}.{prec}f}"
                  f"{a - b:>+{width}.{prec}f}    {ci_note(cis, 'head', 'identity', k)}{direction}")
        print("    CI overlap 'yes' = the two arms are not yet separable at 95%.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
