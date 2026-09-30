#!/usr/bin/env python
"""
lib_windows.py -- SINGLE SOURCE OF TRUTH for the THINGS-EEG2 time axis.

=== THE TRAP THIS MODULE PREVENTS ===========================================
info.json in the preprocessed data declares

    times           = -0.200 .. +0.996 s   (300 samples)
    baseline_duration = 0.2
    after_duration    = 1.0
    sfreq             = 250

but the STORED ARRAYS have only 250 timepoints, and their sample 0 IS STIMULUS
ONSET.  The -0.2 s baseline has already been removed.  Trusting info.json's
`times` therefore shifts every window by 200 ms.

We were bitten by exactly this.  Using the declared axis, the window deemed
"pre-stimulus" (samples 0-50) is really the 0-200 ms early evoked response -- the
STRONGEST neural signal in the dataset.  The A1 negative control duly reported a
large effect there (held-out canonical correlation 0.27, p < 0.001 vs a
1200-sample permutation null, and still present in a fully CLIP-free split-half
reliability of 0.078) and the pipeline was wrongly declared contaminated.

Three independent checks fixed the axis (scripts/detect_onset.py, 3 subjects):
  1. The across-concept F-ratio (CLIP-free, whitening-free, label-free) is BELOW
     the median at sample 0 (F ~ 0.57) and peaks at samples 25-30 (F ~ 1.25),
     i.e. 100-120 ms after sample 0 -- the classic P1 latency.
  2. Fitting the expected ERP shape (quiet for ~50 ms after onset, then a peak at
     75-175 ms) selects sample 0 in every subject, with peak/quiet 1.22-1.30.
     Forcing onset = 50 gives ratio 0.92-0.96, i.e. it does not fit.
  3. Split-half RDM reliability peaks at ~100 ms and decays to ~0 by 300 ms, and
     is ~0 in the first 20 samples -- impossible for a pre-stimulus interval.
=============================================================================
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np

EEG_ROOT = "/project/peilab/why/NeuroBridge/data/things_eeg"

SFREQ = 250.0
N_SAMPLES = 250                    # what the arrays actually contain
ONSET_SAMPLE = 0                   # VERIFIED: sample 0 == t = 0 s
EPOCH_END_S = (N_SAMPLES - 1) / SFREQ      # ~0.996 s

# There is NO pre-stimulus interval in the stored arrays.  This constant exists so
# that no script can quietly reintroduce one.
HAS_PRE_STIMULUS = False


def sample_to_s(i: int) -> float:
    """Sample index -> seconds relative to stimulus onset (VERIFIED axis)."""
    return (i - ONSET_SAMPLE) / SFREQ


def s_to_sample(t: float) -> int:
    return int(round(t * SFREQ)) + ONSET_SAMPLE


# ---------------------------------------------------------------------------
# Canonical windows.  (name, lo_sample, hi_sample) with onset at sample 0.
# ---------------------------------------------------------------------------
WINDOWS: List[Tuple[str, int, int]] = [
    ("w0_000_100ms", 0, 25),      # early evoked, P1 rising
    ("w1_100_200ms", 25, 50),     # P1/N1 peak complex
    ("w2_200_300ms", 50, 75),
    ("w3_300_500ms", 75, 125),
    ("w4_500_800ms", 125, 200),
    ("w5_800_1000ms", 200, 250),
    ("full_000_800ms", 0, 200),
    ("full_000_1000ms", 0, 250),
]

# The primary confirmatory window: where the visual response is strongest and
# most reliable (verified by split-half reliability, docs A2).
PRIMARY_WINDOW = "w1_100_200ms"

# Windows that must NOT be used as "pre-stimulus" controls, kept here with an
# explicit warning so the mistake cannot be repeated silently.
FORBIDDEN_AS_PRE_STIMULUS = ["w0_000_100ms", "w1_100_200ms"]


def get_window(name: str) -> Tuple[int, int]:
    for n, lo, hi in WINDOWS:
        if n == name:
            return lo, hi
    raise KeyError(f"unknown window {name!r}; known: {[w[0] for w in WINDOWS]}")


def slice_eeg(E: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """(n, T, ch, time) -> (n, T, ch*time), baseline-corrected where possible.

    Since the stored arrays start at onset, there is no baseline to subtract for
    the early windows; we remove the per-window channel mean instead, which is the
    only unbiased option available.  For later windows this is nearly equivalent
    to a real baseline correction because the response has decayed.
    """
    W = E[:, :, :, lo:hi].astype(np.float32)
    W = W - W.mean(-1, keepdims=True)
    n, T, c, t = W.shape
    return W.reshape(n, T, c * t)


def verify_axis(quiet: bool = True) -> Dict:
    """Re-run the axis verification and report.  Cheap enough to call in CI."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from detect_onset import f_curve, load_eeg  # noqa: E402
    res = {}
    for sub in ["sub-01", "sub-02"]:
        E = load_eeg(sub)
        F = f_curve(E, 5).mean(0)
        F = F / np.median(F)
        best, bs = None, -1e9
        for c in range(0, 200):
            q = F[c:c + 10].mean()
            p = F[c + 19:c + 44].mean()
            if p / max(q, 1e-9) > bs:
                best, bs = c, p / max(q, 1e-9)
        res[sub] = {"shape_fit_onset": int(best),
                    "ratio": float(bs),
                    "F_at_sample0": float(F[0]),
                    "F_peak_sample": int(np.argmax(F[20:50]) + 20)}
        if not quiet:
            print(f"  {sub}: onset={best} ratio={bs:.3f} "
                  f"F[0]={F[0]:.3f} peak_sample={res[sub]['F_peak_sample']}")
    ok = all(v["shape_fit_onset"] == 0 for v in res.values())
    return {"verified_onset_zero": ok, "subs": res,
            "expected_onset_sample": ONSET_SAMPLE}


if __name__ == "__main__":
    print("lib_windows -- THINGS-EEG2 axis (VERIFIED)")
    print(f"  onset sample   = {ONSET_SAMPLE}  (sample 0 == t=0)")
    print(f"  n_samples      = {N_SAMPLES}  -> 0.000 .. {EPOCH_END_S:.3f} s")
    print(f"  pre-stimulus available? {HAS_PRE_STIMULUS}")
    print(f"  primary window = {PRIMARY_WINDOW}")
    print("\n  windows:")
    for n, lo, hi in WINDOWS:
        print(f"    {n:18s} samples {lo:3d}-{hi:3d}  "
              f"t {sample_to_s(lo):+.3f} .. {sample_to_s(hi):+.3f} s")
    print("\n  re-verifying axis from data...")
    print(json.dumps({k: v for k, v in verify_axis(quiet=False).items()},
                     indent=2, default=str))
