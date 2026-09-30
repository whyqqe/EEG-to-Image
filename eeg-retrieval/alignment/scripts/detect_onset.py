#!/usr/bin/env python
"""
Determine the TRUE stimulus onset index in the stored EEG arrays.

WHY THIS IS URGENT
    Time-resolved split-half reliability peaks at -0.09 s in the metadata time
    axis, i.e. a reproducible concept-specific response ~90 ms BEFORE the nominal
    stimulus onset.  A genuine anticipatory response is impossible here (trial
    order is randomised), so the only physical explanation is that the stored
    arrays are NOT aligned with the `times` field in info.json.

    info.json declares:  times = -0.200 .. +0.996 s (300 samples), baseline 0.2 s,
    after 1.0 s.  But the arrays hold only 250 samples.  The arrays are therefore
    a 250-sample sub-epoch, and the baseline was very likely already removed,
    making sample 0 the stimulus onset rather than -0.2 s.

    EVERY WINDOW DEFINITION IN THE PIPELINE DEPENDS ON THIS.  If onset is at 0
    rather than 50, then the "pre-stimulus" window I used for the A1 negative
    control is actually the 0-200 ms early evoked response -- the STRONGEST
    neural signal in the data.  That would fully explain the A1 "contaminated"
    result: the gate was testing the wrong interval, and the pipeline is not
    contaminated at all.

HOW THE ONSET IS FOUND
    The across-concept F-ratio, F(t) = Var_between / Var_within_per_trial, has
    expectation ~1 wherever there is no stimulus-driven concept-specific signal
    and rises where there is.  It needs no image features, no CLIP, no whitening.
    A visual ERP must show F ~ 1 for the initial ~50 ms after onset (before the
    cortical response) and rise to a peak around 100-150 ms.  Fitting that shape
    locates the onset to the sample.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

EEG_ROOT = "/project/peilab/why/NeuroBridge/data/things_eeg"


def load_eeg(sub, which="train"):
    a = np.load(f"{EEG_ROOT}/preprocessed_eeg/{sub}/{which}.npy")
    if a.ndim == 5:
        a = a.mean(2)
    return a.astype(np.float32)


def f_curve(E: np.ndarray, win: int = 5) -> np.ndarray:
    """F(t) = across-concept variance / (within-concept variance / T)."""
    n, T, ch, nt = E.shape
    m = E.mean(1)                                   # (n, ch, nt)
    var_b = m.var(0)                                # (ch, nt)
    res = E - m[:, None]
    var_w = (res ** 2).mean((0, 1)) / max(T, 1)     # (ch, nt)
    F = var_b / np.maximum(var_w, 1e-30)
    # smooth over time
    if win > 1:
        k = np.ones(win) / win
        F = np.apply_along_axis(lambda x: np.convolve(x, k, mode="same"), 1, F)
    return F


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/project/peilab/why/eeg-retrieval/alignment/outputs/onset")
    ap.add_argument("--subs", nargs="+", default=["sub-01", "sub-02"])
    ap.add_argument("--win", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("=" * 100)
    print("TRUE ONSET DETECTION in the stored EEG arrays")
    print(f"  metric: across-concept F-ratio (CLIP-free, whitening-free, label-free)")
    print(f"  info.json claims onset (t=0) at sample 50; testing that claim")
    print("=" * 100)

    out = {"candidates": {}, "subs": {}}
    results = {}
    for sub in args.subs:
        E = load_eeg(sub)
        n, T, ch, nt = E.shape
        F = f_curve(E, args.win)
        Fm = F.mean(0)                                   # (nt,) averaged over channels
        Fm = Fm / np.nanmedian(Fm)

        print(f"\n--- {sub}  shape={E.shape} ---")
        print("  F/F_median by sample (first 90 samples, ~4 ms steps):")
        s = "   "
        for i in range(0, 90, 5):
            s += f" {i:4d}:{Fm[i]:.3f}"
            if (i - 0) % 25 == 20:
                print(s); s = "   "
        if s.strip():
            print(s)

        # A visual ERP must be quiet for the first ~50 ms AFTER onset, then rise.
        # For each candidate onset c, score how well F matches that shape:
        #   quiet window  = c .. c+50ms (<=12 samples)   -> low
        #   peak window   = c+75ms .. c+175ms            -> high
        best, best_score = None, -1e9
        for c in range(0, nt - 60):
            quiet = Fm[c:c + 10].mean()
            peak = Fm[c + 19:c + 44].mean()
            score = peak / max(quiet, 1e-9)
            if score > best_score:
                best, best_score = c, score
        onset_samples = {"0 (arrays start at onset)": 0, "50 (info.json claim)": 50,
                         f"{best} (shape-fit)": best}
        print(f"\n  shape-fit best onset = sample {best} ({best/250*1000:.0f} ms), "
              f"peak/quiet = {best_score:.3f}")
        for lab, c in onset_samples.items():
            q = Fm[max(c, 0):c + 10].mean() if c + 10 <= nt else np.nan
            p = Fm[c + 19:c + 44].mean() if c + 44 <= nt else np.nan
            print(f"    onset={lab:32s} quiet[+0,+40ms]={q:.3f}  "
                  f"peak[+75,+175ms]={p:.3f}  ratio={p/max(q,1e-9):.3f}")

        results[sub] = {"F": Fm.tolist(), "n": n, "T": T, "nt": nt,
                        "best_shape_fit": int(best), "best_score": float(best_score)}
        out["subs"][sub] = results[sub]

    # ---- verdict: which candidate onset is consistent across subjects? -----
    bf = [results[s]["best_shape_fit"] for s in results]
    verdict = {
        "shape_fit_onsets": bf,
        "shape_fit_mean": float(np.mean(bf)),
        "info_json_claim": 50,
        "arrays_start_at_onset": bool(np.mean(bf) < 15),
    }
    # the decisive comparison
    print("\n" + "=" * 100)
    print("VERDICT")
    print(f"  shape-fit onset samples per subject: {bf}  (mean {np.mean(bf):.1f})")
    if np.mean(bf) < 15:
        verdict["conclusion"] = (
            "ARRAYS START AT STIMULUS ONSET (sample 0 == t=0). The `times` field "
            "in info.json describes a longer epoch and does NOT apply to the "
            "stored 250-sample arrays; the -0.2 s baseline was already removed. "
            "CONSEQUENCE: every window in the pipeline is shifted by 200 ms. The "
            "window used as the A1 'pre-stimulus' negative control (samples 0-50) "
            "is in fact the 0-200 ms early evoked response, the strongest signal "
            "in the data. A1's 'contamination' was a mislabelled window, not a "
            "pipeline defect. A true pre-stimulus negative control does NOT exist "
            "in this dataset and must be replaced by the shuffled-EEG control."
        )
        print("  >>> ARRAYS START AT ONSET (sample 0 == t=0)")
        print("  >>> info.json `times` does not apply to the stored arrays.")
        for line in verdict["conclusion"].split(". "):
            print(f"      {line.strip()}.")
    else:
        verdict["conclusion"] = (
            f"Onset appears to be near sample {np.mean(bf):.0f}, consistent with "
            "the info.json claim; the earlier finding needs a different "
            "explanation."
        )
        print(f"  onset near sample {np.mean(bf):.0f}")
    out["verdict"] = verdict

    with open(os.path.join(args.out, "onset.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  wrote {os.path.join(args.out, 'onset.json')}")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
