#!/usr/bin/env python
"""
A1 -- CONTROL BATTERY                                        [GATE 2: no-go if fail]

A pipeline that always returns k* = 0 passes any lone negative control, so a
negative control on its own proves nothing.  This gate therefore runs a PAIRED
battery and only passes when the pipeline demonstrably discriminates:
true EEG gives k* > 0 somewhere, while every control that breaks the
concept<->brain link gives k* = 0 everywhere.

=== IMPORTANT: WHY THERE IS NO PRE-STIMULUS CONTROL HERE =====================
The obvious negative control -- a pre-stimulus window -- DOES NOT EXIST in this
dataset.  The stored arrays begin at stimulus onset (sample 0 == t = 0 s); the
-0.2 s baseline was already removed, contrary to what info.json's `times` field
implies.  Using the declared axis makes samples 0-50 look "pre-stimulus" when
they are really the 0-200 ms early evoked response, the strongest signal in the
data.  An earlier version of this gate made exactly that mistake and reported
"contamination" on a working pipeline.  See lib_windows.py and
scripts/detect_onset.py for the three independent verifications.

The controls below are strictly stronger anyway, because they attack the actual
claim ("this EEG structure tracks THESE images") rather than merely asserting
that something happens before onset:

  C1  SHUFFLED EEG (primary).  Permute the concept order of the EEG, destroying
      the concept<->brain link while preserving every nuisance structure --
      artefacts, drift, subject-specific patterns, the temporal covariance of the
      field, the trial noise.  If the pipeline reports k* > 0 here it is
      manufacturing signal from nuisance structure.  This is the strongest
      available control.
  C2  SHUFFLED CLIP.  The same permutation applied to the image features instead.
      Equivalent in effect, included to catch asymmetry bugs in the pipeline.
  C3  CIRCULAR TIME SHIFT.  Roll each concept's EEG along the time axis by a
      random per-concept offset.  Preserves the spectrum and the temporal
      autocorrelation of every trial, but destroys the alignment between the
      response time course and the stimulus.  This catches pipelines that latch
      onto slow temporal covariation rather than an evoked response.
  C4  POSITIVE CONTROL.  The early evoked windows must give k* > 0.  Without
      this, a broken-but-silent pipeline would "pass" C1-C3.
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_rsca import rsca  # noqa: E402
from lib_windows import EEG_ROOT, WINDOWS, get_window, slice_eeg  # noqa: E402


def load_eeg(sub: str, which: str = "train") -> np.ndarray:
    a = np.load(f"{EEG_ROOT}/preprocessed_eeg/{sub}/{which}.npy")
    if a.ndim == 5:                      # (n, T, sessions, ch, time)
        a = a.mean(2)
    return a.astype(np.float32)


def load_clip(name: str = "ViT-H-14", which: str = "train") -> np.ndarray:
    f = np.load(f"{EEG_ROOT}/image_feature/{name}/image_{which}.npy")
    if f.ndim == 3:
        f = f.mean(1)
    return f.astype(np.float32)


def circular_shift(E: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Roll each concept's time axis by an independent random offset.

    Must be applied to the 4-D (n, T, ch, time) array BEFORE window slicing,
    because after slicing the time axis is already flattened into the feature
    dimension and rolling it would mix channels with timepoints.
    """
    n, T, ch, nt = E.shape
    offs = rng.integers(1, nt, size=n)
    out = np.empty_like(E)
    for i in range(n):
        out[i] = np.roll(E[i], int(offs[i]), axis=2)
    return out


def run_one(E: np.ndarray, C: np.ndarray, args, seed: int) -> Dict:
    n = min(E.shape[0], C.shape[0])
    r = rsca(E[:n], C[:n], D=args.D, K=args.K, ridge=args.ridge,
             n_perm=args.n_perm, alpha=args.alpha, seed=seed)
    s = r.summary()
    s["rho_cv_sym_top6"] = [round(float(x), 4) for x in r.rho_cv_sym[:6]]
    s["rho_insample_top6"] = [round(float(x), 4) for x in r.rho_insample[:6]]
    s["n_insample_above_thr"] = int((r.rho_insample > r.threshold).sum())
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/project/peilab/why/eeg-retrieval/alignment/outputs/a1")
    ap.add_argument("--subs", nargs="+",
                    default=["sub-01", "sub-02", "sub-03", "sub-04", "sub-05"])
    ap.add_argument("--clip", default="ViT-H-14")
    ap.add_argument("--D", type=int, default=256)
    ap.add_argument("--K", type=int, default=24)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--n-perm", type=int, default=500)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.quick:
        args.n_perm = min(args.n_perm, 60)
        args.subs = args.subs[:1]

    print("=" * 104)
    print("A1 -- control battery   [GATE 2]")
    print(f"     subs={args.subs} clip={args.clip} D={args.D} K={args.K} "
          f"n_perm={args.n_perm} seeds={args.seeds}")
    print("     NOTE: no pre-stimulus control exists in this dataset "
          "(arrays start at onset).")
    print("=" * 104)

    t0 = time.time()
    out: Dict = {"config": vars(args), "subjects": {}}

    for sub in args.subs:
        print(f"\n--- {sub} ---", flush=True)
        E_all = load_eeg(sub)
        C = load_clip(args.clip)
        n = min(E_all.shape[0], C.shape[0])
        E_all, C = E_all[:n], C[:n]
        so: Dict = {"windows": {}, "controls": {}}

        # ---------------- true EEG: positive control -----------------------
        for name, lo, hi in WINDOWS:
            E = slice_eeg(E_all, lo, hi)
            rows = [run_one(E, C, args, s) for s in range(args.seeds)]
            k = np.array([r["k_star"] for r in rows], float)
            so["windows"][name] = {
                "k_star_mean": float(k.mean()), "k_star_all": k.tolist(),
                "threshold": float(np.mean([r["threshold"] for r in rows])),
                "rho_cv_sym_top6": rows[0]["rho_cv_sym_top6"],
                "rho_insample_top6": rows[0]["rho_insample_top6"],
                "n_insample_above_thr": rows[0]["n_insample_above_thr"],
                "rows": rows,
            }
            print(f"  [TRUE EEG] {name:18s} k*={k.mean():5.2f} "
                  f"thr={so['windows'][name]['threshold']:.3f} "
                  f"rho={so['windows'][name]['rho_cv_sym_top6'][0] if so['windows'][name]['rho_cv_sym_top6'] else 'na'}",
                  flush=True)

        # ---------------- controls -----------------------------------------
        # run controls on the primary window AND the early window; a control only
        # needs to be null where the true pipeline is non-null.
        probe_windows = ["w0_000_100ms", "w1_100_200ms", "full_000_1000ms"]
        rng = np.random.default_rng(4242)
        # the time shift must act on the 4-D array, before slicing
        E_all_cs = circular_shift(E_all, rng)
        for name in probe_windows:
            lo, hi = get_window(name)
            E = slice_eeg(E_all, lo, hi)
            base_k = so["windows"][name]["k_star_mean"]
            entries = {}

            # C1 shuffled EEG (primary negative control)
            Esh = E[rng.permutation(E.shape[0])]
            r1 = run_one(Esh, C, args, 0)
            entries["C1_shuffled_eeg"] = r1

            # C2 shuffled CLIP
            Csh = C[rng.permutation(C.shape[0])]
            r2 = run_one(E, Csh, args, 0)
            entries["C2_shuffled_clip"] = r2

            # C3 circular time shift (per concept), sliced from the pre-shifted array
            Ecs = slice_eeg(E_all_cs, lo, hi)
            r3 = run_one(Ecs, C, args, 0)
            entries["C3_time_shift"] = r3

            so["controls"][name] = {"k_star_true": base_k, "entries": entries}
            print(f"  [CONTROLS ] {name:18s} true k*={base_k:.0f} | "
                  f"C1 shuffled-EEG k*={r1['k_star']} | "
                  f"C2 shuffled-CLIP k*={r2['k_star']} | "
                  f"C3 time-shift k*={r3['k_star']}", flush=True)

        out["subjects"][sub] = so

    # ------------------------------ gate ---------------------------------
    det = {}
    for sub, so in out["subjects"].items():
        ctrl_max = max(
            e["k_star"]
            for w in so["controls"].values() for e in w["entries"].values()
        )
        best_true = max(
            so["windows"][w]["k_star_mean"] for w in ["w0_000_100ms", "w1_100_200ms"]
        )
        det[sub] = {"k_star_best_true": best_true,
                    "k_star_max_control": ctrl_max,
                    "k_star_full": so["windows"]["full_000_1000ms"]["k_star_mean"]}

    neg_ok = all(d["k_star_max_control"] == 0 for d in det.values())
    pos_ok = all(d["k_star_best_true"] > 0 for d in det.values())

    out["gate"] = {
        "criterion": "every control gives k*=0 everywhere AND early windows give k*>0",
        "negative_control_passed": bool(neg_ok),
        "positive_control_passed": bool(pos_ok),
        "details": det,
        "verdict": ("PASS" if (neg_ok and pos_ok) else
                    ("INCONCLUSIVE_NO_SIGNAL" if neg_ok else "FAIL_CONTAMINATED")),
        "elapsed_s": round(time.time() - t0, 1),
    }

    with open(os.path.join(args.out, "a1_results.json"), "w") as f:
        json.dump(out, f, indent=2)

    g = out["gate"]
    print("\n" + "=" * 104)
    print(f"GATE 2 VERDICT: {g['verdict']}")
    print(f"  negative controls (C1 shuffled-EEG, C2 shuffled-CLIP, C3 time-shift) "
          f"all k*=0: {'PASS' if neg_ok else 'FAIL'}")
    print(f"  positive control (early evoked windows k*>0):              "
          f"{'PASS' if pos_ok else 'FAIL'}")
    for s, d in det.items():
        print(f"    {s}: k*_true(early)={d['k_star_best_true']:.0f}  "
              f"k*_max_control={d['k_star_max_control']:.0f}  "
              f"k*_full={d['k_star_full']:.0f}")
    if g["verdict"] == "FAIL_CONTAMINATED":
        print("\n  *** PIPELINE MANUFACTURES SIGNAL -- ALL DOWNSTREAM RESULTS VOID ***")
    elif g["verdict"] == "INCONCLUSIVE_NO_SIGNAL":
        print("\n  Controls clean but no true signal detected. Per docs 5.3 this is a")
        print("  legitimate branch: either k* is genuinely small, or n/K/D are too")
        print("  small. Consult A2's ceiling before concluding.")
    print(f"  wrote {os.path.join(args.out, 'a1_results.json')}  ({g['elapsed_s']}s)")
    print("=" * 104)
    return 0 if g["verdict"] in ("PASS", "INCONCLUSIVE_NO_SIGNAL") else 4


if __name__ == "__main__":
    sys.exit(main())
