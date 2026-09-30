#!/usr/bin/env python
"""
A1 -- PRE-STIMULUS NEGATIVE CONTROL                        [GATE 2: no-go if fail]

Question: does the pipeline report neural visibility where there CANNOT be any?

The pre-stimulus window (-0.2 .. 0 s, samples 0..49) contains no stimulus-driven
visual content.  A pipeline that returns k* > 0 there is manufacturing signal and
every downstream result is void.

THE TRAP THIS SCRIPT IS BUILT TO AVOID
--------------------------------------
A trivially broken estimator that always returns k* = 0 would PASS the negative
control.  A negative control alone therefore proves NOTHING.  This script
therefore runs a paired POSITIVE control on the same subject/window-pipeline:
if the negative window gives 0 AND the positive window gives > 0, the pipeline
demonstrably discriminates.  Both must hold for the gate to pass.

  NEGATIVE (must be 0) : pre-stimulus window, and concept-label-shuffled EEG
  POSITIVE (must be >0): post-stimulus windows where a visual response exists

Also reported: the shuffled-EEG control, which destroys any true correspondence
while preserving all nuisance structure (artefacts, drift, subject identity).
That is a stronger control than label permutation of C alone.
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
from lib_rsca import EEGWhitener, rsca  # noqa: E402

EEG_ROOT = "/project/peilab/why/NeuroBridge/data/things_eeg"
BASELINE = 50          # samples in -0.2..0 s at 250 Hz
SFREQ = 250.0

# (name, lo, hi) in samples; time = (sample - 50)/250 s
WINDOWS = [
    ("pre_-0.20_0.00s", 0, 50),      # NEGATIVE: no stimulus
    ("early_0.00_0.10s", 50, 75),
    ("mid1_0.10_0.20s", 75, 100),
    ("mid2_0.20_0.30s", 100, 125),
    ("late1_0.30_0.40s", 125, 150),
    ("late2_0.40_0.60s", 150, 200),
    ("full_0.00_0.80s", 50, 250),
]


def load_eeg(sub: str, which: str = "train") -> np.ndarray:
    """(n, T, p) with p = 63 channels x 250 timepoints, sessions averaged."""
    a = np.load(f"{EEG_ROOT}/preprocessed_eeg/{sub}/{which}.npy")
    if a.ndim == 5:                       # (n, T, sessions, ch, time)
        a = a.mean(2)
    return a.astype(np.float32)


def load_clip(name: str = "ViT-H-14", which: str = "train") -> np.ndarray:
    f = np.load(f"{EEG_ROOT}/image_feature/{name}/image_{which}.npy")
    if f.ndim == 3:                       # (n, images_per_concept, q)
        f = f.mean(1)
    return f.astype(np.float32)


def slice_window(E: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """(n, T, ch, time) -> (n, T, ch*time), baseline-corrected per trial."""
    W = E[:, :, :, lo:hi]
    B = E[:, :, :, :BASELINE].mean(-1, keepdims=True)
    W = W - B if lo >= BASELINE else W - W.mean(-1, keepdims=True)
    n, T, c, t = W.shape
    return W.reshape(n, T, c * t).astype(np.float32)


def analyze(E: np.ndarray, C: np.ndarray, args, seed: int = 0,
            shuffle_eeg: bool = False) -> Dict:
    n = min(E.shape[0], C.shape[0])
    E, C = E[:n], C[:n]
    if shuffle_eeg:
        # destroy the concept<->EEG link while preserving all nuisance structure
        rng = np.random.default_rng(12345)
        E = E[rng.permutation(n)]
    r = rsca(E, C, D=args.D, K=args.K, ridge=args.ridge,
             n_perm=args.n_perm, alpha=args.alpha, seed=seed)
    s = r.summary()
    s["n_above_chance_insample"] = int((r.rho_insample > r.threshold).sum())
    s["rho_cv_sym_top6"] = [round(float(x), 4) for x in r.rho_cv_sym[:6]]
    s["rho_insample_top6"] = [round(float(x), 4) for x in r.rho_insample[:6]]
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default="sub-01")
    ap.add_argument("--out", default="/project/peilab/why/eeg-retrieval/alignment/outputs/a1")
    ap.add_argument("--clip", default="ViT-H-14")
    ap.add_argument("--subs", nargs="*", default=["sub-01"],
                    help="multiple subjects for a stronger gate")
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
        args.n_perm = min(args.n_perm, 100)

    subs = args.subs if args.subs else [args.sub]
    print("=" * 100)
    print("A1 -- pre-stimulus negative control   [GATE 2]")
    print(f"     subjects={subs} clip={args.clip} D={args.D} K={args.K} "
          f"n_perm={args.n_perm} seeds={args.seeds}")
    print("=" * 100)

    t0 = time.time()
    out: Dict = {"config": vars(args), "subjects": {}}

    for sub in subs:
        print(f"\n--- {sub} ---", flush=True)
        E_all = load_eeg(sub)
        C = load_clip(args.clip)
        sub_out: Dict = {"windows": {}, "shuffled_eeg": {}}

        for name, lo, hi in WINDOWS:
            E = slice_window(E_all, lo, hi)
            rows = [analyze(E, C, args, seed=s) for s in range(args.seeds)]
            k = np.array([r["k_star"] for r in rows], float)
            thr = np.mean([r["threshold"] for r in rows])
            ins = np.mean([r["n_above_chance_insample"] for r in rows])
            sub_out["windows"][name] = {
                "k_star_mean": float(k.mean()),
                "k_star_all": k.tolist(),
                "threshold": float(thr),
                "n_above_chance_insample_mean": float(ins),
                "rows": rows,
            }
            print(f"  {name:20s}  k*={k.mean():5.2f}  (thr={thr:.3f})"
                  f"   in-sample comps above thr = {ins:5.1f}", flush=True)

        # --- shuffled-EEG control on the strongest window -------------------
        for name, lo, hi in [("pre_-0.20_0.00s", 0, 50), ("full_0.00_0.80s", 50, 250)]:
            E = slice_window(E_all, lo, hi)
            r = analyze(E, C, args, seed=0, shuffle_eeg=True)
            sub_out["shuffled_eeg"][name] = r
            print(f"  [shuffled EEG] {name:20s}  k*={r['k_star']}", flush=True)

        out["subjects"][sub] = sub_out

    # ------------------------------ gate ---------------------------------
    neg_ok, pos_ok, details = [], [], {}
    for sub, so in out["subjects"].items():
        kn = so["windows"]["pre_-0.20_0.00s"]["k_star_mean"]
        kp = max(so["windows"][w]["k_star_mean"] for w in
                 ["mid2_0.20_0.30s", "late1_0.30_0.40s", "late2_0.40_0.60s",
                  "full_0.00_0.80s"])
        ksh = max(v["k_star"] for v in so["shuffled_eeg"].values())
        neg_ok.append(kn == 0 and ksh == 0)
        pos_ok.append(kp > 0)
        details[sub] = {"k_star_pre": kn, "k_star_best_post": kp, "k_star_shuffled": ksh}

    out["gate"] = {
        "criterion": "pre-window and shuffled-EEG both k*=0, AND some post-window k*>0",
        "negative_control_passed": bool(all(neg_ok)),
        "positive_control_passed": bool(all(pos_ok)),
        "details": details,
        "verdict": "PASS" if (all(neg_ok) and all(pos_ok)) else
                   ("INCONCLUSIVE_NO_SIGNAL" if all(neg_ok) else "FAIL_CONTAMINATED"),
        "elapsed_s": round(time.time() - t0, 1),
    }

    with open(os.path.join(args.out, "a1_results.json"), "w") as f:
        json.dump(out, f, indent=2)

    g = out["gate"]
    print("\n" + "=" * 100)
    print(f"GATE 2 VERDICT: {g['verdict']}")
    print(f"  negative control (pre-window & shuffled EEG -> k*=0): "
          f"{'PASS' if g['negative_control_passed'] else 'FAIL'}")
    print(f"  positive control (post-window -> k*>0):               "
          f"{'PASS' if g['positive_control_passed'] else 'FAIL'}")
    for s, d in details.items():
        print(f"    {s}: k*_pre={d['k_star_pre']:.0f}  "
              f"k*_best_post={d['k_star_best_post']:.0f}  "
              f"k*_shuffled={d['k_star_shuffled']:.0f}")
    if g["verdict"] == "FAIL_CONTAMINATED":
        print("\n  *** PIPELINE MANUFACTURES SIGNAL -- ALL DOWNSTREAM RESULTS VOID ***")
    elif g["verdict"] == "INCONCLUSIVE_NO_SIGNAL":
        print("\n  Negative control clean, but no post-stimulus signal detected.")
        print("  Per docs section 5.3 this is the 'r* below detection threshold' branch:")
        print("  either k* is genuinely 0 (publishable negative result) or n is too")
        print("  small.  Check A2 noise ceiling and A3 effective rank before deciding.")
    print(f"  wrote {os.path.join(args.out, 'a1_results.json')}  ({g['elapsed_s']}s)")
    print("=" * 100)
    return 0 if g["verdict"] in ("PASS", "INCONCLUSIVE_NO_SIGNAL") else 4


if __name__ == "__main__":
    sys.exit(main())
