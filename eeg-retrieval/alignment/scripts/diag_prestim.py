#!/usr/bin/env python
"""
Diagnose the A1 pre-stimulus window anomaly.

SYMPTOM
    The pre-stimulus window (which cannot contain stimulus-driven visual content)
    yields a held-out canonical correlation of ~0.34 against CLIP features and
    k* = 2, while the mid2/late1 windows yield nothing.  The shuffled-EEG control
    is clean (k* = 0).  Either the gate is miscalibrated or the pre-window is
    contaminated by design.

TWO DECISIVE TESTS
-----------------
  TEST 1 -- null calibration with enough permutations.
      n_perm=60 cannot estimate a 95% quantile of a max-over-12-statistic; the
      threshold was unreliable.  Recompute with n_perm=2000 and locate the
      observed pre-window rho inside the null distribution.  If rho sits at the
      99.9th percentile the structure is real; if the null's tail reaches it, the
      gate was simply miscalibrated.

  TEST 2 -- subspace overlap between the pre-window and post-window solutions.
      If the pre-window signal is an ECHO of the stimulus response, it must be
      carried by the same neural directions as the post-window signal, so the two
      k*-dimensional image-side subspaces will overlap strongly.  If the overlap
      is at chance, the pre-window contains independent structure and something
      else is going on.

WHY THIS MATTERS FOR THE SCIENCE
    A zero-phase (filtfilt) high-pass at 0.1 Hz has an effective impulse response
    spanning seconds, so it smears the stimulus response BOTH ways in time.  That
    makes the nominal -0.2..0 s "baseline" not pre-stimulus at all.  If confirmed,
    the pre-window must be dropped as a negative control and the shuffled-EEG
    control becomes the primary one.  That is a methodological finding in its own
    right, and it is exactly the kind of thing that silently invalidates naive
    EEG-encoding analyses.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_rsca import (  # noqa: E402
    EEGWhitener, cca_ridge, held_out_cca_corr, permutation_null, subspace_overlap,
)

EEG_ROOT = "/project/peilab/why/NeuroBridge/data/things_eeg"
BASELINE = 50


def load_eeg(sub, which="train"):
    a = np.load(f"{EEG_ROOT}/preprocessed_eeg/{sub}/{which}.npy")
    if a.ndim == 5:
        a = a.mean(2)
    return a.astype(np.float32)


def load_clip(name="ViT-H-14", which="train"):
    f = np.load(f"{EEG_ROOT}/image_feature/{name}/image_{which}.npy")
    if f.ndim == 3:
        f = f.mean(1)
    return f.astype(np.float32)


def slice_window(E, lo, hi):
    W = E[:, :, :, lo:hi]
    if lo >= BASELINE:
        B = E[:, :, :, :BASELINE].mean(-1, keepdims=True)
        W = W - B
    n, T, c, t = W.shape
    return W.reshape(n, T, c * t).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/project/peilab/why/eeg-retrieval/alignment/outputs/a1diag")
    ap.add_argument("--sub", default="sub-01")
    ap.add_argument("--D", type=int, default=256)
    ap.add_argument("--K", type=int, default=24)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--ridge", type=float, default=1e-3)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    E = load_eeg(args.sub)
    C = load_clip("ViT-H-14")
    n = min(E.shape[0], C.shape[0]); E, C = E[:n], C[:n]

    print("=" * 104)
    print(f"A1 anomaly diagnosis  sub={args.sub}  n={n}  D={args.D} K={args.K} "
          f"n_perm={args.n_perm}")
    print("=" * 104)

    out = {"sub": args.sub, "n": n, "D": args.D, "K": args.K, "n_perm": args.n_perm}
    T = E.shape[1]

    # Each window has a different p (channels x timepoints) and a different noise
    # covariance, so the whitener MUST be fitted per window.  Sharing one across
    # windows would be wrong in principle, not just awkward.
    def whiten_window(lo, hi):
        W = E[:, :, :, lo:hi]
        if lo >= BASELINE:
            W = W - E[:, :, :, :BASELINE].mean(-1, keepdims=True)
        X = W.reshape(n, T, -1).astype(np.float32)
        w = EEGWhitener(D=min(args.D, n - 1, X.shape[-1])).fit(X)
        return w.transform_trials_flat(X), w

    def window_mean(lo, hi):
        Xw, _ = whiten_window(lo, hi)
        return Xw[:, : T // 2].mean(1)

    # ---------------- TEST 1: proper null calibration --------------------
    print("\nTEST 1 -- where does the observed rho sit in the null?")
    print(f"  {'window':20s} {'rho_sym[0]':>10} {'thr(95%)':>9} {'null p50':>9} "
          f"{'null p99':>9} {'null max':>9} {'p-value':>9} {'k*':>4}")
    nA = n // 2
    idx = np.arange(n)
    for name, lo, hi in [("pre_-0.20_0.00s", 0, 50), ("mid2_0.20_0.30s", 100, 125),
                         ("full_0.00_0.80s", 50, 250)]:
        xa = window_mean(lo, hi)

        r_ab = cca_ridge(xa[idx[:nA]], C[idx[:nA]], K=args.K, ridge=args.ridge)
        rho = np.abs(held_out_cca_corr(r_ab, xa[idx[nA:]], C[idx[nA:]]))
        r_ba = cca_ridge(xa[idx[nA:]], C[idx[nA:]], K=args.K, ridge=args.ridge)
        rho_rev = np.abs(held_out_cca_corr(r_ba, xa[idx[:nA]], C[idx[:nA]]))
        rho_sym = np.minimum(rho, rho_rev)

        nul = permutation_null(xa, C, K=args.K, ridge=args.ridge,
                               n_perm=args.n_perm, alpha=0.05,
                               rng=np.random.default_rng(0))
        mx = nul["max_rho"]
        pval = float((mx >= rho_sym[0]).mean())
        k_at = int((rho_sym > nul["threshold"]).sum())
        print(f"  {name:20s} {rho_sym[0]:>10.4f} {nul['threshold']:>9.4f} "
              f"{np.quantile(mx,0.5):>9.4f} {np.quantile(mx,0.99):>9.4f} "
              f"{mx.max():>9.4f} {pval:>9.4f} {k_at:>4d}")
        out.setdefault("test1", {})[name] = {
            "rho_sym_top5": [float(v) for v in rho_sym[:5]],
            "threshold_95": float(nul["threshold"]),
            "null_p50": float(np.quantile(mx, 0.5)),
            "null_p99": float(np.quantile(mx, 0.99)),
            "null_max": float(mx.max()),
            "p_value": pval,
            "k_star_at_thr": k_at,
        }

    # ---------------- TEST 2: pre vs post subspace overlap ---------------
    print("\nTEST 2 -- is the pre-window an ECHO of the post-window?")
    print("  If pre is smeared post, the neural directions must overlap strongly.")
    print(f"  {'pair':34s} {'overlap':>9} {'chance':>9} {'ratio':>7}")

    def fit_subspace(lo, hi, k):
        xa = window_mean(lo, hi)
        r = cca_ridge(xa, C, K=k, ridge=args.ridge)
        return r.B[:, :k]

    rng = np.random.default_rng(7)
    q = C.shape[1]
    kk = 3
    chance = []
    for _ in range(300):
        A = rng.standard_normal((q, kk)); B = rng.standard_normal((q, kk))
        chance.append(subspace_overlap(A, B))
    chance_m = float(np.mean(chance))
    out["test2"] = {"chance": chance_m}

    for lab, (l1, h1), (l2, h2) in [
        ("pre vs post(full)", (0, 50), (50, 250)),
        ("pre vs early(0-0.1s)", (0, 50), (50, 75)),
        ("pre vs late1(0.3-0.4s)", (0, 50), (125, 150)),
        ("early vs late1", (50, 75), (125, 150)),
        ("pre vs pure-noise control", (0, 50), (0, 50)),
    ]:
        if lab.endswith("noise control"):
            V1 = fit_subspace(l1, h1, kk)
            R = rng.standard_normal((q, kk))
            o = subspace_overlap(V1, R)
        else:
            o = subspace_overlap(fit_subspace(l1, h1, kk), fit_subspace(l2, h2, kk))
        print(f"  {lab:34s} {o:>9.3f} {chance_m:>9.3f} {o/max(chance_m,1e-9):>7.2f}")
        out["test2"][lab] = float(o)

    # ---------------- TEST 3: is pre linearly predictable from post? -----
    print("\nTEST 3 -- how much of the pre-window EEG is linearly explained by the "
          "post-window EEG?")
    G = E.mean(1)
    nch = E.shape[2]
    Gpre = G[:, :50, :].reshape(n, -1)
    Gpost = G[:, 50:250, :].reshape(n, -1)
    Gpost_c = Gpost - Gpost.mean(0, keepdims=True)
    Gpre_c = Gpre - Gpre.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Gpost_c, full_matrices=False)
    r = 50
    B = Vt[:r].T
    Z = Gpost_c @ B
    coef, *_ = np.linalg.lstsq(Z, Gpre_c, rcond=None)
    r2 = float(1 - ((Gpre_c - Z @ coef) ** 2).sum() / max((Gpre_c ** 2).sum(), 1e-30))
    print(f"  R^2 of concept-mean pre-window EEG predicted from post-window "
          f"(rank {r}): {r2:.4f}")
    print(f"  -> a large R^2 is the signature of temporal smearing / carryover.")
    out["test3"] = {"r2_pre_from_post": r2, "rank": r}

    with open(os.path.join(args.out, "diagnosis.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {os.path.join(args.out, 'diagnosis.json')}")
    print("=" * 104)
    return 0


if __name__ == "__main__":
    sys.exit(main())
