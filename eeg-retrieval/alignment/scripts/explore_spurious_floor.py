#!/usr/bin/env python
"""Empirical investigation of the spurious canonical-correlation floor.

The analytic expression in docs/01_theory_and_plan.md (Prop. 4) was stated as
rho_max ~ sqrt(min(p,q)/n).  That is WRONG in general and this script measures
what the truth is, so the documented floor is not mis-stated.

Frame: X (n,p), Y (n,q) independent standard normal, near-zero ridge.
Squared canonical correlations = eigenvalues of Sxx^-1 Sxy Syy^-1 Syx.
Under independence Sxy = G/sqrt(n) with G (p,q) iid N(0,1), so
    rho_max^2 ~ lambda_max(Wishart_p(q, I)) / n.
For p,q LARGE the Wishart top eigenvalue -> (sqrt(q)+sqrt(p))^2, giving
    rho_max ~ (sqrt(p)+sqrt(q))/sqrt(n).
For p,q small that asymptotic under-estimates, so the formula is regime-dependent
and the only safe procedure is to SIMULATE at the actual (n,p,q).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_rsca import cca_ridge, permutation_null  # noqa: E402

rng = np.random.default_rng(0)
np.set_printoptions(precision=4, suppress=True, linewidth=200)

print("=" * 104)
print("A. scaling of the spurious floor vs the two candidate formulas")
print("=" * 104)
print(f"{'n':>6} {'p':>5} {'q':>6} {'rho_max':>9} {'(sp+sq)/sn':>11} {'sq(m/n)':>9} "
      f"{'wishart/sn':>11} {'ratio real/wish':>16}")

REPS = int(os.environ.get("REPS", "8"))
for n in [200, 500, 1654, 5000]:
    for (p, q) in [(20, 1024), (64, 1024), (256, 1024), (400, 1024), (1024, 1024), (1, 1)]:
        if p > n or q > n:
            continue
        vals, wish = [], []
        for _ in range(REPS):
            X = rng.standard_normal((n, p))
            Y = rng.standard_normal((n, q))
            r = cca_ridge(X, Y, K=1, ridge=1e-9)
            vals.append(r.rho[0])
            # Wishart prediction for lambda_max(G G^T)
            G = rng.standard_normal((p, q))
            wish.append((np.linalg.eigvalsh(G @ G.T).max()) / n)
        rm = float(np.mean(vals)) ** 2
        sp, sq = np.sqrt(p), np.sqrt(q)
        f_wish = np.sqrt(np.mean(wish))
        f_sum = (sp + sq) / np.sqrt(n)
        f_min = np.sqrt(min(p, q) / n)
        print(f"{n:>6} {p:>5} {q:>6} {np.sqrt(rm):>9.3f} {f_sum:>11.3f} {f_min:>9.3f} "
              f"{f_wish:>11.3f} {np.sqrt(rm)/max(f_wish,1e-9):>16.2f}")

print()
print("=" * 104)
print("B. does the permutation null track the in-sample independent-CCA max?")
print("   (this is the operationally relevant check -- the threshold must be simulated)")
print("=" * 104)
print(f"{'n':>6} {'p':>5} {'q':>6} {'perm threshold':>15} {'perm mean':>11} "
      f"{'insample null':>14} {'agree?':>8}")
for n in [300, 1000]:
    for (p, q) in [(32, 256), (128, 512)]:
        X = rng.standard_normal((n, p))
        Y = rng.standard_normal((n, q))
        nul = permutation_null(X, Y, K=8, ridge=1e-3, n_perm=200, alpha=0.05,
                               rng=np.random.default_rng(1))
        # in-sample null: independent views, full-data CCA
        ins = [cca_ridge(X, rng.standard_normal((n, q)), K=8, ridge=1e-3).rho[0]
               for _ in range(30)]
        agree = abs(nul["threshold"] - np.mean(ins)) < 0.25
        print(f"{n:>6} {p:>5} {q:>6} {nul['threshold']:>15.3f} {nul['mean']:>11.3f} "
              f"{np.mean(ins):>14.3f} {str(agree):>8}")

print()
print("=" * 104)
print("C. the regime that actually matters for THINGS-EEG2")
print("=" * 104)
print("After whitening the EEG side is D=400 (capped at n-1) and q=1024, n=1654.")
for (p, q, n) in [(400, 1024, 1654), (1024, 1024, 1654), (1653, 1024, 1654)]:
    X = rng.standard_normal((n, min(p, n - 1)))
    Y = rng.standard_normal((n, min(q, n - 1)))
    r = cca_ridge(X, Y, K=1, ridge=1e-3)
    print(f"  p={p:5d} q={q:5d} n={n}:  in-sample spurious rho_1 = {r.rho[0]:.4f}"
          f"   -> in-sample CCA is USELESS here (saturated)")
    print(f"      that is why the held-out screening (A1/T4) is not optional.")
