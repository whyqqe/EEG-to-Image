#!/usr/bin/env python
"""
Unit check for lib_rsca.cca_ridge.

The CCA implementation was rewritten after a derivation error, so it must be
verified against a case with a KNOWN answer before anything depends on it.

THREE CHECKS
  T1 reconstruction: X and Y built from a shared k-dim latent with known noise
     variances.  The population canonical correlations of the scalar-rank-1
     channel are rho = C / sqrt((C+N/d)(C+M/d)); check the fitted spectrum
     matches at large n.
  T2 orthogonality/normalisation: the returned variates have unit variance and
     the reported rho equals corr(Xa, Yb) computed independently -- this catches
     the operator-precedence class of bug.
  T3 null spectrum: two independent views.  Largest canonical correlation must
     match sqrt(min(p,q)/n) in order of magnitude.  If it is wildly off, the
     spurious floor used to justify the whole project is mis-stated.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_rsca import cca_ridge, held_out_cca_corr, center, permutation_null  # noqa: E402

FAILED = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'ok ' if cond else 'FAIL'}] {name}  {detail}")
    if not cond:
        FAILED.append(name)


def t1() -> None:
    print("T1 population canonical correlation reconstruction")
    rng = np.random.default_rng(0)
    n, p, q = 20000, 60, 80
    C_s, N_s, M_s, d = 0.5, 2.0, 3.0, 10
    b = rng.standard_normal(d); b /= np.linalg.norm(b)
    z = rng.standard_normal((n, 1))
    X = z * np.sqrt(C_s) * b[None, :] + rng.standard_normal((n, d)) * np.sqrt(N_s / d)
    Y = z * np.sqrt(C_s) * b[None, :] + rng.standard_normal((n, d)) * np.sqrt(M_s / d)
    # pad so p, q differ from d
    X = np.hstack([X, rng.standard_normal((n, p - d)) * np.sqrt(N_s / d)])
    Y = np.hstack([Y, rng.standard_normal((n, q - d)) * np.sqrt(M_s / d)])

    rho_theory = C_s / np.sqrt((C_s + N_s / d) * (C_s + M_s / d))
    r = cca_ridge(X, Y, K=5, ridge=1e-8)
    got = r.rho[0]
    print(f"     theory rho_1 = {rho_theory:.4f}   fitted rho_1 = {got:.4f}")
    check("T1 rho_1 within 5%", abs(got - rho_theory) / rho_theory < 0.05,
          f"rel err = {abs(got-rho_theory)/rho_theory:.4f}")
    check("T1 spectrum non-increasing", bool(np.all(np.diff(r.rho) <= 1e-9)))


def t2() -> None:
    print("T2 variate normalisation and rho consistency")
    rng = np.random.default_rng(1)
    n, p, q = 400, 30, 50
    z = rng.standard_normal((n, 3))
    X = z @ rng.standard_normal((3, p)) + rng.standard_normal((n, p)) * 0.5
    Y = z @ rng.standard_normal((3, q)) + rng.standard_normal((n, q)) * 0.5
    r = cca_ridge(X, Y, K=8, ridge=1e-6)

    vx = r.Xva.std(0)
    vy = r.Yvb.std(0)
    check("T2 variates unit variance (view1)", bool(np.allclose(vx, 1, atol=1e-6)),
          f"range [{vx.min():.6f}, {vx.max():.6f}]")
    check("T2 variates unit variance (view2)", bool(np.allclose(vy, 1, atol=1e-6)))

    # independent recomputation of rho from the primal directions
    Xc, Yc = center(X), center(Y)
    manual = np.array([
        np.corrcoef(Xc @ r.A[:, j], Yc @ r.B[:, j])[0, 1] for j in range(r.rho.size)
    ])
    # tolerance 1e-4: the two routes differ only by float64 accumulation
    check("T2 rho == corr(Xa, Yb)", bool(np.allclose(np.abs(manual), r.rho, atol=1e-4)),
          f"max diff = {np.abs(np.abs(manual)-r.rho).max():.2e}")

    # held_out_cca_corr on the same data must reproduce the same numbers
    ho = held_out_cca_corr(r, X, Y)
    check("T2 held_out_cca_corr consistent",
          bool(np.allclose(np.abs(ho), r.rho, atol=1e-4)))


def t3() -> None:
    """Verify the two floors measured by explore_spurious_floor.py:
       (i)  IN-SAMPLE spurious CCA  ~ (sqrt(p)+sqrt(q))/sqrt(n)   [Wishart top eig]
       (ii) HELD-OUT permutation threshold  ~ c/sqrt(n), with c ~ 2-4
    The second is the one that matters operationally, and the whole methodology
    rests on (ii) << (i).  An earlier version of the docs stated
    sqrt(min(p,q)/n) for (i), which is wrong by up to 4.5x at the relevant sizes.
    """
    print("T3 spurious floors: in-sample Wishart law vs held-out screening")
    rng = np.random.default_rng(2)

    print("  (i) in-sample floor vs (sqrt(p)+sqrt(q))/sqrt(n)")
    print(f"      {'n':>6} {'p':>5} {'q':>6} {'fitted':>8} {'wishart':>9} {'ratio':>7}")
    for n, p, q in [(500, 64, 512), (1000, 128, 512), (1654, 256, 1024)]:
        if p > n or q > n:
            continue
        v = []
        for _ in range(3):
            X = rng.standard_normal((n, p))
            Y = rng.standard_normal((n, q))
            v.append(cca_ridge(X, Y, K=1, ridge=1e-9).rho[0])
        fitted = float(np.mean(v))
        pred = (np.sqrt(p) + np.sqrt(q)) / np.sqrt(n)
        ratio = fitted / pred
        print(f"      {n:>6} {p:>5} {q:>6} {fitted:>8.3f} {pred:>9.3f} {ratio:>7.2f}")
        # asymptotic Wishart law is a large-(p,q) statement: allow 0.5-1.3
        check(f"T3i n={n} p={p} q={q} ratio in [0.5,1.3]", 0.5 < ratio < 1.3)

    print("  (ii) held-out threshold must be FAR below the in-sample floor")
    print(f"      {'n':>6} {'p':>5} {'q':>6} {'in-sample':>10} {'held-out':>9} "
          f"{'suppress':>9} {'thr*sqrt(n)':>12}")
    for n, p, q in [(300, 32, 256), (1000, 64, 512)]:
        X = rng.standard_normal((n, p))
        Y = rng.standard_normal((n, q))
        nul = permutation_null(X, Y, K=8, ridge=1e-3, n_perm=150, alpha=0.05,
                               rng=np.random.default_rng(1))
        ins = float(np.mean([cca_ridge(X, rng.standard_normal((n, q)), K=8,
                                       ridge=1e-3).rho[0] for _ in range(10)]))
        thr = nul["threshold"]
        print(f"      {n:>6} {p:>5} {q:>6} {ins:>10.3f} {thr:>9.3f} "
              f"{ins/max(thr,1e-9):>9.1f} {thr*np.sqrt(n):>12.2f}")
        check(f"T3ii n={n} screening suppresses >2x", ins / max(thr, 1e-9) > 2.0,
              f"suppression = {ins/max(thr,1e-9):.1f}x")


def t4() -> None:
    print("T4 held-out screening actually discriminates (the core mechanism)")
    rng = np.random.default_rng(3)
    n = 400
    # a) true shared signal: held-out rho must NOT collapse
    z = rng.standard_normal((n, 5))
    X = z @ rng.standard_normal((5, 40)) + rng.standard_normal((n, 40)) * 0.3
    Y = z @ rng.standard_normal((5, 60)) + rng.standard_normal((n, 60)) * 0.3
    ia, ib = np.arange(n // 2), np.arange(n // 2, n)
    r_ab = cca_ridge(X[ia], Y[ia], K=5, ridge=1e-6)
    ho_signal = held_out_cca_corr(r_ab, X[ib], Y[ib])

    # b) independent: held-out rho must collapse toward ~0
    X2 = rng.standard_normal((n, 40))
    Y2 = rng.standard_normal((n, 60))
    r2 = cca_ridge(X2[ia], Y2[ia], K=5, ridge=1e-6)
    ho_noise = held_out_cca_corr(r2, X2[ib], Y2[ib])

    print(f"     held-out rho_1:  signal = {ho_signal[0]:.4f}   noise = {ho_noise[0]:.4f}")
    print(f"     in-sample rho_1: signal = {r_ab.rho[0]:.4f}   noise = {r2.rho[0]:.4f}")
    check("T4 signal survives held-out (>0.5)", ho_signal[0] > 0.5)
    check("T4 noise collapses on held-out (<0.3)", ho_noise[0] < 0.3,
          f"got {ho_noise[0]:.4f}")
    check("T4 screening separates signal from noise",
          ho_signal[0] - ho_noise[0] > 0.3,
          f"gap = {ho_signal[0]-ho_noise[0]:.4f}")


if __name__ == "__main__":
    print("=" * 90)
    print("lib_rsca self-check")
    print("=" * 90)
    t1(); t2(); t3(); t4()
    print("=" * 90)
    if FAILED:
        print(f"SELF-CHECK FAILED: {len(FAILED)} check(s): {FAILED}")
        sys.exit(1)
    print("SELF-CHECK PASSED -- CCA implementation verified")
    print("=" * 90)
