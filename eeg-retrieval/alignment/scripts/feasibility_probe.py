#!/usr/bin/env python
"""
Feasibility probe: can THINGS-EEG2 support a Bayesian shared-latent study of
"which image features are neurally visible"?

Measures the four numbers that jointly decide feasibility. All of them are
upper bounds or hard floors -- they constrain every possible downstream model,
so they must be established BEFORE any modelling.

  1. NOISE CEILING (split-half RDM reliability)
     The representational geometry of the EEG response, measured on two disjoint
     halves of the repetitions, and the rank correlation between them
     (Spearman-Brown corrected). This is the maximum RSA score any image-feature
     model can ever reach against this data. Everything else is bounded by it.

  2. EFFECTIVE DIMENSIONALITY (neural bottleneck)
     Participation ratio of the 63-channel covariance. Volume conduction makes
     the scalp field a smooth low-rank mixture of sources, so the number of
     recoverable latent directions is bounded by this, not by 63.

  3. SPURIOUS CCA FLOOR vs n
     The largest canonical correlation obtainable between two INDEPENDENT
     Gaussian views of dimension m and d with n samples. Full-dimensional
     CCA(EEG, CLIP) at n=1654 is dominated by this floor, which is the single
     most common way this kind of study produces false positives.

  4. RSA(EEG, CLIP) vs the ceiling
     Where the real alignment sits relative to (1) and (3).
"""
import glob
import os
import sys

import numpy as np
from scipy.stats import spearmanr

D = "/project/peilab/why/NeuroBridge/data/things_eeg"
SUB = sys.argv[1] if len(sys.argv) > 1 else "sub-01"
NPERM = 2000

np.set_printoptions(precision=4, suppress=True, linewidth=200)


def upper_tri(A):
    iu = np.triu_indices(A.shape[0], k=1)
    return A[iu]


def rdm(X, metric="correlation"):
    """Concept x concept representational dissimilarity matrix."""
    X = np.asarray(X, np.float64)
    if metric == "correlation":
        X = X - X.mean(0, keepdims=True)
        n = np.linalg.norm(X, axis=1, keepdims=True)
        C = (X @ X.T) / np.maximum(n @ n.T, 1e-30)
        R = 1.0 - np.clip(C, -1, 1)
    elif metric == "euclidean":
        sq = (X ** 2).sum(1)
        D2 = sq[:, None] + sq[None, :] - 2 * X @ X.T
        R = np.sqrt(np.maximum(D2, 0))
    else:
        raise ValueError(metric)
    np.fill_diagonal(R, 0.0)
    return R


def spearman_brown(r):
    return 2 * r / (1 + r) if r > -1 else np.nan


# ------------------------------------------------------------------ load EEG
tr = np.load(f"{D}/preprocessed_eeg/{SUB}/train.npy")
tr = tr.mean(2) if tr.ndim == 5 else tr          # average over sessions
ntr, nrep, nch, ntime = tr.shape
print("=" * 96)
print(f"{SUB}  train {tr.shape}  (concept x repetition x channel x time)")
print("=" * 96)

# baseline-correct using the pre-stimulus window (-0.2 .. 0 s = samples 0..49)
tr = tr - tr[:, :, :, :50].mean(-1, keepdims=True)

h1 = tr[:, : nrep // 2].mean(1)   # (ntr, 63, 250)
h2 = tr[:, nrep // 2 :].mean(1)
full = tr.mean(1)

# ------------------------------------------------- 1. split-half RDM reliability
print("\n=== 1. NOISE CEILING (split-half RDM reliability) ===")
print(f"{'window':>14} {'rho(half1,half2)':>17} {'Spearman-Brown':>15} {'=> ceiling R':>13}")
windows = [(50, 75, "0.00-0.10s"), (75, 100, "0.10-0.20s"), (100, 125, "0.20-0.30s"),
           (125, 150, "0.30-0.40s"), (150, 200, "0.40-0.60s"), (200, 250, "0.60-0.80s"),
           (50, 250, "0.00-0.80s"), (0, 50, "PRE (-0.2-0s)")]
ceilings = {}
for lo, hi, name in windows:
    R1 = rdm(h1[:, :, lo:hi].reshape(ntr, -1))
    R2 = rdm(h2[:, :, lo:hi].reshape(ntr, -1))
    r = spearmanr(upper_tri(R1), upper_tri(R2)).correlation
    sb = spearman_brown(r)
    ceilings[name] = sb
    print(f"{name:>14} {r:>17.4f} {sb:>15.4f} {sb:>13.4f}")

# ------------------------------------------------- 2. effective dimensionality
print("\n=== 2. EFFECTIVE DIMENSIONALITY of the 63-channel field (neural bottleneck) ===")
print(f"{'window':>14} {'participation ratio':>20} {'d_eff @90% var':>15} {'d_eff @99%':>12}")
for lo, hi, name in windows:
    X = full[:, :, lo:hi].transpose(1, 0, 2).reshape(nch, -1)   # ch x (concept*time)
    X = X - X.mean(1, keepdims=True)
    S = X @ X.T / X.shape[1]
    w = np.sort(np.maximum(np.linalg.eigvalsh(S), 0))[::-1]
    pr = (w.sum() ** 2) / max((w ** 2).sum(), 1e-30)
    cs = np.cumsum(w) / w.sum()
    print(f"{name:>14} {pr:>20.2f} {int(np.searchsorted(cs,0.90)+1):>15d} "
          f"{int(np.searchsorted(cs,0.99)+1):>12d}")

# ------------------------------------------------- 3. spurious CCA floor vs n
print("\n=== 3. SPURIOUS CCA FLOOR (two independent views, n samples) ===")
print("    Why this matters: with n=1654 concepts and 1024-dim CLIP features,")
print("    full-dimensional CCA reports large 'canonical correlations' even for")
print("    pure noise. sqrt(m/n) is the rule of thumb.")
rng = np.random.default_rng(0)
print(f"\n{'n':>7} {'m=63':>9} {'m=128':>9} {'m=256':>9} {'m=512':>9} {'m=1024':>9}   {'sqrt(m/n) m=1024':>17}")
for n in [200, 500, 1654, 5000, 20000]:
    row = []
    for m in [63, 128, 256, 512, 1024]:
        vals = []
        for _ in range(5):
            A = rng.standard_normal((n, m))
            B = rng.standard_normal((n, 1024))
            A = A - A.mean(0, keepdims=True)
            B = B - B.mean(0, keepdims=True)
            Saa = A.T @ A / n
            Sbb = B.T @ B / n
            Sab = A.T @ B / n
            Ma = Saa + 1e-3 * np.trace(Saa) / m * np.eye(m)
            Mb = Sbb + 1e-3 * np.trace(Sbb) / 1024 * np.eye(1024)
            wa, Va = np.linalg.eigh(Ma)
            wb, Vb = np.linalg.eigh(Mb)
            Ma_i = Va @ np.diag(1 / np.sqrt(np.maximum(wa, 1e-10))) @ Va.T
            Mb_i = Vb @ np.diag(1 / np.sqrt(np.maximum(wb, 1e-10))) @ Vb.T
            sv = np.linalg.svd(Ma_i @ Sab @ Mb_i, compute_uv=False)
            vals.append(sv[0])
        row.append(np.mean(vals))
    print(f"{n:>7} " + " ".join(f"{v:>9.3f}" for v in row)
          + f"   {np.sqrt(1024/n):>17.3f}")

# ------------------------------------------------- 4. RSA(EEG, CLIP)
print("\n=== 4. RSA(EEG, CLIP) against the ceiling ===")
feats = {}
for name in ["RN50", "ViT-H-14"]:
    f = np.load(f"{D}/image_feature/{name}/image_train.npy")
    feats[name] = f.mean(1) if f.ndim == 3 else f

print(f"{'window':>14} {'feat':>10} {'rho(EEG,CLIP)':>14} {'ceiling':>9} {'%of ceiling':>12} {'p (perm)':>10}")
rngp = np.random.default_rng(1)
for lo, hi, wname in [(50, 100, "0.00-0.20s"), (100, 150, "0.20-0.40s"),
                      (75, 125, "0.10-0.30s"), (150, 200, "0.40-0.60s")]:
    Re = rdm(full[:, :, lo:hi].reshape(ntr, -1))
    re = upper_tri(Re)
    for fname, F in feats.items():
        Rc = rdm(F)
        rc = upper_tri(Rc)
        rho = spearmanr(re, rc).correlation
        # permutation null: shuffle concept order of F
        null = []
        for _ in range(200):
            p = rngp.permutation(ntr)
            null.append(spearmanr(re, upper_tri(rdm(F[p]))).correlation)
        null = np.array(null)
        pval = float((null >= rho).mean())
        ceil = ceilings.get(wname, np.nan)
        print(f"{wname:>14} {fname:>10} {rho:>14.4f} {ceil:>9.4f} "
              f"{100*rho/max(ceil,1e-9):>11.1f}% {pval:>10.4f}")

print("""
READING THIS OUTPUT
-------------------
* (1) is the hard ceiling. An RSA of X% of ceiling means the image feature
  explains X% of the *reproducible* neural geometry.
* (3) is the false-positive floor. Any canonical correlation below the
  corresponding entry is not evidence of anything.
* If (4) is far below (1), the bottleneck is the *feature side* (CLIP geometry
  is the wrong description of what EEG carries). If (4) is close to (1), the
  bottleneck is *neural noise* and no modelling improvement can help.
""")
