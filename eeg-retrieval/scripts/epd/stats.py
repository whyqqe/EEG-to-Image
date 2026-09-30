"""Paired statistics over the 200 test concepts.

Why every arm comparison in this project needs these
---------------------------------------------------
The generation metrics are two-way identification accuracies on 200 concepts, and
the same 200 concepts are used for every arm. Concepts differ enormously in how
decodable they are, and that difficulty is a property of the STIMULUS, so it is
shared by every arm. Comparing two arms' aggregate accuracies as if they were
independent draws charges that shared term to the effect and makes the intervals
~14x wider than they need to be -- which is why, before the per-concept
decomposition existed, every structural arm looked "indistinguishable" from
`sem_only` at differences of 0.02-0.05 that later turned out to be real and
consistent.

`eval_official_seven_dir.py` writes `q_i` per concept (the fraction of the other
199 gallery entries that ground truth i beats under the official Pearson
similarity); `mean(q)` reproduces the reported two-way accuracy exactly, asserted
at the writer. `metrics.retrieval_per_concept` does the same for the retrieval
top-k. This module turns those vectors into CIs and p-values.

The bootstrap does not manufacture evidence that is not in the data -- it removes
a nuisance term from the interval. Every function here is deliberately kept free
of thresholds: `paired_bootstrap` reports the interval, `sign_test` reports a
p-value, and the caller decides what those mean for its own comparison.
"""
from __future__ import annotations

import math

import numpy as np

# A 95% bootstrap interval on a paired difference over 200 concepts is the right
# default: it is what the published two-way protocol is scored at.
ALPHA = 0.05
N_RESAMPLES = 10000


def paired_bootstrap(a: np.ndarray, b: np.ndarray, *, n_resamples: int = N_RESAMPLES,
                     alpha: float = ALPHA, seed: int = 0) -> dict:
    """Difference `a - b` per concept, with a percentile bootstrap interval.

    Paired by POSITION, so the two arrays must be in the same concept order (both
    writers state their order in the JSON). Returns the point estimate, the
    interval, and whether the interval excludes zero -- which is the only thing the
    interval is for.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"paired test needs matched shapes, got {a.shape} and {b.shape}")
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_resamples, d.size))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return {"delta": float(d.mean()), "lo": float(lo), "hi": float(hi),
            "excludes_zero": bool(lo > 0.0 or hi < 0.0), "n": int(d.size)}


def sign_test(d: np.ndarray) -> dict:
    """Two-sided sign test on the per-concept differences.

    A normal approximation to the binomial, not an exact test, and reported as
    such: with n=200 concepts the approximation error is far below the resolution
    of any claim made from it. Its purpose is to fail differently from the
    bootstrap -- it is blind to the SIZE of the shift and sensitive only to how
    often the direction flips -- so when the two disagree, the shift is driven by a
    few large concepts rather than by a consistent one.
    """
    d = np.asarray(d, dtype=np.float64)
    nz = d[d != 0.0]
    n = int(nz.size)
    if n == 0:
        return {"p": 1.0, "n_nonzero": 0, "frac_positive": float("nan")}
    k = int((nz > 0).sum())
    z = abs(k - n / 2.0) / math.sqrt(n * 0.25)
    return {"p": float(math.erfc(z / math.sqrt(2.0))), "n_nonzero": n,
            "frac_positive": float(k) / float(n)}


def verdict(a: np.ndarray, b: np.ndarray, *, min_effect: float = 0.02,
            **kw) -> dict:
    """Bootstrap + sign test together, with one conservative label.

    `min_effect` exists because a paired interval over 200 concepts can exclude
    zero on a shift of 0.005, which is real and too small to care about. The label
    therefore requires BOTH an interval excluding zero and a shift of at least
    `min_effect`; everything else is reported as `within noise`, which is not the
    same as "no difference" and is not claimed to be.
    """
    bs = paired_bootstrap(a, b, **kw)
    st = sign_test(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
    if bs["excludes_zero"] and abs(bs["delta"]) >= min_effect:
        label = "shifted"
    elif bs["excludes_zero"]:
        label = "shifted (below the 0.02 threshold)"
    else:
        label = "within noise"
    return {**bs, "sign_p": st["p"], "frac_positive": st["frac_positive"], "verdict": label}
