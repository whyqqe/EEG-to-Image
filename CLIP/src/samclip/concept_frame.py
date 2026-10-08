"""Concept coordinate + per-subject loading frame (the SCC representation).

THE MODEL THIS FILE IMPLEMENTS
-----------------------------
    z_{s,i,r} = mu_s + A_s c_i + eps_{s,i,r},     eps ~ N(0, Sigma_n)

`c_i` is a CONCEPT COORDINATE, shared by every subject and every repetition -- this is the
cross-subject concept invariant, and it is the whole point of the project. `A_s` is a
PER-SUBJECT LOW-RANK LOADING that says how subject `s`'s measured space is oriented relative
to that shared concept space: the explicit form of "a subject is a modality". `mu_s` is the
subject offset the existing SMN already removes; `Sigma_n` is the trial-noise covariance.

WHY THIS IS DIFFERENT FROM WHAT WE HAD, IN ONE EQUATION
------------------------------------------------------
The concept cloud's covariance is

    Cov(query cloud) = A Sigma_c A^T + Sigma_n / R

SCORE averages its repetitions before it ever looks at a cloud, so `Sigma_n / R` is baked
into its estimate and CANNOT be separated out. We keep the repetitions, so `Sigma_n` is
directly estimable from the within-concept scatter, and the concept term can therefore be
recovered by subtraction. That is the cross-trial half of the subject-as-modality framing,
and it is not a hyper-parameter -- it is an object only the un-averaged repetitions can
produce.

WHAT SUBTRACTING `Sigma_n / R` ACTUALLY DOES (this was not obvious before it was measured)
-----------------------------------------------------------------------------------------
`Sigma_n / R` is small in absolute terms -- R = 80, so it is ~1/80 of the trial noise. It is
NOT a global denoiser. What it does is act as a DIRECTION SELECTOR: in a direction where the
concept signal is absent, the query variance is exactly `lambda_n,j / R`, so subtracting
leaves ~0 there and the direction is KILLED rather than whitened-and-amplified. In a direction
where the concept signal lives, `lambda_c >> lambda_n / R`, so subtracting barely moves it.
Whitening a killed direction is impossible; whitening a full-covariance direction that happens
to be noise-only divides by a near-zero eigenvalue and BLOWS IT UP. That is why the measured
condition number of the 200-concept cloud is 2.1e5-6.7e5 while the 16000-repetition cloud is
only 1.1e4-2.0e4.

MEASURED RANK (seed 2025, all 10 folds)
---------------------------------------
Participation ratio of the concept-means cloud: **4.9-7.1** (not the 16 the project's older
`spec_r0` assumed); pooled within-concept noise: 14.3-16.0. So the signal subspace is SMALLER
than assumed and the noise subspace is larger -- which is exactly the regime where a
full-covariance whitening amplifies noise and a noise-corrected low-rank frame does not.

WHAT IS DELIBERATELY NOT HERE
-----------------------------
The CROSS-SUBJECT PRIOR over `A_s` (shrink this fold's frame toward the other subjects') is
NOT in this module. It needs all folds' frames expressed in one common space, and each fold's
model is trained separately, so its 64-d output space is its own. Making the frames comparable
is a TRAINING-TIME design (share the concept-coordinate head across folds), not something a
frozen-feature probe can do. The population-level part that *is* available -- a fixed rank
`r` estimated once across folds instead of per fold -- is exposed as `rank` and swept by
`scripts/probe_g_a.py`, and is the honest stand-in until the shared head exists.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def within_subject_noise(z_reps: np.ndarray, shrink: float = 0.1) -> np.ndarray:
    """Pooled within-concept covariance of the repetitions -- `Sigma_n`.

    `z_reps` is `(C, R, d)`. The scatter is `sum_i sum_r (z_ir - zbar_i)^2` over `C*(R-1)`
    degrees of freedom. `R - 1`, not `R`: the per-concept mean is itself estimated, and using
    `R` would bias the estimate low by a factor `(R-1)/R`, which at `R = 80` is only 1.25% but
    is a bias with no compensating advantage.

    THIS FUNCTION IS THE ENTIRE CROSS-TRIAL NOVELTY: it is not computable from averaged
    queries, because averaging is precisely the operation that deletes the within-concept
    scatter it measures.
    """
    C, R, d = z_reps.shape
    xc = z_reps - z_reps.mean(axis=1, keepdims=True)
    flat = xc.reshape(C * R, d)
    cov = (flat.T @ flat) / max(1, C * (R - 1))
    if shrink > 0:
        cov = (1.0 - shrink) * cov + shrink * (np.trace(cov) / d) * np.eye(d)
    return cov


def noise_corrected_cov(z_reps: np.ndarray, shrink: float = 0.1
                        ) -> tuple[np.ndarray, dict]:
    """`Cov(concept means) - Sigma_n / R` -- the concept covariance with the noise removed.

    The subtraction can make small eigenvalues NEGATIVE (the sample estimate of a genuinely
    zero direction fluctuates around zero and the noise estimate has its own error), and that
    is a feature: a negative eigenvalue is direct evidence that the direction carries no
    concept signal, and it is the criterion `rank_from_positivity` uses.
    """
    C, R, d = z_reps.shape
    xi = z_reps.mean(axis=1)
    xc = xi - xi.mean(axis=0, keepdims=True)
    cov_means = (xc.T @ xc) / max(1, C - 1)
    cov_n = within_subject_noise(z_reps, shrink=shrink)
    corr = cov_means - cov_n / R
    vals = np.linalg.eigvalsh(corr)
    return corr, {"cov_means_trace": float(np.trace(cov_means)),
                  "noise_share": float(np.trace(cov_n) / R / max(np.trace(cov_means), 1e-30)),
                  "n_positive": int((vals > 0).sum()),
                  "lam_max": float(vals.max()), "lam_min": float(vals.min())}


@dataclass
class Frame:
    """A per-subject loading frame: `c = (x - mu) @ W`, with `W` of shape `(d, d)`.

    `W` is expressed in the ORIGINAL feature basis, not the eigenbasis, and that is not a
    detail. The baseline operator pairs a whitened query against the RAW gallery, so the query
    must come back to the gallery's coordinates; building `W = V diag(s)` instead returns
    eigenvector coordinates and silently destroys the comparison. Measured on fold 8/seed 2025:
    the eigenbasis form scores 0.50 (chance) where the original-basis form scores 34.50 -- so
    a `W` that looks like a legal whitening can be a broken one, and the shape check does not
    catch it because `r` is small rather than mismatched.

    Kept directions have `s_j = 1/sqrt(lambda_j)` (or 1 when `whiten=False`); dropped
    directions have `s_j = 0`, i.e. the direction is KILLED rather than rescaled, which is the
    whole point of the noise correction.
    """
    mu: np.ndarray
    W: np.ndarray
    lam: np.ndarray
    keep: np.ndarray
    diag: dict = field(default_factory=dict)

    @property
    def rank(self) -> int:
        return int(self.keep.sum())

    def embed(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) - self.mu) @ self.W


def subject_frame(
    z_reps: np.ndarray,
    rank: int | None = None,
    shrink: float = 0.1,
    whiten: bool = True,
    max_cond: float = 1e3,
    min_eig: float = 0.0,
    fill: str = "kill",
) -> Frame:
    """Build the frame from a fold's repetition cloud, label-free.

    `rank=None` keeps every eigen-direction whose noise-corrected eigenvalue exceeds
    `min_eig` -- the data decides the rank. A fixed integer `rank` is a POPULATION prior: it is
    the one cross-subject statement available on frozen features, and it removes a per-fold
    degree of freedom, which is the mechanism by which it should also reduce fold-to-fold
    spread.

    `whiten=False` keeps every surviving direction's own spread instead of equalising it,
    which isolates how much of the gain is direction SELECTION (dropping noise-only
    directions) versus the whitening SCALE. That distinction turned out to matter: recovery's
    per-dimension moment matching rescales every surviving axis to the gallery's spread, so a
    query-side SCALE choice is largely erased downstream, and only the SURVIVING SUBSPACE
    can carry information through it.
    """
    corr, diag = noise_corrected_cov(z_reps, shrink=shrink)
    mu = z_reps.reshape(-1, z_reps.shape[-1]).mean(axis=0, keepdims=True)
    vals, vecs = np.linalg.eigh(corr)
    order = np.argsort(-vals)
    vals, vecs = vals[order], vecs[:, order]

    keep = vals > min_eig
    if rank is not None:
        keep = np.zeros_like(vals, dtype=bool)
        keep[:int(rank)] = True

    d = vals.shape[0]
    if whiten:
        floor = max(float(vals[keep].max()) / max_cond, 1e-12) if keep.any() else 1e-12
        scale = np.where(keep, 1.0 / np.sqrt(np.maximum(vals, floor)), 0.0)
    else:
        scale = np.where(keep, 1.0, 0.0)
    # `fill` decides what a DROPPED direction becomes, and the two options are not cosmetic.
    # "kill" zeroes it. "unit" passes it through unscaled, which keeps all `d` dimensions
    # active and therefore changes how the surviving columns MIX -- and mixing is the only
    # thing that can survive `moment_match`, which absorbs any per-output-dimension
    # rescaling. The distinction is what separates "the noise directions genuinely carry no
    # signal" (kill wins or ties) from "they carry signal but must not be amplified"
    # (unit wins).
    if fill == "unit":
        scale = np.where(keep, scale, 1.0)
    elif fill != "kill":
        raise ValueError(f"unknown fill mode: {fill!r} (expected 'kill' or 'unit')")
    w = vecs @ np.diag(scale) @ vecs.T
    return Frame(mu=mu, W=w, lam=vals, keep=keep,
                 diag={**diag, "rank_used": int(keep.sum()), "whiten": whiten,
                       "max_cond": max_cond, "fill": fill,
                       "kept_trace_share": float(vals[keep].sum() / max(np.trace(corr), 1e-30))})
