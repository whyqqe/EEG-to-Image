"""MVNN -- multivariate noise normalisation (Guggenmos et al., NeuroImage 2018).

Why this file exists
--------------------
Every inter-subject paper on this benchmark runs MVNN, and our pipeline does not.
Two consequences, both bad:

  * EEG channels have wildly different noise levels. The occipital electrodes sit
    on a low-impedance, high-SNR region and the frontal ones sit on muscle and
    blink artefacts. Without whitening, the encoder's first layer spends its
    budget on whichever channel happens to shout loudest, and the inter-subject
    ranking is decided by the noisiest electrode rather than by the signal.
  * Our tokenizer's docstring already claims "the input is already MVNN-whitened"
    -- that claim was transcribed from EEGiT, which consumed the official
    ``Preprocessed_data_250Hz_whiten`` release. Our cache is the unwhitened OSF
    release, so the comment has been false since it was written. This module is
    what makes it true.

The method, exactly
-------------------
Guggenmos compares three ways to estimate the noise covariance P; the "epoch
method" is the one the field adopted and the one implemented here:

  1. take the residuals of each trial about its own condition's mean. Only the
     *within-condition* variability is noise -- the mean carries the evoked
     response, which is signal and must not enter a noise estimate.
  2. for each time point t, form the covariance of those residuals across trials,
     ``S_t = (1/n) sum_i r_{i,t} r_{i,t}^T``;
  3. average across time;
  4. whiten with the inverse symmetric square root, ``W = P^{-1/2}``, applied as
     ``x -> W x`` on the channel axis.

Two details are not optional:

  * **Shrinkage.** With 4 repetitions per condition the per-time-point covariance
    is badly conditioned, and inverting it amplifies exactly the noise directions
    it was meant to suppress. Guggenmos: "shrinkage improved the performance of
    all normalisation methods". We use Ledoit-Wolf towards the scaled identity,
    which needs no tuning and no validation set -- important, because our
    inter-subject protocol has no validation split to tune a shrinkage constant on.
  * **Fit on training data only.** The whitener transfers like a z-score statistic:
    estimating it from labelled training trials and applying it to unlabelled test
    trials is label-free, estimating it from the test trials' *labels* is not. See
    ``load_subject_std`` for which split feeds which fit.

Repetitions are retained on purpose
-----------------------------------
The cache that feeds the encoder has already averaged repetitions (4 -> 1 in
train, 80 -> 1 in test), which destroys the very residuals step 1 needs. So the
whitener is fitted from the raw, un-averaged file and then applied to the averaged
one. That order is legal because W is linear and therefore commutes with the
averaging: ``W mean_r(x_r) = mean_r(W x_r)``. The alternative -- whitening the raw
file and re-caching it -- would cost 4 GiB per subject per channel-set and buy
nothing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


__all__ = ["Whitener", "ledoit_wolf_intensity", "fit_from_blocks", "apply", "identity"]


def ledoit_wolf_intensity(x: np.ndarray, s: np.ndarray | None = None) -> tuple[float, float]:
    """Ledoit-Wolf shrinkage intensity of ``x`` towards ``mu * I``.

    ``x`` is ``(n, p)``. Returns ``(lam, mu)`` such that the shrunk estimate is
    ``(1 - lam) S + lam * mu * I``, with ``S = x^T x / n`` and ``mu = tr(S) / p``.

    This is Ledoit & Wolf (2004), the identity-target case, in closed form:

        d^2 = ||S - mu I||_F^2 / p                     how far S is from the target
        b^2 = (1/n^2) sum_i ||x_i x_i^T - S||_F^2 / p  how noisy S itself is
        lam = min(b^2, d^2) / d^2

    ``b^2 >= d^2`` is possible for small n and would give lam > 1, i.e. a negative
    weight on the data; the ``min`` is the guard, not a heuristic. The inner norm
    is expanded as ``(x_i^T x_i)^2 - 2 x_i^T S x_i + tr(S^2)`` so that no ``p x p``
    matrix is ever formed per sample.

    Computed here rather than called from sklearn because the fit loops over 250
    time points and sklearn re-validates and re-estimates ``S`` on every call.
    """
    x = np.asarray(x)
    n, p = x.shape
    if n < 2:
        raise ValueError(f"need at least 2 trials to estimate a covariance, got {n}")
    s = x.T @ x / n if s is None else s
    mu = float(np.trace(s) / p)

    dev = s - mu * np.eye(p, dtype=s.dtype)
    d2 = float((dev * dev).sum() / p)
    if d2 <= 0.0:
        # S is already exactly the target: there is nothing to shrink towards.
        return 0.0, mu

    sq = np.einsum("ij,ij->i", x, x)
    xsx = np.einsum("ij,ij->i", x @ s, x)
    tr_s2 = float(np.trace(s @ s))
    bbar2 = float((sq * sq).sum() - 2.0 * xsx.sum() + n * tr_s2) / (n * n) / p
    lam = min(bbar2, d2) / d2
    return float(np.clip(lam, 0.0, 1.0)), mu


@dataclass
class Whitener:
    """A fitted ``P^{-1/2}`` plus the diagnostics needed to distrust it.

    The diagnostics are not decoration. A whitener built from too few residual
    trials is nearly singular in the directions it amplifies, and the failure is
    invisible downstream: the encoder trains, the loss falls, and the ranking is
    worse. ``lam`` near 1 means the data covariance was uninformative and the
    whitener is essentially a scalar; ``cond`` (condition number of P) in the
    thousands means the square root is amplifying a direction we cannot estimate.
    """

    w: np.ndarray                    # (C, C) whitening matrix, apply as x -> W x
    sigma: np.ndarray                # (C, C) shrunk covariance it came from
    lam: float                       # mean Ledoit-Wolf intensity across time
    lam_min: float
    lam_max: float
    cond: float                      # condition number of sigma
    n_trials: int                    # residual trials used
    n_cond: int                      # conditions those trials came from
    n_rep: int
    n_time: int
    shrinkage: str
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "lam": self.lam, "lam_min": self.lam_min, "lam_max": self.lam_max,
            "cond": self.cond, "n_trials": self.n_trials, "n_cond": self.n_cond,
            "n_rep": self.n_rep, "n_time": self.n_time, "shrinkage": self.shrinkage,
            **self.extra,
        }

    def describe(self) -> str:
        return (f"lam {self.lam:.4f} [{self.lam_min:.4f},{self.lam_max:.4f}] "
                f"cond {self.cond:.1f} from {self.n_cond} cond x {self.n_rep} rep")


def _inverse_sqrt(sigma: np.ndarray, rel_floor: float) -> np.ndarray:
    """Symmetric ``P^{-1/2}``, with near-null directions floored rather than freed.

    ``eigh`` is the right factorisation for a symmetric PSD matrix: ``svd`` would
    sort by magnitude and mix signs, and ``inv`` would not preserve symmetry, which
    matters because the whitened covariance is only guaranteed to be the identity
    if W is symmetric.

    Eigenvalues below ``rel_floor * lam_max`` are raised to that floor. Without the
    floor a single round-off-level eigenvalue produces a factor of 1e6 and the
    "whitened" data is dominated by a direction that carries no noise estimate at
    all. With Ledoit-Wolf shrinkage and n >> C the floor is never reached; it exists
    so that a degenerate fit degrades into "shrink harder" instead of into NaNs.
    """
    lam, vec = np.linalg.eigh(sigma)
    floor = max(float(lam.max()) * rel_floor, 1e-12)
    lam = np.maximum(lam, floor)
    return (vec * (1.0 / np.sqrt(lam))) @ vec.T


def fit_from_blocks(
    blocks: np.ndarray,
    shrinkage: str = "lw",
    fixed: float = 0.1,
    rel_floor: float = 1e-10,
    max_cond: int = 0,
    verbose: bool = False,
) -> Whitener:
    """Fit a whitener from raw trials laid out as ``(n_cond, n_rep, C, T)``.

    The balanced-block layout is not a convenience, it is the shape the dataset
    actually has -- 4 repetitions per (concept, image) condition in train, 80 in
    test -- and taking it seriously removes the need to carry condition ids and the
    ``np.add.at`` scatter that would come with them. Repetitions within a condition
    are required to be meaningful; unequal counts would silently weight some
    conditions more heavily in the covariance.

    Shrinkage is applied in CORRELATION space, not covariance space
    -----------------------------------------------------------------
    The obvious implementation -- Ledoit-Wolf the covariance towards ``mu * I`` with
    ``mu = tr(S)/p`` -- is actively harmful on EEG, and measurably so. EEG channel
    variances span two orders of magnitude (measured on sub-08 residuals: 4.9 at the
    quietest electrode, 473 at the loudest, a 97x spread), so ``mu`` is set by the
    loud channels, and adding ``lam * mu`` to a quiet channel injects far more
    variance than that channel ever had. On a synthetic fixture with a 64x spread and
    a perfectly ordinary ``lam = 0.008``, it inflated the quietest channel's variance
    by 36%, and the whitener -- the one object whose entire job is to equalise
    channels -- left the quiet channel 5x under-whitened.

    The fix is the standard one, and it is what MNE's ``regularize(..., 'shrunk')``
    does:     normalise each channel to unit variance *first* (univariate noise
    normalisation), shrink the resulting correlation matrix towards ``I`` -- which is
    now exactly the right target, since every diagonal entry is 1 by construction --
    and unscale. The returned ``sigma`` is ``D R_s D``, whose diagonal is the sample
    variances *unchanged*: UNN is preserved exactly and only the correlations are
    shrunk. ``W sigma W^T = I`` still holds by construction, which is the property
    the tests pin down.

    Consequence for ``W``: it is a whitener, not a symmetric one. ``W R_s^{1/2} D``
    is orthogonal but not the identity, because ``D`` does not commute with
    ``R_s^{-1/2}``. The covariance of ``W x`` is ``W sigma W^T``, which is why that
    -- not ``W sigma W`` -- is what the tests assert. Everything downstream only
    ever left-multiplies, so the asymmetry costs nothing.

    ``shrinkage`` is ``"lw"`` (Ledoit-Wolf per time point, the default) or
    ``"fixed"`` (a constant ``fixed``, kept so the unshrunk ``lam=0`` end of the
    axis stays reachable for an ablation).

    ``max_cond`` subsamples conditions; it exists for smoke runs, and a whitener
    fitted from a subsample is stored under a different key by the caller.
    """
    blocks = np.asarray(blocks)
    if blocks.ndim != 4:
        raise ValueError(f"expected (n_cond, n_rep, C, T), got {blocks.shape}")
    n_cond, n_rep, C, T = blocks.shape
    if n_rep < 2:
        raise ValueError(
            f"n_rep={n_rep}: MVNN needs within-condition residuals, and a condition "
            f"seen once has none. This is a sign the caller passed an already-averaged "
            f"array instead of the raw file.")
    if max_cond and max_cond < n_cond:
        blocks = blocks[:max_cond]
        n_cond = max_cond
    if shrinkage not in ("lw", "fixed"):
        raise ValueError(f"shrinkage must be 'lw' or 'fixed', got {shrinkage!r}")

    # Residuals about each condition's own mean, in float32 (passing the block
    # through float64 would double a 4 GiB allocation for a 1e-7 difference).
    x = blocks.astype(np.float32, copy=True)
    x -= x.mean(axis=1, keepdims=True)          # (n_cond, n_rep, C, T)
    x = x.reshape(n_cond * n_rep, C, T)
    n = x.shape[0]

    eye = np.eye(C, dtype=np.float64)
    corr = np.zeros((C, C), dtype=np.float64)   # shrunk correlation, averaged over t
    var = np.zeros(C, dtype=np.float64)         # per-channel variance, averaged over t
    lams = np.empty(T, dtype=np.float64)
    for t in range(T):
        xt = x[:, :, t].astype(np.float64)      # (n, C)
        var_t = np.einsum("ij,ij->j", xt, xt) / n
        var += var_t
        sd = np.sqrt(np.maximum(var_t, 1e-30))
        zt = xt / sd                            # UNN: unit variance on every channel
        r = zt.T @ zt / n                       # correlation: diagonal is exactly 1
        if shrinkage == "lw":
            lam, _ = ledoit_wolf_intensity(zt, r)
        else:
            lam = float(fixed)
        corr += (1.0 - lam) * r + lam * eye
        lams[t] = lam
    corr /= T
    var /= T

    d = np.diag(np.sqrt(var))
    sigma = d @ corr @ d                        # unscale: diagonal is the sample variance
    w = (_inverse_sqrt(corr, rel_floor) @ np.diag(1.0 / np.sqrt(var))).astype(np.float64)
    eig = np.linalg.eigvalsh(sigma)
    cond = float(eig.max() / max(eig.min(), 1e-30))
    wh = Whitener(w=w.astype(np.float32), sigma=sigma, lam=float(lams.mean()),
                  lam_min=float(lams.min()), lam_max=float(lams.max()),
                  cond=cond, n_trials=int(n), n_cond=int(n_cond), n_rep=int(n_rep),
                  n_time=int(T), shrinkage=shrinkage)
    if verbose:
        print(f"[mvnn ] {wh.describe()}")
    return wh


def apply(x: np.ndarray, wh: Whitener | np.ndarray) -> np.ndarray:
    """Whiten ``x`` on its channel axis.

    ``x`` is ``(..., C, T)`` (an averaged trial, a batch of them, or the raw
    ``(n_cond, n_rep, C, T)`` file -- the channel axis is resolved from the end, so
    all three work without a reshape).

    ``einsum`` with an explicit index list rather than a ``@``: the channel axis is
    not the last one, and ``np.tensordot``/``moveaxis`` would allocate an
    intermediate of the full array. Note that ``W`` is a whitener and not in general
    a symmetric matrix (see ``fit_from_blocks``), so this is a genuine left-multiply
    -- ``"cd,...dt->...ct"`` -- and not an abbreviation for a transpose trick.
    """
    w = wh.w if isinstance(wh, Whitener) else np.asarray(wh)
    x = np.asarray(x)
    if x.shape[-2] != w.shape[0]:
        raise ValueError(
            f"channel mismatch: x has {x.shape[-2]}, whitener was fitted for "
            f"{w.shape[0]}. A 17-channel whitener cannot whiten a 63-channel cache.")
    out = np.einsum("cd,...dt->...ct", w.astype(np.float32), x, optimize=True)
    return out.astype(np.float32, copy=False)


def identity(n_channels: int) -> Whitener:
    """The no-op whitener, so callers can hold one code path instead of two."""
    return Whitener(w=np.eye(n_channels, dtype=np.float32),
                    sigma=np.eye(n_channels, dtype=np.float64), lam=0.0,
                    lam_min=0.0, lam_max=0.0, cond=1.0, n_trials=0, n_cond=0,
                    n_rep=0, n_time=0, shrinkage="identity")
