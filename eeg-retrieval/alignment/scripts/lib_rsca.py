"""
lib_rsca.py -- Reproducibility-Screened Canonical Alignment

Core library for the neural-visibility study. Implements the estimator described
in docs/01_theory_and_plan.md section 6.

WHY THIS EXISTS (the two failure modes it must defeat)
------------------------------------------------------
1. SPURIOUS CANONICAL CORRELATION FLOOR.  With n concepts, EEG effective rank
   p_E and CLIP dim q, independent views already give
       rho_max ~ (sqrt(p_E)+sqrt(q))/sqrt(n) / ((1+sqrt(p_E/n))(1+sqrt(q/n)))
   Plugging in THINGS-EEG2 (n=1654, q=1024, p_E<=1653) gives rho ~ 0.5-0.8.
   A naive CCA on raw features therefore reports large "alignment" that is pure
   noise. We never trust a single-fit rho; everything is screened.

2. PRIVATE-VARIANCE DOMINANCE.  Martens et al. (PMLR 2024, v240) proved that
   shared/private disentanglement fails when modality-specific variation
   dominates the shared signal -- exactly the EEG regime (subject identity,
   artefacts, impedance).  We therefore whiten by the *directly estimable*
   trial-residual covariance rather than assuming iid noise.

THE TWO LEVERS
--------------
  lever 1: repeat structure  -> trial-residual covariance is estimable with NO
           distributional assumption, so private variance can be whitened away.
  lever 2: split reproducibility -> spuriously fitted directions do NOT survive a
           held-out split; real ones do.  This structure is what the spurious
           floor does not have, which is why screening works.

CONVENTIONS
-----------
  n : number of concepts (stimuli)
  T : number of repetitions per concept
  p : EEG feature dimension (channels x timepoints, or any linear feature)
  q : image-embedding dimension
  K : number of canonical components fitted / screened
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse.linalg import svds

# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------


def center(X: np.ndarray) -> np.ndarray:
    """Column-center."""
    return X - X.mean(0, keepdims=True)


def _inv_sqrt_psd(S: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    """Symmetric inverse square root of a PSD matrix via eigendecomposition."""
    S = 0.5 * (S + S.T)
    if ridge > 0:
        S = S + ridge * np.eye(S.shape[0])
    w, V = np.linalg.eigh(S)
    w = np.maximum(w, 1e-12)
    return (V * (1.0 / np.sqrt(w))) @ V.T


def principal_angles(V1: np.ndarray, V2: np.ndarray) -> np.ndarray:
    """Principal angles (radians) between column spaces of V1 and V2.

    cos(theta_i) = i-th singular value of Q1^T Q2, with Q the orthonormal bases.
    Small angles => the two subspaces coincide.  Used for cross-subject
    reproducibility (A7) and split stability.
    """
    Q1, _ = np.linalg.qr(V1)
    Q2, _ = np.linalg.qr(V2)
    s = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    return np.arccos(np.clip(s, -1.0, 1.0))


def subspace_overlap(V1: np.ndarray, V2: np.ndarray) -> float:
    """Mean cos^2 of principal angles, in [0,1]. 1 = identical subspace."""
    ang = principal_angles(V1, V2)
    return float(np.mean(np.cos(ang) ** 2))


# ---------------------------------------------------------------------------
# Step 1: EEG reduction + noise whitening
# ---------------------------------------------------------------------------


@dataclass
class EEGWhitener:
    """Reduces EEG to D components and whitens by the trial-residual covariance.

    Fit on a *nuisance* basis (unsupervised, uses only repeat structure), so it
    is safe to fit once and share across splits.  See docs section 6.2 step 1.

    Attributes
    ----------
    W  : (p, D) PCA basis of the concept-mean EEG
    L  : (D, D) whitening factor,  ẽ = Ẽ @ L
    M  : (p, D) composite map,      ẽ = (ē - mu) @ M
    """

    D: int = 400
    ridge: float = 1e-6
    mu: np.ndarray = field(default=None, repr=False)
    W: np.ndarray = field(default=None, repr=False)
    L: np.ndarray = field(default=None, repr=False)
    M: np.ndarray = field(default=None, repr=False)
    evr: np.ndarray = field(default=None, repr=False)

    def fit(self, E: np.ndarray) -> "EEGWhitener":
        """E: (n, T, p) -- trials for each concept."""
        n, T, p = E.shape
        Ebar = E.mean(1)                                   # (n, p)
        self.mu = Ebar.mean(0, keepdims=True)              # (1, p)
        Ec = Ebar - self.mu

        # --- PCA reduction of the concept-mean (total covariance) -----------
        # D is chosen well above the EEG effective rank (measured in A3, ~15-25)
        # so this reduction discards noise directions, not signal.
        d = min(self.D, n - 1, p)
        # economy SVD: Ec = U S V^T
        U, S, Vt = np.linalg.svd(Ec, full_matrices=False)
        self.W = Vt[:d].T                                  # (p, d)
        self.evr = (S ** 2)[:d] / max((S ** 2).sum(), 1e-30)

        # --- trial-residual covariance in the reduced space ------------------
        # r_i^(t) = e_i^(t) - ē_i    (captures trial noise + session effects)
        Rc = (E - Ebar[:, None, :]).reshape(-1, p)         # (n*T, p)
        Rc = Rc - Rc.mean(0, keepdims=True)
        Rp = Rc @ self.W                                   # (n*T, d)
        Sres = (Rp.T @ Rp) / max(Rp.shape[0] - 1, 1)       # (d, d)

        # --- whiten ---------------------------------------------------------
        L = _inv_sqrt_psd(Sres, ridge=self.ridge * np.trace(Sres) / d)
        self.L = L
        self.M = self.W @ L                                # (p, d)
        return self

    def transform(self, Ebar: np.ndarray) -> np.ndarray:
        """Ebar: (n, p) -> (n, d) whitened."""
        return (Ebar - self.mu) @ self.M

    def transform_trials(self, E: np.ndarray) -> np.ndarray:
        """E: (n, T, p) -> (n, T, d) whitened (for per-trial splits)."""
        n, T, p = E.shape
        return (E - self.mu[:, None, :]).reshape(n * T, p) @ self.M

    def transform_trials_flat(self, E: np.ndarray) -> np.ndarray:
        n, T, p = E.shape
        return self.transform_trials(E).reshape(n, T, -1)


# ---------------------------------------------------------------------------
# Step 2: ridge-regularised CCA (dual / kernel form, n << p)
# ---------------------------------------------------------------------------


@dataclass
class CCAResult:
    rho: np.ndarray            # (K,) canonical correlations
    A: np.ndarray              # (p, K) EEG-side directions (primal)
    B: np.ndarray              # (q, K) image-side directions (primal)
    Xva: np.ndarray            # (n, K) canonical variates, view 1
    Yvb: np.ndarray            # (n, K) canonical variates, view 2
    alpha: np.ndarray = field(default=None, repr=False)  # dual weights
    beta: np.ndarray = field(default=None, repr=False)


def cca_ridge(
    X: np.ndarray, Y: np.ndarray, K: int = 32, ridge: float = 1e-3
) -> CCAResult:
    """Ridge-regularised CCA of X (n, p) and Y (n, q).

    FORMULATION (derived, not copied -- the naive dual form is WRONG here)

    There is no natural "cross-Gram" X Y^T between two feature spaces of
    different dimension (it would require p == q).  The correct route is via the
    SVDs of the two centred views.  Write

        X_c = Ux Sx Vx^T,   Y_c = Uy Sy Vy^T        (thin SVDs)

    The canonical correlations are the singular values of

        T_norm = (X^T X + lx I)^{-1/2} X^T Y (Y^T Y + ly I)^{-1/2}
               = Vx Dx (Ux^T Uy) Dy Vy^T,
        Dx = diag( sx_i / sqrt(sx_i^2 + lx) ),  Dy likewise,

    so the spectrum is that of the small matrix  T = Dx (Ux^T Uy) Dy  of size
    rx x ry.  Primal directions are recovered as

        a_j = Vx (Sx^2 + lx I)^{-1/2} u_j,   b_j = Vy (Sy^2 + ly I)^{-1/2} v_j

    This is exact, works for any (p, q, n), never forms a p x p or q x q matrix,
    and reduces the permutation null (called thousands of times) to a tiny SVD.
    """
    Xc, Yc = center(X), center(Y)

    Ux, sx, Vtx = np.linalg.svd(Xc, full_matrices=False)   # (n,rx),(rx,),(rx,p)
    Uy, sy, Vty = np.linalg.svd(Yc, full_matrices=False)

    # ridge scaled by the mean squared singular value (dimension-free)
    lx = ridge * float((sx ** 2).mean()) if sx.size else ridge
    ly = ridge * float((sy ** 2).mean()) if sy.size else ridge

    dx = sx / np.sqrt(sx ** 2 + lx)
    dy = sy / np.sqrt(sy ** 2 + ly)

    T = (dx[:, None] * (Ux.T @ Uy)) * dy[None, :]          # (rx, ry)
    K_eff = max(1, min(K, min(T.shape)))
    Ut, s, Vtt = np.linalg.svd(T, full_matrices=False)
    Ut, s, Vtt = Ut[:, :K_eff], s[:K_eff], Vtt[:K_eff]

    # primal directions
    # NOTE: `@` and `*` share precedence and are left-associative in Python, so
    # the inner product must be parenthesised or this silently broadcasts wrong.
    A = Vtx.T @ (((sx ** 2 + lx) ** -0.5)[:, None] * Ut)      # (p, K)
    B = Vty.T @ (((sy ** 2 + ly) ** -0.5)[:, None] * Vtt.T)   # (q, K)

    Xva = Xc @ A
    Yva = Yc @ B

    # normalise variates to unit variance so correlations are read directly
    # (the derivation gives unit *norm*; unit variance is what is needed)
    sxv = Xva.std(0, keepdims=True)
    syv = Yva.std(0, keepdims=True)
    sxv[sxv < 1e-12], syv[syv < 1e-12] = 1.0, 1.0
    Xva, Yva = Xva / sxv, Yva / syv
    A = A / sxv
    B = B / syv

    return CCAResult(rho=np.clip(s, 0, 1), A=A, B=B, Xva=Xva, Yvb=Yva,
                     alpha=None, beta=None)


def held_out_cca_corr(
    rho_dirs: CCAResult, X_test: np.ndarray, Y_test: np.ndarray
) -> np.ndarray:
    """Apply directions fitted on split A to split B and measure correlations.

    This is the single mechanism that separates real shared directions from
    over-fitted ones: a spurious direction was fitted to noise in A, so its
    correlation on B collapses; a real direction persists.

    Returns (K,) per-component held-out correlations.
    """
    Xc = center(X_test)
    Yc = center(Y_test)
    u = Xc @ rho_dirs.A                       # (n_b, K)
    v = Yc @ rho_dirs.B
    out = np.zeros(u.shape[1])
    for j in range(u.shape[1]):
        a, b = u[:, j], v[:, j]
        sa, sb = a.std(), b.std()
        if sa < 1e-12 or sb < 1e-12:
            out[j] = 0.0
            continue
        out[j] = np.corrcoef(a, b)[0, 1]
    return out


# ---------------------------------------------------------------------------
# Step 3: permutation null
# ---------------------------------------------------------------------------


def permutation_null(
    X: np.ndarray,
    Y: np.ndarray,
    K: int = 32,
    ridge: float = 1e-3,
    n_perm: int = 2000,
    alpha: float = 0.05,
    split_frac: float = 0.5,
    rng: Optional[np.random.Generator] = None,
    return_all: bool = False,
) -> Dict[str, np.ndarray]:
    """Null distribution of the held-out canonical correlation.

    MUST be simulated, never read off a formula: the constant depends on the
    whitening, the ridge and the preprocessing, so the analytic
    (sqrt(p)+sqrt(q))/sqrt(n) expression is only the leading order.  The
    operationally correct threshold is a quantile of this empirical null.

    Protocol: permute the concept labels of Y (breaking any true correspondence),
    fit on split A, evaluate on split B -- identical to the real pipeline.  The
    threshold is the (1-alpha) quantile of the max-over-components held-out rho.
    """
    rng = rng or np.random.default_rng(0)
    n = X.shape[0]
    nA = int(round(n * split_frac))
    idx = np.arange(n)

    max_rho = np.empty(n_perm)
    all_rho = np.empty((n_perm, K)) if return_all else None
    for b in range(n_perm):
        perm = rng.permutation(n)
        Yp = Y[perm]
        iA = idx[:nA]
        res = cca_ridge(X[iA], Yp[iA], K=K, ridge=ridge)
        r = held_out_cca_corr(res, X[nA:], Yp[nA:])
        max_rho[b] = np.nanmax(r) if r.size else 0.0
        if return_all:
            all_rho[b] = r
    out = {
        "max_rho": max_rho,
        "threshold": float(np.quantile(max_rho, 1 - alpha)),
        "alpha": alpha,
        "n_perm": n_perm,
        "mean": float(max_rho.mean()),
        "std": float(max_rho.std()),
    }
    if return_all:
        out["all_rho"] = all_rho
    return out


# ---------------------------------------------------------------------------
# Step 4: the estimator
# ---------------------------------------------------------------------------


@dataclass
class RSCAResult:
    k_star: int                          # screened rank estimate
    rho_cv: np.ndarray                   # (K,) held-out correlations, A->B
    rho_cv_rev: np.ndarray               # (K,) held-out correlations, B->A
    rho_cv_sym: np.ndarray               # symmetric (min of both) -- conservative
    rho_insample: np.ndarray             # (K,) in-sample (inflated) for contrast
    threshold: float
    null: Dict[str, np.ndarray]
    V_hat: np.ndarray                    # (q, k_star) neural-visible subspace in CLIP space
    B_full: np.ndarray                   # (q, K) all fitted image-side directions
    A_full: np.ndarray                   # (d, K) all fitted EEG-side directions
    stability: float                     # subspace overlap between A->B and B->A
    meta: Dict = field(default_factory=dict)

    def summary(self) -> Dict:
        return {
            "k_star": int(self.k_star),
            "threshold": float(self.threshold),
            "rho_cv_sym": [float(x) for x in self.rho_cv_sym],
            "rho_insample": [float(x) for x in self.rho_insample],
            "stability": float(self.stability),
            **self.meta,
        }


def rsca(
    E: np.ndarray,
    C: np.ndarray,
    *,
    D: int = 400,
    K: int = 32,
    ridge: float = 1e-3,
    n_perm: int = 2000,
    alpha: float = 0.05,
    split_frac: float = 0.5,
    seed: int = 0,
    whitener: Optional[EEGWhitener] = None,
    null: Optional[Dict[str, np.ndarray]] = None,
) -> RSCAResult:
    """Reproducibility-Screened Canonical Alignment.

    Parameters
    ----------
    E : (n, T, p) EEG trials per concept, or (n, p) if already averaged.
    C : (n, q) CLIP features.
    whitener : reuse a pre-fitted EEGWhitener (fit once per subject/window).
    null : reuse a pre-computed permutation null (same n, K, ridge).

    Returns
    -------
    RSCAResult with k_star = #{ j : rho_cv_j > threshold }, and V_hat the
    corresponding subspace of image-embedding space.
    """
    n = C.shape[0]
    if whitener is None:
        if E.ndim == 3:
            whitener = EEGWhitener(D=D).fit(E)
        else:
            # already averaged: synthesise a T=2 view by splitting channels? no.
            # Require 3-D input for honest noise estimation.
            raise ValueError("E must be (n, T, p) so the noise floor is estimable")
    Ew = whitener.transform_trials_flat(E) if E.ndim == 3 else E   # (n, T, d)

    # --- split repetitions (A/B) ------------------------------------------
    T = Ew.shape[1]
    half = [0, T // 2] if split_frac == 0.5 else [0, int(round(T * split_frac))]
    Ea = Ew[:, half[0]:half[1]].mean(1)      # (n, d)
    Eb = Ew[:, half[1]:].mean(1)

    idx = np.arange(n)
    nA = int(round(n * split_frac))

    # --- fit on A, evaluate on B ------------------------------------------
    res_ab = cca_ridge(Ea[idx[:nA]], C[idx[:nA]], K=K, ridge=ridge)
    rho_ab = held_out_cca_corr(res_ab, Ea[idx[nA:]], C[idx[nA:]])

    # --- swap (fit on B, evaluate on A) ----------------------------------
    res_ba = cca_ridge(Ea[idx[nA:]], C[idx[nA:]], K=K, ridge=ridge)
    rho_ba = held_out_cca_corr(res_ba, Ea[idx[:nA]], C[idx[:nA]])

    rho_sym = np.minimum(np.abs(rho_ab), np.abs(rho_ba))

    # --- in-sample fit for contrast (full data) --------------------------
    res_full = cca_ridge(Ea, C, K=K, ridge=ridge)

    # --- null -------------------------------------------------------------
    if null is None:
        null = permutation_null(Ea, C, K=K, ridge=ridge, n_perm=n_perm,
                                alpha=alpha, split_frac=split_frac,
                                rng=np.random.default_rng(seed))

    k_star = int((rho_sym > null["threshold"]).sum())

    # stability of the two independent fits (fitted on disjoint halves)
    stability = float("nan")
    if k_star > 0:
        stability = subspace_overlap(res_ab.B[:, :k_star], res_ba.B[:, :k_star])

    return RSCAResult(
        k_star=k_star,
        rho_cv=np.abs(rho_ab),
        rho_cv_rev=np.abs(rho_ba),
        rho_cv_sym=rho_sym,
        rho_insample=res_full.rho,
        threshold=null["threshold"],
        null=null,
        V_hat=res_full.B[:, :k_star].copy(),
        B_full=res_full.B,
        A_full=res_full.A,
        stability=stability,
        meta={
            "n": int(n), "T": int(T), "D_eff": int(whitener.L.shape[0]),
            "q": int(C.shape[1]), "K": int(K), "ridge": float(ridge),
            "n_above_half_insample": int((res_full.rho > 0.5).sum()),
        },
    )


# ---------------------------------------------------------------------------
# Step 5: NV score
# ---------------------------------------------------------------------------


def nv_score(V_hat: np.ndarray, v: np.ndarray) -> float:
    """Neural-visibility score NV(v) = ||P_V v||^2 for a unit direction v.

    This is the payoff of the whole construction: although the individual axes
    of V are unidentifiable (docs Prop. 3, the O(k) gauge freedom), NV is
    invariant under V -> VR for orthogonal R, hence well defined.  It is the
    only kind of quantity about "which image features are neurally visible" that
    the data can legitimately support.
    """
    v = np.asarray(v, float).ravel()
    nv = np.linalg.norm(v)
    if nv < 1e-12:
        return 0.0
    v = v / nv
    if V_hat.size == 0:
        return 0.0
    B, _ = np.linalg.qr(V_hat)
    return float(np.sum((B.T @ v) ** 2))


def nv_scores(V_hat: np.ndarray, V: np.ndarray) -> np.ndarray:
    """NV for each column of V (q, m)."""
    return np.array([nv_score(V_hat, V[:, j]) for j in range(V.shape[1])])


def nv_bootstrap_ci(
    V_hat: np.ndarray, v: np.ndarray, n_boot: int = 1000, seed: int = 0,
    n: Optional[int] = None, n_concepts: Optional[int] = None,
) -> Tuple[float, float]:
    """Bootstrap CI for NV(v).  Resamples concept rows (n must be supplied).

    Kept separate from nv_score so that callers who hold the data can produce a
    CI and callers who only hold the subspace do not silently get a fake one.
    """
    raise NotImplementedError(
        "call nv_ci_with_data() which has access to the paired samples"
    )


def nv_ci_with_data(
    Ebar: np.ndarray, C: np.ndarray, v: np.ndarray, *,
    D: int = 400, K: int = 32, ridge: float = 1e-3,
    n_boot: int = 200, alpha: float = 0.05, seed: int = 0,
) -> Dict[str, float]:
    """Bootstrap CI for NV(v) by resampling concepts.

    Note this resamples *concepts*, which is the correct unit: the canonical
    directions are estimated across concepts.
    """
    rng = np.random.default_rng(seed)
    n = C.shape[0]
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            r = cca_ridge(center(Ebar[idx]), center(C[idx]), K=K, ridge=ridge)
            vals[b] = nv_score(r.B[:, :K], v)
        except Exception:
            vals[b] = np.nan
    vals = vals[~np.isnan(vals)]
    return {
        "nv": nv_score(cca_ridge(center(Ebar), center(C), K=K, ridge=ridge).B[:, :K], v),
        "lo": float(np.quantile(vals, alpha / 2)) if vals.size else float("nan"),
        "hi": float(np.quantile(vals, 1 - alpha / 2)) if vals.size else float("nan"),
        "n_boot_valid": int(vals.size),
    }


# ---------------------------------------------------------------------------
# Diagnostics used by A2/A3
# ---------------------------------------------------------------------------


def participation_ratio(S: np.ndarray) -> float:
    """(sum lambda)^2 / sum lambda^2 -- effective dimensionality."""
    w = np.maximum(np.linalg.eigvalsh(0.5 * (S + S.T)), 0)
    return float((w.sum() ** 2) / max((w ** 2).sum(), 1e-30))


def effective_rank_frac(S: np.ndarray, frac: float = 0.90) -> int:
    w = np.sort(np.maximum(np.linalg.eigvalsh(0.5 * (S + S.T)), 0))[::-1]
    cs = np.cumsum(w) / max(w.sum(), 1e-30)
    return int(np.searchsorted(cs, frac) + 1)


def rdm(X: np.ndarray, metric: str = "correlation") -> np.ndarray:
    """Concept x concept representational dissimilarity matrix."""
    X = np.asarray(X, np.float64)
    if metric == "correlation":
        X = X - X.mean(0, keepdims=True)
        nrm = np.linalg.norm(X, axis=1, keepdims=True)
        Cm = (X @ X.T) / np.maximum(nrm @ nrm.T, 1e-30)
        R = 1.0 - np.clip(Cm, -1, 1)
    elif metric == "euclidean":
        sq = (X ** 2).sum(1)
        R = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0))
    else:
        raise ValueError(metric)
    np.fill_diagonal(R, 0.0)
    return R


def upper_tri(A: np.ndarray) -> np.ndarray:
    return A[np.triu_indices(A.shape[0], k=1)]


def spearman_brown(r: float) -> float:
    """Correction from half-length to full-length reliability."""
    return 2 * r / (1 + r) if r > -1 else float("nan")
