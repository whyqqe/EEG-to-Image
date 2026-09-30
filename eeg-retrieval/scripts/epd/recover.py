#!/usr/bin/env python
"""SCORE's label-free coordinate recovery at deployment (arXiv 2608.19134, Eq. 3-9).

Why this module is the whole point of the inter-subject pipeline
---------------------------------------------------------------
SCORE's ablation (their Table 4) separates the two phases, and the split is lopsided:

    train   SAMGA objective            26.22 Top-1
            + multi-positive           28.63        (+2.41)
            + simulated recovery       29.33        (+0.70)
    test    CSLS ranking               39.08        (+9.75)
            + mean and scale           43.80        (+4.72)
            + recovery                 50.98        (+7.18)
            + identity regularization  53.23        (+2.25)

Everything below the "test" line is CPU post-processing on FROZEN features: 23.90 of
the 27.01 total points, against 3.11 for all the training changes. So the largest
single lever in this problem is not an architecture, it is a change of coordinates --
which is why this file exists before any new encoder does.

What it assumes, and the assumption that makes it work
-----------------------------------------------------
Different subjects preserve the same relationships between concepts but express them
along differently oriented frames. An orthogonal map cannot change distances, so if
the per-subject difference were mostly a nonlinear distortion this would be the wrong
model and Table 1's 16.89 -> 28.22 would not hold. It is fitted from pseudo-pairs
(landmarks), never from labels, and it reorients ALL target features, not just the
landmark rows.

The identity regularization is the part that is easy to get wrong by omitting. A
deployment batch has a few hundred queries, so m << d and the alignment term alone is
rank deficient: R is unconstrained along every direction no landmark supports, and the
minimiser is not unique. Adding lambda*||R - I||_F^2 leaves those directions at the
identity, so a direction with no evidence is left alone rather than set arbitrarily.
That is worth +2.25 Top-1 on its own.

Numerical conventions that are load-bearing
-------------------------------------------
* Eq. 3 matches per-DIMENSION statistics, not global ones. Matching a global scale
  instead leaves each dimension's spread different between the two spaces, and the
  orthogonal map is then fitted to compare vectors that were never put on a common
  per-axis scale.
* Eq. 4's CSLS terms are means over the k nearest neighbours in the OTHER space,
  computed on the full similarity matrix BEFORE any landmark selection. Recomputing
  them on the landmark subset would change what the penalty means.
* Eq. 9 recomputes CSLS on the recovered features. The neighbourhood terms must come
  from where the queries ended up, not from where they started; reusing Eq. 4's
  scores would score the recovered queries with the pre-recovery hubness correction.
* The landmark weights are margins over the whole gallery (top-2 of the query's full
  CSLS row), not margins between landmarks.

Run:  python scripts/test_epd_recover.py     (no GPU, no dataset)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _l2(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1)


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cosine similarity of every row of `a` against every row of `b`, (n, m).

    Matching SAMGA's `retrieve_all`, which uses `cosine_similarity` -- i.e. both sides
    are normalised for scoring. Our EEG features are deliberately NOT L2-normalised
    (SAMGA's `inter.sh` does not pass `--eeg_l2norm`), so normalising here rather than
    upstream keeps that choice from silently becoming a different retrieval metric.
    """
    return _l2(a) @ _l2(b).t()


def moment_match(q: torch.Tensor, g: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Eq. 3: put `q` on `g`'s per-dimension mean and scale.

    An orthogonal map cannot express a translation, so the two origins have to be made
    to coincide before the orientation is estimated. Per-dimension rather than global:
    a single global scale would leave the two spaces with different spreads on each
    axis, and the rotation would then be fitted across vectors that do not share a
    coordinate scale -- which is the thing being estimated.

    `eps` only matters for a dimension that is (near) constant, where it keeps the
    division finite. It is added to BOTH standard deviations, so a degenerate
    dimension maps to the gallery's mean rather than being amplified by 1/eps.
    """
    if q.shape[1] != g.shape[1]:
        raise ValueError(f"feature width mismatch: q {q.shape[1]} vs g {g.shape[1]}")
    return (((q - q.mean(0, keepdim=True)) / (q.std(0, unbiased=False, keepdim=True) + eps))
            * (g.std(0, unbiased=False, keepdim=True) + eps) + g.mean(0, keepdim=True))


def csls_scores(q: torch.Tensor, g: torch.Tensor, k: int = 10) -> torch.Tensor:
    """Eq. 4: `2*cos(q_i, g_j) - r_G(q_i) - r_Q(g_j)`.

    Cross-domain similarity local scaling. High-dimensional retrieval suffers from
    hubness: a few gallery items sit near many queries and attract matches regardless
    of content, so a query's raw nearest neighbour is often the same few "hub" images.
    Subtracting each query's mean similarity to its k nearest gallery items penalises
    a query that is broadly similar to everything, and subtracting each gallery item's
    mean similarity to its k nearest queries penalises a hub image. Both terms are the
    reason plain CSLS ranking is worth +9.75 over cosine on its own.

    `k` is clamped to the available width so a gallery smaller than k does not raise;
    SCORE uses k=10 with a 200-item gallery.
    """
    s = cos_sim(q, g)
    if s.shape[1] == 0 or s.shape[0] == 0:
        # Nothing to penalise against. Return the empty matrix rather than letting
        # `topk(1)` raise on a zero-length dimension, so the caller's own "no gallery"
        # message is the one that surfaces.
        return s
    kk = max(1, min(int(k), s.shape[1]))
    kq = max(1, min(int(k), s.shape[0]))
    r_g = s.topk(kk, dim=1).values.mean(dim=1, keepdim=True)   # per query
    r_q = s.topk(kq, dim=0).values.mean(dim=0, keepdim=True)   # per gallery item
    return 2.0 * s - r_g - r_q


def mutual_nn_pairs(s: torch.Tensor) -> torch.Tensor:
    """Mutual nearest-neighbour pairs of a score matrix, as (m, 2) [query, gallery].

    Only mutual pairs become landmarks. A one-sided nearest neighbour is exactly the
    hubness failure CSLS is meant to reduce, so accepting it would put the least
    trustworthy matches into the fit that defines the map.

    Ties are broken by `argmax`'s first-index rule. That is deterministic, and a tie
    between two gallery items for the same query means those two are interchangeable
    for the fit, so the choice does not need to be randomised.
    """
    if s.ndim != 2:
        raise ValueError(f"expected a 2-D score matrix, got {tuple(s.shape)}")
    if s.shape[0] == 0 or s.shape[1] == 0:
        # `argmax` raises on a zero-length dimension, which would turn "this fold had
        # no gallery" into a crash instead of an empty landmark set. Returning the
        # empty result lets `orthogonal_recovery` produce the real complaint.
        return s.reshape(0, 2)
    fwd = s.argmax(dim=1)                       # best gallery item per query
    bwd = s.argmax(dim=0)                       # best query per gallery item
    keep = bwd[fwd] == torch.arange(s.shape[0], device=s.device)
    rows = torch.nonzero(keep, as_tuple=False).flatten()
    return torch.stack([rows, fwd[rows]], dim=1) if rows.numel() else rows.reshape(0, 2)


def landmark_margins(s: torch.Tensor, pairs: torch.Tensor, eta: float = 1e-6) -> torch.Tensor:
    """Eq. 5: `max(s_i,(1) - s_i,(2), eta)` for each landmark query.

    The margin is the gap between the query's best and second-best CSLS score over the
    WHOLE gallery, not the gap to the runner-up landmark. It is a statement about how
    confident that query's match is, and a query whose top two candidates are nearly
    tied is uncertain regardless of how the other landmarks scored.

    `eta` floors the weight above zero so that a landmark contributes something rather
    than being dropped, and so that the normalisation below cannot divide by zero.
    """
    if pairs.numel() == 0:
        return torch.zeros(0, device=s.device, dtype=s.dtype)
    top2 = s.topk(min(2, s.shape[1]), dim=1).values
    if top2.shape[1] < 2:
        margins = torch.ones(pairs.shape[0], device=s.device, dtype=s.dtype)
    else:
        margins = (top2[:, 0] - top2[:, 1])[pairs[:, 0]]
    deltas = margins.clamp_min(eta)
    return pairs.shape[0] * deltas / deltas.sum()


def select_landmarks(s: torch.Tensor, k: int = 10, max_landmarks: int | None = 160,
                     eta: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """Mutual-NN landmarks with margin weights, capped at `max_landmarks`.

    The cap takes the highest-margin pairs first, because when more mutual pairs exist
    than the budget allows the informative ones are the confident ones. SCORE's
    experiments use 12 to 160 landmarks and a deployment batch of a few hundred
    queries, so the cap is normally slack; it matters when a query batch is large.
    """
    pairs = mutual_nn_pairs(s)
    if pairs.numel() == 0:
        return pairs, torch.zeros(0, device=s.device, dtype=s.dtype)
    w = landmark_margins(s, pairs, eta=eta)
    if max_landmarks is not None and pairs.shape[0] > int(max_landmarks):
        keep = torch.topk(w, int(max_landmarks)).indices
        keep, _ = torch.sort(keep)          # stable order, so results are reproducible
        pairs, w = pairs[keep], w[keep]
        w = pairs.shape[0] * w / w.sum()    # renormalise to `m` after the cap
    return pairs, w


def orthogonal_recovery(x: torch.Tensor, y: torch.Tensor, w: torch.Tensor,
                        rho: float = 0.1,
                        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Eq. 6-7: weighted orthogonal Procrustes with identity regularisation.

    Solves ``R* = argmin_{R^T R = I} ||W^{1/2}(X R - Y)||_F^2 + lambda ||R - I||_F^2``
    with ``lambda = rho * ||X^T W Y||_2``, then returns ``(R*, mu_X, mu_Y)``.

    `rho` is dimensionless, which is the reason lambda is scaled by the spectral norm
    rather than set directly: the cross-covariance's energy grows with the feature
    scale and the number of landmarks, so a fixed lambda would mean a different amount
    of regularisation in every configuration. `rho = 0` recovers unregularised
    Procrustes, and the two arms are what SCORE's "+ identity regularization" row
    (+2.25 Top-1) compares.

    The closed form is `R* = U V^T` from `M = X^T W Y + lambda*I = U Sigma V^T`. Adding
    lambda*I to the cross-covariance is exactly what makes the directions carrying
    little energy stay near the identity: where `X^T W Y` is small, M is dominated by
    lambda*I and `R*` reduces to `I` there. That is the whole content of the term --
    in a rank-deficient fit it is the difference between a defined answer and an
    arbitrary one.

    `X` and `Y` are centred here, weighted by `w`, rather than being expected centred
    by the caller: the weighted centre is what Eq. 8 needs returned, and computing it
    anywhere else would risk centring with unweighted means.
    """
    if x.shape != y.shape:
        raise ValueError(f"landmark shapes must match, got {tuple(x.shape)} vs "
                         f"{tuple(y.shape)}")
    if x.shape[0] == 0:
        raise ValueError("no landmarks: the map is undetermined. Refuse rather than "
                         "return the identity, because a silent identity here looks "
                         "exactly like a correctly regularised no-op map")
    w = w.to(x.dtype)
    wsum = w.sum().clamp_min(1e-12)
    mu_x = (w[:, None] * x).sum(0, keepdim=True) / wsum
    mu_y = (w[:, None] * y).sum(0, keepdim=True) / wsum
    xc = (x - mu_x) * w[:, None].sqrt()
    yc = (y - mu_y) * w[:, None].sqrt()
    c = xc.t() @ yc
    lam = float(rho) * float(torch.linalg.matrix_norm(c, ord=2))
    m = c + lam * torch.eye(c.shape[0], device=c.device, dtype=c.dtype)
    u, _s, vt = torch.linalg.svd(m)
    return u @ vt, mu_x, mu_y


def apply_recovery(q: torch.Tensor, r: torch.Tensor,
                   mu_x: torch.Tensor, mu_y: torch.Tensor) -> torch.Tensor:
    """Eq. 8: `Q_hat = (Q - mu_X) R* + mu_Y`, applied to ALL target features.

    Note this is `q @ R`, i.e. R acts on the right, because Eq. 6 is written as
    ``||X R - Y||`` with X holding rows. Getting the side wrong is not a crash -- it
    is a valid rotation applied in the wrong basis, which retrieves at (or below) the
    unrecovered accuracy.
    """
    return (q - mu_x) @ r + mu_y


@torch.no_grad()
def recover(q: torch.Tensor, g: torch.Tensor, *, k: int = 10, rho: float = 0.1,
            eps: float = 1e-5, max_landmarks: int | None = 160, eta: float = 1e-6,
            moment: bool = True, orientation: bool = True,
            min_landmark_rate: float = 0.0,
            ) -> tuple[torch.Tensor, dict]:
    """Eq. 3-8 on frozen features. Returns the recovered queries and diagnostics.

    The `moment` / `orientation` switches exist so the deployment-side ablation of
    SCORE's Table 4 can be reproduced step by step from one implementation:

        moment=False, orientation=False   -> plain cosine/CSLS on raw features
        moment=True,  orientation=False   -> "+ mean and scale"
        moment=True,  orientation=True    -> "+ recovery"

    `rho=0` with `orientation=True` is the "+ recovery" row and `rho=0.1` the
    "+ identity regularization" row, so all four steps are reachable without a second
    code path.

    Why there is an abstention gate
    -------------------------------
    Recovery is NOT a free win, and the failure is not graceful. `test_epd_recover.py`
    locates a sharp threshold in a synthetic subject whose frame is rotated by an angle
    theta: at theta <= 0.2 the mutual nearest neighbours are 98-100% correct and
    recovery lifts Top-1 from 0.81 to 1.00, while at theta = 0.30 the mutual-pair
    accuracy collapses to 4% and recovery scores 0.025 BELOW the 0.035 unrecovered
    baseline. A map fitted from wrong correspondences is a real rotation in the wrong
    direction, so applying it is worse than doing nothing.

    The precondition is that the un-recovered features already retrieve well enough for
    mutual nearest neighbours to be mostly correct -- which is why SCORE's CSLS ranking
    (+9.75) comes before its recovery (+7.18) in the ablation, and why recovery is a
    poor fit for a weak representation. `min_landmark_rate` turns the observable proxy
    into a guard: the fraction of queries that became landmarks separates the two
    regimes cleanly (0.90 against 0.55 in the synthetic sweep) without needing labels.
    When it is not met the features are returned UNRECOVERED and `abstained` is set, so
    a run reports the baseline rather than a confident wrong number.

    Diagnostics are meant to be logged, not just available: the landmark count, the
    landmark rate and the weight concentration are what say whether the map was fitted
    from evidence or from noise.
    """
    if q.ndim != 2 or g.ndim != 2:
        raise ValueError("recover expects (n_queries, d) and (n_gallery, d)")
    qt = moment_match(q, g, eps=eps) if moment else q
    s = csls_scores(qt, g, k=k)
    pairs, w = select_landmarks(s, k=k, max_landmarks=max_landmarks, eta=eta)
    rate = (pairs.shape[0] / int(q.shape[0])) if q.shape[0] else 0.0
    diag: dict = {
        "n_queries": int(q.shape[0]),
        "n_gallery": int(g.shape[0]),
        "n_mutual_pairs": int(pairs.shape[0]),
        "landmark_rate": float(rate),
        "k_csls": int(k),
        "rho": float(rho),
        "moment_match": bool(moment),
        "orientation": bool(orientation),
        "min_landmark_rate": float(min_landmark_rate),
        "abstained": False,
    }
    if not orientation:
        return qt, diag
    if pairs.numel() == 0:
        raise SystemExit(
            "coordinate recovery found 0 mutual nearest-neighbour pairs, so there is no "
            "map to estimate. Refusing rather than returning the identity, because a "
            "silent identity is indistinguishable from a correctly regularised no-op "
            "map and would be read as 'recovery ran'")
    if rate < float(min_landmark_rate):
        diag["abstained"] = True
        diag["abstain_reason"] = (
            f"landmark rate {rate:.3f} < {min_landmark_rate:.3f}: the un-recovered "
            f"features do not retrieve well enough for the pseudo-matches to be "
            f"trustworthy, and a map fitted from wrong pairs scores BELOW the "
            f"baseline. Returning the unrecovered features")
        return qt, diag
    idx_q, idx_g = pairs[:, 0], pairs[:, 1]
    r, mu_x, mu_y = orthogonal_recovery(qt[idx_q], g[idx_g], w, rho=rho)
    wshare = w / w.sum()
    diag["weight_share_top1"] = float(wshare.max().item())
    diag["weight_entropy"] = float((-(wshare * torch.log(wshare.clamp_min(1e-12))).sum()).item())
    # How far R* moved from the identity, in Frobenius terms. Near 0 means the
    # regularisation dominated and recovery is a no-op; sqrt(2d) means the landmarks
    # overrode it. Either extreme is worth seeing in a log.
    diag["r_minus_i_frobenius"] = float(torch.linalg.matrix_norm(
        r - torch.eye(r.shape[0], device=r.device, dtype=r.dtype)).item())
    return apply_recovery(qt, r, mu_x, mu_y), diag
