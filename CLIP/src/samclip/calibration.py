"""Deployment-time geometric calibration (label-free).

WHAT THE MEASUREMENT ACTUALLY SAYS
----------------------------------
This module's header used to assert that "the largest single lever in this problem is a
change of coordinates, not an architecture", citing SCORE's 26.22 -> 53.23 on a frozen
SAMGA encoder. That statement is WRONG, and the correction changes what the project
should build -- `probe_signal_weights.py` separated the two things ``saw_whiten`` does,
on this fold and three seeds:

    metric                          seed2025   seed2026   seed2027
    raw cosine                         13.50      17.50      18.00
    centre the QUERY cloud only        20.00      22.00      20.50   <-- almost all of it
    centre the gallery cloud only          --      16.50      17.00
    centre both                        20.50      21.00      22.00
    saw_whiten (best shrink)           19.50      21.50      24.50   <-- ~+0..3 on top
    oracle w_j = rho_j^2 / lambda_j    21.00      20.00      19.50   <-- the whole family's ceiling

Three facts follow, and each one is load-bearing:

  1. Subtracting the query cloud's mean accounts for nearly ALL of the gain
     (+6.5 / +4.5 / +2.5 points). The covariance half is worth at most ~1 point and on
     two of three seeds the best shrinkage merely ties or loses to centring.
  2. The per-dimension signal-to-noise-optimal diagonal reweighting -- which needs the
     LABELS -- lands at or BELOW plain centring on 2 of 3 seeds. That bounds the entire
     family (whitening, CSLS's local scaling, covariance-based adaptation, coordinate
     recovery, any fitted map) and the bound is not higher than "subtract a mean".
  3. So the lever is a TRANSLATION, not a change of coordinates; and a translation is
     something an architecture should not need in the first place. That is the argument
     for the v4 redesign (`docs/eeg2image_v4_architecture.md` §1.2), and it is why the
     evaluation ladder below starts at "+ centre" instead of jumping to whitening.

Three composable, label-free operations on the similarity structure:

  0. ``center_queries``  -- subtract the query cloud's mean. The measured strongest
     single operation, and the rung every later one has to beat. Kept as its own
     function because the previous version of this file bundled it into ``saw_whiten``
     and credited the covariance.
  1. ``saw_whiten``      -- subject-adaptive whitening of the query embeddings using
     the target subject's *unlabelled* calibration statistics.
  2. ``csls_scores``     -- cross-domain similarity local scaling, which removes the
     hubness that makes a few gallery items attract every query.
  3. ``coordinate_recovery`` -- weighted orthogonal Procrustes with identity
     regularisation, fitted from mutual-NN pseudo-pairs (SCORE Eq. 3-9).

(3) is imported from the shared `eeg-retrieval/epd.recover` implementation, which
already encodes the numerical conventions that are load-bearing (per-dimension
moment matching; CSLS recomputed after recovery; the abstention gate that refuses to
apply a map fitted from untrustworthy pseudo-pairs). Re-implementing it would be a
second place for those conventions to drift.
"""
from __future__ import annotations

import sys

import numpy as np
import torch

from . import config


# ------------------------------------------------------------------ centring
def center_queries(q: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, dict]:
    """Subtract the query cloud's own mean.

    Legitimate under a strict LOSO protocol for the same reason ``saw_whiten`` is: it
    uses the target subject's EEG only, never which stimulus a trial was, and it is
    transductive over the query set rather than fitted to anything. It is also the one
    operation whose benefit is measured to be large, stable across seeds, and identical
    whether it is applied here or inside the model's forward pass (the v4 ``SMN``).

    The diagnostic reports the quantity the operation removes:
    ``||mean(q)|| / mean||q||``. On the v3 checkpoints it is 0.27-0.42 and the gain from
    removing it is real; if a future model has a SHARED cross-modal head
    (``arch: v4``) this number should be small and this rung should buy nothing. That
    pair of facts is the falsifiable prediction for C1/C2, which is why the number is
    reported rather than just the score.
    """
    q = np.asarray(q, dtype=np.float64)
    mu = q.mean(axis=0, keepdims=True)
    row = np.linalg.norm(q, axis=-1).mean()
    diag = {
        "offset_ratio": float(np.linalg.norm(mu) / max(float(row), eps)),
        "n_samples": int(q.shape[0]),
    }
    return (q - mu).astype(np.float32), diag


# ----------------------------------------------------------------- whitening
def saw_whiten(q: np.ndarray, shrink: float = 0.1, max_cond: float = 1e3,
               eps: float = 1e-8) -> tuple[np.ndarray, dict]:
    """Whiten the query embeddings by their OWN covariance (subject-adaptive).

    Legitimate under a strict LOSO protocol because it uses the EEG only -- never
    which stimulus a trial was -- so it is not a hidden label leak.

    THIS FUNCTION DOES TWO THINGS AND ONLY ONE OF THEM IS MEASURED TO MATTER. It centres
    (``qc = q - mu``) and it rescales by the inverse square root of the covariance.
    Separated on three seeds, the centre alone is worth +6.5/+4.5/+2.5 points while the
    rescale adds -0.5 to +4.0 -- and the per-dimension SNR-optimal diagonal weighting,
    computed WITH labels, lands at 21.0/20.0/19.5, i.e. at or below plain centring. Use
    ``center_queries`` when the question is "how much does centring buy?", and this
    function when the question is the specific bundled operation the earlier runs
    reported. See the module docstring.

    Returns `(whitened, diagnostics)`. Two guards are needed because the covariance
    here is severely rank-deficient: 200 query trials against a `d_embed = 512`
    embedding spans at most 199 dimensions, so ~313 of the 512 eigenvalues are
    numerical zero. A fixed absolute floor (`cov + 1e-5 * I`, as an earlier version
    used) inverts them to `1/sqrt(1e-5) ~ 316`, i.e. it amplifies pure noise in the
    null space by more than two orders of magnitude -- which is why "whitening" could
    score *below* the raw cosine it was supposed to improve.

    `shrink` pulls the covariance toward a scaled identity (the standard remedy for a
    singular sample covariance) and `max_cond` caps the condition number of the
    whitening map, so the result stays a rotation-and-rescale rather than a noise
    amplifier.
    """
    q = np.asarray(q, dtype=np.float64)
    mu = q.mean(axis=0, keepdims=True)
    qc = q - mu
    n, d = qc.shape
    cov = (qc.T @ qc) / max(1, n - 1)
    if shrink > 0:
        cov = (1.0 - shrink) * cov + shrink * (np.trace(cov) / d) * np.eye(d)
    cov = cov + eps * np.eye(d)
    vals, vecs = np.linalg.eigh(cov)
    lo = max(float(vals.max()) / max_cond, eps)
    vals = np.maximum(vals, lo)
    inv_sqrt = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
    diag = {
        "eig_max": float(vals.max()),
        "eig_min": float(vals.min()),
        "cond": float(vals.max() / vals.min()),
        "n_samples": int(n),
        "d_embed": int(d),
        "rank_deficient": bool(n - 1 < d),
    }
    return (qc @ inv_sqrt).astype(np.float32), diag


def moment_match(q: np.ndarray, g: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Put `q` on `g`'s per-dimension mean and scale."""
    q = np.asarray(q, dtype=np.float32)
    g = np.asarray(g, dtype=np.float32)
    qs = q.std(axis=0, keepdims=True).astype(np.float64)
    gs = g.std(axis=0, keepdims=True).astype(np.float64)
    return (((q - q.mean(0, keepdims=True)) / (qs + eps)) * (gs + eps)
            + g.mean(0, keepdims=True)).astype(np.float32)


# --------------------------------------------------------------------- CSLS
def csls_scores(q: np.ndarray, g: np.ndarray, k: int = 10) -> np.ndarray:
    """`2*cos(q_i, g_j) - r_G(q_i) - r_Q(g_j)` -- the hubness correction."""
    qn = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8)
    gn = g / np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-8)
    s = qn @ gn.T
    if s.size == 0:
        return s
    kk = max(1, min(int(k), s.shape[1]))
    kq = max(1, min(int(k), s.shape[0]))
    r_g = np.sort(s, axis=1)[:, -kk:].mean(axis=1, keepdims=True)
    r_q = np.sort(s, axis=0)[-kq:, :].mean(axis=0, keepdims=True)
    return 2.0 * s - r_g - r_q


def report_with_scores(scores: np.ndarray) -> dict:
    """Top-k on an already-computed score matrix (e.g. CSLS output)."""
    order = np.argsort(-scores, axis=1)
    rk = np.diag(np.argsort(order, axis=1)) + 1
    return {"top1": float((rk <= 1).mean() * 100.0),
            "top5": float((rk <= 5).mean() * 100.0),
            "mean_rank": float(rk.mean()), "n": int(scores.shape[0])}


# ------------------------------------------------------- structural expert (SATTC)
def _ranks(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Forward and backward ranks of every (query, gallery) pair, 0-based.

    ``rf[i, j]`` is how many gallery items query ``i`` ranks above ``j``; ``rb[i, j]`` is
    the same on the transposed problem. Both are needed because the structural signal is
    *agreement between the two directions*, and a one-sided rank cannot express it.
    """
    nq, ng = scores.shape
    # A DOUBLE argsort IS the rank, and the obvious fancy-index form is a TRAP: writing
    # `rb[np.argsort(-scores, axis=0), np.arange(ng)[:, None]] = np.arange(nq)[:, None]`
    # silently produces GARBAGE on a 200x200 matrix (measured: values in [-4.7, 199] and
    # non-integer, dtype-dependent), because the two advanced index arrays broadcast to
    # (nq, ng) in a way that does not correspond to the (rank, column) pairs the assignment
    # intends. It happened to be correct on a 4x4 toy, which is exactly why it survived. The
    # double-argsort form is provably the rank and is verified to reproduce `rf` bit-for-bit.
    rf = np.argsort(np.argsort(-scores, axis=1), axis=1).astype(np.float64)
    rb = np.argsort(np.argsort(-scores, axis=0), axis=0).astype(np.float64)
    return rf, rb


def _zscore_rows(s: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Standardise every row: subtract its mean, divide by its std.

    This is the fusion's unit fix, and it is the difference between a working PoE and a
    broken one. The two experts live on incomparable scales -- a log-probability (order
    -5) and a reciprocal-rank product (order -4 to 0) -- so adding them unweighted lets
    whichever happens to have the larger spread dominate, and the measured result of
    doing that was a collapse from 35.50 to 18.00 Top-1. Standardising each row makes the
    two comparable, and it costs nothing: within a row, adding a constant cannot change
    ``argmax``, so the row mean is free information to remove.
    """
    mu = s.mean(axis=1, keepdims=True)
    sd = s.std(axis=1, keepdims=True)
    return (s - mu) / np.maximum(sd, eps)


def structural_scores(
    scores: np.ndarray,
    k: int = 10,
    hub_alpha: float = 1.0,
    lam: float = 0.2,
    eps: float = 1e-12,
) -> tuple[np.ndarray, dict]:
    """SATTC's structural expert, fused with the geometric scores by a weighted PoE.

    WHY THIS EXISTS. CSLS repairs the *scale* of a similarity, but it is still a purely
    geometric object: it never asks whether the two sides *agree on each other*. A
    shortlist built from geometry alone stays unreliable for the small ``k`` a real
    system would use, which is the failure SATTC (CVPR 2026) names directly. Its
    structural expert adds three cheap, label-free signals and fuses them with the
    geometric score, so a candidate has to look good to BOTH rather than being carried by
    one.

    The three signals, computed on the score matrix alone:

      * **mutual agreement** -- the product of the two reciprocal ranks. A pair that is
        rank 0 forward but rank 80 backward is not a match, and the product says so.
      * **bidirectional top-k** -- the hard indicator that both sides put each other in
        their top ``k``.
      * **hubness / popularity** -- how often a gallery item is claimed as a top-k by
        *any* query. Dividing its score by its degree is the structural counterpart of
        CSLS's local scaling, and it operates on a statistic CSLS cannot see: a count,
        not a mean similarity.

    FUSION. Each expert is standardised per row (see ``_zscore_rows``) and combined as
    ``z_geom + lam * z_struct``. ``lam`` is exposed precisely because the honest state of
    this component is "the signal is weak and its weight has to be measured": on this fold
    only ~2% of pairs are mutual-top-k (a ~8x enrichment over chance, but still mostly
    noise), so an unweighted fusion *loses* 17 points. ``lam=0`` recovers pure geometry,
    which is the correct fallback whenever the structural expert is not paying for itself.

    Everything here is label-free and uses only the target subject's query set and the
    gallery, so it stays inside the LOSO protocol (a T2 / transductive operation).

    Returns ``(fused_scores, diagnostics)``. ``mutual_topk_enrichment`` is the diagnostic
    that decides whether this component is worth keeping at all.
    """
    s = np.asarray(scores, dtype=np.float64)
    nq, ng = s.shape
    if nq == 0 or ng == 0:
        return s.astype(np.float32), {"applied": False, "reason": "empty"}

    kk = max(1, min(int(k), ng, nq))
    rf, rb = _ranks(s)

    mutual = (1.0 / (1.0 + rf)) * (1.0 / (1.0 + rb))
    both_topk = ((rf < kk) & (rb < kk)).astype(np.float64)
    degree = (rf < kk).sum(axis=0)                       # (ng,)

    struct = (np.log(mutual + eps) + np.log1p(both_topk)
              - hub_alpha * np.log1p(degree)[None, :])

    z_geo = _zscore_rows(s)
    z_str = _zscore_rows(struct)
    fused = z_geo + float(lam) * z_str

    # enrichment over chance: observed mutual-top-k rate / the rate a random ranking
    # would produce. <2 means the expert is noise and `lam` should be 0.
    rate = float(both_topk.mean())
    chance = float((kk / ng) * (kk / nq))
    diag = {
        "applied": True, "k": int(kk), "lam": float(lam), "hub_alpha": float(hub_alpha),
        "mutual_topk_rate": rate,
        "mutual_topk_chance": chance,
        "mutual_topk_enrichment": float(rate / max(chance, eps)),
        "max_degree_over_k_nq": float(degree.max() / max(1, kk * nq)),
        "mean_degree": float(degree.mean()),
    }
    return fused.astype(np.float32), diag




# -------------------------------------------------------- coordinate recovery
def _sinkhorn_plan(s: np.ndarray, tau: float = 0.05, iters: int = 50,
                   eps: float = 1e-12) -> np.ndarray:
    """Doubly-stochastic soft matching plan from a similarity matrix.

    WHY SINKHORN AND NOT ARGMAX, in one paragraph. The deployed recovery picks landmarks by
    mutual-nearest-neighbour and then fits a ``d x d`` rotation from them. On this task that
    is ~42 pairs out of 200 (:meth:`coordinate_recovery`'s own diagnostics report
    ``n_mutual = 30-42``) for a map with ``d(d-1)/2 = 2016`` free parameters -- and the
    project's own spectral measurement puts the concept manifold at 16 dimensions, so ~48 of
    the 64 directions are noise as far as this alignment is concerned. The estimate is
    therefore not "a rotation fitted to 42 constraints"; it is a rotation fitted to a
    handful of constraints and regularised back to the identity everywhere else, which is
    exactly what ``rho`` does. Sinkhorn replaces the hard 0/1 assignment with the maximum-entropy
    doubly-stochastic one: every query contributes to the estimate with a weight, the row and
    column marginals enforce a one-to-one retrieval's balance, and the effective sample size
    rises from ~42 toward ``C``.

    The temperature ``tau`` interpolates hard (``tau -> 0``, recovers argmax matching) and
    uniform (``tau -> inf``, no information); it is swept rather than assumed.

    STABILISATION IS ROW-WISE, NOT GLOBAL, AND IT IS LOAD-BEARING. Subtracting the global
    maximum leaves ``exp((s_ij - s_max)/tau)`` underflowing to exactly 0 for every entry more
    than a few ``tau`` below the peak, so ``K`` becomes a near-permutation, Sinkhorn's scaling
    vectors blow up (measured: ``sum(P**2)`` overflowed to inf, ``n_effective`` reported 0.0),
    and the plan it returns has row sums in ``0.26..1.05`` instead of 1. Subtracting each
    ROW's maximum makes ``K`` row-stochastic before the iterations start, keeps every
    scaling vector O(1), and the returned plan is doubly stochastic to machine precision.
    The final ``u`` update is applied after the loop so the ROW marginal is exact rather than
    the column one -- the weighted Procrustes below consumes row-normalised weights.
    """
    k = np.exp((s - s.max(axis=1, keepdims=True)) / max(float(tau), eps))
    u = np.ones(k.shape[0], dtype=np.float64)
    v = np.ones(k.shape[1], dtype=np.float64)
    for _ in range(int(iters)):
        v = 1.0 / (k.T @ u + eps)
        u = 1.0 / (k @ v + eps)
    plan = (u[:, None] * k) * v[None, :]
    # NORMALISED TO UNIT MASS, matching `_hard_mutual_plan`. This is not cosmetic: the
    # weighted Procrustes consumes `q^T (P g)`, which scales linearly with the plan's mass,
    # so a doubly-stochastic plan (mass C) against a hard plan (mass 1) would compare the two
    # arms at a 200x different data-to-`rho` ratio -- i.e. the comparison would silently be
    # about `rho`, which is the parameter the whole subspace argument is trying to hold
    # fixed. Unit mass makes `P g` a weighted AVERAGE of gallery rows in both arms.
    return plan / max(float(plan.sum()), eps)


def _sq_cos_dist(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Standardised squared chordal distance matrix. Twin of the training-side helper.

    Standardised because `alpha` has to trade a score against a squared-distance sum, and
    without a common scale `alpha` would be measuring the units rather than the trade-off.
    """
    n = x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)
    d = (n @ n.T - 1.0) ** 2
    return (d - d.mean()) / max(float(d.std()), eps)


def _eff_rank(dmat: np.ndarray) -> float:
    """Participation ratio of the eigenvalue spectrum: ``(sum l)^2 / sum l^2``.

    PERMUTATION-INVARIANT BY CONSTRUCTION: relabelling the rows/columns of `dmat` conjugates it
    by a permutation, which leaves the spectrum -- and therefore this scalar -- unchanged. That
    is the whole reason it is usable as a source prior (see `_fgw_plan`), because it carries NO
    index correspondence and can therefore not smuggle in the target's labels the way the M1
    template did.
    """
    w = np.clip(np.linalg.eigvalsh(np.asarray(dmat, dtype=np.float64)), 0.0, None)
    return float((w.sum() ** 2) / max(float((w ** 2).sum()), 1e-12))


def _spectral_truncate(dmat: np.ndarray, k: int) -> np.ndarray:
    """Top-`k` eigen-approximation, ``V_k L_k V_k^T``.

    NOT the same operation as the stage-2 shrinkage that failed, and the difference is why this
    is worth measuring at all. Shrinkage, ``(1-l) S + l I``, INFLATES the noise subspace toward
    `l`; measured, it reduced the structural gain monotonically at every R (3.08 -> 1.13pp at
    R=80), because inflating noise is exactly what the GW term must not be fed. Truncation
    DELETES the noise subspace instead. With the concept manifold measured at d_M ~ 8-16 out of
    d = 64, that is 48 directions of pure estimation noise removed rather than rescaled.
    """
    w, v = np.linalg.eigh(np.asarray(dmat, dtype=np.float64))
    idx = np.argsort(w)[::-1][:int(k)]
    return (v[:, idx] * w[idx]) @ v[:, idx].T


def _topo_graph(dmat: np.ndarray, eps: float) -> np.ndarray:
    """The eps-threshold graph of a metric, as 0/1 adjacency.

    This is the TOPOLOGICAL structural reference (docs/eeg2image_v10_m1_theory.md §9.3): it
    throws away every distance magnitude and keeps only which concepts are within `eps` of each
    other. That is a strictly coarser object than the metric, and coarser is the point -- the
    metric prior failed *because its fine magnitudes are whitening-flattened*, so the surviving
    structure is expected to live in the COARSE connectivity, not in the spectrum.
    """
    a = (np.asarray(dmat, dtype=np.float64) <= float(eps)).astype(np.float64)
    np.fill_diagonal(a, 0.0)
    return a


def _topo_eps_from(xn: np.ndarray, q: float = 0.10) -> float:
    """Threshold for the eps-graph: the `q`-quantile of the pairwise distance multiset.

    WHY A FIXED QUANTILE AND NOT A PERSISTENCE GAP. The first version of this function used the
    largest gap in the sorted merge distances -- the H0 persistence gap -- on the theory that it
    lands on a "natural" cluster scale. On the real data it does not: the 200 THINGS concepts are
    spread over the sphere without pronounced clusters, so there is no large gap to find, and the
    criterion then fires on whatever the largest small gap happens to be. Measured on sub-08 that
    produced a graph with **11940 of 19900 possible edges -- 60% density**, i.e. a near-complete
    graph that carries almost no connectivity information. A criterion that cannot abstain when
    its own premise (a gap) is absent will silently return a useless scale, which is worse than
    returning none.

    A QUANTILE HAS NO SUCH FAILURE MODE and is the standard construction for this object: an
    `eps`-graph (equivalently a mutual-kNN graph) at a fixed density. Setting `eps` to the
    `q`-quantile of the upper-triangle distances gives EXACTLY `q * C(C-1)/2` undirected edges by
    construction, so the graph's sparsity is a chosen constant rather than a byproduct of noise
    in the distance distribution. `q = 0.10` keeps it sparse enough that an edge is a statement.

    SCALE-FREE ACROSS DOMAINS. The EEG cloud and the image gallery get `eps` from their OWN
    multisets, so the two graphs are compared at the same *density* while living at different
    *scales*. That is what makes the term topological (which concepts are neighbours) rather than
    metric (how far apart they are), and it is why the two-scale design in `_fgw_plan` is not a
    convenience but the point.

    PERMUTATION-INVARIANT: a quantile of a distance multiset is unchanged by relabelling the
    concepts, so this scalar carries no index correspondence and the M1 leakage channel cannot
    open through it.
    """
    n = xn.shape[0]
    d = np.asarray(_sq_cos_dist(xn), dtype=np.float64)
    iu = np.triu_indices(n, 1)
    return float(np.quantile(d[iu], float(q)))


def _fgw_plan(s: np.ndarray, qn: np.ndarray, gn: np.ndarray, alpha: float,
              tau: float, iters: int, outer: int = 10, eps: float = 1e-9,
              de_ref: np.ndarray | None = None, de_mix: float = 0.0,
              spec_rank: int | None = None,
              topo_eps: tuple[float, float] | None = None,
              di_ref: np.ndarray | None = None) -> np.ndarray:
    """Fused Gromov-Wasserstein matching plan, solved in SCORE space by conditional gradient.

    THE UNIFICATION. A coupling `pi` is a correspondence, and there are three orders of
    information available to constrain it:

        FGW(pi) = (1-a) <pi, C>  +  a * sum_ijkl pi_ij pi_kl (D^e_ik - D^i_jl)^2

    * the anchored term `<pi, C>` is the cross-modal similarity (CSLS) -- 1st order, and the
      ONLY term `subspace_soft_recovery` has ever used (`alpha = 0` is exactly the v8 operator);
    * the structural term is the intra-domain metric comparison -- 2nd order, invariant to any
      orthogonal re-embedding of either cloud, which is precisely the `phi_s` this task has to
      defeat. Measured on frozen features the concept metric IS shared across subjects
      (`corr(D_eeg_s, D_eeg_t) = +0.565` over 45 pairs, chance -0.003), and nothing in the
      deployed ladder used it.

    WHY THE OPTIMUM IS INTERIOR, NOT AT EITHER END. The anchored cost is `proportional to
    Z_e Z_i^T`, whose rank is at most the aligned dimension (64, and in practice ~8-16 after the
    concept-manifold measurement). A rank-`r` object cannot distinguish two correspondences that
    differ only in the orthogonal complement of an `r`-dimensional subspace of matrix space,
    whereas the structural term lives in the `C(C-1)/2`-dimensional space of pairwise
    comparisons. Neither can express what the other does. Measured on 5 subjects: alpha=0 gives
    37.00 top-1, alpha=0.25 gives 38.80 (+1.80), alpha=1.0 COLLAPSES to 1.90 -- the interior peak
    plus the collapse at pure structure is the predicted profile and is why this is fused.

    SOLVER. The GW gradient factorises, so one conditional-gradient step is two matrix products:
        grad_ij = sum_kl pi_kl (D^e_ik - D^i_jl)^2
                = [D^e^2 @ rowsum(pi)]_ij + [D^i^2 @ colsum(pi)^T]_ij - 2 [D^e @ pi @ D^i^T]_ij
    The re-solve is the SAME row-stabilised Sinkhorn already property-tested, so no new solver
    is introduced and the two agree by construction.
    """
    de = _sq_cos_dist(qn)
    di = _sq_cos_dist(gn)
    # G1 -- THE GALLERY-SIDE MULTI-VIEW REFERENCE. The FGW structural term is a comparison
    # of TWO estimates of the shared concept metric, and until now only the QUERY side had
    # a multi-view estimator (`rep_blocks` pools the target's own repetition blocks). The
    # gallery side was always the metric of ONE fused embedding. `di_ref` injects a
    # precomputed gallery-side metric -- built the same way (reliability-weighted fusion
    # over the gallery's own independent views, here the K target layers, which are indexed
    # by the SAME image and therefore carry no cross-domain correspondence) -- so BOTH sides
    # of the coupling can be denoised. `di_ref=None` is bit-identical to the shipped
    # operator (asserted by the caller's fidelity gate and smoke_test §19).
    if di_ref is not None:
        di = np.asarray(di_ref, dtype=np.float64)
    # M1 -- THE SOURCE-METRIC TEMPLATE (v10 stage 2.5, docs/eeg2image_v10_pipeline.md).
    #
    # `de` is the EEG-side metric, and it is estimated from C = 200 whitened query means. That
    # estimate is the quantity stage 1 showed the whole structural gain is hostage to, and stage
    # 2 showed shrinkage CANNOT fix it: increasing shrink monotonically reduced the gain at every
    # R (3.08 -> 1.13 at R=80), because shrinkage buys variance with ISOTROPIC BIAS, degrading the
    # very metric the term exists to use. Pooling across subjects is the one variance-reduction
    # route that carries no such bias -- and it is licensed by a measurement, not an assumption:
    # the concept metric is shared across subjects at corr(D_eeg_s, D_eeg_t) = +0.565 over 45
    # pairs (chance -0.003). So `de_ref` is the SOURCE subjects' average concept metric, expressed
    # in the SAME whitened space, and `de_mix` interpolates toward it.
    #
    # `de_mix = 0` is bit-identical to the shipped operator (asserted by the caller's fidelity
    # gate), and `de_mix = 1` replaces the target's own metric with the source template entirely
    # -- which is the pure form of the hypothesis and is measured, not assumed, to be the right
    # end of the range.
    if de_ref is not None and float(de_mix) > 0.0:
        de = (1.0 - float(de_mix)) * de + float(de_mix) * np.asarray(de_ref, dtype=np.float64)
    # P3 -- THE TOPOLOGICAL REFERENCE. Both sides are thresholded to 0/1 adjacency, EACH AT ITS
    # OWN scale. Two eps values rather than one is the whole point: the EEG cloud and the image
    # gallery live in different spaces with different distance scales, so a single shared eps
    # would mean one graph is a singleton cloud and the other a blob. Topology is scale-free, so
    # each domain gets the characteristic scale of its OWN distance multiset and the term then
    # compares SHAPE (which concepts are neighbours) rather than magnitude -- which is the
    # property that survives a whitener that flattens magnitudes.
    #
    # A `0/1` matrix also cannot be low-rank-flattened, so this is why P1's obstruction does not
    # reach here; `spec_rank` is therefore meaningless on it and the caller disables it.
    if topo_eps is not None:
        e_de, e_di = float(topo_eps[0]), float(topo_eps[1])
        de = (de <= e_de).astype(np.float64)
        di = (di <= e_di).astype(np.float64)
    # P1 -- THE SPECTRAL RANK PRIOR. Applied AFTER any blend, so it is the last thing to touch
    # the metric and "the rank changed the plan" is the only possible explanation.
    if spec_rank is not None and 0 < int(spec_rank) < de.shape[0]:
        de = _spectral_truncate(de, int(spec_rank))
    s_std = (s - s.mean()) / max(float(s.std()), eps)
    plan = _sinkhorn_plan(s_std, tau=tau, iters=iters)
    for _ in range(int(outer)):
        r = plan.sum(1, keepdims=True)
        c = plan.sum(0, keepdims=True)
        # THE GW GRADIENT, WRITTEN OUT. With C1 = D^e and C2 = D^i as the distance matrices,
        #     grad_ij = sum_kl (C1_ik - C2_jl)^2 pi_kl
        #             = (C1^2 @ r)_i + (C2^2 @ c)_j - 2 (C1 @ pi @ C2^T)_ij
        # where r_k = sum_l pi_kl and c_l = sum_k pi_kl. The first two terms are row- and
        # column-constant respectively, so they enter as an (N,1) broadcast down the columns and
        # an (1,N) broadcast across the rows -- NOT as two (N,1) vectors added together. An
        # earlier version of this line wrote `C1 @ r + C2 @ c.T` (missing BOTH elementwise
        # squares, and transposing nothing, which silently broadcast the C2 term along i instead
        # of j). Verified against a finite-difference gradient of the bilinear form: that version
        # had correlation -0.48 with the true gradient -- it was a CONDITIONAL-GRADIENT step in a
        # partially wrong direction, i.e. worse than no structural term. This version matches to
        # 1.4e-10 with correlation +1.000000.
        grad = (de * de) @ r + ((di * di) @ c.T).T - 2.0 * (de @ plan @ di.T)
        g_std = (grad - grad.mean()) / max(float(grad.std()), eps)
        # structural consistency is a REWARD in score space: high when a query and a gallery row
        # have matching distance profiles to all other concepts.
        s_eff = (1.0 - alpha) * s_std - alpha * g_std
        plan = _sinkhorn_plan(s_eff, tau=tau, iters=iters)
    return plan


def _hard_mutual_plan(s: np.ndarray) -> np.ndarray:
    """The deployed landmark rule as a normalised plan: mutual-NN rows, weight ``1/L``.

    Expressed as a matrix so it plugs into the SAME weighted-Procrustes code as the soft
    plan. That is not cosmetic: it makes "hard vs soft" a comparison of two settings of one
    estimator rather than two code paths, so the 2x2 attribution below cannot be confounded
    by an implementation difference.
    """
    n_q, n_g = s.shape
    fwd = s.argmax(axis=1)
    bwd = s.argmax(axis=0)
    gal = np.arange(n_g)
    g_idx = np.nonzero(fwd[bwd] == gal)[0]
    plan = np.zeros_like(s)
    if g_idx.size == 0:
        return plan
    q_idx = bwd[g_idx]
    plan[q_idx, g_idx] = 1.0 / float(g_idx.size)
    return plan


def subspace_soft_recovery(
    q: np.ndarray,
    g: np.ndarray,
    k: int = 10,
    rho: float = 0.1,
    rank: int | None = 16,
    tau: float = 0.05,
    iters: int = 50,
    hard_landmarks: bool = False,
    moment: bool = True,
    min_landmarks: int = 8,
    alpha: float = 0.0,
    fgw_outer: int = 10,
    fgw_de_ref: np.ndarray | None = None,
    fgw_de_mix: float = 0.0,
    fgw_spec_rank: int | None = None,
    fgw_topo_eps: tuple[float, float] | None = None,
    fgw_di_ref: np.ndarray | None = None,
    eps: float = 1e-8,
) -> tuple[np.ndarray, dict]:
    """Subspace-regularised soft recovery (S3R): recover coordinates without over-fitting.

    TWO DEFECTS OF THE DEPLOYED OPERATOR, AND WHY THEY ARE ONE FIX.

    ``coordinate_recovery`` fits an orthogonal map by orthogonal Procrustes on mutual-NN
    landmarks. Its measured behaviour on the full 30-run G3 grid is the thing that motivates
    this function: the recovery rung contributes ``+3.62 +- 1.72`` Top-1 and that gain is
    **flat** -- ``corr(raw, gain) = -0.19``, ``corr(landmark_rate, gain) = -0.21`` over 30
    runs. A gain that does not move when the encoder gets better, does not move when the
    landmark rate moves, and has no fold structure is the signature of an estimator problem
    rather than a representation problem. (And it is why every attempt to raise the landmark
    rate by TRAINING -- T2', T2'', SCORE's source-only episode -- failed to move the score:
    they were pushing a quantity this operator does not consume.)

      * **Scarcity.** ~42 landmarks. The polar factor of ``q^T g + rho I`` sets the remaining
        directions to the identity, so the effective gain is bounded by how much of the map
        those 42 pairs can identify.
      * **Over-parameterisation.** ``d = 64`` while the concept manifold measured at 16 dims:
        2016 rotation parameters, ~120 of which lie in the signal subspace.

    The two are the same statement seen from two sides, so this function fixes them together
    and the 2x2 probe (`scripts/probe_s3r.py`) attributes each half:

      * ``hard_landmarks=False`` + ``rank=None``  -> soft matching, full dimension (defect 1)
      * ``hard_landmarks=True``  + ``rank=16``    -> hard matching, subspace (defect 2)
      * ``hard_landmarks=False`` + ``rank=16``    -> S3R (both)
      * ``hard_landmarks=True``  + ``rank=None``  -> the deployed operator, reproduced

    THE SUBSPACE IS NOT A TRUNCATION, WHICH IS THE POINT. The map is
    ``R = I + V (R_r - I_r) V^T``: identity on the noise complement and free only inside the
    top-``rank`` subspace ``V``. Nothing is deleted -- the recovered cloud keeps all ``d``
    coordinates -- so this cannot be the "we threw away signal" failure that the noise-corrected
    low-rank frame family was falsified for (G-a, 30 runs, `docs/eeg2image_v7_architecture.md`
    §1). What changes is only how many parameters the 42-200 landmarks are asked to identify.

    WEIGHTED PROCRUSTES. ``maximise tr(R^T q^T P g)`` with ``P`` the matching plan. For the
    hard plan ``P`` is ``1/L`` on mutual-NN positions and this IS Procrustes; for the soft plan
    it is Sinkhorn's doubly-stochastic assignment, so every query contributes with weight.
    Returns ``(recovered, diagnostics)``. A degenerate plan, a non-finite solve, or too few
    effective landmarks abstains and returns the input unchanged, so a failure degrades to
    "no recovery for this fold" rather than to a silently bad rotation.
    """
    q = np.asarray(q, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    diag: dict = {"kind": "s3r", "rank": None if rank is None else int(rank),
                  "tau": float(tau), "hard_landmarks": bool(hard_landmarks),
                  "k": int(k), "rho": float(rho)}
    if q.shape[0] < 2 or g.shape[0] < 2:
        return q, {"abstained": True, "reason": "too few rows", **diag}

    qn = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), eps)
    gn = g / np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), eps)
    s = csls_scores(qn, gn, k=k)          # same hubness correction the deployed ladder uses
    # `alpha` swaps the PLAN and nothing else -- not the model class, not the fit, not the
    # scoring. That is deliberate: it makes "the plan changed" the only possible explanation for
    # a change in the output, and it makes `alpha = 0` bit-identical to the shipped operator
    # (verified), so the structural term is measured against a reproduction rather than against
    # a reimplementation. The `alpha == 0` branch is separate rather than falling through with
    # `alpha = 0`, because `_fgw_plan` standardises its input and standardising rescales the
    # effective temperature by the std -- a silent change to a tuned parameter.
    if hard_landmarks:
        plan = _hard_mutual_plan(s)
    elif float(alpha) > 0.0:
        plan = _fgw_plan(s, qn, gn, float(alpha), tau=tau, iters=iters, outer=fgw_outer,
                         de_ref=fgw_de_ref, de_mix=fgw_de_mix, spec_rank=fgw_spec_rank,
                         topo_eps=fgw_topo_eps, di_ref=fgw_di_ref)
    else:
        plan = _sinkhorn_plan(s, tau=tau, iters=iters)
    diag["alpha"] = float(alpha)
    diag["fgw_de_mix"] = float(fgw_de_mix)
    diag["fgw_spec_rank"] = None if fgw_spec_rank is None else int(fgw_spec_rank)
    diag["plan_mass"] = float(plan.sum())
    # Effective number of matched pairs = participation ratio of the unit-mass plan. For the
    # hard plan (`1/L` on `L` mutual-NN entries) this returns exactly `L`, so the two arms
    # are read on one ruler and "soft uses more landmarks" is a measured statement: the
    # deployed operator sees ~30-42 pairs here. `plan_acc` is the plan's mass on the true
    # diagonal, i.e. a soft-accuracy that needs no labels to compute at deployment.
    diag["n_effective"] = float(1.0 / max(float((plan ** 2).sum()), eps))
    diag["plan_acc"] = float(np.trace(plan)) if plan.shape[0] == plan.shape[1] else None
    n_landmark = int(np.count_nonzero(plan))
    if n_landmark < int(min_landmarks):
        return q, {"abstained": True, "reason": "validated landmarks below floor",
                   "n_landmarks": n_landmark, **diag}

    qc, gc = q, g
    if moment:
        # Per-dimension moment matching, as deployment does, so the two clouds are in a
        # comparable frame before the rotation is solved for.
        qc = (q - q.mean(0, keepdims=True)) / (q.std(0, keepdims=True) + 1e-5)
        gc = (g - g.mean(0, keepdims=True)) / (g.std(0, keepdims=True) + 1e-5)

    m = qc.T @ (plan @ gc) + float(rho) * np.eye(q.shape[1])
    if rank is not None and int(rank) < q.shape[1]:
        c = qc.T @ qc + gc.T @ gc
        vals, vecs = np.linalg.eigh(c)
        r = max(2, min(int(rank), q.shape[1]))
        v = vecs[:, -r:]                   # top-`rank` signal directions
        mr = v.T @ m @ v
        ur, sr, vr = np.linalg.svd(mr)
        rr = ur @ vr                       # polar factor INSIDE the subspace
        rot = np.eye(q.shape[1]) + v @ (rr - np.eye(r)) @ v.T
        diag["subspace_rank"] = int(r)
        diag["subspace_energy"] = float(vals[-r:].sum() / max(vals.sum(), eps))
    else:
        u, sv, vt = np.linalg.svd(m)
        rot = u @ vt
        diag["subspace_rank"] = int(q.shape[1])

    out = qc @ rot
    if not np.isfinite(out).all():
        return q, {"abstained": True, "reason": "non-finite recovery", **diag}
    if moment:
        out = moment_match(out, g)
    diag.update({"abstained": False, "n_landmarks": n_landmark,
                 "orthogonality_err": float(np.abs(rot @ rot.T - np.eye(q.shape[1])).max())})
    return out, diag


def coordinate_recovery(
    q: np.ndarray,
    g: np.ndarray,
    k: int = 10,
    rho: float = 0.1,
    moment: bool = True,
    orientation: bool = True,
    max_landmarks: int | None = 160,
    min_landmark_rate: float = 0.0,
    **_ignored,
) -> tuple[np.ndarray, dict]:
    """SCORE Eq. 3-9 on frozen features. Returns `(recovered_queries, diagnostics)`.

    Falls back to a clearly-labelled no-op if the shared `epd` implementation is not
    importable, rather than silently returning the input as if recovery had run.
    """
    try:
        root = str(config.EPD_ROOT)
        if root not in sys.path:
            sys.path.insert(0, root)
        from epd.recover import recover as _recover  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "coordinate recovery needs `eeg-retrieval/epd` on sys.path; it was not "
            f"importable ({exc}). Set config.EPD_ROOT or run with recovery disabled."
        ) from exc

    qt = torch.as_tensor(q, dtype=torch.float32)
    gt = torch.as_tensor(g, dtype=torch.float32)
    out, diag = _recover(qt, gt, k=k, rho=rho, moment=moment, orientation=orientation,
                         max_landmarks=max_landmarks, min_landmark_rate=min_landmark_rate)
    return out.numpy(), diag


def calibrate(
    q: np.ndarray,
    g: np.ndarray,
    *,
    center: bool = False,
    whiten: bool = False,
    csls: bool = False,
    recovery: bool = False,
    k: int = 10,
    rho: float = 0.1,
    min_landmark_rate: float = 0.0,
    recovery_fn=None,
) -> tuple[np.ndarray, dict]:
    """Compose the label-free calibration steps and return the final scores + report.

    The order is ``center -> whiten -> recovery -> CSLS``, and it is not arbitrary:

      * centring and whitening are both changes of the query coordinates, so they have to
        happen before anything that measures angles in those coordinates. ``whiten``
        centres internally, so ``center`` alongside it is redundant (not harmful) -- the
        rung exists to attribute the gain to the mean rather than to the covariance;
      * SCORE's ``recover`` fits its Procrustes map from mutual-NN pseudo-pairs and
        recomputes CSLS *internally* while doing so, so running CSLS before it would
        be thrown away;
      * the final CSLS is a re-scoring of the recovered similarity matrix, which is
        only meaningful once the coordinates are final.

    (An earlier version of this docstring claimed the opposite order -- CSLS first. The
    code has always run recovery first; the comment was wrong, not the code.)
    """
    diag: dict = {"center": center, "whiten": whiten, "csls": csls, "recovery": recovery}
    qq = q
    if center:
        qq, cdiag = center_queries(qq)
        diag["center_diag"] = cdiag
    if whiten:
        qq, wdiag = saw_whiten(qq)
        diag["whiten_diag"] = wdiag
    if recovery:
        # `recovery_fn` swaps the operator without touching the ladder's composition, so
        # "the operator changed" cannot be confused with "the rung changed".
        rec = coordinate_recovery if recovery_fn is None else recovery_fn
        qq, rdiag = rec(qq, g, k=k, rho=rho,
                        min_landmark_rate=min_landmark_rate)
        diag["recovery_diag"] = rdiag
    if csls:
        scores = csls_scores(qq, g, k=k)
    else:
        qn = qq / np.maximum(np.linalg.norm(qq, axis=-1, keepdims=True), 1e-8)
        gn = g / np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-8)
        scores = qn @ gn.T
    return scores, diag


# ------------------------------------------------ multi-repetition refinement (T2)
def _whiten_from_cloud(x: np.ndarray, shrink: float = 0.1, max_cond: float = 1e3,
                       eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray, dict]:
    """``(mu, W, diag)`` of the SAW whitening map estimated from a cloud `x`."""
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean(axis=0, keepdims=True)
    xc = x - mu
    n, d = xc.shape
    cov = (xc.T @ xc) / max(1, n - 1)
    if shrink > 0:
        cov = (1.0 - shrink) * cov + shrink * (np.trace(cov) / d) * np.eye(d)
    cov = cov + eps * np.eye(d)
    vals, vecs = np.linalg.eigh(cov)
    lo = max(float(vals.max()) / max_cond, eps)
    vals = np.maximum(vals, lo)
    w = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
    return mu.astype(np.float64), w, {"cond": float(vals.max() / vals.min()),
                                      "n_samples": int(n), "d": int(d),
                                      "rank_deficient": bool(n - 1 < d)}


def fuse_scores(
    score_list: list[np.ndarray],
    normalize: bool = True,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Sum-rule late fusion of per-route score matrices (v6 pillar B, deployment side).

    This is the numpy twin of ``losses.contrastive.score_fusion_loss``, and the two must
    agree. The training term fuses *within each subject block* because CSLS's density is a
    retrieval-set property; at eval there is exactly one held-out subject, so the whole
    ``(C, C)`` matrix IS the block -- the same operator over one block rather than several.

    ``normalize`` divides each route's matrix by its own standard deviation, a single
    GLOBAL scalar per route. The reason is in `score_fusion_loss`: a CSLS-corrected matrix
    is unbounded while a cosine matrix lies in ``[-1, 1]``, so an unnormalised sum weights
    the routes by the scale of their scores rather than by their information. The failure
    is not hypothetical -- `structural_scores` records a 35.50 -> 18.00 collapse when two
    experts on incomparable scales were added unweighted.

    Weights default to uniform and are NOT fitted here. Fitting them on the fold being
    reported would be selecting on the test set; CORTIVA's own weight sweep landed inside
    interval-crossing-zero of each other with uniform on top, so the sum rule is
    first-order insensitive to them.
    """
    if not score_list:
        raise ValueError("fuse_scores needs at least one score matrix")
    w = np.ones(len(score_list)) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (len(score_list),):
        raise ValueError(f"weights must have one entry per route, got {w.shape} for "
                         f"{len(score_list)} routes")
    total: np.ndarray | None = None
    for s, wi in zip(score_list, w):
        s = np.asarray(s)
        if s.ndim != 2 or s.shape[0] != s.shape[1]:
            raise ValueError(f"each score matrix must be square (C, C), got {s.shape}")
        t = s / max(float(s.std()), 1e-6) if normalize else s
        total = wi * t if total is None else total + wi * t
    return total


def rep_cloud_scores(
    z_reps: np.ndarray,
    g: np.ndarray,
    k: int = 10,
    rho: float = 0.1,
    shrink: float = 0.1,
    min_landmark_rate: float = 0.0,
    recovery_fn=None,
    rep_subsample: int | None = None,
    src_means: np.ndarray | None = None,
    src_mix: float = 0.0,
    src_ref_mode: str = "eeg",
    src_calib: bool = False,
    spec_rank: int | None = None,
    spec_from_src: bool = False,
    raw_metric: bool = False,
    topo_eps: float | None = None,
    topo_auto: bool = False,
    topo_q: float = 0.10,
    rep_blocks: int = 0,
    gallery_di_ref: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """T2: the repetition cloud's own moment matching + recovery, scored with CSLS.

    ``z_reps`` is ``(C, R, d)`` -- the encoder's embeddings of the ``R`` UNAVERAGED
    repetitions of each of the ``C`` test concepts -- and ``g`` is ``(C, d)``, the image
    gallery. Returns ``(scores, recovery_diag)``.

    ONE implementation, because there were two. `scripts/probe_c2_levers.py` inlined this
    operator and `scripts/probe_rep_dose.py` carried a copy annotated "verbatim from
    `probe_c2_levers`"; a project that has already been bitten by a formula living in three
    places should not acquire a third copy the moment the operator becomes load-bearing.
    Both probes and `scripts/run_eval.py` now call THIS function.

    WHAT THE OPERATOR IS, AND WHY IT IS NOT `refine_with_reps`
    --------------------------------------------------------
    A strictly weaker-gated sibling of `refine_with_reps`, and the two are NOT
    interchangeable -- they score differently (33.50 vs 29.50 Top-1 on sub-08), so mixing
    them up silently changes the number:

      * moment matching is fitted on the whole ``(C*R, d)`` cloud, i.e. on ``C*R`` = 16000
        samples rather than the ``C`` = 200 the averaged deployment query has. That is the
        whole point of using repetitions: it is the same mean/covariance estimator with
        ``R`` times the sample count, so its variance is lower, not its bias.
      * the repetitions are then averaged and passed through the deployed
        `coordinate_recovery` -- the SAME recovery as the T1 rung, so a T1-vs-T2 comparison
        is mechanism-for-mechanism.
      * **no agreement gate.** `refine_with_reps` trusts a pseudo-pair only when the
        repetitions agree; here the disagreement is 0.09 on sub-08 (see
        `outputs/probe/c2_levers_v5a1k20.json`), so the gate discards almost everything it
        is given. Ungated is the variant that carries the gain, and the two are reported
        side by side rather than one silently standing in for the other.

    `shrink` stays at the probes' 0.1 so the number that motivated sharing this code --
    33.50 alone, 39.50 fused with the T1 best rung on sub-08 -- is reproducible from here.
    `min_landmark_rate` is forwarded to the recovery so `--min-landmark-rate` gates this
    path the same way it gates the T1 ladder; its default of 0.0 is the probes' behaviour.
    """
    z = np.asarray(z_reps, dtype=np.float64)
    if z.ndim != 3:
        raise ValueError(f"z_reps must be (C, R, d), got {z.shape}")
    C, R, d = z.shape
    # `rep_subsample` keeps only the first R' repetitions. This is the VARIANCE-GATING KNOB of
    # v10 stage 1, and it is deliberately a *prefix* slice rather than a random draw: the
    # question is whether the structural term's gain falls off as the target-side geometry
    # estimate gets noisier, and a fixed prefix makes the sweep exactly reproducible. With R'=1
    # this operator degenerates to the averaged-query condition the T1 rung already measures.
    if rep_subsample is not None and 0 < int(rep_subsample) < R:
        R = int(rep_subsample)
        z = z[:, :R]
    g = np.asarray(g, dtype=np.float64)
    zf = z.reshape(C * R, d)
    mu, w_map, _ = _whiten_from_cloud(zf, shrink=shrink)
    q_bar = (z.mean(axis=1) - mu) @ w_map
    # `recovery_fn` lets the recovery operator be swapped without touching this estimator.
    # It exists because the operator is used at TWO rows of the ladder -- the T1 rung and
    # here -- so a change to it has to be applied in both places or the two rows stop being
    # comparable, and because `coordinate_recovery`'s contribution is measured as FLAT
    # (+3.62 +- 1.72, uncorrelated with raw and with landmark rate over 30 runs), which is an
    # estimator defect that a better operator may fix.
    rec = coordinate_recovery if recovery_fn is None else recovery_fn
    # ---- M1: the source-metric template, and the diagnostic that predicts whether it can work.
    #
    # `src_means` is `(S, C, d)` -- `S` SOURCE subjects' per-concept mean embeddings in the
    # encoder's raw space, on the SAME C concepts (THINGS-EEG2 uses one shared 200-concept test
    # set, so source and target concept indices line up by construction, not by alignment).
    # They are whitened here by THIS target's map so the template lives in the same space as
    # `q_bar`; whitening them by their own subjects' maps would put the template in a different
    # frame and the blend would be a geometric non-sequitur.
    de_ref = None
    m1_diag: dict = {}
    if src_means is not None and src_mix > 0.0:
        sm = np.asarray(src_means, dtype=np.float64)
        zs = (sm - mu) @ w_map                                  # (S, C, d) in target's frame
        zs = zs / np.clip(np.linalg.norm(zs, axis=2, keepdims=True), 1e-9, None)
        de_ref = np.mean([_sq_cos_dist(zs[i]) for i in range(zs.shape[0])], axis=0)
        # Is the template an estimate of the SAME metric? Measured, not assumed: correlation
        # between the source-template and the target's own `de`. High means the template carries
        # the target's signal (variance reduction can help); low means blending it in would
        # import a differently-shaped metric and must hurt. This single number is what predicts
        # whether `src_mix > 0` can pay off, so it is reported before any accuracy is read.
        qn_d = q_bar / np.clip(np.linalg.norm(q_bar, axis=1, keepdims=True), 1e-9, None)
        de_t = _sq_cos_dist(qn_d)
        iu = np.triu_indices(de_t.shape[0], 1)
        m1_diag["src_vs_target_metric_corr"] = float(np.corrcoef(de_t[iu], de_ref[iu])[0, 1])
        m1_diag["src_vs_target_metric_corr_n"] = int(iu[0].size)
        m1_diag["src_n_subjects"] = int(sm.shape[0])
        # THE DECISIVE MECHANISM TEST. M1's gain is +15pp and FLAT in R, so it is NOT variance
        # reduction (that would pay off at low R and vanish at high R). The competing explanation
        # is that the source template is simply a BETTER PROXY FOR THE GALLERY METRIC `di` than
        # the target's own estimate -- unsurprising, since the encoder was TRAINED on source
        # subjects to align exactly this EEG metric to the image metric. Those two stories imply
        # opposite architectures (ensemble the target's estimate vs. replace it with the source
        # frame), so the correlation against `di` is recorded and is the number that decides.
        gn_d = np.asarray(g, dtype=np.float64)
        gn_d = gn_d / np.clip(np.linalg.norm(gn_d, axis=1, keepdims=True), 1e-9, None)
        di_c = _sq_cos_dist(gn_d)
        iu2 = np.triu_indices(di_c.shape[0], 1)
        # ---- THE ADJUDICATION ARMS (2026-10-05). `de_ref` normally IS the source subjects'
        # EEG metric, so `src_mix=1` substitutes the source EEG view for the target's own view
        # of the query-side metric. Whether the +15pp is a LEGITIMATE shared-concept-frame effect
        # or the source EEG acting as an ANSWER KEY is decided by WHICH of the following four
        # references it needs to be to reproduce the gain:
        #   "eeg"     -- the source EEG metric (the arm under adjudication; the default)
        #   "gallery" -- `di_c`, the metric of the PUBLIC image gallery, which carries ZERO EEG
        #                information and is available at test. If this reproduces the gain, the
        #                effect needs no source EEG at all and the arm is not a neural leak.
        #   "self"    -- `de_t`, the target's own metric, i.e. a NO-OP reference (mix=1 == mix=0).
        #                Guards bit-identity of the plumbing.
        #   "rand"    -- a fixed-seed random symmetric metric, the floor that any geometry must
        #                beat; a gain here would mean the structural term is being driven by
        #                something other than the reference.
        # Every arm is a SINGLE-argument change from "eeg", so the contrast is paired by
        # construction. `src_ref_mode="eeg"` is bit-identical to the shipped operator.
        if src_ref_mode == "gallery":
            de_ref = di_c
        elif src_ref_mode == "self":
            de_ref = de_t
        elif src_ref_mode == "rand":
            rng = np.random.default_rng(0)
            rmat = rng.standard_normal(di_c.shape)
            de_ref = (rmat + rmat.T) / 2.0
        elif src_ref_mode != "eeg":
            raise ValueError(f"unknown src_ref_mode {src_ref_mode!r}")
        m1_diag["src_ref_mode"] = str(src_ref_mode)
        de_used = (1.0 - src_mix) * de_t + src_mix * de_ref
        m1_diag["corr_de_src_di"] = float(np.corrcoef(de_ref[iu2], di_c[iu2])[0, 1])
        m1_diag["corr_de_target_di"] = float(np.corrcoef(de_t[iu2], di_c[iu2])[0, 1])
        m1_diag["corr_de_used_di"] = float(np.corrcoef(de_used[iu2], di_c[iu2])[0, 1])

    # ---- C1: SOURCE-CALIBRATED METRIC TRANSFER (docs/eeg2image_v10_m1_theory.md §9) --------
    #
    # WHY THIS IS THE RIGHT REPAIR FOR WHAT M1 GOT WRONG. The permutation control showed M1's
    # +15pp was an INDEX correspondence, not geometry: `de_src[i,j]` carries the identity of
    # concept i in its own subscript, `di` carries the gallery slot in its own, and `de_src` is
    # a near-perfect proxy for `di` (measured +0.78), so the FGW optimum became the IDENTITY plan
    # -- which is the answer key under this protocol. Destroying the correspondence (shuffle the
    # concept axis) turned +15.4pp into -14.5pp.
    #
    # C1 keeps the part that was real and removes the part that was leakage. The real part is the
    # measured fact `corr(de_src, di) = 0.78 > corr(de_target, di) = 0.59`: the source subjects'
    # metric IS a better proxy for the gallery metric. C1 learns that relationship ON THE SOURCES,
    # where using the source<->gallery correspondence is legitimate inductive transfer, and then
    # applies the learned map to the TARGET'S OWN metric.
    #
    #     phi = argmin_phi  sum over source subjects, over (i,j) : ( phi(de_s[i,j]) - di[i,j] )^2
    #     de_used = phi( de_target )
    #
    # `phi` is applied ELEMENTWISE, i.e. it never sees an index. Its input is the target's metric
    # in the target's own frame and its output stays in that frame: there is no cross-frame
    # correspondence anywhere in the application, so the M1 failure mode is absent BY CONSTRUCTION
    # rather than by an added penalty. The correspondence enters only inside the fitting, on source
    # data, which is what makes this transfer rather than leakage.
    #
    # PRE-REGISTERED CRITERION, and it can fail on the first fold: corr(phi(de_target), di) must
    # EXCEED corr(de_target, di). If phi does not transfer across subjects the correlation will not
    # move and the whole direction dies here for the cost of one GPU job. Note the failure mode is
    # informative in the other direction too: a map that transfers but does NOT raise the
    # correlation says the source/target metric gap is noise, not shape.
    if src_calib and src_means is not None:
        from sklearn.isotonic import IsotonicRegression
        sm_c = np.asarray(src_means, dtype=np.float64)
        zs_c = (sm_c - mu) @ w_map
        zs_c = zs_c / np.clip(np.linalg.norm(zs_c, axis=2, keepdims=True), 1e-9, None)
        gn_d2 = np.asarray(g, dtype=np.float64)
        gn_d2 = gn_d2 / np.clip(np.linalg.norm(gn_d2, axis=1, keepdims=True), 1e-9, None)
        di_flat = _sq_cos_dist(gn_d2).ravel()
        des_flat = np.concatenate([_sq_cos_dist(zs_c[i]).ravel()
                                   for i in range(zs_c.shape[0])])
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        iso.fit(des_flat, np.tile(di_flat, zs_c.shape[0]))
        qn_c = q_bar / np.clip(np.linalg.norm(q_bar, axis=1, keepdims=True), 1e-9, None)
        de_t_c = _sq_cos_dist(qn_c)
        de_cal = iso.predict(de_t_c.ravel()).reshape(de_t_c.shape)
        di_mat = di_flat.reshape(de_t_c.shape)
        iu3 = np.triu_indices(de_t_c.shape[0], 1)
        m1_diag["corr_de_target_di_calib"] = float(np.corrcoef(de_cal[iu3], di_mat[iu3])[0, 1])
        m1_diag["corr_de_target_di_precalib"] = float(np.corrcoef(de_t_c[iu3],
                                                                  di_mat[iu3])[0, 1])
        m1_diag["c1_transfer_gain_corr"] = (m1_diag["corr_de_target_di_calib"]
                                            - m1_diag["corr_de_target_di_precalib"])
        de_ref = de_cal
        src_mix = 1.0
    # ---- P1: THE SPECTRAL RANK PRIOR, estimated from the SOURCES --------------------------
    #
    # WHAT P1 IS REPAIRING. Two falsifications bracket the whole space of "use the source data":
    #
    #   M1 (job 645455): source metric with its indices participating -> +15.4pp, and the
    #     permutation control (job 645593) turned that into -14.5pp with corr(de_src, di)
    #     collapsing 0.782 -> 0.001. The gain was an INDEX correspondence -- `de_src[i,j]` carries
    #     concept i's identity in its subscript, so the FGW optimum became the identity plan,
    #     which is the answer key here. LEAKAGE.
    #
    #   C1 (job 645606): a monotone elementwise map phi fitted on sources -> +0.00pp Top-1 and
    #     +0.0019 correlation. Provably bounded: a monotone map cannot reorder, so the most it can
    #     buy is corr - Spearman, and the measured post-calibration correlation (0.594) sits on
    #     that ceiling. The 0.593 -> 0.782 gap is ORDER, not SHAPE. NO HEADROOM.
    #
    # The gap between those two closes onto one shape of usable source information: something that
    # is NEITHER an index correspondence NOR a monotone rescaling. The eigenvalue SPECTRUM is
    # exactly that -- relabelling concepts conjugates the metric by a permutation, which leaves
    # the spectrum invariant, so a scalar read off the spectrum carries no correspondence at all
    # and cannot leak. And because it is not monotone, the C1 ceiling does not apply.
    #
    # The prior is the intrinsic dimension: the source subjects' metrics say how many directions
    # of `de` are signal. `de` is then replaced by its top-k eigen-approximation, k taken from the
    # SOURCES rather than tuned on the target. With d_M measured at 8-16 out of d=64, ~48
    # directions are estimation noise; truncation DELETES them where stage-2 shrinkage INFLATED
    # them, which is why the stage-2 failure does not predict this one.
    #
    # PRE-REGISTERED CRITERION: Top-1(k_used) > Top-1(rank = full), paired per fold. k = full is
    # bit-identical to the shipped m=0 cell (asserted), so the comparison is against a
    # reproduction and not a reimplementation.
    p1_diag: dict = {}
    if spec_from_src and src_means is not None:
        sm_r = np.asarray(src_means, dtype=np.float64)
        zs_r = (sm_r - mu) @ w_map
        zs_r = zs_r / np.clip(np.linalg.norm(zs_r, axis=2, keepdims=True), 1e-9, None)
        ranks = np.array([_eff_rank(_sq_cos_dist(zs_r[i])) for i in range(zs_r.shape[0])])
        qn_r = q_bar / np.clip(np.linalg.norm(q_bar, axis=1, keepdims=True), 1e-9, None)
        ranks = np.append(ranks, _eff_rank(_sq_cos_dist(qn_r)))
        p1_diag["eff_rank_src_mean"] = float(ranks[:-1].mean())
        p1_diag["eff_rank_src_sd"] = float(ranks[:-1].std(ddof=1)) if len(ranks) > 2 else 0.0
        p1_diag["eff_rank_target"] = float(ranks[-1])
        # Rounded UP and floored at 8 (the measured lower end of d_M): participation ratio is a
        # soft count and tends to sit just below the true rank, and a prior that under-cuts the
        # signal subspace would discard the structure the term exists to use.
        spec_rank = int(max(8, np.ceil(float(ranks[:-1].mean()))))
        p1_diag["spec_rank_from_src"] = int(spec_rank)

    # ---- P2: THE SAME PRIOR, MOVED OUT OF THE WHITENED SPACE (docs §9.3) -------------------
    #
    # WHY P1 FAILED, AND WHY THAT FALSIFICATION POINTED HERE RATHER THAN AT "STOP". P1 demanded
    # that the whitened metric `de` be low rank. It is not, and the pipeline is why: `q_bar` has
    # already been through `_whiten_from_cloud`, so its covariance is (shrunk toward) identity.
    # A whiten step is a map that FLATTENS a spectrum by construction, so truncating the whitened
    # metric discards directions the whitener just manufactured -- 0/30 folds positive, and the
    # structural term's sign inverted. The premise "`de` is low rank (d_M ~ 8-16)" is true only
    # in the space where it was measured: the RAW encoder output, before whitening. P1 read the
    # number in one space and acted on another.
    #
    # The repair is not a new prior -- it is the SAME spectral prior evaluated where the low-rank
    # claim actually holds. `q_raw` is the unwhitened concept mean, `de_raw` its metric, and the
    # effective rank of the two is recorded side by side so the premise is checked on every run
    # rather than assumed: if `eff_rank(de_raw)` is not materially below `eff_rank(de_whitened)`
    # the whole family is dead and the `p2_premise_ok` flag says so.
    #
    # KILL SWITCH (pre-registered, docs §9.3): the family is abandoned if the unwhitened metric's
    # effective rank exceeds 50. That is the same test the P1 falsification was read through, so
    # a positive result can never be produced by quietly moving the threshold.
    #
    # NO-LEAKAGE: `q_raw` is the TARGET's own feature, `de_raw` its own metric, and the truncation
    # index set is a permutation-invariant spectral object. Nothing here is indexed by concept, so
    # the M1 index-leakage channel is absent by construction.
    p2_diag: dict = {}
    if raw_metric:
        qn_w = q_bar / np.clip(np.linalg.norm(q_bar, axis=1, keepdims=True), 1e-9, None)
        de_w = _sq_cos_dist(qn_w)
        q_raw = z.mean(axis=1)                            # BEFORE the whitener, C x d
        qn_raw = q_raw / np.clip(np.linalg.norm(q_raw, axis=1, keepdims=True), 1e-9, None)
        de_raw = _sq_cos_dist(qn_raw)
        p2_diag["eff_rank_de_whitened"] = _eff_rank(de_w)
        p2_diag["eff_rank_de_unwhitened"] = _eff_rank(de_raw)
        p2_diag["p2_premise_ok"] = bool(_eff_rank(de_raw) <= 50.0)
        # The reference the structural term is fed. `spec_rank` (when set) is applied INSIDE
        # `_fgw_plan`, i.e. to this unwhitened object, which is exactly the relocation P2 claims
        # and is why the same `--fgw-spec-rank` number means something different here.
        de_ref = de_raw
        src_mix = 1.0
        if spec_rank is not None:
            p2_diag["spec_rank_applied_to"] = "unwhitened"

    # ---- P3: THE TOPOLOGICAL REFERENCE (docs §9.4) ----------------------------------------
    #
    # The metric prior's failure was that the FINE magnitudes of `de` are whitening-flattened and
    # noisy. The coarsest thing that survives that is CONNECTIVITY: throw away every distance and
    # keep only which concepts are within `eps` of each other. That object -- the eps-threshold
    # graph -- is the topological structural reference, and it is `0/1`, so a whitener cannot
    # flatten it and it has no spectrum to truncate.
    #
    # `eps` is a SINGLE SCALAR per domain, read off that domain's OWN distance multiset as a
    # fixed quantile (`_topo_eps_from`, default q=0.10), so both graphs have the same DENSITY
    # while living at their own SCALE. It is permutation-invariant, is computed without any test
    # query or label, and therefore carries no index correspondence -- the M1 channel again cannot
    # open. Using the GALLERY (not the query) to fix the scale is deliberate: the gallery defines
    # the reference structure, and the query is then read in the gallery's topology rather than
    # the gallery being read in the query's.
    if topo_auto or (topo_eps is not None):
        if raw_metric:
            qn_t = qn_raw
        else:
            qn_t = q_bar / np.clip(np.linalg.norm(q_bar, axis=1, keepdims=True), 1e-9, None)
        gn_t = np.asarray(g, dtype=np.float64)
        gn_t = gn_t / np.clip(np.linalg.norm(gn_t, axis=1, keepdims=True), 1e-9, None)
        # TWO scales, one per domain. See `_fgw_plan`'s P3 note: the EEG cloud and the image
        # gallery have different distance scales, so a single shared eps would build a blob on
        # one side and a singleton cloud on the other. Each side gets the characteristic scale of
        # its OWN multiset, which is what makes the comparison topological (shape) rather than
        # metric (magnitude). `--fgw-topo-eps` overrides BOTH with one number, which is the
        # shared-scale ablation and is labelled as such in the diagnostics.
        if topo_eps is not None:
            eps_de = eps_di = float(topo_eps)
            p2_diag["topo_mode"] = "shared_eps"
        else:
            eps_de = _topo_eps_from(qn_t, q=float(topo_q))
            eps_di = _topo_eps_from(gn_t, q=float(topo_q))
            p2_diag["topo_mode"] = "auto_per_domain"
        p2_diag["topo_q"] = float(topo_q)
        de_topo = _topo_graph(_sq_cos_dist(qn_t), eps_de)
        di_topo = _topo_graph(_sq_cos_dist(gn_t), eps_di)
        p2_diag["topo_eps_query"] = eps_de
        p2_diag["topo_eps_gallery"] = eps_di
        p2_diag["topo_query_edges"] = int(de_topo.sum())
        p2_diag["topo_gallery_edges"] = int(di_topo.sum())
        # Overlap of the two graphs is a LABEL-FREE proxy for whether the topology agrees at all
        # (both are 0/1 and symmetric, so the Jaccard needs no correspondence beyond the shared
        # concept axis, which is the legitimate axis here -- the query and the gallery index the
        # SAME 200 test concepts). It is a diagnostic, not a gate.
        inter = float((de_topo * di_topo).sum())
        union = float(((de_topo + di_topo) > 0).sum())
        p2_diag["topo_graph_jaccard"] = inter / max(union, 1.0)
        topo_pair: tuple[float, float] | None = (eps_de, eps_di)
        # A binary adjacency has no meaningful spectrum to cut: `spec_rank` would be a no-op at
        # best and destructive at worst, so it is switched off here and the fact is recorded
        # rather than left to a reader to infer from a shared flag name.
        if spec_rank is not None:
            p2_diag["spec_rank_disabled_by_topo"] = True
            spec_rank = None
    else:
        topo_pair = None

    # ---- L2/L3/L4: STRUCTURE ENSEMBLE + RELIABILITY-WEIGHTED FUSION (v11) ------------------
    #
    # THE MEASUREMENT THIS IMPLEMENTS. A CPU/GPU probe over the ten banked encoders
    # (`scripts/probe_fusion.py`, job 645697) asked whether POOLING independent structure
    # estimates raises their agreement with a held-out view's structure. It does:
    #
    #     cross-modal, pool of K  0.2093 (K=1) -> 0.2471 (K=8), MONOTONE, 10/10 folds
    #     the correspondence-destroying control stays at ~0.000 across all K
    #     within-EEG leave-one-view-out  0.678 -> 0.806
    #
    # The control is the load-bearing part: an UNORDERED pool gains nothing, so the gain is
    # genuinely the shared structure and not an artefact of averaging. That is what licenses
    # building this, and it is the same control discipline that killed M1.
    #
    # WHAT IS BEING POOLED, AND WHY IT CARRIES NO INDEX LEAKAGE. The target's own `R`
    # repetitions are split into `B` contiguous blocks; each block is an INDEPENDENT estimate of
    # the same 200 concepts, so their metrics live in one shared concept order and averaging them
    # is index-consistent BY CONSTRUCTION. This is categorically different from the M1 template,
    # whose index correspondence came from a DIFFERENT subject aligned to the gallery: there the
    # index was an external correspondence and it leaked; here it is the same trials of the same
    # subject, which is exactly what T2 already relies on. The prior we are replacing (P1/P2/A4)
    # was permutation-invariant too, and this one keeps that property because a block average
    # commutes with relabelling -- asserted in smoke §16.
    #
    # WHY FUSION AND NOT TRUNCATION, GIVEN THAT FOUR REDUCTIONS ALREADY FAILED. P1, A1/A2 and A4
    # all REDUCED the structural object (drop eigenvalues, binarise to a graph). The measured
    # lesson of job 645630 was that the term is monotonically better with a RICHER object
    # (on-off +3.85pp whitened vs +2.50pp unwhitened), so reduction was the wrong direction.
    # Fusion is the opposite operation: it does not delete a single direction, it averages away
    # the estimation noise that the recovery rung was measured to be limited by (its gain is flat
    # in input quality, corr ~ -0.19 over 30 runs -- the signature of an estimator-noise limit).
    #
    # L3's WEIGHTS ARE MEASURED, NOT TUNED. Each block's weight is its agreement with the
    # leave-one-out mean of the others, so a block whose metric is corrupted contributes little.
    # No hyperparameter is fitted on the target's labels, and the weights are recorded so the
    # "the mechanism responds to input quality" prediction can be read off directly.
    struct_diag: dict = {}
    if int(rep_blocks) > 1:
        B = int(min(int(rep_blocks), R))
        edges = np.linspace(0, R, B + 1).astype(int)
        bm, bc, bmeans = [], [], []
        for b in range(B):
            sl = slice(int(edges[b]), int(edges[b + 1]))
            if sl.start >= sl.stop:
                continue
            qb = (z[:, sl].mean(axis=1) - mu) @ w_map
            qbn = qb / np.clip(np.linalg.norm(qb, axis=1, keepdims=True), 1e-9, None)
            bm.append(_sq_cos_dist(qbn))
            bmeans.append(qbn)
            bc.append(int(sl.stop - sl.start))
        if len(bm) > 1:
            # leave-one-out agreement -> weights (inverse residual, normalised)
            rel = []
            for b in range(len(bm)):
                others = [bm[j] for j in range(len(bm)) if j != b]
                ref = np.mean(others, axis=0)
                rel.append(float(np.linalg.norm(bm[b] - ref)
                                 / max(float(np.linalg.norm(ref)), 1e-12)))
            inv = 1.0 / (np.asarray(rel) + 1e-6)
            wgt = inv / inv.sum()
            de_fused = np.tensordot(wgt, np.asarray(bm), axes=(0, 0))
            struct_diag["struct_fuse_blocks"] = len(bm)
            struct_diag["struct_fuse_block_sizes"] = bc
            struct_diag["struct_fuse_weights"] = [float(x) for x in wgt]
            struct_diag["struct_fuse_block_resid"] = [float(x) for x in rel]
            struct_diag["struct_fuse_weight_range"] = float(wgt.max() - wgt.min())
            # the "does the mechanism respond to input quality" check: spread of the weights is
            # the mechanism's sensitivity, and it is recorded rather than assumed non-zero.
            struct_diag["struct_fuse_is_flat"] = bool(wgt.max() - wgt.min() < 1e-3)
            if not struct_diag["struct_fuse_is_flat"]:
                de_ref = de_fused
                src_mix = 1.0

    q_rec, diag = rec(q_bar, g, k=k, rho=rho, min_landmark_rate=min_landmark_rate,
                      fgw_de_ref=de_ref, fgw_de_mix=(src_mix if de_ref is not None else 0.0),
                      fgw_spec_rank=spec_rank, fgw_topo_eps=topo_pair,
                      fgw_di_ref=gallery_di_ref)
    diag.update(m1_diag)
    diag.update(p1_diag)
    diag.update(p2_diag)
    diag.update(struct_diag)
    diag["rep_subsample"] = R
    diag["src_mix"] = float(src_mix)
    # Recorded because `shrink` turned out to be load-bearing: the v10 stage-1 sweep showed the
    # structural gain is GATED by how good this whitening estimate is, and `shrink` is the one
    # free regulariser in it. A report that does not carry the value cannot be compared to one
    # that used a different one.
    diag["whiten_shrink"] = float(shrink)
    # G1: whether a precomputed gallery-side metric reference reached the FGW term. Recorded
    # because a silently-dropped argument is this project's most expensive recurring bug
    # (the A4 topo knob died in `**kw` once and looked like a *result*): a `gallery_di_ref`
    # that did not arrive would make the G1 cell bit-identical to its twin.
    diag["gallery_di_ref"] = bool(gallery_di_ref is not None)
    return csls_scores(q_rec, g, k=k), diag


def cloud_metric_views(z_reps: np.ndarray, shrink: float = 0.1,
                       rep_blocks: int = 8,
                       mode: str = "cont") -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """One route's repetition cloud, exposed as a LIST OF METRIC VIEWS plus its pooled query.

    Why this exists rather than being read out of `rep_cloud_scores`: the O2-MVE estimator
    (docs/eeg2image_v13_croma.md §7) fuses metric views ACROSS view families, and it can only do
    that if the per-family METRICS are addressable. A metric is `(C, C)` in every family, so
    metrics pool across families whose embedding spaces are mutually incompatible -- the `(C, d)`
    query vectors cannot be, and averaging them would be a geometric non-sequitur. This is the
    whole reason the estimator must act on the order-2 object and not on the features.

    `mode` selects the PARTITION of the `R` repetitions, and it is the new axis of this function:
      * `cont`   -- `B` contiguous slices (what `rep_cloud_scores` already fuses);
      * `stride` -- interleaved, block `b` takes repetitions `b::B`. A DIFFERENT view family, not
                    a relabelling: equipment drift is slowly varying in trial index, so contiguous
                    blocks each carry a different drift offset while strided blocks spread the same
                    drift across all of them. The two families' errors are therefore not the same
                    random variable, which is what makes pooling them an increase in view count
                    rather than a duplicate;
      * `rand`   -- a seeded random partition, the unbiased control between the two.
    Returns `(views, q_bar, w_map)`: `views` is a list of `(C, C)` estimates, `q_bar` the pooled
    whitened query `(C, d)` the recovery needs, `w_map` the whitening map.
    """
    z = np.asarray(z_reps, dtype=np.float64)
    if z.ndim != 3:
        raise ValueError(f"z_reps must be (C, R, d), got {z.shape}")
    C, R, d = z.shape
    mu, w_map, _ = _whiten_from_cloud(z.reshape(C * R, d), shrink=shrink)
    B = int(min(int(rep_blocks), R))
    if str(mode) == "stride":
        parts = [np.arange(b, R, B) for b in range(B)]
    elif str(mode) == "rand":
        rng = np.random.default_rng(1000 + B)
        parts = [np.nonzero(rng.permutation(R) % B == b)[0] for b in range(B)]
    elif str(mode) == "cont":
        edges = np.linspace(0, R, B + 1).astype(int)
        parts = [np.arange(int(edges[b]), int(edges[b + 1])) for b in range(B)]
    else:
        raise ValueError(f"unknown partition mode {mode!r}")
    views: list[np.ndarray] = []
    for idx in parts:
        if idx.size == 0:
            continue
        qb = (z[:, idx].mean(axis=1) - mu) @ w_map
        qbn = qb / np.clip(np.linalg.norm(qb, axis=1, keepdims=True), 1e-9, None)
        views.append(_sq_cos_dist(qbn))
    q_bar = (z.mean(axis=1) - mu) @ w_map
    return views, q_bar, w_map


def mve_fuse_views(view_groups: list[list[np.ndarray]],
                   min_views: int = 2) -> tuple[np.ndarray | None, dict]:
    """Pool metric views from SEVERAL route families into one reliability-weighted estimate.

    `view_groups` is one list of `(C, C)` metrics per family (e.g. one list per DINOv2 layer, each
    list holding that layer's repetition blocks). Every view is an independent estimate of the SAME
    latent concept metric, which is what licenses pooling them: they share the concept axis by
    construction, and each block is a disjoint slice of the same subject's trials, so nothing here
    is indexed externally and the M1 leakage channel cannot open (the same property that made the
    within-route fusion permutation-safe, asserted in smoke §16).

    Weights are each view's agreement with the leave-one-out mean of all OTHER views, in inverse
    residual form. This is the same rule the existing block fusion uses, extended over routes; it is
    deliberately NOT a tuned constant. `mve_weight_range` is reported because a near-flat range is
    the honest signal that the views are interchangeable, i.e. that the extra families bought
    nothing -- a diagnostic, since a fusion that cannot be distinguished from a plain average is not
    evidence for the mechanism.
    """
    views = [np.asarray(x, dtype=np.float64) for grp in view_groups for x in grp]
    if len(views) < int(min_views) or any(v.shape != views[0].shape for v in views):
        return None, {"mve_n_views": 0, "mve_reason": "too_few_or_mismatched"}
    stack = np.stack(views, axis=0)
    total = stack.sum(axis=0)
    rel = []
    for i in range(stack.shape[0]):
        ref = (total - stack[i]) / max(stack.shape[0] - 1, 1)
        rel.append(float(np.linalg.norm(stack[i] - ref)
                         / max(float(np.linalg.norm(ref)), 1e-12)))
    inv = 1.0 / (np.asarray(rel) + 1e-6)
    w = inv / inv.sum()
    fused = np.tensordot(w, stack, axes=(0, 0))
    return fused, {"mve_n_views": int(stack.shape[0]),
                   "mve_weight_range": float(w.max() - w.min()),
                   "mve_weights": [float(x) for x in w],
                   "mve_is_flat": bool(w.max() - w.min() < 1e-3)}


def mve_scores(view_groups: list[list[np.ndarray]], q_bar: np.ndarray, g: np.ndarray,
               k: int = 10, rho: float = 0.1, min_landmark_rate: float = 0.0,
               recovery_fn=None, extra_de_views: list[np.ndarray] | None = None,
               ) -> tuple[np.ndarray, dict]:
    """O2-MVE: recover coordinates under the metric fused from ALL view families.

    `q_bar` is the PRIMARY route's pooled query -- one route has to supply the `(C, d)` query the
    recovery operates on, but the STRUCTURAL REFERENCE is the cross-family fused metric. That
    asymmetry is intentional and is the same shape `src_mix=1.0` already uses (`de_ref` replaced
    while `q_bar` stays the target's own): the reference is what the plan is fitted against, so
    pooling references is well-defined even when queries are not poolable across routes.
    """
    de_ref, diag = mve_fuse_views(
        view_groups + ([[np.asarray(x)] for x in extra_de_views] if extra_de_views else []))
    rec = coordinate_recovery if recovery_fn is None else recovery_fn
    # `coordinate_recovery` takes `**_ignored` and therefore DROPS `fgw_de_ref`/`fgw_de_mix`
    # silently -- the fused reference would never reach the plan and the cell would be
    # bit-identical to a structural-off twin while carrying a name that says otherwise. That is
    # this project's most expensive recurring bug, so it is flagged rather than left to be
    # inferred from a flat number.
    diag["mve_fgw_path"] = "default_operator_ignores_fgw" if recovery_fn is None else "wired"
    q_rec, rdiag = rec(np.asarray(q_bar, dtype=np.float64), np.asarray(g, dtype=np.float64),
                       k=k, rho=rho, min_landmark_rate=min_landmark_rate,
                       fgw_de_ref=de_ref,
                       fgw_de_mix=(1.0 if de_ref is not None else 0.0))
    diag.update({f"mve_{kk}": vv for kk, vv in rdiag.items()
                 if not hasattr(vv, "shape")})
    return csls_scores(q_rec, np.asarray(g, dtype=np.float64), k=k), diag


def refine_with_reps(
    z_reps: np.ndarray,
    g: np.ndarray,
    k: int = 10,
    rho: float = 0.1,
    shrink: float = 0.1,
    whiten: bool = True,
    rep_agreement: float = 0.5,
    min_landmarks: int = 8,
    max_landmarks: int = 160,
    moment: bool = True,
) -> tuple[np.ndarray, dict]:
    """Label-free test-time refinement using repeated trials (v6 pillar C2).

    ``z_reps`` is ``(C, R, d)`` -- the encoder's embeddings of the ``R`` un-averaged
    repetitions of each of the ``C`` test concepts -- and ``g`` is ``(C, d)``, the image
    gallery. Returns ``(scores, diag)``.

    WHY REPETITIONS, WHEN THE STANDARD PATH ALREADY AVERAGES THEM
    ------------------------------------------------------------
    The standard query is the encoder's output on the repetition-average, so the
    repetitions look like pure redundancy. They are not, and they buy two things the
    averaged trial cannot:

      1. **Better-conditioned statistics.** The whitening/moment estimates deployment
         applies are fitted from ``C`` = 200 rows. Here they are fitted from all
         ``C * R`` repetitions, which is the same quantity with ``R`` times the samples.
         The estimate is a mean and a covariance, so this is not a new mechanism -- it is
         the same mechanism with less variance.
      2. **A validated landmark set.** Deployment's coordinate recovery fits its rotation
         from mutual-NN pseudo-pairs, which are pseudo-labels and therefore wrong some of
         the time. A pseudo-pair ``(concept i -> gallery j)`` can be CHECKED against the
         repetitions: if the majority of concept ``i``'s repetitions independently rank
         ``j`` first under the same CSLS statistic, the pair is trustworthy. That is a
         strictly higher-precision landmark set for the same operator.

    Both halves are label-free. Nothing here reads a test label or a gallery identity: the
    repetitions are the target subject's own EEG, and the "check" is agreement between
    independent measurements of the SAME trial. This is a T2 / transductive operation and
    must be reported separately from every T1 number.

    Order matches `calibrate` (estimate, then apply) and the episode in
    `losses.episode`, so the three stay comparable: whiten -> landmark select -> fit ->
    apply -> CSLS.
    """
    z = np.asarray(z_reps, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    if z.ndim != 3:
        raise ValueError(f"z_reps must be (C, R, d), got {z.shape}")
    if z.shape[0] != g.shape[0]:
        raise ValueError(f"z_reps and gallery disagree on C: {z.shape[0]} vs {g.shape[0]}")

    C, R, d = z.shape
    zf = z.reshape(C * R, d)
    diag: dict = {"n_concepts": C, "n_reps": R, "k": int(k), "rho": float(rho)}

    if whiten:
        mu, w_map, wdiag = _whiten_from_cloud(zf, shrink=shrink)
        diag["whiten_diag"] = wdiag
    else:
        mu, w_map, wdiag = np.zeros((1, d)), np.eye(d), {"whiten": False}

    q_bar = (z.mean(axis=1) - mu) @ w_map                       # (C, d)
    reps_w = (zf - mu) @ w_map
    qn = q_bar / np.maximum(np.linalg.norm(q_bar, axis=-1, keepdims=True), 1e-8)
    gn = g / np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-8)

    # ---- landmark selection, then VALIDATION against the repetitions -------------
    s = csls_scores(qn, gn, k=k)
    fwd = s.argmax(axis=1)
    bwd = s.argmax(axis=0)
    gal = np.arange(C)
    mutual = fwd[bwd] == gal
    g_idx = np.nonzero(mutual)[0]
    n_mutual = int(g_idx.size)
    diag["n_mutual"] = n_mutual

    if n_mutual >= int(min_landmarks):
        q_idx = bwd[g_idx]
        # per-repetition agreement: score every rep against the gallery with the SAME
        # CSLS statistic and ask how often concept i's reps independently pick j.
        rn = reps_w / np.maximum(np.linalg.norm(reps_w, axis=-1, keepdims=True), 1e-8)
        # (C*R, C) then resolved per concept
        rep_scores = csls_scores(rn, gn, k=k)
        rep_top = rep_scores.argmax(axis=1).reshape(C, R)
        agree = np.array([float((rep_top[i] == g_idx[t]).mean())
                          for t, i in enumerate(q_idx)])
        # confidence = CSLS margin (top1 - top2) on the averaged query
        part = np.partition(s, -2, axis=1)
        margin = (part[:, -1] - part[:, -2])[q_idx]
        keep = agree >= float(rep_agreement)
        if int(keep.sum()) >= int(min_landmarks):
            order = np.argsort(-margin[keep])[:int(max_landmarks)]
            sel = np.nonzero(keep)[0][order]
            x = qn[q_idx[sel]]
            y = gn[g_idx[sel]]
            if moment:
                x = (x - x.mean(0, keepdims=True)) / (x.std(0, keepdims=True) + 1e-8)
                y = (y - y.mean(0, keepdims=True)) / (y.std(0, keepdims=True) + 1e-8)
            # Same solver as the episode and as `coordinate_recovery`'s contract.
            import torch  # noqa: PLC0415  (kept local: calibration stays numpy-first)
            from .losses.recovery import orthogonal_procrustes
            with torch.no_grad():
                r, rdiag = orthogonal_procrustes(
                    torch.as_tensor(x, dtype=torch.float32),
                    torch.as_tensor(y, dtype=torch.float32), rho=float(rho))
            q_rec = qn @ r.numpy()
            diag.update({"n_validated": int(sel.size),
                         "landmark_rate": float(sel.size) / max(1, C),
                         "mean_rep_agreement": float(agree[keep].mean()),
                         "orthogonality_err": rdiag["orthogonality_err"],
                         "det": rdiag["det"]})
            return csls_scores(q_rec, gn, k=k), diag
        diag["reason"] = "validated landmarks below floor"

    # Fall back to the un-refined (but rep-whitened) query: better statistics, no fitted
    # rotation. A silently bad rotation is worse than no rotation.
    diag["n_validated"] = 0
    return csls_scores(qn, gn, k=k), diag
