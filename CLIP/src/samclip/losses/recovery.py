"""Differentiable coordinate recovery, for SCORE-style source-only episodes.

WHY THIS MODULE EXISTS
----------------------
The deploy-time recovery in `samclip.calibration.coordinate_recovery` is a NumPy/torch
routine that runs on FROZEN features. It lifted `v5-a1-k20` from 27.00 to 35.50 Top-1 on
sub-08 with no retraining (docs/eeg2image_v5_master_plan.md §11.1), which is the largest
single measured gain in this project. But it is used *post hoc*, on an encoder that was
never told recovery would happen.

SCORE (arXiv 2608.19134) does the other half, and it is the half that matters for a
model trained end-to-end: during training it treats one source subject as a temporary
target, hides that subject's EEG-image matches, applies **the same recovery procedure it
will use at deployment**, and only then reveals the matches to compute the loss. The
encoder is therefore optimised in a coordinate frame it can actually recover, instead of
one that happens to be convenient.

The earlier post-hoc attempt in this project failed (v4, `coordinate_recovery` was judged
a no-op). That measurement was on a model with no reason to have recoverable coordinates,
which is exactly the configuration SCORE's ablations predict will fail. Re-running it on a
v5 checkpoint (§11.1) reversed the verdict. This module is the training-side counterpart.

DIFFERENTIABILITY
-----------------
A recovery step has one discrete choice -- which pairs are landmarks -- and one smooth
solve -- the orthogonal map. Only the smooth part can carry gradient, so the split is
explicit:

  * landmark SELECTION (argmax / mutual-NN) runs under `torch.no_grad()`. Its output is a
    set of indices. Gradient does not flow through *which* pairs were chosen, only through
    the embeddings that were chosen.
  * the Procrustes solve is FORWARD-ONLY; its result is detached. See the long note on
    `orthogonal_procrustes` for why this is the correct design rather than a concession.
    The APPLICATION `Z @ R` is an ordinary matmul and carries gradient normally, so the
    encoder still learns to place its coordinates where a chosen landmark set gives a good
    map. Deployment also never differentiates through R -- it estimates and then applies --
    so this is what makes the episode a faithful simulation in the first place.

This is the standard treatment for a non-differentiable correspondence step and it is
what makes the episode trainable at all. It is also why the module ships with an
abstention gate: if too few mutual landmarks survive, fitting a map from them would be
fitting noise, and the forward returns the input unchanged with `abstained=True` rather
than a silently bad rotation.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def orthogonal_procrustes(
    x: torch.Tensor,
    y: torch.Tensor,
    rho: float = 0.1,
    ns_iters: int = 3,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict]:
    """Orthogonal ``R`` minimising ``||x R - y||_F^2 + rho ||R - I||_F^2``. DETACHED.

    Expansion: ``||xR - y||^2 = ||x||^2 - 2 tr(R^T x^T y) + ||y||^2`` and
    ``rho ||R - I||^2 = rho(||R||^2 - 2 tr(R^T) + ||I||^2)``. With ``R`` orthogonal,
    ``||R||^2`` is constant, so both terms collapse onto one objective:

        maximise  tr(R^T (x^T y + rho I))

    whose solution is the ORTHOGONAL POLAR FACTOR of ``M = x^T y + rho I``.

    WHY THE RETURNED MATRIX IS DETACHED, AND WHY THAT IS THE CORRECT DESIGN
    ----------------------------------------------------------------------
    The first implementation here differentiated through the solve, and it destroyed a run:
    the training loss went NaN 50 steps in, and the failure surfaced far from its cause (as
    an unrelated ``eigh`` "ill-conditioned" error in a diagnostic, after the parameters were
    already poisoned). The root cause is that ``M`` is built from ``L`` landmarks in a
    ``d``-dimensional space, and ``L < d`` is the NORMAL case (42 landmarks, ``d = 64``);
    the embeddings are also genuinely low-rank -- this project's own measurement puts the
    concept manifold at 16 dimensions. ``x^T y`` therefore has many repeated singular
    values, and ``rho I`` makes that worse rather than better: it moves them from an exactly
    repeated 0 to an *approximately* repeated ``rho``, and both SVD's and ``eigh``'s
    backward passes divide by eigenvalue/singular-value differences, so they return NaN.
    Measured on a rank-16 input: SVD backward NaN; ``eigh`` backward NaN at every ``rho``
    outside a narrow band. Note that a forward-only finiteness check cannot catch any of
    this -- the FORWARD value was finite in every failing case; only the GRADIENT was NaN.

    Rather than defend a narrow band of numerically lucky ``rho`` values, the solve is now
    FORWARD-ONLY and the result is detached. This is not a concession, it makes the episode
    match deployment: at deployment ``R`` is estimated from the target subject's features
    and then simply APPLIED -- nothing differentiates through it. A training step that
    backpropagated through ``R`` would optimise a graph that deployment does not have. It is
    also the same treatment the other two estimated quantities in this module already get:
    landmark SELECTION and the moment statistics are both detached, and ``R`` is the third
    member of that class. Gradient still flows, and flows strongly, through the APPLICATION
    ``z_eeg @ R`` (it is a matmul with a constant, finite matrix), which is precisely the
    signal we want: "place your coordinates so that, once a landmark set is chosen, the
    estimated rotation lines them up with the gallery".

    The polar factor is computed as ``M (M^T M)^{-1/2}`` -- a symmetric eigendecomposition,
    then a few Newton-Schulz steps. The polish is not cosmetic: ``eigh`` alone left a 7.4e-3
    orthogonality error (float32 precision through ``rsqrt`` of small eigenvalues) and three
    steps bring it to 3e-7. Orthogonality matters because the retrieval score is a cosine,
    so a non-orthogonal ``R`` would silently change the geometry instead of rotating it.

    ``rho`` must be strictly positive. With ``rho = 0`` the polar factor of a rank-deficient
    ``M`` is a partial isometry rather than a rotation, and the Newton-Schulz polish would
    then bend it into a rotation the objective never asked for. Large ``rho`` is safe and
    simply drives ``R`` toward the identity -- i.e. toward "no recovery" -- which is the
    intended behaviour of the regulariser, not an error.
    """
    if not (float(rho) > 0.0):
        raise ValueError(
            f"rho must be > 0; got {rho}. With rho = 0 the polar factor of a rank-deficient "
            f"M is a partial isometry, not a rotation, and the Newton-Schulz polish below "
            f"would bend it into a rotation the objective never requested.")
    # ---- solve in float64 --------------------------------------------------------
    # The polar factor is `M (M^T M)^-1/2`, so its accuracy is set by the SMALLEST
    # eigenvalue of `M^T M`. `M = x^T y + rho I` with `x^T y` rank-`L` in a `d`-space
    # makes those eigenvalues ~rho^2 while the head is ~||x^T y||^2; on unnormalised
    # landmarks that ratio exceeded 1e-9 and float32 `eigh` (precision ~1e-7 RELATIVE)
    # returned garbage there -> `rsqrt` blew up -> NaN, in the FORWARD pass. float64
    # costs nothing here (d is 64, the call is detached and forward-only) and removes the
    # whole class of failure. This is not a precision nicety: the failing configuration
    # is the DEFAULT one (L < d, rho small).
    m = x.transpose(0, 1) @ y                      # (d, d) = x^T y
    in_dtype = m.dtype
    m = m.to(torch.float64)
    d = m.shape[0]
    eye = torch.eye(d, dtype=m.dtype, device=m.device)
    m = m + float(rho) * eye

    gram = m.transpose(0, 1) @ m                   # symmetric PD, eigenvalues >= rho^2
    w, q = torch.linalg.eigh(gram)
    w = w.clamp_min(eps)
    inv_sqrt = (q * w.rsqrt().unsqueeze(0)) @ q.transpose(0, 1)
    r = m @ inv_sqrt

    # Newton-Schulz polish: R is already nearly orthogonal, so this converges immediately
    # and removes the float32 error the inverse square root introduced.
    for _ in range(int(ns_iters)):
        r = 0.5 * r @ (3.0 * eye - r.transpose(0, 1) @ r)

    det = float(torch.linalg.det(r))
    if det < 0:
        # a reflection is not a coordinate change: flip the sign of the direction carrying
        # the least energy, which leaves the fit essentially untouched
        u, _s, vh = torch.linalg.svd(r)
        u = u.clone()
        u[:, -1] = -u[:, -1]
        r = u @ vh
        det = float(torch.linalg.det(r))

    r = r.to(in_dtype).detach()
    return r, {"det": det, "orthogonality_err": float(
        (r.transpose(0, 1) @ r - torch.eye(d, dtype=r.dtype, device=r.device)).abs().max())}


def recovery_episode(
    z_eeg: torch.Tensor,
    z_gallery: torch.Tensor,
    k: int = 10,
    rho: float = 0.1,
    min_landmarks: int = 8,
    moment: bool = True,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict]:
    """Recover ``z_eeg``'s coordinates into ``z_gallery``'s frame. Label-free.

    This is the differentiable analogue of `calibration.coordinate_recovery`, intended to
    be called INSIDE training on a source-only episode: the caller passes one source
    subject's EEG block and the image gallery, and gets back that subject's EEG in the
    shared coordinates. The caller must HIDE the EEG-image pairing from this function --
    it only ever sees the two clouds, which is what makes it a faithful simulation of
    deployment.

    Steps:
      1. l2-normalise both clouds and form the cosine matrix;
      2. correct it with the same CSLS local scaling deployment uses (`csls_correct`), so
         the landmarks are chosen in the metric the score is produced in -- using raw
         cosine here would select hub landmarks that deployment would not;
      3. keep MUTUAL nearest neighbours as pseudo-landmarks (detached);
      4. fit the regularised orthogonal map from those pairs and apply it.

    Returns ``(z_recovered, diagnostics)``. When fewer than `min_landmarks` mutual pairs
    survive, returns the input unchanged with ``abstained=True`` -- fitting a rotation
    from a handful of pairs is fitting noise, and a silent wild rotation is worse than no
    recovery at all.
    """
    from .contrastive import csls_correct

    if z_eeg.shape[0] < 2 or z_gallery.shape[0] < 2:
        return z_eeg, {"abstained": True, "reason": "too few rows"}

    ze = F.normalize(z_eeg, dim=-1)
    zg = F.normalize(z_gallery, dim=-1)
    s = csls_correct(ze @ zg.T, k=k)

    n_e, n_g = s.shape

    # ---- discrete part: detached, cannot carry gradient -------------------------
    with torch.no_grad():
        fwd = s.argmax(dim=1)                          # (n_e,) gallery index per query
        bwd = s.argmax(dim=0)                          # (n_g,) query index per gallery
        # mutual NN: gallery j whose best query i also picks j
        gal = torch.arange(n_g, device=s.device)
        mutual = fwd[bwd] == gal                       # (n_g,) bool
        g_idx = torch.nonzero(mutual, as_tuple=False).squeeze(-1)
        n_mutual = int(g_idx.numel())

        if n_mutual < int(min_landmarks):
            return z_eeg, {"abstained": True, "n_mutual": n_mutual,
                           "min_landmarks": int(min_landmarks)}
        q_idx = bwd[g_idx]

    # ---- smooth part: differentiable -------------------------------------------
    x = ze[q_idx]                                      # (L, d) EEG landmarks
    y = zg[g_idx]                                      # (L, d) image landmarks
    if moment:
        # per-dimension moment matching, as deployment does (`moment_match`): put the
        # landmark coordinates on the gallery's mean and scale before solving for R.
        x = (x - x.mean(0, keepdim=True)) / (x.std(0, keepdim=True) + eps)
        y = (y - y.mean(0, keepdim=True)) / (y.std(0, keepdim=True) + eps)
    r, rdiag = orthogonal_procrustes(x, y, rho=rho, eps=eps)

    # apply the map to the WHOLE cloud, not just the landmarks: the landmarks only
    # estimate R, and it is the full cloud that the loss is computed on.
    z_rec = z_eeg @ r

    # NON-FINITE GUARD. The recovery is a differentiable block inside a training loop, so
    # a non-finite value here does not stay here -- it poisons the parameters, and the
    # failure then surfaces as a NaN loss or an unrelated linear-algebra error several
    # steps later. Returning the input unchanged keeps the episode inert for that step
    # (the encoder simply gets no recovery signal) instead of destroying the run.
    if not torch.isfinite(z_rec).all():
        return z_eeg, {"abstained": True, "reason": "non-finite recovery",
                       "n_mutual": n_mutual, "n_queries": n_e, "n_gallery": n_g}

    diag = {
        "abstained": False,
        "n_mutual": n_mutual,
        "n_queries": n_e,
        "n_gallery": n_g,
        "landmark_rate": n_mutual / max(1, min(n_e, n_g)),
        "rho": float(rho),
        "moment": bool(moment),
        **rdiag,
    }
    return z_rec, diag
