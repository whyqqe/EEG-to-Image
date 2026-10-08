"""Faithful deploy-matched source-only episode (v6 pillar A, "Generalized RAT").

THE FAILURE THIS MODULE FIXES, IN ONE PARAGRAPH
-----------------------------------------------
`recovery_episode` (v5) ran SCORE's source-only episode as ``CSLS -> mutual-NN ->
Procrustes``.  Deployment runs ``whiten -> recovery -> CSLS``.  The two are not the same
operator, and the measured cost of the mismatch was exact: ``v5-ep2`` was the first arm
to move the *raw* number (+3.00) and to shrink the subject gap (-5.67pp), yet its
**deployed** Top-1 FELL 35.50 -> 32.00.  The mechanism worked; the episode taught the
encoder to be recoverable in a coordinate frame deployment never enters (un-whitened,
k = 20) instead of the one it does (SAW-whitened, k = 10).  A training signal and a
deployment operator that disagree by a change of coordinates are two different
operators.

WHAT "FAITHFUL" MEANS HERE, AND WHAT IT DOES NOT CLAIM
------------------------------------------------------
This module re-implements the deployment ladder's ESTIMATE-AND-APPLY structure in a
differentiable form:

    whiten  ->  CSLS(k)  ->  mutual-NN landmarks  ->  moment match  ->  Procrustes  ->  apply

Every estimated quantity is detached, and that is the faithful choice rather than a
gradient-saving concession: deployment estimates ``mu``, the whitening map ``W``, the
landmark set and the rotation ``R`` from the target subject's features and then APPLIES
them.  Nothing differentiates through an estimate at deployment.  So the episode must
not either, or it optimises a computation graph deployment does not have.  Gradient
flows through exactly one thing -- the application ``(z - mu) @ W @ R`` -- which is the
signal we want: "place your coordinates so that, once this subject's statistics are
estimated and a rotation is fitted, the recovered cloud lines up with the gallery".

THE HONEST LIMIT OF THE FIDELITY CLAIM.  ``whiten`` is asserted bit-close to the numpy
``calibration.saw_whiten`` (same closed form, float64 statistics -- see
``smoke_v6.py``); the final scoring CSLS is literally the same ``csls_correct`` the
deployment ladder's torch path uses.  The Procrustes step is NOT the external
``epd.recover`` routine, so the two are asserted on their shared contract (orthogonality,
detachment, recovery of a known rotation) rather than to machine epsilon.  Claiming
bit-identity there would be false; the order, the k, and the estimated-quantity
semantics are what match.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .contrastive import csls_correct
from .recovery import orthogonal_procrustes


@torch.no_grad()
def whiten_map(
    q: torch.Tensor,
    shrink: float = 0.1,
    max_cond: float = 1e3,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """``(mu, W, diag)`` for subject-adaptive whitening, mirroring ``saw_whiten``.

    Reimplemented rather than imported because the numpy version is not differentiable
    and -- more importantly -- because the ESTIMATE is detached at deployment, so the
    torch version only ever needs to run under ``no_grad`` on the query cloud's own
    statistics.  The closed form is copied from ``calibration.saw_whiten`` including
    both guards: the shrinkage toward a scaled identity (the sample covariance of 200
    trials in a 64-d space is rank-deficient whenever ``N - 1 < d``) and the condition
    cap (a fixed absolute floor inverts the null space to ~316x and *amplifies* noise).

    Statistics are computed in float64 and cast back, matching the numpy path, because
    ``eigh`` on a near-singular 64x64 covariance in float32 is where the two would drift.
    """
    if q.dim() != 2:
        raise ValueError(f"whiten_map expects (N, D), got {tuple(q.shape)}")
    q64 = q.detach().to(torch.float64)
    mu = q64.mean(dim=0, keepdim=True)
    qc = q64 - mu
    n, d = qc.shape
    cov = (qc.transpose(0, 1) @ qc) / max(1, n - 1)
    eye = torch.eye(d, dtype=q64.dtype, device=q64.device)
    if shrink > 0:
        cov = (1.0 - float(shrink)) * cov + float(shrink) * (torch.trace(cov) / d) * eye
    cov = cov + float(eps) * eye
    vals, vecs = torch.linalg.eigh(cov)
    lo = max(float(vals.max()) / float(max_cond), float(eps))
    vals = vals.clamp_min(lo)
    inv_sqrt = (vecs * vals.rsqrt().unsqueeze(0)) @ vecs.transpose(0, 1)
    diag = {
        "eig_max": float(vals.max()),
        "eig_min": float(vals.min()),
        "cond": float(vals.max() / vals.min()),
        "n_samples": int(n),
        "d_embed": int(d),
        "rank_deficient": bool(n - 1 < d),
        "shrink": float(shrink),
    }
    return mu.to(q.dtype), inv_sqrt.to(q.dtype), diag


def deploy_stack_episode(
    z_eeg: torch.Tensor,
    z_gallery: torch.Tensor,
    k: int = 10,
    rho: float = 0.1,
    min_landmarks: int = 8,
    whiten: bool = True,
    shrink: float = 0.1,
    max_cond: float = 1e3,
    moment: bool = True,
    ns_iters: int = 3,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict]:
    """Apply the FULL deployment ladder to ``z_eeg``, differentiably, and return it.

    The caller must hide the EEG-image pairing: this function only ever sees the two
    clouds.  Returns ``(z_recovered, diagnostics)``; a non-finite or below-floor episode
    returns the input unchanged with ``abstained=True`` rather than a silently bad
    rotation (a rotation fitted from a handful of pseudo-pairs is fitted noise).
    """
    if z_eeg.shape[0] < 2 or z_gallery.shape[0] < 2:
        return z_eeg, {"abstained": True, "reason": "too few rows"}

    diag: dict = {"whiten": bool(whiten), "k": int(k), "rho": float(rho)}
    if whiten:
        mu, w_map, wdiag = whiten_map(z_eeg, shrink=shrink, max_cond=max_cond, eps=eps)
        z_w = (z_eeg - mu) @ w_map
        diag["whiten_diag"] = wdiag
    else:
        z_w = z_eeg

    ze = F.normalize(z_w, dim=-1)
    zg = F.normalize(z_gallery, dim=-1)
    # CSLS in the SAME (whitened) metric the deployment ladder chooses landmarks in.
    # Using raw cosine here would select hub landmarks deployment would not.
    s = csls_correct(ze @ zg.t(), k=int(k))

    n_e, n_g = s.shape
    with torch.no_grad():
        fwd = s.argmax(dim=1)
        bwd = s.argmax(dim=0)
        gal = torch.arange(n_g, device=s.device)
        mutual = fwd[bwd] == gal
        g_idx = torch.nonzero(mutual, as_tuple=False).squeeze(-1)
        n_mutual = int(g_idx.numel())
        if n_mutual < int(min_landmarks):
            return z_eeg, {"abstained": True, "n_mutual": n_mutual,
                           "min_landmarks": int(min_landmarks), **diag}
        q_idx = bwd[g_idx]

    x = ze[q_idx]
    y = zg[g_idx]
    if moment:
        # per-dimension moment matching, as deployment's `epd.recover` does: put the
        # landmark coordinates on the gallery's mean and scale before solving for R.
        x = (x - x.mean(0, keepdim=True)) / (x.std(0, keepdim=True) + eps)
        y = (y - y.mean(0, keepdim=True)) / (y.std(0, keepdim=True) + eps)
    # Estimates are detached (see the module docstring): R is an estimated statistic.
    r, rdiag = orthogonal_procrustes(x.detach(), y.detach(), rho=rho, ns_iters=ns_iters,
                                     eps=eps)

    z_rec = z_w @ r

    if not torch.isfinite(z_rec).all():
        return z_eeg, {"abstained": True, "reason": "non-finite recovery",
                       "n_mutual": n_mutual, **diag}

    diag.update({
        "abstained": False,
        "n_mutual": n_mutual,
        "n_queries": int(n_e),
        "n_gallery": int(n_g),
        "landmark_rate": n_mutual / max(1, min(n_e, n_g)),
        "moment": bool(moment),
        **rdiag,
    })
    return z_rec, diag
