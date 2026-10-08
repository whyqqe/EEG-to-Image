"""Soft-plan alignment: the first training term that is aligned with the DEPLOYED operator.

THE MISSING LINK THIS FILE CLOSES. Four training-side attempts to lift the recovery rung have
now been falsified on this project -- T2' (stimulus-grouped repetition collapse, -4.5pp), T2''
(row-grouped, gate decay), SCORE's source-only episode (-0.48pp named rung / -1.21pp best
rung, 2 positive vs 6 negative folds), and the noise-corrected low-rank concept frame (G-a, 30
runs). Read against the measured behaviour of the operator they were all trying to help, the
pattern is not "these ideas were wrong" but "these ideas were pointed at the wrong quantity":

    the deployed recovery's contribution is +3.62 +- 1.72 Top-1 and FLAT across 30 runs
    -- corr(raw, gain) = -0.19, corr(landmark_rate, gain) = -0.21.

Every one of those four terms tried to raise the HARD mutual-NN landmark rate. The estimator
that consumes the landmarks, meanwhile, was fitting a 64-dimensional rotation from ~42 pairs
-- and the swappable-operator probe (`scripts/probe_s3r.py`) shows what happens when it stops:
replacing hard mutual-NN matching with a Sinkhorn soft plan raises the effective number of
matched pairs from ~42 to ~465 and the 10-fold x 3-seed headline from 45.53 to 48.42
(+3.20 +- 2.01, 28/30 runs positive, t = 8.72; 9/10 folds). The operator's bottleneck was the
ESTIMATOR, not the representation.

Which flips the four falsifications into a prediction. The operator now consumes a SOFT plan.
A training term that shapes that plan is therefore no longer pushing a quantity the operator
ignores -- it is shaping the exact object the recovery fit is computed from. That is the whole
of `soft_plan_loss`, and it is why this term is different in kind from T2'/T2''/G-a rather
than a fifth attempt at the same thing.

WHAT THE SINKHORN CONSTRAINT BUYS OVER InfoNCE. `_multi_positive` is a row-softmax with no
column constraint: several queries are free to concentrate on one gallery row. A 1-to-1
retrieval has the opposite prior -- the correct assignment is a permutation -- and the
doubly-stochastic marginal is exactly that constraint, relaxed. It also matters mechanically:
`coordinate_recovery`/`subspace_soft_recovery` fit a rotation THROUGH the plan, and a plan
that collapses several queries onto one gallery row supplies the fit with duplicated
correspondences, which is a rank-deficiency the `rho` shrinkage then has to absorb. Balanced
assignment is what makes the plan usable as a correspondence set.

A NOTE ON `tau`, WHICH IS NOT THE DEPLOYMENT `tau`. Deployment uses a very small temperature
(0.01-0.02) because it wants the sharpest possible plan, gradients be damned. Training is the
opposite case: at tau = 0.05 the exponent `(s - rowmax)/tau` spans ~-80 and the plan
degenerates to a near-permutation whose Jacobian is ~0, so the term would contribute nothing
but were reported as active -- this project has already paid once for a flag that was accepted
and silently did nothing. `tau` here defaults to a value chosen for gradient flow, and
`diag_mass` is returned so a saturated plan is visible rather than assumed.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .contrastive import csls_correct


def sinkhorn_plan(sim: torch.Tensor, tau: float = 0.1, iters: int = 20,
                  eps: float = 1e-9) -> torch.Tensor:
    """Row-normalised doubly-stochastic plan. Torch twin of `calibration._sinkhorn_plan`.

    Row-wise (not global) max subtraction, for the same reason recorded there: a global
    subtraction underflows most of the matrix to exactly zero at small `tau`, Sinkhorn's
    scaling vectors blow up, and the plan stops being doubly stochastic. The scaling vectors
    are kept differentiable here -- the loop is unrolled rather than detached -- because the
    entire point of the loss is to send gradient back into the embeddings through the plan.
    """
    k = torch.exp((sim - sim.amax(dim=1, keepdim=True)) / max(float(tau), 1e-6))
    u = torch.ones(k.shape[0], dtype=k.dtype, device=k.device)
    v = torch.ones(k.shape[1], dtype=k.dtype, device=k.device)
    for _ in range(int(iters)):
        v = 1.0 / (k.t() @ u + eps)
        u = 1.0 / (k @ v + eps)
    return u[:, None] * k * v[None, :]      # row sums == 1


def _sq_cos_dist(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Standardised squared chordal distance matrix. Twin of `calibration._sq_cos_dist`."""
    n = F.normalize(x, dim=-1, eps=eps)
    d = (n @ n.t() - 1.0) ** 2
    return (d - d.mean()) / d.std().clamp_min(eps)


def _blocked_sq_dist(x: torch.Tensor, block: torch.Tensor | None) -> torch.Tensor:
    """Distance matrix restricted to within-block pairs; cross-block entries set to the mean.

    WHY THE STRUCTURAL TERM MUST BE RESTRICTED, AND WHY IT IS NOT AUTOMATIC. The plan is
    block-diagonal, but the GW gradient is NOT automatically block-local: its first two terms
    are `D @ rowsum(pi)` and `D @ colsum(pi)^T`, and a marginal `rowsum(pi)` is positive for
    EVERY row regardless of block, so `(D @ rowsum(pi))[i,j]` for `i,j` inside one subject
    still sums `D[i,k]` over rows `k` in other subjects. Restricting requires zeroing the
    cross-block distances, which is what this does. The alternative -- a Python loop over
    subjects -- is the same maths with more ways to index wrongly.
    """
    d = _sq_cos_dist(x)
    if block is None:
        return d
    block = block.reshape(-1)
    same = (block[:, None] == block[None, :])
    # Cross-block entries go to the ROW MEAN rather than 0: a squared chordal distance is
    # non-negative, so 0 would read as "maximally close" and pull the gradient toward pairing
    # across subjects. The mean is the neutral value under the standardisation above.
    return torch.where(same, d, d.mean(dim=1, keepdim=True).expand_as(d))


def _std_valid(x: torch.Tensor, same: torch.Tensor | None = None) -> torch.Tensor:
    """Standardise using ONLY the in-block entries; return the FULL matrix, unmasked.

    Order matters and the wrong order is silent. If the off-block sentinel is still in `x` when
    the mean and sd are taken, the sentinel dominates both and `(x - mean)/sd` maps EVERY entry
    into a narrow band around zero: on a 9-subject batch the masked `-1e4` entries came out at
    -0.354 and the valid ones at +2.828 -- no longer a mask at all, and the Sinkhorn then put
    86.7% of the plan's mass across subjects instead of 0.0%, i.e. it trained each concept
    against a DIFFERENT SUBJECT's concepts. That collapsed the encoder (top-1 45 -> 3.5,
    `spec_top_frac` 0.999).

    Equally, the mask must NOT be applied to both terms of the final linear combination: masking
    `s_std` and `g_std` separately and then computing `(1-a)*s_std - a*g_std` scales the sentinel
    by `(1 - 2a)`, which REVERSES ITS SIGN for a > 0.5 and hands the plan 100% cross-subject mass.
    So this function standardises only; masking happens once, on the combination.
    """
    if same is None:
        return (x - x.mean()) / x.std().clamp_min(1e-9)
    vals = x[same]
    return (x - vals.mean()) / vals.std().clamp_min(1e-9)


def fgw_plan(sim: torch.Tensor, a: torch.Tensor, b: torch.Tensor, alpha: float,
             tau: float, iters: int, outer: int = 10,
             block: torch.Tensor | None = None) -> torch.Tensor:
    """Torch conditional-gradient FGW plan. Twin of `calibration._fgw_plan`.

    Gradient is kept flowing through the whole loop (nothing detached) because the point is to
    shape the embeddings the plan is built from. `alpha = 0` is NOT special-cased here: the
    caller skips this function entirely at alpha 0, because standardising rescales the
    effective temperature and that would silently change a tuned parameter.
    """
    de = _blocked_sq_dist(a, block)
    di = _blocked_sq_dist(b, block)
    same = None
    if block is not None:
        block = block.reshape(-1)
        same = (block[:, None] == block[None, :])

    def masked(x):
        # applied ONCE, to the combination -- never to the two terms separately
        return x if same is None else x.masked_fill(~same, -1e4)

    s_std = _std_valid(sim, same)
    plan = sinkhorn_plan(masked(s_std), tau=tau, iters=iters)
    for _ in range(int(outer)):
        r = plan.sum(1, keepdim=True)
        c = plan.sum(0, keepdim=True)
        # Same GW gradient as `calibration._fgw_plan`, including the two elementwise squares and
        # the column-broadcast of the C2 term. See the long note there for what the un-squared,
        # untransposed version did (correlation -0.48 with the true gradient).
        grad = (de * de) @ r + ((di * di) @ c.t()).t() - 2.0 * (de @ plan @ di.t())
        g_std = _std_valid(grad, same)
        plan = sinkhorn_plan(masked((1.0 - alpha) * s_std - alpha * g_std),
                             tau=tau, iters=iters)
    return plan


def soft_plan_loss(a: torch.Tensor, b: torch.Tensor,
                   groups: torch.Tensor | None = None,
                   block: torch.Tensor | None = None,
                   tau: float = 0.1, iters: int = 20,
                   csls_k: int | None = 20,
                   alpha: float = 0.0, fgw_outer: int = 10,
                   return_diag: bool = False):
    """Negative log plan-mass on the true correspondences, in the score's own metric.

    `a`/`b` are the two modalities' embeddings for one batch (EEG and the image features of
    the same stimuli, row-aligned). Two optional groupings, and they are different things:

      * `groups` marks rows that are POSITIVES of each other -- the multi-repetition case,
        where several queries share a stimulus and `_multi_positive`'s argument applies: a plan
        that sends mass to another view of the SAME stimulus is not making an error.
      * `block` marks membership of a RETRIEVAL SET. The deployed operator is fitted on one
        held-out subject, so its plan is solved inside a single subject's `(C, C)` matrix.
        A pooled training batch spans subjects, and an unblocked plan would make the column
        marginal equate different subjects' concept distributions -- a constraint with no
        basis in the task (`recovery_aware.block_per_subject` records the same argument, and
        measured the unblocked variant as the one that works).

    Blocking is done by masking the similarity, not by looping over subjects: with
    cross-block entries driven to a large negative, Sinkhorn's row/column scalings normalise
    independently inside each block, so the plan factorises into per-block Sinkhorn exactly,
    at the cost of one masked fill. A loop would be the same maths with more ways to index
    wrongly.

    `csls_k` is applied BEFORE the plan for the same reason `InfoNCE` applies it before the
    temperature: the plan should be built in the metric the deployment operator builds it in,
    or the term shapes a plan nobody will ever compute.
    """
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    sim = a @ b.t()
    if csls_k is not None:
        sim = csls_correct(sim, k=int(csls_k))
    if block is not None:
        block = block.reshape(-1)
    if float(alpha) > 0.0:
        # v9: solve the plan as a FUSED Gromov-Wasserstein coupling, so the object the encoder
        # is shaped against is the same alpha-fused coupling deployment solves. `alpha = 0`
        # takes the plain Sinkhorn path, which is bit-identical to v8's term.
        #
        # `sim` is passed UNMASKED here on purpose: `fgw_plan` has to standardise before it
        # masks (see `_std_keep_mask`), so masking it here would reintroduce exactly the bug
        # that collapsed the encoder.
        plan = fgw_plan(sim, a, b, float(alpha), tau=tau, iters=iters,
                        outer=fgw_outer, block=block)
    else:
        if block is not None:
            sim = sim.masked_fill(block[:, None] != block[None, :], -1e4)
        plan = sinkhorn_plan(sim, tau=tau, iters=iters)
    if groups is None:
        mask = torch.eye(plan.shape[0], dtype=plan.dtype, device=plan.device)
    else:
        groups = groups.reshape(-1)
        mask = (groups[:, None] == groups[None, :]).to(plan.dtype)
    mass = (plan * mask).sum(dim=1)
    loss = -torch.log(mass.clamp_min(1e-6)).mean()
    if not return_diag:
        return loss
    with torch.no_grad():
        off = plan * (1.0 - mask)
        diag = {
            "soft_plan_diag_mass": float(mass.mean()),
            "soft_plan_entropy": float(-(plan.clamp_min(1e-12) * plan.clamp_min(1e-12).log())
                                       .sum(dim=1).mean()),
            "soft_plan_offdiag_mass": float(off.sum() / max(plan.shape[0], 1)),
        }
    return loss, diag
