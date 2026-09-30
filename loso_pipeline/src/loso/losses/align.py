"""Alignment, invariance and disentanglement losses.

Soft-label InfoNCE
------------------
A batch of EEG trials has no reason to contain only one example per concept, and
even when it does, several distinct concepts are near-synonyms (dog/wolf,
car/bus).  Treating every non-diagonal entry as a negative therefore pushes apart
pairs the teacher itself considers similar.  The reference implementation
(CognitionCapturerPro `ClipLoss_Modified`, `utils.py:184-219`) handles this by
building *soft* targets instead of a one-hot diagonal:

  1. take the pairwise similarity of the (frozen) text features,
  2. keep each row's top-k most similar entries (excluding self), then re-add self,
  3. intersect with the same-class mask,
  4. row-normalise -> a uniform distribution over {self} U {top-k same-class peers},
  5. soft cross-entropy against the logits.

`soft_label_contrastive` reproduces that construction.  Here the similarity is taken
over the *teacher* targets (CLIP embeddings of the batch's images) rather than over
an encoder's own text features, which is what makes it usable for an EEG encoder:
the neighbourhood is a property of the target space, so it is stable and requires no
gradient.

Why the teacher similarity, not the encoder's
---------------------------------------------
CogCapPro computes its similarity from the text tower's own output because that
tower is a *trained* branch of the model.  Here the text/image targets are frozen
external teachers, so the neighbourhood must be computed from them and detached --
otherwise the soft targets would drift during training and the loss would chase a
moving objective.

Weighting
---------
Defaults follow the design's Phase-2 schedule verbatim:
``L_img + 0.2 L_text + 0.5 L_dino + 0.1 L_vae + 0.1 L_orth``, plus the two terms the
design marks as deliberately dominant (``L_time`` and the repeated-trial contrastive,
both at 3x) and the adversarial/distribution terms, which the design caps to keep
them from overwhelming the alignment objective.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

# --- similarity helpers ------------------------------------------------------
def l2_normalize(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return F.normalize(x, dim=dim)


@torch.no_grad()
def teacher_similarity(targets: torch.Tensor) -> torch.Tensor:
    """Cosine similarity among frozen teacher targets, for soft-label mining.

    Must stay under `no_grad`: these similarities define the loss's target
    distribution, and letting them carry gradient would make the objective
    self-referential (the model could minimise the loss by moving the targets).
    """
    z = l2_normalize(targets.float(), dim=-1)
    return z @ z.t()


def neighbour_mask(similarity: torch.Tensor, topk: int = 10,
                   same_class: torch.Tensor | None = None) -> torch.Tensor:
    """Rows of {self} U {top-k most similar, restricted to `same_class`}.

    The class mask is applied *before* the top-k, not after.  The reference
    implementation intersects afterwards (`utils.py:196-210`), but the two orders
    coincide whenever same-class members have identical features -- which is exactly
    its setting, since it compares a text tower's output for samples that all share
    one stimulus.  They diverge here: a THINGS concept has ten *distinct* images, so
    same-concept features are merely similar (not identical) and can lose the global
    top-k race to unrelated images.  Masking first makes "top-k" mean "the k most
    similar peers among this concept's own samples", which is the intended
    mining behaviour, and is a no-op for the reference case.
    """
    b = similarity.shape[0]
    k = max(0, min(topk, b - 1))
    if same_class is not None:
        # -inf (not 0) so masked entries cannot be selected by top-k and do not
        # resemble a legitimate "very dissimilar" score.
        sim = similarity.masked_fill(same_class <= 0, float("-inf"))
    else:
        sim = similarity.clone()
    # Self is excluded before top-k so it cannot consume a slot that should go to a
    # genuine peer; it is restored explicitly afterwards.
    sim.fill_diagonal_(float("-inf"))

    mask = torch.zeros_like(similarity)
    if k > 0:
        # Only rows that still have at least one eligible peer can contribute.
        rows = torch.isfinite(sim).any(dim=1).nonzero(as_tuple=True)[0]
        if rows.numel() > 0:
            vals, idx = sim[rows].topk(k, dim=1, sorted=False)
            # `topk` pads its output with -inf when a row has fewer than k eligible
            # peers (a concept with few members in this batch).  Those slots are not
            # positives -- they are the *least* similar items in the batch -- so they
            # must be filtered out rather than scattered in.
            keep = torch.isfinite(vals).to(mask.dtype)
            # Boolean/integer indexing produces a copy, so the update is assigned
            # back explicitly; an in-place scatter_ on the copy would be a no-op.
            mask[rows] = mask[rows].scatter(1, idx, keep)
    # Every row keeps self: a row of all zeros would produce a NaN soft target after
    # row normalisation.
    mask.fill_diagonal_(1.0)
    return mask


def normalise_rows(mask: torch.Tensor) -> torch.Tensor:
    row_sums = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    return mask / row_sums


def center_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Subtract the batch's mean row, then L2-normalise.

    The teacher spaces are strongly anisotropic, and this is the measured fix rather
    than a stylistic one.  `scripts/probe_geometry.py` reports, on this data:

        target space      mean off-diagonal cosine   energy in the shared direction
        clip_image                     +0.391                    39.7%
        clip_text_caption              +0.350                    35.9%
        dino                           +0.004                     0.9%

    A component present in *every* key adds an almost query-independent offset to
    every logit, so it contributes no discrimination; what it does do is give
    systematically higher similarity to whichever keys happen to point along it.
    Retrieval then prefers those hub targets over the correct one, which is the
    concrete reason a run can have a positive-but-tiny margin and still score at
    chance.  Removing the shared direction costs nothing that helps and deletes the
    hub bias.

    The same measurement applied to the model's own output is why this matters here
    and not only in theory: after the failed run, 85.4% of `p_img`'s energy sat in
    its batch-mean direction, leaving a retrieval margin of +0.0034 against a target
    space whose off-diagonal cosines average +0.37.  There is no accuracy to be had
    from a signal 1% the size of the background.
    """
    x = x.float()
    return F.normalize(x - x.mean(dim=0, keepdim=True), dim=-1, eps=eps)


def soft_label_contrastive(query: torch.Tensor, key: torch.Tensor,
                           logit_scale: torch.Tensor,
                           teacher_sim: torch.Tensor | None = None,
                           topk: int = 10,
                           same_class: torch.Tensor | None = None,
                           center: bool = False) -> torch.Tensor:
    """Symmetric soft-label InfoNCE between (B, D) query/key batches.

    Both sides must be L2-normalised.  Falls back to a hard diagonal when
    `teacher_sim` is None, which is the plain CLIP objective.

    `center=True` removes the batch-mean direction from both sides before the
    similarity, which is required for the CLIP branches and is the reason it is a
    flag rather than the unconditional behaviour.  See `center_normalize` for the
    measurement behind it; when it is set, `teacher_sim` must have been built from
    the same centred vectors, or the soft targets would describe a different
    similarity than the logits they supervise.
    """
    if center:
        query = center_normalize(query)
        key = center_normalize(key)
    logits_qk = logit_scale * query @ key.t()
    logits_kq = logits_qk.t()

    if teacher_sim is None:
        labels = torch.eye(query.shape[0], device=query.device)
    else:
        labels = normalise_rows(
            neighbour_mask(teacher_sim, topk=topk, same_class=same_class)
        )

    # Soft cross-entropy: cross_entropy already accepts a probability target, it
    # just requires the label tensor to share the logits' dtype.
    loss_qk = F.cross_entropy(logits_qk, labels.to(logits_qk.dtype))
    loss_kq = F.cross_entropy(logits_kq, labels.t().to(logits_kq.dtype))
    return 0.5 * (loss_qk + loss_kq)


def hard_info_nce(query: torch.Tensor, key: torch.Tensor,
                  logit_scale: torch.Tensor) -> torch.Tensor:
    return soft_label_contrastive(query, key, logit_scale, teacher_sim=None)


# --- regression-style alignment ----------------------------------------------
def huber_alignment(pred: torch.Tensor, target: torch.Tensor,
                    delta: float = 1.0) -> torch.Tensor:
    """Huber on L2-normalised vectors, reduced over features but averaged over the batch.

    Used for the DINOv2 branch.  The design specifies Huber rather than cosine
    because DINOv2 is a self-supervised feature: its components have real variance
    structure (unlike CLIP's projection space, which is trained to be
    cosine-comparable), so a regression objective preserves more of it.  Huber caps
    the influence of outliers, which DINOv2 features do contain.

    The reduction is deliberate, and it was wrong before.  `F.huber_loss`'s default
    `reduction="mean"` averages over *every* element, i.e. divides by `B * d`.  For
    unit-norm vectors in `d = 1024` dimensions the per-element residual is ~0.035, so
    the term came out at 0.00098 while every other term sat between 1 and 11 -- a
    factor of ~1000 -- and `dino` took 2e-2 of the gradient where `img` took 1.9e2 in
    the smoke test's gradient-reach table.  At weight 0.5 that is a switched-off term
    wearing a weight that says it is the second most important one, and nothing in the
    logs distinguishes it from a term that is merely easy to satisfy.

        reduction="mean" (all elements) : 0.000975   <- previous
        reduction="sum"  (all elements) : 63.8731
        sum / B                         : 0.9980    <- used here

    `delta=1.0` is unchanged and is effectively inactive on this data: the maximum
    possible per-element residual between unit vectors is 2 and the measured maximum is
    0.21, so the objective sits in its quadratic regime and behaves as a squared error
    with outlier clipping held in reserve.
    """
    if pred.shape[0] == 0:
        return pred.new_zeros(())
    return F.huber_loss(pred, target, delta=delta, reduction="sum") / pred.shape[0]


def cosine_mse(pred: torch.Tensor, target: torch.Tensor,
               mse_weight: float = 1.0) -> torch.Tensor:
    """1 - cosine + a bounded MSE, for the coarse VAE branch."""
    cos = 1.0 - F.cosine_similarity(pred, target, dim=-1).mean()
    mse = F.smooth_l1_loss(pred, target)
    return cos + mse_weight * mse


#: Smoothing temperature for the patch-side log-sum-exp inside `set_alignment`.
#:
#: Must be on the scale of the *differences between cosines*, which are order 1, not on
#: the scale of `sqrt(d)` -- `sqrt(d)` is the right normalisation for the dot product of
#: unnormalised vectors, and is meaningless for a log-sum-exp over already-bounded
#: cosines.  The distinction is not cosmetic; it is the difference between a term that
#: measures alignment and one that measures nothing.
#:
#: Measured on 64 patches with a matching cosine of 0.92 against 0.24 for a random
#: pair, aligned prediction versus a global constant (scripts/smoke_align.py):
#:
#:     tau        aligned   constant   margin
#:     sqrt(d)     2.348     2.499      0.151   <- the previous default
#:     1.0         2.276     2.500      0.225
#:     0.3         1.777     2.506      0.729
#:     0.1         0.135     2.532      2.397   <- chosen
#:     0.02        0.025     2.631      2.606
#:
#: At `tau = sqrt(d) = 16` for the real `vae_patch_dim = 256`, one perfect patch match is
#: diluted by the other 63 patches: `exp(1/16) = 1.065` against 63 terms of ~1.0, so the
#: per-candidate scores differ by ~0.02 out of ~29 and the term is very nearly constant
#: across candidates.  A term that is flat in the quantity it claims to measure still
#: produces gradient -- on the component shared by every candidate -- so it pushes the
#: representation toward uniformity while its loss value looks entirely reasonable.
#: That is a plausible mechanism for the 58% gradient share and the collapse attributed
#: to `L_time`, and it is why the default is no longer `sqrt(d)`.
#:
#: Below ~0.02 the margin keeps improving but the soft-max approaches a hard max, and
#: the gradient to the non-arg-max patches (the reason for smoothing at all) vanishes.
#: 0.1 keeps a usable soft-max while giving a 2.4-nat margin.
SET_ALIGNMENT_TAU: float = 0.1


def set_alignment(pred_tokens: torch.Tensor, target_patches: torch.Tensor,
                  logit_scale: torch.Tensor | None = None,
                  tau: float | None = None) -> torch.Tensor:
    """Time-resolved alignment against VAE latent patches, as a *set* matching.

    This replaces an earlier index-wise formulation that asked EEG time step i to
    match VAE patch i in raster order.  That correspondence does not exist, and it
    was measurable: the true index correspondence scored 0.9542 on the positional
    term versus 0.9557 +/- 0.0002 for a *randomly permuted* patch sequence, i.e. a
    gap of -0.0015.  An objective whose optimum is reached equally well by an
    arbitrary permutation can only be improved by making both sides uniform, and
    because it carried the largest weight it was the main driver of the observed
    collapse (all tokens ending up parallel, cross-sample cosine 0.91).

    The replacement keeps the intent -- each EEG token must correspond to *some*
    part of the correct image -- while removing the false ordering.  With
    `sim[b,i,c,j]` the cosine between EEG token `i` of sample `b` and patch `j` of
    candidate stimulus `c`, and `L` the log-sum-exp over the named axis:

        token_side[b, c]  = mean_i  L_j  cos(pe[b,i], pt[c,j]) / tau
        patch_side[b, c]  = mean_j  L_i  cos(pe[b,i], pt[c,j]) / tau

    so every token must find a patch, and -- the reverse view -- every patch must be
    claimed by some token, each discriminated against the other stimuli in the batch.

    Two properties matter:

    * Permutation-invariant in both i (EEG time) and j (patch order).  Shuffling
      either axis now leaves the loss unchanged, which is the honest statement of
      what this term knows.  Asserted in `scripts/smoke_align.py`, because the two
      axes are reduced in different places and reducing the wrong one is exactly the
      bug this function previously had: `token_side` reduced over the *candidate*
      axis rather than the patch axis, so the "logits" came out `(b, n_patch)` and
      were scored against `b`-way labels.  That is an `IndexError: Target 62 is out
      of bounds` on CPU, and a device-side assert on GPU whose traceback blames
      whichever kernel synchronised next rather than this line.
    * It has *negatives* (the batch dimension).  Any pure alignment term without
      negatives -- cosine, Huber, Chamfer -- is minimised by a constant prediction,
      so it cannot be used here at all.  With negatives, a constant prediction gives
      uniform logits and loss log(B), which is the worst possible value; the term
      therefore *resists* collapse instead of causing it.

    Both sides are reduced by an explicitly named `einsum` axis rather than by
    permuting and indexing dims.  The previous version's forward/backward asymmetry
    came from permuting `sim` and then indexing it, where an off-by-one in the axis
    is invisible until it produces a shape that happens to be legal.

    What this term does *not* enforce
    ---------------------------------
    Because the token axis is averaged *after* the log-sum-exp, the term is satisfied
    by any prediction whose per-sample mean direction is right -- it constrains which
    trial the tokens belong to, not how they vary within a trial.  A per-sample
    constant scores well below a global constant and is therefore not excluded by this
    term alone (measured in `scripts/smoke_align.py`).  Within-trial structure is not
    something this objective can demand; what it does establish is that the tokens
    carry *trial-identifying* information, and that is the property the pipeline needs
    from it.  This is stated here because the term's name ("time-resolved alignment")
    invites the opposite assumption.
    """
    pe = l2_normalize(pred_tokens.float(), dim=-1)      # (b, n_tok, d)
    pt = l2_normalize(target_patches.float(), dim=-1)   # (b, n_patch, d)
    if logit_scale is None:
        raise ValueError("set_alignment requires logit_scale (an alignment term "
                         "without negatives is minimised by a constant prediction)")
    if logit_scale.shape != ():
        raise ValueError(
            f"logit_scale must be a scalar, got shape {tuple(logit_scale.shape)}")
    if tau is None:
        tau = SET_ALIGNMENT_TAU
    b = pe.shape[0]
    if b < 2:
        # With a single sample there is nothing to discriminate against, so the loss
        # is exactly log(1) and carries no gradient.  Returning zero is the honest
        # value; letting it through would silently add a constant to the total.
        return pe.new_zeros(())
    # (b, n_tok, c, n_patch): sim[b, i, c, j]
    sim = torch.einsum("bid,cjd->bicj", pe, pt)
    labels = torch.arange(b, device=pe.device)
    # Each EEG token picks its best patch of each candidate stimulus, then tokens are
    # averaged.  The patch axis is the last one (j), so reduce dim=3.
    token_side = (torch.logsumexp(sim / tau, dim=3) * tau).mean(dim=1)   # (b, c)
    forward = F.cross_entropy(logit_scale * token_side, labels)
    # Each patch picks its best token of each candidate sample, then patches are
    # averaged.  Reduce over the token axis (dim=1), then mean over patches (dim=2).
    # Without this direction a single well-placed token could satisfy the whole term
    # and the rest of the temporal axis would be unconstrained.
    patch_side = (torch.logsumexp(sim / tau, dim=1) * tau).mean(dim=2)   # (b, c)
    backward = F.cross_entropy(logit_scale * patch_side, labels)
    return 0.5 * (forward + backward)


# --- subject invariance ------------------------------------------------------
def coral_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """CORAL: squared Frobenius distance between the two groups' covariances.

    Second-order statistic matching.  Complements MMD, which is a kernel-based
    *distribution* distance and is dominated by mean/lower-order differences.
    """
    d = x.shape[1]
    x = x - x.mean(0, keepdim=True)
    y = y - y.mean(0, keepdim=True)
    cx = (x.t() @ x) / max(1, x.shape[0] - 1)
    cy = (y.t() @ y) / max(1, y.shape[0] - 1)
    return ((cx - cy) ** 2).sum() / (4.0 * d * d)


def _rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    x2 = (x ** 2).sum(1, keepdim=True)
    y2 = (y ** 2).sum(1, keepdim=True)
    d2 = (x2 + y2.t() - 2.0 * x @ y.t()).clamp_min(0.0)
    return torch.exp(-d2 / (2.0 * sigma ** 2))


def mmd_loss(x: torch.Tensor, y: torch.Tensor,
             sigmas: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)) -> torch.Tensor:
    """Multi-bandwidth RBF maximum mean discrepancy (unbiased estimator).

    The unbiased estimator can return small *negative* values, including when the
    two samples come from the same distribution (observed: -0.02 on 64 samples of
    N(0, 1)).  That is expected rather than a bug -- the statistic is unbiased
    around zero, not non-negative -- and it is harmless as a training loss because
    the gradient still points toward smaller discrepancy.  Clamping at zero would
    be worse: it would kill the gradient precisely on the batches that are already
    well matched.
    """
    if x.shape[0] < 2 or y.shape[0] < 2:
        return x.new_zeros(())
    total = x.new_zeros(())
    for s in sigmas:
        kxx = _rbf_kernel(x, x, s)
        kyy = _rbf_kernel(y, y, s)
        kxy = _rbf_kernel(x, y, s)
        n, m = x.shape[0], y.shape[0]
        # Drop the diagonal for the unbiased estimator.
        kxx = (kxx.sum() - kxx.diag().sum()) / (n * (n - 1))
        kyy = (kyy.sum() - kyy.diag().sum()) / (m * (m - 1))
        total = total + kxx + kyy - 2.0 * kxy.mean()
    return total / len(sigmas)


def orthogonality_loss(z_inv: torch.Tensor, z_sub: torch.Tensor) -> torch.Tensor:
    """||z_inv^T z_sub||_F, scale-normalised.

    Drives the invariant and subject-specific subspaces apart so the split is real
    rather than a cosmetic partition of one representation.  Normalising by the
    product of the two Frobenius norms makes the term scale-free, so it cannot be
    driven to zero by shrinking either branch.
    """
    a = z_inv - z_inv.mean(0, keepdim=True)
    b = z_sub - z_sub.mean(0, keepdim=True)
    cross = a.t() @ b
    denom = (a.norm() * b.norm()).clamp_min(1e-6)
    return cross.norm() / denom


def subject_adversarial_loss(logits: torch.Tensor,
                             subject_id: torch.Tensor) -> torch.Tensor:
    """Cross-entropy for the GRL subject classifier.

    Note the gradient direction is inverted inside the model (see `grad_reverse`),
    not here: this is an ordinary classification loss on the *reversed* features.
    """
    return F.cross_entropy(logits, subject_id)


def repeated_trial_contrastive(z: torch.Tensor, image_slot: torch.Tensor,
                               logit_scale: torch.Tensor) -> torch.Tensor:
    """Pull repeated trials of the same stimulus together.

    Trials of one image are the natural positives for an EEG encoder: they share
    the entire stimulus and differ only by neural noise, so they define a
    within-image invariance the cross-modal objective cannot express (the image
    target is identical for all of them, so it provides no gradient separating
    "same image, different trial" from "same image, same trial").

    Rows whose image appears only once in the batch are dropped: for those the
    objective degenerates to a one-hot diagonal, which is just the plain contrastive
    term again and would double-count it.
    """
    same = image_slot.unsqueeze(0) == image_slot.unsqueeze(1)
    same.fill_diagonal_(False)
    keep = same.any(dim=1)
    if int(keep.sum()) < 2:
        return z.new_zeros(())
    zk = l2_normalize(z[keep].float(), dim=-1)
    same_k = same[keep][:, keep]
    # Explicit normalisation makes the term invariant to the scale of z_inv, so it
    # cannot be down-weighted by shrinking the representation instead of by
    # actually clustering the repetitions.
    logits = logit_scale * zk @ zk.t()
    # Masked mean over the positive entries only: averaging zero loss into empty
    # rows would let the encoder ignore the term by spreading trials out.
    log_prob = F.log_softmax(logits, dim=-1)
    positives = same_k.float()
    denom = positives.sum(dim=1).clamp_min(1.0)
    return -(log_prob * positives).sum(dim=1).div(denom).mean()


# --- anti-collapse regularisers ----------------------------------------------
# The alignment terms above are *not* sufficient to keep `z_inv` informative, and
# the previous run proved it: they were all satisfied by a near-constant vector.
# Contrastive terms resist collapse only as far as their negatives reach; a term
# whose positives are a large fraction of the batch (the grouped sampler makes 36 of
# 512 rows share a stimulus, and the repeated-trial term makes all 36 mutual
# positives) has a weak repulsive component, and the purely regression-shaped terms
# (dino, vae) have none at all -- their global optimum is `pred = mean(target)`,
# a constant.  So the representation needs a regulariser that acts on the *marginal*
# statistics of `z_inv` rather than on any pairwise comparison.
#
# VICReg (Bardes, Ponce & LeCun, ICLR 2022) supplies exactly that, as two terms:
#
#   variance     hinge pushes every dimension's batch std up to `gamma`
#   covariance   drives the off-diagonal of the covariance matrix to zero
#
# Together they forbid both failure modes: a dimension that never varies (variance
# term) and a set of dimensions that only vary together (covariance term).  The
# failed run's signature -- 0.867 of the variance in one direction -- is precisely a
# violation of the second.
#
# Both use the standard VICReg stop-gradient on the violating side, which turns them
# into *constraints* rather than objectives: the representation is pushed out of the
# violating region while the regulariser itself is not minimised further.  Without
# the detach, `cov` would be minimised by scaling the whole representation toward
# zero, which the variance term would then fight.
def vicreg_variance(z: torch.Tensor, gamma: float = 1.0,
                    eps: float = 1e-4) -> torch.Tensor:
    """Hinge on the per-dimension batch standard deviation: `mean_j relu(gamma - std_j)`.

    `z` is *not* normalised first.  A dimension with zero variance contributes
    `gamma` and the term is at its maximum, which is the intended behaviour; if the
    input were L2-normalised per sample the per-dimension std would be bounded well
    below `gamma` for a high-dimensional vector and the term could never be
    satisfied.

    `gamma=1.0` is the VICReg default and matches the scale `z_inv` has after the
    terminal LayerNorm in `EEGEncoder.head_inv` (unit-ish variance per dimension
    before training pushes it around).  Verified against the actual std at
    initialisation in `scripts/smoke_align.py`, because a `gamma` ten times the
    achievable scale would make this term a constant with a zero gradient.

    Callers must pass a **centred** `z`.  This is not optional on a LayerNorm output,
    and the reason is arithmetic rather than empirical.  `head_inv` ends in a
    LayerNorm, so every `z_inv` lies on a sphere of radius `sqrt(d_inv)` and the
    identity

        sum_j var_j + sum_j mean_j^2 = d_inv        (per dimension: var + mean^2 = 1)

    holds exactly.  A hinge that only rewards per-dimension spread is therefore
    satisfied *most cheaply* by moving all the energy into the shared mean direction:
    that raises every `var_j` without requiring any structure orthogonal to it.  So on
    the raw representation this term does not merely fail to prevent collapse, it
    pays for it.

    That is what the preview run did, and it is visible in the two numbers together:
    `terms["var"]` fell 0.646 -> 0.538 while the energy in `z_inv`'s batch-mean
    direction rose 49.5% -> 80.3% (`scripts/probe_geometry.py`).  Both are correct
    readings of the same drift -- the term was being minimised exactly as written.

    Centring removes the degenerate direction from the objective entirely: a shift
    along the mean is subtracted away, so the only way to reduce the hinge is to spread
    the representation across directions orthogonal to it.  `terms["cov"]` pins the
    centred form from the other side, since centring is also what makes a constant
    representation score zero there.
    """
    if z.shape[0] < 2:
        return z.new_zeros(())
    std = torch.sqrt(z.float().var(dim=0) + eps)
    return F.relu(gamma - std).mean()


def vicreg_covariance(z: torch.Tensor) -> torch.Tensor:
    """Squared off-diagonal covariance, normalised by the dimension.

    `1/d * sum_{i != j} C_ij^2`, which is the VICReg form (Bardes, Ponce & LeCun,
    ICLR 2022).  The normaliser was `1/(d*(d-1))` here, which is 511x too small at
    `d_inv = 512`: the term was present, finite, non-zero and taking 0.02% of the
    encoder gradient -- the same *class* of defect as `huber_alignment`'s reduction,
    where a plausible-looking normaliser silently switches a term off.  The published
    value is used verbatim rather than re-derived, because the normaliser *is* the
    effective weight and re-deriving it is how it went wrong.

    This term is minimised by a *constant* representation, not just by a decorrelated
    one: centring a constant gives zero, so its covariance is exactly zero.  Measured
    on the real encoder, `cov(real) = 0.001886` against `cov(constant) = 0.000000` --
    the collapsed representation scores strictly better.  The term is therefore not
    anti-collapse on its own, and the pair is not a redundancy: `vicreg_variance` is
    the half that makes collapse expensive, and its hinge is what stops the encoder
    from taking the `cov`-preferred exit.  It must never be used without `var`.
    """
    n, d = z.shape
    if n < 2 or d < 2:
        return z.new_zeros(())
    centred = z.float() - z.float().mean(dim=0, keepdim=True)
    cov = (centred.t() @ centred) / (n - 1)
    off_diagonal = cov - torch.diag(torch.diagonal(cov))
    return (off_diagonal ** 2).sum() / d


# --- UCK (Unified Concept Kernel) gallery terms ------------------------------
# Ported from NeuroBridge `scripts/nda/uck_train.py`.  UCK's measured recipe for
# Stage-2 alignment is: one query against a *train-concept* gallery of 1,654
# rows, not only in-batch negatives.  That is what these two helpers provide.
def gallery_nce(query: torch.Tensor, gallery: torch.Tensor,
                concept_id: torch.Tensor, logit_scale: torch.Tensor,
                center: bool = True) -> torch.Tensor:
    """InfoNCE of a query against a fixed concept gallery.

    Positive for row `b` is `gallery[concept_id[b]]`.  The gallery is the set of
    train concepts (1,654 rows), asserted disjoint from the 200 test concepts, so
    this term cannot leak evaluation-set targets.  Centring matches the CLIP
    branch used by `soft_label_contrastive` -- without it the hub bias of the
    CLIP space reappears.
    """
    if center:
        query = center_normalize(query)
        gallery = center_normalize(gallery)
    else:
        query = F.normalize(query.float(), dim=-1)
        gallery = F.normalize(gallery.float(), dim=-1)
    return F.cross_entropy(logit_scale * query @ gallery.t(), concept_id)


def memory_retrieve(query: torch.Tensor, gallery: torch.Tensor,
                    logit_scale: torch.Tensor, k: int = 16,
                    center: bool = True) -> torch.Tensor:
    """Differentiable top-k soft retrieval over a concept gallery.

    Full softmax over a flat similarity profile collapses to the gallery mean
    (UCK measured mem→IP cosine 0.64 vs a constant 0.61).  top-k keeps per-row
    mass on the nearest concepts.  Returns L2-normalised retrieved vectors of the
    same shape as `query`.
    """
    if center:
        q = center_normalize(query)
        g = center_normalize(gallery)
    else:
        q = F.normalize(query.float(), dim=-1)
        g = F.normalize(gallery.float(), dim=-1)
    sim = q @ g.t()
    topv, topi = sim.topk(min(k, g.shape[0]), dim=-1)
    # Temperature is applied in the soft-max over the *selected* keys only, so a
    # large logit_scale still sharpens within the neighbourhood rather than
    # collapsing onto the single nearest neighbour of the full gallery.
    weights = torch.softmax(topv * logit_scale, dim=-1)
    retrieved = (weights.unsqueeze(-1) * g[topi]).sum(dim=1)
    return F.normalize(retrieved, dim=-1)


def memory_alignment(query: torch.Tensor, gallery: torch.Tensor,
                     logit_scale: torch.Tensor, k: int = 16) -> torch.Tensor:
    """Pull the query toward its top-k retrieved concept prototype.

    Complements `gallery_nce`: the NCE term decides *which* concept the query
    should retrieve, this term asks the query to sit near that soft prototype so
    the retrieved vector (what Stage 3 would condition on) stays informative.
    """
    retrieved = memory_retrieve(query, gallery, logit_scale, k=k, center=True)
    q = center_normalize(query)
    return (1.0 - (q * retrieved).sum(dim=-1)).mean()


# --- loss assembly -----------------------------------------------------------
@dataclass
class LossWeights:
    """Weights for the nine alignment/invariance terms plus the two anti-collapse ones.

    Changed from the design's Phase-2 defaults in exactly one place, for a measured
    reason.  The design gave `L_time` weight 3.0 as "dominant", and in the run that
    collapsed it produced ~58% of the total gradient.  That dominance was justified
    by treating the term as the main supervision; the audit showed the term's
    *formulation* was wrong -- it matched EEG token i to VAE raster patch i, a
    correspondence that does not exist, and a randomly permuted patch sequence
    scored 0.9557 against the true correspondence's 0.9542 (gap -0.0015).  So the
    largest share of the gradient was spent on an objective with no signal, and the
    only way to reduce it was to make both sides uniform, i.e. collapse.

    `set_alignment` replaces it with a permutation-invariant set matching that
    carries batch negatives, which removes the reason for the 3.0 weight: a term
    whose optimum is *not* a constant does not need to outvote the others to be
    respected.  `time` and `trial` are therefore at 1.0, in line with the primary
    term, and the freed budget goes to `var`/`cov`, which is what actually prevents
    the failure.  `text`/`vae` stay low because category-level text under-determines
    which of ten images of a dog was shown, and pixel-level appearance is not the
    encoder's job; `adv`/`dist` stay capped because an unbounded domain-adversarial
    loss discards task-relevant signal along with subject-specific signal.

    If a run still shows `time` or `trial` dominating, `GradientBudget` in
    `loso.diagnostics` measures the actual share -- the weights alone are not the
    budget, since a term's share also depends on the gradient magnitude it produces.

    `var`/`cov` are the one pair whose balance is not a judgement call.  VICReg's
    published coefficients are `var_coeff=25, cov_coeff=1`, i.e. a gradient ratio of
    1/25, and those coefficients are defined against a specific normaliser (the
    covariance term divides by `d`, not by `d*(d-1)`; see `loso.losses.align`).  The
    weights here are 1.0 and 0.1, which were *measured* to give a `cov`/`var` gradient
    ratio of 0.0446 against the published 0.0400 -- so they are right as they stand, and
    the 511x discrepancy that made `cov` inert was in the normaliser rather than here.
    Changing either weight without re-measuring the ratio breaks that correspondence.
    """

    img: float = 1.0
    text: float = 0.2
    dino: float = 0.3           # kept, but below the UCK primary
    vae: float = 0.1
    #: Off by default under the UCK recipe.  `set_alignment` materialises a
    #: (B, n_tok, B, n_patch) similarity of ~4 GB at batch 512, and the measured
    #: forward share on CPU was already 12% of the step with a 4x-smaller batch --
    #: on the real path it dominates wall time without moving retrieval.  UCK's
    #: measured substitute is the concept-gallery NCE below.
    time: float = 0.0
    #: Off by default, and this is the measured conclusion rather than a preference.
    #: `measure_gradient_budget` on the real path reported `trial=70-73%` of the
    #: encoder gradient.  Two facts make that a defect rather than a strong signal:
    #:
    #: 1. It is redundant.  The grouped sampler puts 36 trials of one image in the
    #:    batch, and those 36 rows already share one identical CLIP target, so
    #:    `terms["img"]` pulls exactly the same pairs together.  `trial` adds no
    #:    constraint the primary term does not already impose.
    #: 2. It is a pure attractor at this group size.  All 36 rows are mutual
    #:    positives, so within a group the term has no repulsive component at all --
    #:    its only opposition comes from the 476 rows outside the group, and reducing
    #:    *those* similarities is the cheapest way to lower it.  The gradient it
    #:    produces is therefore dominated by "make distant trials more similar".
    #:
    #: The consequence was measured end to end: with `trial=1.0` the share of
    #: `z_inv`'s energy in its batch-mean direction rose from 49.5% to 80.3% over the
    #: preview, while `terms["trial"]` itself never left its uniform plateau (6.30
    #: against ln(512) = 6.24 for a constant).  So the term took 70% of the gradient,
    #: bought nothing on its own objective, and spent that gradient on the shared
    #: direction the regularisers were supposed to be removing.
    #:
    #: Set to a non-zero value only together with a group size small enough that the
    #: term retains negatives inside each group; the weight is not the safe knob here,
    #: because at weight 0.1 the measured share was still large.
    trial: float = 0.0
    #: UCK primary: InfoNCE against the 1,654-row train-concept gallery.  Measured
    #: in NeuroBridge to be the term that actually moves retrieval when the in-batch
    #: soft-label InfoNCE alone plateaued near its constant-encoder baseline.
    gallery: float = 1.0
    #: UCK auxiliary: pull the query toward its top-k soft prototype over the same
    #: gallery.  Kept below `gallery` so the NCE decides the ranking and this only
    #: densifies the retrieved vector Stage 3 would condition on.
    mem: float = 0.3
    adv: float = 0.3            # design: must not dominate
    dist: float = 0.3
    orth: float = 0.1
    var: float = 1.0            # VICReg variance hinge (anti-collapse)
    cov: float = 0.1            # VICReg covariance penalty (anti-collapse)
    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def weighted_total(terms: dict[str, torch.Tensor], weights: LossWeights,
                   active: set[str] | None = None) -> torch.Tensor:
    """Sum weighted terms, ignoring weights with no corresponding term.

    Raises when a term named in `active` produced no value.  That is the dangerous
    direction: a silently missing term turns an intended ablation into an accident,
    e.g. a run that claims adversarial training is on while no adversarial gradient
    is ever applied.
    """
    if active is not None:
        missing = sorted(active - set(terms))
        if missing:
            raise KeyError(
                f"loss terms {missing} are enabled but produced no value; "
                f"available: {sorted(terms)}"
            )
    total = None
    for name, value in terms.items():
        if active is not None and name not in active:
            continue
        w = getattr(weights, name, None)
        if w is None:
            raise KeyError(f"loss term {name!r} has no weight in {type(weights).__name__}")
        if w == 0.0:
            continue
        total = value * w if total is None else total + value * w
    if total is None:
        raise ValueError(f"no active loss term produced a value; terms={sorted(terms)}")
    return total


def same_concept_mask(concept_index: torch.Tensor) -> torch.Tensor:
    """Same-concept mask over a batch, for restricting soft positives to peers."""
    return (concept_index.unsqueeze(0) == concept_index.unsqueeze(1)).float()


def build_soft_positive_mask(image_slot: torch.Tensor) -> torch.Tensor:
    """Same-image mask over a batch (all repetitions of one stimulus)."""
    return (image_slot.unsqueeze(0) == image_slot.unsqueeze(1)).float()
