"""Losses for cross-modal alignment."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCE(nn.Module):
    """Symmetric contrastive loss (CLIP-style) with a learnable temperature.

    The image side is L2-normalised, so this reduces to scaled cosine logits.

    `softplus` reproduces SAMGA's `--softplus` flag, and it matters more than the
    name suggests. The scale is applied to a *temperature* parameter:

        softplus : scale = log(1 + exp(theta))   ->  theta=log(1/0.07) gives 2.73
        exp      : scale = exp(theta)            ->  theta=log(1/0.07) gives 14.29

    So softplus makes the objective ~5x softer at the same initialisation. That is
    a strong regulariser: it caps how peaked the logits can become, which is
    exactly the failure mode of a high-capacity encoder on a small EEG set.

    NOTE: this parameter is the temperature. Applying softplus to the *embeddings*
    instead (a bug that once lived in model.py) is a different and harmful thing:
    it makes every embedding non-negative, so after L2 normalisation all vectors
    crowd into the positive orthant and every pairwise cosine similarity is
    inflated -- the contrastive signal loses its resolution.
    """

    def __init__(self, init_temp: float = 0.07, softplus: bool = True,
                 learnable: bool = True,
                 l2norm_a: bool = True, l2norm_b: bool = True) -> None:
        super().__init__()
        # store log(1/temp) as a parameter, as in CLIP / SAMGA
        self.logit_scale = nn.Parameter(
            torch.tensor(float(torch.log(torch.tensor(1.0 / init_temp)))), requires_grad=learnable
        )
        self.softplus = softplus
        # `l2norm_a=False` reproduces the released EEGiT code, which normalises only
        # the image side before the loss (`img_z = img_z / img_z.norm(...)`) and
        # passes the EEG embedding in raw. The EEG embedding's norm therefore stays
        # free and multiplies the logits, i.e. it is a learned per-sample scale on
        # top of the fixed temperature. Retrieval normalises both sides, official
        # included, so this is a training-only difference -- and a large one: it
        # changes what "similar" means during optimisation, not just how peaked the
        # softmax is.
        self.l2norm_a = l2norm_a
        self.l2norm_b = l2norm_b

    def effective_scale(self) -> torch.Tensor:
        """The multiplier currently applied to the cosine logits (for logging)."""
        s = self.logit_scale
        return F.softplus(s) if self.softplus else s.exp()

    def forward(self, a: torch.Tensor, b: torch.Tensor,
                groups: torch.Tensor | None = None) -> torch.Tensor:
        """Symmetric contrastive loss, optionally multi-positive.

        `groups` is the STIMULUS INDEX of each row -- SCORE's `y_i` -- and it is the
        whole difference between an inter-subject run that is set up correctly and one
        that fights itself. In a LOSO fold `expand_loso_images` tiles every image
        feature once per source subject, so nine rows in a batch share one stimulus.
        With `groups=None` those nine rows are each other's NEGATIVES (the `arange`
        labels say exactly one of them is right), and the objective spends its
        gradient pushing apart nine subjects who looked at the same picture. SCORE
        Eq. 1 is the fix: rows with the same stimulus are all positives.

        Scale, and why this is an average rather than SCORE's sum
        --------------------------------------------------------
        SCORE writes ``L_MP = L_{E->I} + L_{I->E}``, a sum. This returns
        ``0.5 * (L_{E->I} + L_{I->E})``, the same average our pairwise path uses. That
        factor of two is deliberate and it is what makes `--multipos` a single-variable
        ablation: with every group a singleton the mask is the identity and this
        expression is *exactly* the pairwise InfoNCE below -- same value, not merely
        the same shape -- so turning the flag on changes only whether co-stimulus rows
        are positives, never the loss scale or the effective learning rate. Matching
        SCORE's literal sum would fold a 2x scale change into the same comparison.
        `test_epd_multipos.py` pins the degeneracy down numerically.
        """
        if self.l2norm_a:
            a = F.normalize(a, dim=-1)
        if self.l2norm_b:
            b = F.normalize(b, dim=-1)
        scale = self.effective_scale().clamp(max=100.0)
        logits = scale * (a @ b.t())
        if groups is None:
            labels = torch.arange(a.shape[0], device=a.device)
            return 0.5 * (
                F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)
            )

        # `mask[i, j]` iff row j is a positive of row i. Includes i itself (SCORE's
        # P(i) contains i), which is what makes `mask.sum(1)` the group size and makes
        # the singleton case collapse to the diagonal.
        mask = groups[:, None] == groups[None, :]
        size = mask.sum(dim=1)
        # The stimulus of a row is the image it saw, so every row belongs to at least
        # one group. A zero would be a silent NaN in the division below.
        if bool((size == 0).any()):
            raise ValueError("multi-positive mask has an empty group; every row is its "
                             "own positive, so this cannot happen and means `groups` "
                             "was built from the wrong tensor")
        logp = F.log_softmax(logits, dim=1)
        e2i = -(logp * mask).sum(dim=1).div(size).mean()
        logp_t = F.log_softmax(logits.t(), dim=1)
        mask_t = mask.t()
        i2e = -(logp_t * mask_t).sum(dim=1).div(mask_t.sum(dim=1)).mean()
        return 0.5 * (e2i + i2e)


def stimulus_groups_from_ids(ids: torch.Tensor) -> torch.Tensor:
    """Relabel an already-correct stimulus id vector to compact group indices.

    Thin on purpose: the identity of a stimulus is decided by the dataset, and this
    only closes the gap between arbitrary id values and the ``0..n_groups-1`` that
    ``InfoNCE``'s mask arithmetic wants. Nothing is inferred from the features.

    `torch.unique` sorts, so the labels are arbitrary but consistent; only equality
    between them is ever used, so the ordering does not matter.
    """
    if ids.ndim != 1:
        raise ValueError(f"expected a 1-D id vector, got {tuple(ids.shape)}")
    return torch.unique(ids, return_inverse=True)[1].to(ids.device)


def stimulus_groups(image_feat: torch.Tensor) -> torch.Tensor:
    """Stimulus index per row, recovered by grouping rows with IDENTICAL features.

    TEST/DIAGNOSTIC ONLY -- do not use this to build the training mask, because it is
    only correct when the image target is subject-independent, and in that case the
    multi-positive loss it feeds is provably equal to the pairwise one (see
    `expand_loso_images`). The two regimes are therefore exactly the ones where it is
    respectively inert and wrong:

      * subject-independent target (`--target-fusion single`): the copies are
        bit-identical, so this recovers the right groups -- and the loss collapses to
        the pairwise baseline regardless, so the grouping changes nothing.
      * subject-dependent target (`--target-fusion routed_sr`): each subject has its
        own target vector for the same picture, so identical features no longer mean
        "same picture" and this splits one stimulus into one group per subject. The
        mask then misses precisely the cross-subject pairs it exists to supply.

    It is kept because it is the cheap way to CHECK tiling in a test
    (`test_epd_multipos.py` uses it to confirm `expand_loso_images` produced the
    layout it claims), where "did the copies come out identical" is the question
    being asked.

    `torch.unique` sorts, so the group ids are arbitrary but consistent. Only
    equality between ids is ever used. O(B log B) rather than O(B^2 D).
    """
    if image_feat.ndim != 2:
        raise ValueError(f"expected (B, D) image features, got {tuple(image_feat.shape)}")
    with torch.no_grad():
        _values, inverse = torch.unique(image_feat, dim=0, return_inverse=True)
    return inverse.to(image_feat.device)


def _rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigmas: tuple[float, ...]) -> torch.Tensor:
    d2 = torch.cdist(x, y, p=2) ** 2
    k = 0.0
    for s in sigmas:
        k = k + torch.exp(-d2 / (2.0 * s * s))
    return k / len(sigmas)


def median_heuristic_sigmas(x: torch.Tensor, y: torch.Tensor) -> tuple[float, ...]:
    """Bandwidths scaled to the actual embedding distances.

    Why this is not optional: the kernel below is `exp(-d^2 / 2s^2)`. With sigmas
    pinned at (1,2,4,8) and unnormalised 512-d embeddings whose pairwise distances
    are in the tens, every term underflows to 0, so xx = yy = xy = 0 and the MMD
    gradient is identically zero. The term was therefore *numerically dead* in
    every run that enabled it -- which also invalidates the earlier reading that
    "SAMGA's MMD warm-up hurts here". That result said nothing about MMD; it
    measured its absence.

    The standard fix is the median heuristic: set the bandwidth from the median
    pairwise squared distance, and spread a few multiples around it so the kernel
    has support over more than one scale.
    """
    with torch.no_grad():
        z = torch.cat([x, y], dim=0)
        # Subsample: the full B x B distance matrix at B=512 is fine, but the
        # heuristic is a median and does not need every pair.
        if z.shape[0] > 256:
            pick = torch.linspace(0, z.shape[0] - 1, 256).long().to(z.device)
            z = z[pick]
        d2 = torch.cdist(z, z, p=2) ** 2
        med = float(d2.median().clamp_min(1e-12))
    s = math.sqrt(max(med, 1e-12))
    return (s * 0.5, s, s * 2.0, s * 4.0)


def mmd_rbf(x: torch.Tensor, y: torch.Tensor, sigmas: tuple[float, ...] | None = None) -> torch.Tensor:
    """Maximum mean discrepancy between two embeddings.

    Used as a coarse alignment term: it matches distributions without requiring
    instance correspondence, which stabilises the shared geometry early in
    training before the contrastive term can separate instances.

    Both sides are L2-normalised first. That is deliberate and it is what makes
    the objective well-posed: the contrastive term lives on the unit sphere (so
    only direction carries signal), whereas an unnormalised MMD would be dominated
    by the embedding norms, which are free to drift -- the loss could then be
    reduced without changing any similarity that retrieval actually uses.
    """
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    if sigmas is None:
        sigmas = median_heuristic_sigmas(x, y)
    xx = _rbf_kernel(x, x, sigmas).mean()
    yy = _rbf_kernel(y, y, sigmas).mean()
    xy = _rbf_kernel(x, y, sigmas).mean()
    return (xx + yy - 2.0 * xy).clamp_min(0.0)


def latent_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 regression on a spatial latent field.

    L1 rather than L2 is a deliberate choice for this target. The VAE latent is
    not a semantic embedding but a *low-frequency layout*: what the generation
    stack needs from it is the coarse spatial arrangement (where the object is,
    how big, which orientation), because SDXL re-synthesises every detail through
    img2img. L2's optimum is the conditional mean, which on a 4x64x64 field is a
    smooth blob -- and, more importantly, L2 spends its gradient on the
    high-variance residual that the decoder is going to overwrite anyway. L1 is
    the conditional median and is markedly sharper on exactly this kind of field.

    Both sides are compared after per-channel normalisation (see train.py), so
    this number is in units of the latent's own standard deviation and is
    comparable across runs.
    """
    if pred.shape != target.shape:
        raise ValueError(f"latent shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    return F.l1_loss(pred, target)


def _as_channel_std(target_std, pred: torch.Tensor, n_ch: int) -> torch.Tensor:
    """Coerce a per-channel target scale to a float tensor on the prediction's device.

    Accepts a tensor OR an array-like, because the production caller does not have a
    tensor: `train.py` computes the fit-split scale straight off the memmapped latent
    cache (`np.load(...).std(axis=(0, 2, 3))`), so it hands over a numpy array. The
    first submitted run died here with `'numpy.ndarray' object has no attribute 'to'`
    -- at the very first optimiser step, after the whole smoke gate had passed,
    because the unit tests exercised the tensor path while the pipeline uses the
    numpy one. Coercing in one place is what keeps those two paths from diverging
    again; `torch.as_tensor` is a no-op when it is already a tensor.
    """
    sd = torch.as_tensor(target_std, dtype=pred.dtype, device=pred.device).reshape(-1)
    if sd.numel() != n_ch:
        raise ValueError(f"target_std has {sd.numel()} entries but the prediction has "
                         f"{n_ch} channels")
    return sd


def latent_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L2 on the VAE latent field. The alternative to `latent_l1`, chosen by `--vae-loss`.

    Why the choice is not cosmetic
    ------------------------------
    L1's optimum is the conditional MEDIAN. For a target as weakly predictable as this
    one -- the VAE latent's best linear read-out from the raw EEG is 5.07% instance
    Top-1 on a 0.50% chance floor -- the conditional median over the training set is
    close to the global mean field, and the loss is therefore nearly flat between "a
    varying best guess" and "that best guess plus zero-mean noise". Three runs through
    the scalp-topography interface collapsed under it, the last at a variance ratio of
    0.6577 with a per-sample correlation of +0.063 against the constant predictor's
    +0.157.

    MSE's optimum is the conditional MEAN, which is not flat in the same way: a
    constant prediction pays the target's entire variance as loss, so the gradient
    pressure toward an instance-specific estimate is first-order rather than absent.

    Both sides are compared after per-channel normalisation (see train.py), so this
    number is in units of the latent's own standard deviation.
    """
    if pred.shape != target.shape:
        raise ValueError(f"latent shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    return F.mse_loss(pred, target)


def variance_floor(pred: torch.Tensor, target_std, margin: float = 1.0) -> torch.Tensor:
    """Hinge penalty on predicting a field that varies LESS than the target does.

    Why this is not optional here
    -----------------------------
    L1's optimum is the conditional median. When the target is only weakly
    predictable -- and the VAE latent is, with a linear ceiling of 5.07%/6.00%
    instance Top-1 against a 0.667% chance floor -- the conditional median is close
    to the conditional mean over the whole training set, i.e. a near-constant field.
    A constant field scores a perfectly respectable L1 while carrying no instance
    information at all, so the objective is satisfied by exactly the failure the
    generation stack cannot use.

    This is not a hypothetical. The shipped VAE head was measured at a
    variance ratio of 0.0068 (prediction std / target std) -- five times WORSE than
    a closed-form ridge fit on the same inputs, which reaches 0.4411. The head had
    genuinely collapsed to the mean while its L1 looked fine.

    The term is a one-sided hinge against a per-channel target scale, not an MSE
    toward that scale. A two-sided term would spend gradient pushing variance DOWN
    whenever a sample legitimately predicts a flat image, which is most of a 200-way
    retrieval set. The hinge is silent once the prediction is as variable as the
    target and only ever pushes variance UP, so it cannot fight the L1 term anywhere
    the L1 term is doing something sensible.

    `margin` < 1 relaxes the floor: with real targets that have heavy-tailed scales,
    matching the global std exactly is a stricter requirement than "did not
    collapse", and reaching 0.44 (the ridge baseline) is the target, not 1.0.
    """
    if pred.ndim != 4:
        raise ValueError(f"expected a (B, C, H, W) field, got {tuple(pred.shape)}")
    sd = _as_channel_std(target_std, pred, pred.shape[1])
    p = pred.std(dim=(0, 2, 3), unbiased=False)
    deficit = F.relu(sd * margin - p)
    return (deficit ** 2).mean()


def variance_ratio(pred: torch.Tensor, target_std) -> float:
    """prediction std / target std, averaged over channels. A diagnostic, not a loss.

    Reported per epoch because it is the one number that says whether the structural
    branch is alive: 0.0068 is the collapse that prompted `variance_floor`, and
    ~0.44 is what a closed-form linear map achieves on the same target.
    """
    with torch.no_grad():
        if pred.ndim != 4:
            raise ValueError(f"expected a (B, C, H, W) field, got {tuple(pred.shape)}")
        sd = _as_channel_std(target_std, pred, pred.shape[1])
        p = pred.std(dim=(0, 2, 3), unbiased=False)
        return float((p / sd.clamp_min(1e-8)).mean())


def grad_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 on finite-difference gradients of a single-channel map.

    A pure per-pixel loss on a 64x64 depth map is minimised by the smooth mean,
    which is what the diffusion prior then has to invent structure from. Matching
    the first differences as well puts gradient mass on the edges, so the predicted
    map keeps the silhouette. Only the depth head uses it; the VAE latent already
    has 4 channels of its own internal structure and the same term there competes
    with the low-frequency objective rather than helping it.
    """
    if pred.ndim != target.ndim:
        raise ValueError(f"depth rank mismatch: {pred.ndim} vs {target.ndim}")
    dx_p = pred[..., :, 1:] - pred[..., :, :-1]
    dx_t = target[..., :, 1:] - target[..., :, :-1]
    dy_p = pred[..., 1:, :] - pred[..., :-1, :]
    dy_t = target[..., 1:, :] - target[..., :-1, :]
    return F.l1_loss(dx_p, dx_t) + F.l1_loss(dy_p, dy_t)
