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

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.l2norm_a:
            a = F.normalize(a, dim=-1)
        if self.l2norm_b:
            b = F.normalize(b, dim=-1)
        scale = self.effective_scale().clamp(max=100.0)
        logits = scale * (a @ b.t())
        labels = torch.arange(a.shape[0], device=a.device)
        return 0.5 * (
            F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)
        )


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
