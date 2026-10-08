"""Subject-invariance terms: HSIC decorrelation, MMD distribution matching, GRL.

The goal of all three is the same -- "keep the image information, drop the subject
identity" -- but they differ in how hard they push and how stable they are:

  * ``hsic_subject``    (recommended default) -- penalises statistical dependence
    between the embedding and the subject label. Smooth, no min-max game.
  * ``mmd_subject``     -- matches the per-subject embedding *distributions*. Coarser
    than HSIC: it removes location/scale shift but not dependence in place. Defaults to
    a LINEAR kernel, i.e. it matches the per-subject MEANS; the RBF form is kept as an
    ablation and is documented in-function as unable to see the subject shift here.
  * ``SubjectAdversary`` (GRL) -- a classifier trained to predict the subject while
    the encoder is trained to defeat it. Strongest and least stable; kept as an
    ablation because subject information and image information are partly entangled,
    so an aggressive adversary can remove signal.

The plan (doc §4.5.3) recommends HSIC as primary precisely because a hard adversary
can cost image information, and the regularisers here are one-line ablations rather
than separate code paths.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------- helpers
def _median_sigma(z: torch.Tensor, max_n: int = 256) -> float:
    with torch.no_grad():
        if z.shape[0] > max_n:
            idx = torch.linspace(0, z.shape[0] - 1, max_n).long().to(z.device)
            z = z[idx]
        d2 = torch.cdist(z, z, p=2) ** 2
        med = float(d2.median().clamp_min(1e-12))
    return math.sqrt(max(med, 1e-12))


def _rbf(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
    return torch.exp(-torch.cdist(x, y, p=2) ** 2 / (2.0 * sigma * sigma))


def hsic(x: torch.Tensor, y: torch.Tensor, sigma_x: float | None = None,
         sigma_y: float | None = None, unbiased: bool = True) -> torch.Tensor:
    """Normalised HSIC between two batches of features (lower = less dependence).

    `x` is the embedding, `y` the subject one-hot (or any label encoding). The
    normalisation by ``sqrt(HSIC(x,x) * HSIC(y,y))`` makes the value comparable
    across batch sizes and feature scales, which matters because it is a loss weight
    that has to mean the same thing in every run.
    """
    n = x.shape[0]
    if n < 3:
        return x.new_zeros(())
    if sigma_x is None:
        sigma_x = _median_sigma(x)
    if sigma_y is None:
        sigma_y = _median_sigma(y)
    k = _rbf(x, x, sigma_x)
    l = _rbf(y, y, sigma_y)
    h = torch.eye(n, device=x.device, dtype=x.dtype)
    if unbiased:
        h = h - 1.0 / n
        k = k.clone()
        l = l.clone()
        k.fill_diagonal_(0.0)
        l.fill_diagonal_(0.0)
    hk = h @ k @ h
    hsic_xy = (hk * l).sum() / ((n - 1) ** 2 if unbiased else n ** 2)
    hsic_xx = (hk * k).sum() / ((n - 1) ** 2 if unbiased else n ** 2)
    hsic_yy = (h @ l @ h * l).sum() / ((n - 1) ** 2 if unbiased else n ** 2)
    return hsic_xy / (hsic_xx.clamp_min(1e-8) * hsic_yy.clamp_min(1e-8)).sqrt()


def subject_onehot(subject: torch.Tensor, n_subjects: int) -> torch.Tensor:
    return F.one_hot(subject.long(), num_classes=n_subjects).to(torch.float32)


def hsic_subject(z: torch.Tensor, subject: torch.Tensor, n_subjects: int) -> torch.Tensor:
    """Decorrelate the shared embedding from the subject label."""
    if z.shape[0] < 3 or subject.unique().numel() < 2:
        return z.new_zeros(())
    zc = z - z.mean(dim=0, keepdim=True)
    y = subject_onehot(subject, n_subjects)
    return hsic(zc, y)


def mmd_subject(z: torch.Tensor, subject: torch.Tensor, kernel: str = "linear",
                sigmas: tuple[float, ...] | None = None,
                max_pairs: int = 6) -> torch.Tensor:
    """Mean MMD across subject pairs (0 when fewer than two subjects are present).

    TWO KERNELS, AND THE CHOICE IS NOT COSMETIC
    -------------------------------------------
    ``kernel="linear"`` (the default) uses ``k(x, y) = <x, y>``, for which
    ``MMD^2(E_a, E_b) = ||E_a[z] - E_b[z]||^2`` -- the squared distance between the two
    subjects' MEAN embeddings. ``kernel="rbf"`` is the multi-bandwidth form that an
    earlier version used, kept as an ablation.

    The RBF form was measured to be unable to see the subject shift on this
    representation, for a reason that no bandwidth choice fixes. On the L2-normalised
    embedding the per-sample spread is ~1.41 (concentration of measure puts every pair
    near ``sqrt(2)``) while the subject-MEAN displacement is only ~0.195 -- a factor of
    seven. A bandwidth wide enough to cover a whole subject cannot resolve the mean
    separation, and one narrow enough to resolve it treats each subject's own samples as
    mutually unrelated. Concretely, on a trained checkpoint:

      * the reported value was ~0.016, of which the ``_rbf(a, a)`` diagonal (identically
        1, so zero gradient) accounts for ``2/N`` = 0.0156 -- i.e. the number was almost
        entirely the self-pair bias, not a measurement of subject mismatch;
      * the *unbiased* RBF-MMD^2 (diagonal removed, so the part that actually carries
        gradient) was ~1e-4 across the whole bandwidth sweep;
      * the term therefore supplied 0.9% of the gradient reaching the encoder while
        holding 26% of the reported loss value.

    The linear kernel has no bandwidth to mis-set, and its gradient share measured
    8.1% at weight 1.0 on the same checkpoint -- an order of magnitude more signal.

    ``max_pairs`` caps the number of subject pairs averaged, so the term costs the same
    when a batch happens to contain many subjects.
    """
    subs = subject.unique()
    if subs.numel() < 2:
        return z.new_zeros(())
    if kernel not in ("linear", "rbf"):
        raise ValueError(f"kernel must be 'linear' or 'rbf', got {kernel!r}")
    z = F.normalize(z, dim=-1)

    mus = None
    if kernel == "linear":
        mus = torch.stack([z[subject == s].mean(dim=0) for s in subs])

    if kernel == "rbf" and sigmas is None:
        s = _median_sigma(z)
        sigmas = (s * 0.5, s, s * 2.0, s * 4.0)

    def _mmd(i: int, j: int) -> torch.Tensor:
        if kernel == "linear":
            return (mus[i] - mus[j]).pow(2).sum()
        a = z[subject == subs[i]]
        b = z[subject == subs[j]]
        if a.shape[0] < 2 or b.shape[0] < 2:
            return z.new_zeros(())
        xx, yy, xy = 0.0, 0.0, 0.0
        for sg in sigmas:
            xx = xx + _rbf(a, a, sg).mean()
            yy = yy + _rbf(b, b, sg).mean()
            xy = xy + _rbf(a, b, sg).mean()
        return (xx + yy - 2.0 * xy).clamp_min(0.0) / len(sigmas)

    terms = []
    for i in range(subs.numel()):
        for j in range(i + 1, subs.numel()):
            terms.append(_mmd(i, j))
            if len(terms) >= max_pairs:
                break
        if len(terms) >= max_pairs:
            break
    return torch.stack(terms).mean() if terms else z.new_zeros(())


# ----------------------------------------------------------------- adversary
class _GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:  # type: ignore[override]
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):  # type: ignore[override]
        return -ctx.lambd * grad, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return _GradientReversal.apply(x, lambd)


class SubjectAdversary(nn.Module):
    """Subject classifier behind a gradient-reversal layer (ablation arm)."""

    def __init__(self, d_in: int, n_subjects: int, hidden: int = 256,
                 dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_subjects),
        )

    def forward(self, z: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
        return self.net(grad_reverse(z, lambd))

    @staticmethod
    def loss(logits: torch.Tensor, subject: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, subject.long())
