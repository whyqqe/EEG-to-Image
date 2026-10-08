"""Cross-modal distribution matching: the EEG cloud against the image cloud.

This is a faithful port of the reference architecture's `module/loss.py::mmd_rbf`
(SAMGA, `third_party/SAMGA`) -- with ONE deliberate difference in the bandwidth
handling, which is explained below because copying the numbers verbatim would have
produced a silently dead term.

WHY THIS TERM EXISTS AT ALL (and why `mmd_subject` is not it)
-------------------------------------------------------------
The reference's Stage 1 is

    loss = w_mmd * MMD(z_eeg cloud, z_image cloud) + (1 - w_mmd) * InfoNCE,
    w_mmd: 0.9 -> 0.5   (`inter.sh`: --stage1_mmd_start 0.9 --stage1_mmd_end 0.5)

i.e. a *distribution-level warm start*: before the contrast is asked to separate
individual trials, the two clouds are asked to occupy the same region. Our v3 objective
instead contained `mmd_subject`, which matches **subject-against-subject** distributions.
That is a different quantity, and the name "coarse-to-fine" in our config referred to it
by accident. The reference's term is cross-MODAL; the thing we shipped was cross-SUBJECT.

Under the paradigm this project is built on -- subjects are modalities -- the cross-modal
form is the coherent one: it asks "does the EEG cloud look like the image cloud?", which
is a statement about the shared space, whereas "do two subjects' clouds look alike?"
can be satisfied inside a subject-specific subspace that the image cloud never occupies.

WHY THE BANDWIDTHS CANNOT BE COPIED NUMERICALLY
-----------------------------------------------
`mmd_rbf` is a multi-bandwidth RBF kernel,

    K_ab = (1/|S|) * sum_{s in S} exp(-||a - b||^2 / (2 s^2)),   S = (0.1, 0.2, 0.5, 1, 2)

so every value in ``S`` is in the same units as the squared distance between two
samples. The reference feeds it features that are NOT L2-normalised (`inter.sh` passes
`--img_l2norm` but not `--eeg_l2norm`), so its absolute distances are set by whatever
scale `share_encoder` happens to produce there.

Our term is evaluated on L2-normalised embeddings -- the space retrieval is actually
scored in, and the only space in which this term cannot be satisfied by shrinking every
feature toward zero. On unit-norm vectors ``||a - b||^2 in [0, 4]`` with a typical value
near 2, so ``s = 0.1`` gives ``exp(-d^2/0.02) ~= 0`` for essentially every pair: a kernel
that contributes a constant ~0 to ``K_xx``, ``K_yy`` and ``K_xy`` alike. It would not
raise an error, it would just damp 1/5 of the average while looking like a faithful port.

So the *ladder* is kept and the *scale* is measured: ``S = sigma_med * (0.1, 0.2, 0.5,
1, 2)`` where ``sigma_med`` is the median pairwise distance of the batch, computed ONCE
over both clouds. One shared bandwidth set is a correctness requirement, not a detail:
with per-cloud bandwidths the three kernel terms would be evaluated at different scales,
and ``K_xx + K_yy - 2 K_xy`` would no longer be an MMD at all (it can even go negative
before the clamp).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

#: The reference's bandwidth ladder, kept as RELATIVE values (see module docstring).
LADDER: tuple[float, ...] = (0.1, 0.2, 0.5, 1.0, 2.0)


def _pairwise_sq_dists(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """``||x_i - y_j||^2`` via the expand trick (avoids a (B,B,D) intermediate)."""
    x2 = (x ** 2).sum(dim=1, keepdim=True)
    y2 = (y ** 2).sum(dim=1, keepdim=True).t()
    return (x2 + y2 - 2.0 * (x @ y.t())).clamp_min_(0.0)


def _mix_rbf_kernel(dist2: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
    """Mean over the bandwidth ladder of the RBF kernel at squared distance ``dist2``."""
    k = torch.zeros_like(dist2)
    for s in sigmas:
        k = k + torch.exp(-dist2 / (2.0 * s * s))
    return k / sigmas.numel()


def _median_sigma(x: torch.Tensor, y: torch.Tensor, max_n: int = 512) -> torch.Tensor:
    """Median pairwise distance over the pooled sample (one bandwidth set, shared).

    Subsampled above ``max_n`` pooled rows because the statistic only has to be right to
    within the factor of ~2 that separates adjacent ladder rungs, while the O(n^2)
    distance matrix is the expensive part.
    """
    with torch.no_grad():
        z = torch.cat([x, y], dim=0)
        if z.shape[0] > max_n:
            idx = torch.linspace(0, z.shape[0] - 1, max_n).long().to(z.device)
            z = z[idx]
        d2 = _pairwise_sq_dists(z, z)
        med = d2.median().clamp_min(1e-12)
    return med.sqrt()


def mmd_crossmodal(
    x: torch.Tensor,
    y: torch.Tensor,
    sigmas: tuple[float, ...] = LADDER,
    unbiased: bool = True,
    normalize: bool = True,
    max_n: int = 1024,
) -> torch.Tensor:
    """``MMD^2`` between the ``x`` cloud and the ``y`` cloud. Scalar, >= 0.

    ``unbiased`` reproduces the reference exactly: the two self-terms have their
    diagonal removed and are divided by ``B(B-1)``, while the cross term is the plain
    mean. That asymmetry is the standard unbiased MMD estimate and is what makes the
    value go to zero when the clouds genuinely coincide (the biased estimator sits at
    ``2/B`` even for identical distributions, which is the trap `mmd_subject`'s RBF
    variant fell into on this project: its reported 0.016 was almost entirely the
    ``2/N`` self-pair bias rather than a measurement).

    ``x`` and ``y`` must have the same number of rows -- the reference asserts this, and
    the asymmetry above is only a valid estimator when it holds. Our batches do satisfy
    it (one EEG row per image-target row).
    """
    if x.dim() != 2 or y.dim() != 2:
        raise ValueError(f"mmd_crossmodal expects (B, D) inputs, got "
                         f"{tuple(x.shape)} and {tuple(y.shape)}")
    if normalize:
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"mmd_crossmodal needs matched batch sizes (unbiased MMD^2 is only a valid "
            f"estimator then), got {x.shape[0]} and {y.shape[0]}")
    n = x.shape[0]
    if n < 2:
        return x.new_zeros(())
    if max_n and n > max_n:
        # Row-wise subsample, NOT a shuffle of the pairing: `x[i]` and `y[i]` are the
        # same stimulus, and pairing must survive the subsample or the cross term would
        # estimate the MMD between two clouds that no longer correspond row-for-row.
        idx = torch.linspace(0, n - 1, max_n).long().to(x.device)
        x, y = x[idx], y[idx]
        n = max_n

    scale = _median_sigma(x, y).to(x.dtype)
    sig = torch.tensor(sigmas, device=x.device, dtype=x.dtype) * scale

    k_xx = _mix_rbf_kernel(_pairwise_sq_dists(x, x), sig)
    k_yy = _mix_rbf_kernel(_pairwise_sq_dists(y, y), sig)
    k_xy = _mix_rbf_kernel(_pairwise_sq_dists(x, y), sig)

    if unbiased:
        sum_xx = (k_xx.sum() - torch.diagonal(k_xx).sum()) / (n * (n - 1))
        sum_yy = (k_yy.sum() - torch.diagonal(k_yy).sum()) / (n * (n - 1))
        sum_xy = k_xy.mean()
    else:
        sum_xx, sum_yy, sum_xy = k_xx.mean(), k_yy.mean(), k_xy.mean()
    return (sum_xx + sum_yy - 2.0 * sum_xy).clamp_min(0.0)
