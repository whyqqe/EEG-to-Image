"""Anti-collapse regularisers, the EMA prototype pool, and spectral concentration.

Small batches + a strong alignment objective is a recipe for representation collapse:
the encoder can drive the alignment loss down by mapping every input to nearly the
same vector. Two independent guards are provided.

  * ``vicreg_terms`` -- VICReg's variance hinge plus a covariance penalty. The
    variance term is one-sided (only pushes variance UP), so it cannot fight the
    contrastive term on samples that legitimately sit close together.

  * ``spectral_concentration`` -- the v4 replacement for the variance hinge. It
    penalises the fraction of second-moment energy sitting BEYOND the top ``r0``
    eigendirections, i.e. it asks the representation to be *low-dimensional* instead of
    asking every one of its ``d`` axes to reach unit variance. See the function
    docstring for the measurement that motivated the switch.

  * ``PrototypeEMA`` -- an EMA pool of neural class anchors (SUP-MCRL's PPA /
    "asymmetric prototype alignment"). It supplies two things that the pairwise
    contrasts cannot: a *denoised* target for a trial (a single EEG trial is a very
    noisy sample of its class, so pulling it toward a running average of its own class
    is a much better-conditioned signal than pulling it toward one other noisy trial),
    and stable extra positives, which raises the effective batch size without raising
    memory.

THE ASYMMETRY IS THE POINT
--------------------------
``proto_contrast`` and ``image_anchor_contrast`` look like the same operation applied
to two tensors. They are not, and the direction is deliberate.

The prototypes are built from EEG. The contrastive image term already pulls EEG toward
the image space; if the prototype term also pulled EEG toward its own EEG prototype, the
two terms would be competing for the same degrees of freedom and the encoder could
satisfy both by collapsing. Instead the *anchor* term pushes the IMAGE embeddings toward
the EEG anchors (the class prototypes), i.e. the frozen-and-learned image head is asked
to agree with where the EEG data actually piles up. That is the direction the published
result reports: aligning visual embeddings to neural class anchors beats aligning EEG to
visual prototypes, because the neural side carries the class structure the retrieval task
actually scores, and the visual side is the one with a learnable head to absorb it.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .contrastive import effective_logit_scale


def vicreg_terms(z: torch.Tensor, var_target: float = 1.0,
                 cov_weight: float = 0.04, eps: float = 1e-4) -> dict[str, torch.Tensor]:
    """Variance hinge + off-diagonal covariance penalty.

    Computed on the *unnormalised* embedding: after L2 normalisation every vector has
    unit norm by construction, so the variance term would be measuring nothing.
    """
    if z.dim() != 2:
        raise ValueError(f"expected (B, D) embeddings, got {tuple(z.shape)}")
    n, d = z.shape
    if n < 2:
        zero = z.new_zeros(())
        return {"var": zero, "cov": zero}
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    var_loss = F.relu(var_target - std).mean()
    zc = z - z.mean(dim=0, keepdim=True)
    cov = (zc.t() @ zc) / max(1, n - 1)
    off = cov - torch.diag_embed(torch.diagonal(cov))
    cov_loss = (off ** 2).sum() / d
    return {"var": var_loss, "cov": cov_weight * cov_loss}


def _cov_eigs(z: torch.Tensor, jitter: float = 1e-7) -> torch.Tensor:
    """Eigenvalues of the batch covariance, ascending, computed on the SHORT side.

    Eigendecomposing ``(D, D)`` when ``n << D`` is 512^3 work to extract 200 meaningful
    directions. The non-zero spectrum of ``Z^T Z`` and ``Z Z^T`` is identical, so when
    the batch is the smaller dimension the ``(n, n)`` Gram matrix gives the same tail
    energies for a fraction of the cost. The jitter keeps `eigvalsh`'s backward pass
    well conditioned when the spectrum has near-repeated values, which it does at
    initialisation.
    """
    n, d = z.shape
    zc = z - z.mean(dim=0, keepdim=True)
    if n <= d:
        gram = (zc @ zc.t()) / max(1, n - 1)
    else:
        gram = (zc.t() @ zc) / max(1, n - 1)
    gram = 0.5 * (gram + gram.t())
    tr = torch.diagonal(gram).sum().clamp_min(1e-12)
    gram = gram + (jitter * tr / gram.shape[0]) * torch.eye(
        gram.shape[0], device=z.device, dtype=z.dtype)
    return torch.linalg.eigvalsh(gram).clamp_min(0.0)


def spectral_concentration(z: torch.Tensor, r0: int = 16) -> torch.Tensor:
    """Fraction of the batch's second-moment energy held BEYOND the top ``r0`` eigendirections.

    ``L_spec = sum_{j > r0} lambda_j / sum_j lambda_j``, to be MINIMISED.

    THIS REPLACES VICReg's VARIANCE HINGE, AND THE DIFFERENCE IS A DIRECTION
    ------------------------------------------------------------------------
    ``relu(var_target - std_d)`` is applied per dimension, so it pushes ALL ``d`` axes
    outward with equal force. Measured on a trained v3 checkpoint
    (`scripts/probe_embedding_scale.py`): 0 of 512 dimensions reached std 1.0 and the
    per-dimension std had cv = 0.127 -- a permanently unsatisfiable push on a
    representation whose task-relevant variance lives in ~16 directions (the subspace
    probe puts 93-96% of query and 98% of target variance in the top 16 PCs). It could
    never win, because the norm is bounded by the contrast, yet it held 18.8% of the
    gradient reaching the encoder. That is a regulariser spending the encoder's capacity
    on a fight it cannot win, in a direction (inflate the tail) opposite to the one the
    task needs (concentrate the head).

    A ratio has two properties the hinge lacks, and each was the reason to switch:

      * **bounded and satisfiable** -- it is a fraction in ``[0, 1]`` with an attainable
        minimum given the data, so it can actually be driven down instead of pushing
        forever;
      * **scale invariant** -- the eigenVECTORS decide the value, not the eigenVALUES'
        magnitude, so it cannot be satisfied by inflating the embedding and cannot fight
        the contrast over the norm.

    KNOWN PATHOLOGY -- DO NOT SHIP THIS WITH ``spec > 0``
    ----------------------------------------------------
    The third property this docstring used to claim ("aligned with the task -- it asks for
    a low-dimensional head") is FALSE, and the reason is a one-line consequence of the
    definition: ``tail = 1 - head``, so MINIMISING the tail is MAXIMISING the head, and
    the maximiser of the head fraction is a rank-1 representation. The term therefore
    cannot tell "concentrated into the 16 measured directions" from "collapsed onto one".
    Measured on the shipped configuration: the value is ``~1e-07`` at rank 1 AND
    ``~2e-07`` at rank 16 -- indistinguishable -- while the trained v4 checkpoint came back
    with an effective rank of 6.5 against the v3.2 baseline's 14.4 on the same fold, and
    ``spec_top`` pinned to 0.999 by epoch 8 and never moved again. That is the collapse
    signature, not the intended concentration.

    An ablation then measured what the term was actually doing: removing it
    (``spec: 0``) changed Top-1 by ``-0.00pp`` (paired, 3 seeds, p = 1.00) and moved the
    effective rank from 6.5 to 6.7. So it was not merely harmful, it was INERT at the
    weight it shipped with -- which is why the correct move was to set ``spec: 0`` rather
    than to tune it. The mathematically correct replacement is
    :func:`spectral_rank_target`, which penalises deviation of the effective rank from
    ``r0`` in BOTH directions and therefore has no degenerate optimum.

    Kept (rather than deleted) so recorded runs and the v3 bit-exactness guarantee still
    load, and because its measured inertness is itself a result worth being able to
    reproduce.

    ``r0`` defaults to 16 because that is the measured dimension of the concept manifold,
    not a tuned constant. ``r0 >= d`` (or an all-but-one-degenerate batch) returns 0 with
    no gradient, so the term is inert rather than NaN at the degenerate end.
    """
    if z.dim() != 2:
        raise ValueError(f"expected (B, D) embeddings, got {tuple(z.shape)}")
    if z.shape[0] < 2:
        return z.new_zeros(())
    lam = _cov_eigs(z)
    if r0 >= lam.numel():
        return z.new_zeros(())
    total = lam.sum()
    if float(total) <= 0.0:
        return z.new_zeros(())
    return lam[: lam.numel() - r0].sum() / total


def spectral_rank_target(z: torch.Tensor, r0: int = 16) -> torch.Tensor:
    """Bilateral rank target: ``((effective_rank - r0) / r0) ** 2``.

    This is the corrected form of :func:`spectral_concentration`. That function
    minimises the TAIL energy fraction, which is the same as maximising the head
    fraction and therefore has total collapse (rank 1) as its global optimum -- it
    cannot distinguish rank 1 from rank 16, both of which measure ~1e-07. This one has
    an interior optimum by construction, so it penalises a too-diffuse spectrum AND a
    too-collapsed one.

    ``effective_rank = exp(H(p))`` with ``p`` the normalised spectrum is the same ruler
    the v3.2 audit and `spectrum_report` already use, so the target and the diagnostic
    are the same quantity -- there is no mismatch between what is optimised and what is
    reported. ``r0 = 16`` is the MEASURED dimension of the concept manifold (93-96% of
    query variance, 98% of target variance in the top 16 PCs), not a tuned constant.

    Note the term is scale-invariant like the ratio it replaces (the spectrum is
    normalised before the entropy), so it cannot be satisfied by inflating the embedding.
    """
    if z.dim() != 2:
        raise ValueError(f"expected (B, D) embeddings, got {tuple(z.shape)}")
    if r0 < 1:
        raise ValueError(f"r0 must be >= 1, got {r0}")
    if z.shape[0] < 2:
        return z.new_zeros(())
    lam = _cov_eigs(z)
    total = lam.sum()
    if float(total) <= 0.0:
        return z.new_zeros(())
    p = (lam / total).clamp_min(1e-12)
    p = p / p.sum()
    # `log_softmax`-style entropy, written explicitly so the gradient is exact even when
    # a direction's share is at the clamp floor.
    eff_rank = torch.exp(-(p * p.log()).sum())
    return ((eff_rank - float(r0)) / float(r0)).pow(2)


@torch.no_grad()
def spectrum_report(z: torch.Tensor, r0: int = 16) -> dict[str, float]:
    """Read-only spectrum summary for the run log.

    ``top_frac`` is the number the spectral term is trying to raise, and
    ``effective_rank`` is ``exp(entropy of the normalised spectrum)`` -- the same ruler
    the v3.2 audit used, kept identical so the two versions are comparable.
    """
    if z.dim() != 2 or z.shape[0] < 2:
        return {"top_frac": 0.0, "tail_frac": 0.0, "effective_rank": 0.0}
    lam = _cov_eigs(z)
    total = lam.sum().clamp_min(1e-12)
    p = (lam / total).clamp_min(1e-12)
    k = min(int(r0), lam.numel())
    top = float(lam[-k:].sum() / total) if k else 0.0
    return {
        "top_frac": top,
        "tail_frac": 1.0 - top,
        "effective_rank": float(torch.exp(-(p * p.log()).sum())),
    }


class PrototypeEMA(nn.Module):
    """EMA prototype per class slot, used as a denoised anchor and an extra positive.

    ``n_classes`` is the size of the group id space passed to ``update``/``gather``.
    Which id that is -- the global stimulus id (``concept * n_images + slot``) or the
    concept id -- is a *data* decision made by the caller, because it changes what a
    prototype means:

      * stimulus-level: each prototype pools the ~9 subjects that saw that exact image.
        Keeps within-concept image differences that the EEG may genuinely encode, but
        every prototype sees only as many updates per step as there are subjects in the
        batch (~9), so it is the noisier of the two.
      * concept-level: each prototype pools all subjects *and* all 10 image slots, so it
        sees ~90 updates per step. Much better conditioned, and it matches the fact that
        the THINGS-EEG2 test set is concept-level (one averaged trial per concept, one
        image per concept). It deliberately discards within-concept image identity.

    Buffers are keyed by the group id and live on the CPU until gathered, so the pool
    does not multiply GPU memory by the dataset size. Prototypes for groups not yet
    seen are simply absent and contribute nothing -- the validity mask is what keeps a
    mostly-empty pool at the start of training from adding noise, and it is why
    ``min_updates`` exists rather than a plain ``count > 0``.
    """

    def __init__(self, n_classes: int, d_embed: int, momentum: float = 0.99,
                 min_updates: int = 2, init_temp: float = 0.07,
                 softplus: bool = True) -> None:
        super().__init__()
        self.n_classes = int(n_classes)
        self.d_embed = int(d_embed)
        self.momentum = float(momentum)
        self.min_updates = int(min_updates)
        self.softplus = bool(softplus)
        self.register_buffer("proto", torch.zeros(self.n_classes, d_embed))
        self.register_buffer("count", torch.zeros(self.n_classes))
        # The prototype contrast IS a contrast, so it needs a temperature for the same
        # reason `InfoNCE` does -- plus one more that is specific to this term. Its
        # logits are cosines between L2-normalised embeddings, so they live in [-1, 1]
        # and WITHOUT a scale the softmax over the class bank is essentially uniform.
        # Measured on a trained `v3` checkpoint: `proto` sat at 5.32-5.50 against a
        # uniform-softmax floor of ``ln(384) = 5.95`` for 50 epochs, i.e. it never left
        # its initialisation, and its gradient carried 1.2% of the total learning
        # signal while the term held 26% of the reported loss VALUE. A cosine contrast
        # needs the scale the same way a raw dot-product one does.
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / init_temp)))

    def effective_scale(self) -> torch.Tensor:
        """Bounded inverse temperature, sharing `InfoNCE`'s clamp."""
        return effective_logit_scale(self.logit_scale, self.softplus)

    # ------------------------------------------------------------------ update
    @torch.no_grad()
    def update(self, z: torch.Tensor, group: torch.Tensor) -> None:
        z = F.normalize(z.detach(), dim=-1).to(self.proto.device)
        for i in range(z.shape[0]):
            j = int(group[i])
            if self.count[j] == 0:
                self.proto[j] = z[i]
            else:
                m = self.momentum
                self.proto[j] = m * self.proto[j] + (1.0 - m) * z[i]
            self.count[j] += 1

    # ------------------------------------------------------------------- gather
    def gather(self, group: torch.Tensor, device: torch.device
               ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return `(proto (B,D), valid (B,) bool)` for the batch's groups."""
        idx = group.to(self.proto.device).long()
        p = self.proto[idx].to(device)
        valid = (self.count[idx].to(device) >= self.min_updates)
        return p, valid

    def _contrast_to_proto(self, z: torch.Tensor, group: torch.Tensor) -> torch.Tensor:
        """Row-wise cross-entropy of `z` against the whole prototype bank.

        The negatives are ALL other prototypes in the bank, not just the ones present
        in this batch. That is the second thing the pool buys over a pairwise contrast:
        with 8 stimuli in a batch the pairwise term sees 7 negatives, while this one
        sees every prototype seen so far.
        """
        p, valid = self.gather(group, z.device)
        if not bool(valid.any()):
            return z.new_zeros(())
        sim = F.normalize(z, dim=-1) @ F.normalize(p, dim=-1).t()
        logits = self.effective_scale() * sim
        target = torch.arange(z.shape[0], device=z.device)
        return F.cross_entropy(logits[valid], target[valid])

    # ------------------------------------------------------------------- losses
    def proto_contrast(self, z_eeg: torch.Tensor, group: torch.Tensor) -> torch.Tensor:
        """Pull each EEG row toward its own class prototype (neural anchoring)."""
        return self._contrast_to_proto(z_eeg, group)

    def image_anchor_contrast(self, z_img: torch.Tensor, group: torch.Tensor
                              ) -> torch.Tensor:
        """Pull each IMAGE row toward the EEG prototype of its class (the asymmetry).

        Gradients flow into the image head, never into the prototypes (they are buffers
        and are updated by ``update``), so this term shapes the mapping from visual
        space into the shared space -- it cannot move the neural anchors to meet the
        images, which is what makes it an anchor rather than a second contrastive term.
        """
        return self._contrast_to_proto(z_img, group)
