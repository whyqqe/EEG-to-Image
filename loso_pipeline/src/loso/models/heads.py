"""Projection heads, gradient reversal and the subject classifier.

The encoder emits two representations and this module maps them onto the frozen
teacher spaces:

  z_inv -> P_img  -> CLIP image space      (global semantics)
        -> P_text -> CLIP text space       (category semantics, same space as above)
        -> P_dino -> DINOv2 space          (layout / shape structure)
        -> P_vae  -> pooled VAE latent     (low-level appearance, coarse)
  z_inv -> P_time -> VAE latent patches    (time-resolved alignment)
  z_inv -> SubjectClassifier  (through a GRL: the adversarial branch)

Two deliberate design points, both from the design document:

* `L_text` and `L_vae` are secondary.  Category text under-determines which of ten
  distinct images of a dog was shown, and pixel-level appearance is not the
  encoder's job; both enter at low weight, and `P_vae` sees a *spatially pooled*
  latent (4x8x8) rather than the full 4x64x64 one so the main trunk is never asked
  to regress pixels.

* The CLIP branches share an output space and therefore a temperature.  The
  reference implementation (CogCapPro) ties a single `logit_scale` across its
  modality branches for exactly this reason: separate temperatures would let one
  branch be systematically easier to fit while the summed loss looks healthy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HeadConfig:
    d_inv: int = 512
    d_sub: int = 128
    d_model: int = 256           # encoder token width, for the time-resolved head
    d_teacher: int = 1024        # CLIP ViT-H-14 and DINOv2-L are both 1024-d
    #: Hidden width of the projectors.  Was 1024, which made the head set 7.9 M
    #: parameters against a 3.7 M encoder trunk -- a 2.11x inversion.  With the
    #: projectors able to represent more than the trunk, "the model cannot fit the
    #: targets" is not a hypothesis the architecture can test, and the failure mode
    #: is the observed one: projectors fitted straight to the frozen targets while
    #: `z_inv` stayed near-constant, because the loss was reduced without the encoder
    #: contributing anything.  At 512 the projectors total ~3.2 M against the same
    #: 3.7 M trunk, so the trunk is once more the larger component.  Asserted in
    #: `scripts/smoke_align.py` so it cannot silently invert again.
    proj_hidden: int = 512
    proj_layers: int = 2
    dropout: float = 0.1
    n_subjects: int = 9
    vae_pool: int = 8            # spatial grid the VAE latent is pooled to
    vae_channels: int = 4
    n_time_patches: int = 64
    grl_lambda: float = 1.0
    #: VICReg variance hinge target, measured rather than assumed.
    #:
    #: `head_inv` ends in a LayerNorm, so every `z_inv` lies on a sphere of radius
    #: `sqrt(d_inv)` -- verified in `scripts/measure_z_scale.py`, which reports a
    #: per-sample norm of 22.6271 (= sqrt(512), std 0.0000 across samples).  On that
    #: sphere, per-dimension batch variance plus squared per-dimension batch mean
    #: sums to exactly 1 per dimension, so a per-dimension std of 1.0 is the
    #: isotropic limit and is the only scale-free choice of `gamma` here: it reads as
    #: "use the sphere evenly".
    #:
    #: This matters because the obvious guess is wrong in a way that cannot be seen
    #: in the loss table.  At initialisation the per-dimension std was measured at
    #: 0.438 (max 0.666), so `gamma=0.25` -- which is roughly the VICReg-style
    #: "1.0 for unnormalised features, scaled down for a unit-norm embedding" guess --
    #: produces a *saturated* hinge: `vicreg_variance = 0.00000` with gradient norm
    #: `0.00000`.  The term would have sat at a plausible-looking constant forever
    #: while contributing nothing, which is worse than omitting it.  `gamma=1.0`
    #: reports 0.56191 against 0.99000 for a constant encoder, with a non-zero
    #: gradient, so it is active from step 0 and never fully saturates.
    vicreg_gamma: float = 1.0

    @property
    def vae_dim(self) -> int:
        return self.vae_channels * self.vae_pool * self.vae_pool

    @property
    def vae_patch_dim(self) -> int:
        # Each 8x8 spatial cell of the 64x64 latent, across all 4 channels.
        cells = 64 // self.vae_pool
        return self.vae_channels * cells * cells


class MLPProjector(nn.Module):
    """Residual MLP with LayerNorm, optionally L2-normalising its output.

    Normalising is not cosmetic: every alignment loss here is a cosine or an
    InfoNCE over directions, and an unnormalised projector can reduce the loss by
    shrinking its outputs rather than by improving alignment.

    There is deliberately **no** input-to-output shortcut.  An earlier version added
    `nn.Linear(d_in, d_out, bias=False)` whenever `d_in != d_out`, as a residual
    path.  That is a straight linear map from `z_inv` into the teacher space, so the
    nonlinear branch could sit near zero and the branch would still fit its target
    through the shortcut alone: the capacity was there (524 K of the 2.10 M
    parameters of a 512->1024 projector) but the *inductive bias* the residual block
    was written for was gone.  Removing it halves each projector and leaves a plain
    two-layer MLP, which is what the design's `ProjMod` specifies.
    """

    def __init__(self, d_in: int, d_out: int, hidden: int, n_layers: int = 2,
                 dropout: float = 0.1, normalize: bool = True):
        super().__init__()
        self.normalize = normalize
        layers: list[nn.Module] = []
        cur = d_in
        for _ in range(max(1, n_layers - 1)):
            layers += [nn.Linear(cur, hidden), nn.LayerNorm(hidden), nn.GELU(),
                       nn.Dropout(dropout)]
            cur = hidden
        layers.append(nn.Linear(cur, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)
        return F.normalize(h, dim=-1) if self.normalize else h


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    """Identity forward, negated-and-scaled gradient backward.

    This is what makes the subject classifier adversarial: it still learns to
    predict the subject from `z_inv`, but the encoder receives the negated gradient
    and is pushed toward features from which the subject is *not* recoverable.
    """
    return _GradReverse.apply(x, lambd)


class SubjectClassifier(nn.Module):
    """Predicts subject identity from z_inv through a gradient reversal layer."""

    def __init__(self, d_in: int, n_subjects: int, hidden: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.n_subjects = max(2, n_subjects)
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, self.n_subjects),
        )

    def forward(self, z_inv: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
        return self.net(grad_reverse(z_inv, lambd))


# VAE latent patch extraction ------------------------------------------------
# The design asks for a time-resolved alignment term (L_time) plus a coarse global
# one (L_vae).  Both are derived from the same latent, tiled two different ways:
#   pooled():    adaptive average pool to (C, P, P)          -> (C*P*P,) global
#   patched():   unfold into a (64/P)^2 grid of cells        -> (N_patch, 4*cell^2)
# Deriving them in one place keeps the two views consistent by construction.
def vae_pooled(latent: torch.Tensor, pool: int = 8) -> torch.Tensor:
    """(B, C, H, W) -> (B, C*pool*pool)."""
    x = F.adaptive_avg_pool2d(latent.float(), (pool, pool))
    return x.flatten(1)


def vae_patched(latent: torch.Tensor, pool: int = 8, n_patches: int = 64) -> torch.Tensor:
    """(B, C, H, W) -> (B, n_patches, C*cell*cell) in raster order.

    With pool=8 on a 64x64 latent, `cell` is 8 and each patch is 4*8*8 = 256 wide,
    i.e. every patch keeps all channels while the patch count matches a typical
    token budget.  `n_patches` only reinterpolates the sequence length.
    """
    b, c, h, w = latent.shape
    cell = h // pool
    if cell * pool != h or cell * pool != w:
        raise ValueError(f"latent {h}x{w} is not evenly divisible by pool={pool}")
    x = latent.float().view(b, c, pool, cell, pool, cell)
    x = x.permute(0, 2, 4, 1, 3, 5).reshape(b, pool * pool, c * cell * cell)
    if n_patches > 0 and pool * pool != n_patches:
        # Interpolate the patch count so any token budget can be matched.
        x = F.interpolate(x.transpose(1, 2), size=n_patches, mode="linear",
                          align_corners=False).transpose(1, 2)
    return x


class AlignmentHeads(nn.Module):
    """All projectors, plus the adversarial subject classifier."""

    def __init__(self, cfg: HeadConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_inv
        self.img = MLPProjector(d, cfg.d_teacher, cfg.proj_hidden, cfg.proj_layers,
                                cfg.dropout, normalize=True)
        # Shares the teacher space with `img`, so InfoNCE can be taken across
        # modalities as well as within one.
        self.text = MLPProjector(d, cfg.d_teacher, cfg.proj_hidden, cfg.proj_layers,
                                 cfg.dropout, normalize=True)
        self.dino = MLPProjector(d, cfg.d_teacher, cfg.proj_hidden, cfg.proj_layers,
                                 cfg.dropout, normalize=True)
        self.vae = MLPProjector(d, cfg.vae_dim, cfg.proj_hidden, cfg.proj_layers,
                                cfg.dropout, normalize=True)
        # The time head consumes Transformer *tokens*, not the pooled z_inv, so it
        # is sized on d_model.  Projecting the pooled vector instead would discard
        # the temporal structure this term exists to constrain.
        self.time = MLPProjector(cfg.d_model, cfg.vae_patch_dim, cfg.proj_hidden,
                                 cfg.proj_layers, cfg.dropout, normalize=True)
        self.subject = SubjectClassifier(d, cfg.n_subjects, hidden=256,
                                         dropout=cfg.dropout)
        # Logit scale for the CLIP-space branches.  Learned in log space and clamped,
        # matching the CLIP/InfoNCE convention; exp(logit_scale) is the temperature
        # inverse.  Shared by img/text per the note in the module docstring.
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def resize_subjects(self, n_subjects: int) -> None:
        if n_subjects == self.cfg.n_subjects:
            return
        device = self.logit_scale.device
        self.cfg.n_subjects = max(2, n_subjects)
        self.subject = SubjectClassifier(
            self.cfg.d_inv, self.cfg.n_subjects, hidden=256, dropout=self.cfg.dropout,
        ).to(device)

    def scaled_logit_scale(self, max_scale: float = 100.0) -> torch.Tensor:
        return self.logit_scale.clamp(max=math.log(max_scale)).exp()
