"""Multi-subject EEG→CLIP with subject-specific latent alignment (v2).

Paradigm (ENIGMA 2026 + stronger shared features):
  Shared backbone → bottleneck latent x ∈ R^{Nz}
  Subject linear alignment: z = W_s x + b_s
  Shared MLP projector → CLIP (MSE + InfoNCE [+ optional ATM distill])

v2 upgrades vs v1:
  - backbone=atm_style (stronger shared features) with Nz bottleneck
  - larger ENIGMA CNN option retained
  - ATM embedding distillation hook in loss
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from eeg_brainit.models.atm_backbone import AtmStyleEEGEncoder


class SpatioTemporalBackbone(nn.Module):
    """ENIGMA-like temporal→spatial conv stack. Input (B, C, T) → (B, Nz)."""

    def __init__(self, n_channels: int = 63, seq_len: int = 250, n_filters: int = 80, emb: int = 8):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv2d(1, n_filters, kernel_size=(1, 5), stride=1, padding=0),
            nn.AvgPool2d(kernel_size=(1, 17), stride=(1, 5)),
            nn.BatchNorm2d(n_filters),
            nn.GELU(),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(n_filters, n_filters, kernel_size=(n_channels, 1), stride=1, padding=0),
            nn.BatchNorm2d(n_filters),
            nn.GELU(),
            nn.Dropout(0.5),
        )
        self.proj = nn.Conv2d(n_filters, emb, kernel_size=1)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, seq_len)
            h = self.proj(self.spatial(self.temporal(dummy)))
            self.nz = int(h.numel())
        self.n_channels = n_channels
        self.seq_len = seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.float().unsqueeze(1)
        return self.proj(self.spatial(self.temporal(y))).flatten(1)


class AtmSharedBackbone(nn.Module):
    """ATM-style shared encoder → fixed Nz bottleneck (before subject W_s)."""

    def __init__(
        self,
        n_channels: int = 63,
        seq_len: int = 250,
        nz: int = 256,
        clip_dim: int = 1024,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.atm = AtmStyleEEGEncoder(
            n_channels=n_channels, seq_len=seq_len, clip_dim=clip_dim, dropout=dropout
        )
        self.bottleneck = nn.Sequential(
            nn.LayerNorm(self.atm.flat_dim),
            nn.Linear(self.atm.flat_dim, nz),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.nz = nz

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bottleneck(self.atm.encode_hidden(x.float()))


class MLPProjector(nn.Module):
    """Residual MLP: Nz → clip_dim."""

    def __init__(self, nz: int, clip_dim: int = 1024, dropout: float = 0.25):
        super().__init__()
        self.fc1 = nn.Linear(nz, clip_dim)
        self.fc2 = nn.Linear(clip_dim, clip_dim)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(clip_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.fc1(z))
        h = self.drop(h)
        return self.norm(self.fc2(h) + h)


class SubjectAlignEEGEncoder(nn.Module):
    """Shared backbone + per-subject linear alignment + shared CLIP projector."""

    def __init__(
        self,
        n_subjects: int = 10,
        n_channels: int = 63,
        seq_len: int = 250,
        clip_dim: int = 1024,
        backbone: str = "atm_style",
        nz: int = 256,
        n_filters: int = 80,
        emb: int = 8,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.backbone_name = backbone
        if backbone == "atm_style":
            self.backbone = AtmSharedBackbone(
                n_channels=n_channels, seq_len=seq_len, nz=nz, clip_dim=clip_dim, dropout=dropout
            )
            self.nz = nz
        elif backbone == "enigma":
            self.backbone = SpatioTemporalBackbone(
                n_channels=n_channels, seq_len=seq_len, n_filters=n_filters, emb=emb
            )
            self.nz = int(self.backbone.nz)
        else:
            raise ValueError(backbone)

        self.clip_dim = clip_dim
        self.n_subjects = n_subjects
        self.align = nn.ModuleList([nn.Linear(self.nz, self.nz, bias=True) for _ in range(n_subjects)])
        for layer in self.align:
            nn.init.eye_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.projector = MLPProjector(self.nz, clip_dim=clip_dim, dropout=dropout)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def encode_pre_align(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def encode_latent(self, x: torch.Tensor, subject_id: torch.Tensor | int) -> torch.Tensor:
        h = self.encode_pre_align(x)
        if isinstance(subject_id, int):
            return self.align[subject_id](h)
        sid = subject_id.long().view(-1)
        out = torch.empty_like(h)
        for s in sid.unique():
            mask = sid == s
            out[mask] = self.align[int(s.item())](h[mask])
        return out

    def forward(
        self,
        x: torch.Tensor,
        subject_id: torch.Tensor | int,
        normalize: bool = True,
    ) -> dict[str, torch.Tensor]:
        z = self.encode_latent(x, subject_id)
        raw = self.projector(z)
        emb = F.normalize(raw, dim=-1) if normalize else raw
        return {
            "latent": z,
            "clip_raw": raw,
            "clip_emb": F.normalize(raw, dim=-1),
            "clip_out": emb,
            "logit_scale": self.logit_scale.exp(),
        }

    def shared_parameters(self):
        for p in self.backbone.parameters():
            yield p
        for p in self.projector.parameters():
            yield p
        yield self.logit_scale

    def projector_parameters(self):
        return self.projector.parameters()

    def subject_parameters(self, subject_idx: int):
        return self.align[subject_idx].parameters()

    def freeze_shared(self) -> None:
        for p in self.shared_parameters():
            p.requires_grad_(False)

    def unfreeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad_(True)


class GradReverse(torch.autograd.Function):
    """Gradient reversal for subject-adversarial (DANN-style) training."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambd * grad_output, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return GradReverse.apply(x, lambd)


class SubjectDiscriminator(nn.Module):
    """Predict subject ID from pre-align latent (to be fooled by backbone)."""

    def __init__(self, nz: int, n_subjects: int = 10, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(nz),
            nn.Linear(nz, hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, n_subjects),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


def enigma_loss(
    clip_raw: torch.Tensor,
    clip_emb: torch.Tensor,
    target: torch.Tensor,
    temp: float = 0.07,
    lambda_nce: float = 0.5,
    atm_emb: torch.Tensor | None = None,
    lambda_atm: float = 0.0,
) -> torch.Tensor:
    """MSE(CLIP) + InfoNCE + optional distill to ATM teacher embeddings."""
    tgt = target.float()
    mse = F.mse_loss(clip_raw, tgt)
    logits = clip_emb @ F.normalize(tgt, dim=-1).T / temp
    labels = torch.arange(logits.size(0), device=logits.device)
    nce = F.cross_entropy(logits, labels)
    loss = mse + lambda_nce * nce
    if atm_emb is not None and lambda_atm > 0:
        loss = loss + lambda_atm * F.mse_loss(clip_emb, F.normalize(atm_emb.float(), dim=-1))
    return loss


def fit_align_ridge(
    h: torch.Tensor,
    z_target: torch.Tensor,
    ridge: float = 1e-2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Closed-form ridge affine map: z ≈ W h + b (row samples).

    h, z_target: (N, Nz). Returns W (Nz, Nz), b (Nz,).
    """
    x = h.float()
    y = z_target.float()
    n, d = x.shape
    x_aug = torch.cat([x, torch.ones(n, 1, device=x.device, dtype=x.dtype)], dim=1)
    xtx = x_aug.T @ x_aug
    reg = ridge * torch.eye(d + 1, device=x.device, dtype=x.dtype)
    reg[-1, -1] = 0.0
    beta = torch.linalg.solve(xtx + reg, x_aug.T @ y)
    w = beta[:-1].T.contiguous()
    b = beta[-1].contiguous()
    return w, b


def estimate_latent_targets(
    model: "SubjectAlignEEGEncoder",
    eeg: torch.Tensor,
    clip_img: torch.Tensor,
    steps: int = 40,
    lr: float = 0.3,
) -> torch.Tensor:
    """Invert projector approximately: find z s.t. projector(z) ≈ clip_img."""
    with torch.no_grad():
        z0 = model.encode_pre_align(eeg).detach().clone()
    z = z0.clone().requires_grad_(True)
    opt = torch.optim.Adam([z], lr=lr)
    tgt = clip_img.float()
    for _ in range(steps):
        pred = model.projector(z)
        loss = F.mse_loss(pred, tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return z.detach()


def mixup_batch(
    eeg: torch.Tensor,
    clip_img: torch.Tensor,
    subject_id: torch.Tensor,
    atm_emb: torch.Tensor | None = None,
    alpha: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Feature-level mixup on EEG + soft CLIP targets (subject-invariant aug)."""
    if alpha <= 0:
        ones = torch.ones(eeg.size(0), device=eeg.device)
        return eeg, clip_img, subject_id, atm_emb, ones
    lam = float(np.random.beta(alpha, alpha))
    lam = max(lam, 1.0 - lam)
    idx = torch.randperm(eeg.size(0), device=eeg.device)
    eeg_m = lam * eeg + (1.0 - lam) * eeg[idx]
    clip_m = lam * clip_img + (1.0 - lam) * clip_img[idx]
    atm_m = None
    if atm_emb is not None:
        atm_m = lam * atm_emb + (1.0 - lam) * atm_emb[idx]
    return eeg_m, clip_m, subject_id, atm_m, torch.full((eeg.size(0),), lam, device=eeg.device)

