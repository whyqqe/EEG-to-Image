"""ST-GATE: Subject-Transport & Geometry Adaptation from EEG Profiles.

Novel claim
-----------
Cross-subject EEG→image retrieval fails from *two* coupled factors:
  (1) latent transport (individual remapping of semantic axes)
  (2) retrieval geometry (subject-dependent hubness / temperature / neighborhood)

ENIGMA-style W_s only models (1) as a full linear map.
SATTC only models (2) post-hoc on frozen embeddings (and empirically hurts
after W_s personalization).

ST-GATE estimates a compact subject profile p_s from *unlabeled* background
EEG, then a hypernetwork emits:
  • low-rank transport (A_s, B_s) applied on a subject residual branch
  • geometry parameters θ_s for a differentiable retrieval head

Optional K labeled pairs refine only (A_s, B_s); θ_s stays label-free.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from eeg_brainit.models.subject_align import AtmSharedBackbone, MLPProjector, SpatioTemporalBackbone


class ResidualSplit(nn.Module):
    """h → (h_sem, h_subj) with soft residual: h_subj = h - h_sem."""

    def __init__(self, nz: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or nz
        self.sem = nn.Sequential(
            nn.LayerNorm(nz),
            nn.Linear(nz, hidden),
            nn.GELU(),
            nn.Linear(hidden, nz),
        )

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h_sem = self.sem(h)
        h_subj = h - h_sem
        return h_sem, h_subj


class SubjectProfileEncoder(nn.Module):
    """Pool M context subject-residual vectors → profile p_s."""

    def __init__(self, nz: int, profile_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(nz),
            nn.Linear(nz, profile_dim),
            nn.GELU(),
            nn.Linear(profile_dim, profile_dim),
            nn.LayerNorm(profile_dim),
        )
        self.profile_dim = profile_dim

    def forward(self, h_subj_ctx: torch.Tensor) -> torch.Tensor:
        # (B, M, Nz) or (M, Nz)
        if h_subj_ctx.dim() == 2:
            pooled = h_subj_ctx.mean(dim=0, keepdim=True)
        else:
            pooled = h_subj_ctx.mean(dim=1)
        return self.net(pooled)


class TransportGeometryHypernet(nn.Module):
    """p_s → low-rank factors (A, B) and geometry θ = (log_tau, k_row, k_col, csls_w)."""

    def __init__(self, profile_dim: int, nz: int, rank: int = 16, geo_dim: int = 4):
        super().__init__()
        self.nz = nz
        self.rank = rank
        self.geo_dim = geo_dim
        hidden = max(profile_dim, 128)
        self.trunk = nn.Sequential(
            nn.Linear(profile_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.to_a = nn.Linear(hidden, nz * rank)
        self.to_b = nn.Linear(hidden, nz * rank)
        self.to_bias = nn.Linear(hidden, nz)
        self.to_geo = nn.Linear(hidden, geo_dim)
        # near-zero init → start close to identity transport
        nn.init.zeros_(self.to_a.weight)
        nn.init.zeros_(self.to_a.bias)
        nn.init.zeros_(self.to_b.weight)
        nn.init.zeros_(self.to_b.bias)
        nn.init.zeros_(self.to_bias.weight)
        nn.init.zeros_(self.to_bias.bias)
        nn.init.zeros_(self.to_geo.weight)
        # log_tau≈log(0.07), soft k offsets, csls weight≈0.5
        nn.init.constant_(self.to_geo.bias, 0.0)

    def forward(self, profile: torch.Tensor) -> dict[str, torch.Tensor]:
        if profile.dim() == 1:
            profile = profile.unsqueeze(0)
        h = self.trunk(profile)
        bsz = h.size(0)
        a = self.to_a(h).view(bsz, self.nz, self.rank)
        b = self.to_b(h).view(bsz, self.nz, self.rank)
        bias = self.to_bias(h)
        geo = self.to_geo(h)
        # map geo to usable ranges
        log_tau = geo[:, 0] + math.log(0.07)
        k_row = 5.0 + 25.0 * torch.sigmoid(geo[:, 1])  # ∈ (5, 30)
        k_col = 5.0 + 25.0 * torch.sigmoid(geo[:, 2])
        csls_w = torch.sigmoid(geo[:, 3])  # ∈ (0, 1)
        return {
            "A": a,
            "B": b,
            "bias": bias,
            "log_tau": log_tau,
            "k_row": k_row,
            "k_col": k_col,
            "csls_w": csls_w,
            "geo_raw": geo,
        }


def apply_low_rank_transport(
    h_sem: torch.Tensor,
    h_subj: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """z = h_sem + A (B^T h_subj) + bias.

    h_*: (B, Nz); A,B: (B, Nz, R) or (1, Nz, R); bias: (B, Nz) or (1, Nz)
    """
    if A.dim() == 2:
        A = A.unsqueeze(0)
        B = B.unsqueeze(0)
    if bias.dim() == 1:
        bias = bias.unsqueeze(0)
    if A.size(0) == 1 and h_sem.size(0) > 1:
        A = A.expand(h_sem.size(0), -1, -1)
        B = B.expand(h_sem.size(0), -1, -1)
        bias = bias.expand(h_sem.size(0), -1)
    bt_h = torch.matmul(B.transpose(1, 2), h_subj.unsqueeze(-1)).squeeze(-1)  # (B, R)
    delta = torch.matmul(A, bt_h.unsqueeze(-1)).squeeze(-1)  # (B, Nz)
    return h_sem + delta + bias


class SubjectDiscriminator(nn.Module):
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


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lambd * grad, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return GradReverse.apply(x, lambd)


class STGateModel(nn.Module):
    """Shared backbone + residual split + profile hypernet transport + CLIP projector."""

    def __init__(
        self,
        n_channels: int = 63,
        seq_len: int = 250,
        clip_dim: int = 1024,
        backbone: str = "atm_style",
        nz: int = 256,
        profile_dim: int = 128,
        rank: int = 16,
        dropout: float = 0.25,
        n_filters: int = 80,
        emb: int = 8,
    ):
        super().__init__()
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
        self.rank = rank
        self.profile_dim = profile_dim
        self.split = ResidualSplit(self.nz)
        self.profile_enc = SubjectProfileEncoder(self.nz, profile_dim=profile_dim)
        self.hyper = TransportGeometryHypernet(profile_dim, self.nz, rank=rank)
        self.projector = MLPProjector(self.nz, clip_dim=clip_dim, dropout=dropout)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))
        # null profile for dropout / zero-shot without context
        self.null_profile = nn.Parameter(torch.zeros(profile_dim))

    def encode_raw(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x.float())

    def split_latent(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.split(h)

    def build_profile(self, ctx_eeg: torch.Tensor, detach_backbone: bool = True) -> torch.Tensor:
        """ctx_eeg: (M,C,T) or (B,M,C,T) → profile (profile_dim,) or (B, profile_dim)."""
        single = ctx_eeg.dim() == 3
        if single:
            ctx_eeg = ctx_eeg.unsqueeze(0)
        b, m, c, t = ctx_eeg.shape
        flat = ctx_eeg.reshape(b * m, c, t)
        with torch.no_grad() if detach_backbone else torch.enable_grad():
            h = self.encode_raw(flat)
        if detach_backbone:
            h = h.detach()
        _, h_subj = self.split_latent(h)
        h_subj = h_subj.reshape(b, m, -1)
        prof = self.profile_enc(h_subj)
        return prof[0] if single else prof

    def transport_from_profile(
        self, h: torch.Tensor, profile: torch.Tensor | None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        h_sem, h_subj = self.split_latent(h)
        if profile is None:
            profile = self.null_profile.expand(h.size(0), -1)
        elif profile.dim() == 1:
            profile = profile.unsqueeze(0).expand(h.size(0), -1)
        params = self.hyper(profile)
        # if batch profile matches batch h, use per-row; else broadcast first
        if params["A"].size(0) == 1 and h.size(0) > 1:
            z = apply_low_rank_transport(
                h_sem, h_subj, params["A"], params["B"], params["bias"]
            )
        elif params["A"].size(0) == h.size(0):
            z = apply_low_rank_transport(
                h_sem, h_subj, params["A"], params["B"], params["bias"]
            )
        else:
            z = apply_low_rank_transport(
                h_sem, h_subj, params["A"][:1], params["B"][:1], params["bias"][:1]
            )
        return z, {**params, "h_sem": h_sem, "h_subj": h_subj}

    def forward(
        self,
        x: torch.Tensor,
        profile: torch.Tensor | None = None,
        ctx_eeg: torch.Tensor | None = None,
        use_null_profile: bool = False,
        normalize: bool = True,
    ) -> dict[str, torch.Tensor]:
        h = self.encode_raw(x)
        if use_null_profile:
            profile = self.null_profile.expand(h.size(0), -1)
        elif ctx_eeg is not None:
            profile = self.build_profile(ctx_eeg, detach_backbone=True)
            if profile.dim() == 1:
                profile = profile.unsqueeze(0).expand(h.size(0), -1)
        z, params = self.transport_from_profile(h, profile)
        raw = self.projector(z)
        emb = F.normalize(raw, dim=-1)
        out = {
            "hidden": h,
            "latent": z,
            "clip_raw": raw,
            "clip_emb": emb,
            "clip_out": emb if normalize else raw,
            "logit_scale": self.logit_scale.exp(),
            "profile": profile if profile is not None else self.null_profile.expand(h.size(0), -1),
            "h_sem": params["h_sem"],
            "h_subj": params["h_subj"],
            "log_tau": params["log_tau"],
            "k_row": params["k_row"],
            "k_col": params["k_col"],
            "csls_w": params["csls_w"],
            "A": params["A"],
            "B": params["B"],
            "bias": params["bias"],
        }
        return out

    def shared_parameters(self):
        for p in self.backbone.parameters():
            yield p
        for p in self.split.parameters():
            yield p
        for p in self.profile_enc.parameters():
            yield p
        for p in self.hyper.parameters():
            yield p
        for p in self.projector.parameters():
            yield p
        yield self.logit_scale
        yield self.null_profile

    def transport_parameters(self):
        """Parameters refined with K-shot (hypernet A/B/bias path + optional profile)."""
        for p in self.hyper.parameters():
            yield p
        for p in self.profile_enc.parameters():
            yield p


def soft_csls_logits(
    q: torch.Tensor,
    g: torch.Tensor,
    k_row: torch.Tensor,
    k_col: torch.Tensor,
    csls_w: torch.Tensor,
    tau: torch.Tensor,
) -> torch.Tensor:
    """Differentiable CSLS-style logits for in-batch / gallery retrieval.

    q: (B, D) L2-normalized; g: (N, D) or (B, N, D)
    Returns logits (B, N).
    """
    if g.dim() == 2:
        sim = q @ g.T  # (B, N)
    else:
        sim = torch.matmul(q.unsqueeze(1), g.transpose(1, 2)).squeeze(1)
    # approximate top-k mean via softmax temperature scaled by k
    # row neighborhood
    kr = k_row.view(-1, 1).clamp(min=1.0)
    # soft top-k: sharpen with temp ∝ 1/k
    row_temp = 1.0 / kr
    row_w = torch.softmax(sim / row_temp, dim=1)
    r = (row_w * sim).sum(dim=1, keepdim=True)
    # column neighborhood (shared across batch queries)
    kc = k_col.view(-1).mean().clamp(min=1.0)
    col_temp = 1.0 / kc
    col_w = torch.softmax(sim / col_temp, dim=0)
    c = (col_w * sim).sum(dim=0, keepdim=True)
    w = csls_w.view(-1, 1)
    s = sim - w * 0.5 * r - w * 0.5 * c
    t = tau.view(-1, 1).clamp(min=1e-3)
    return s / t


def st_gate_losses(
    out: dict[str, torch.Tensor],
    clip_img: torch.Tensor,
    subject_id: torch.Tensor | None = None,
    disc: SubjectDiscriminator | None = None,
    atm_emb: torch.Tensor | None = None,
    gallery: torch.Tensor | None = None,
    temp: float = 0.07,
    lambda_nce: float = 0.5,
    lambda_atm: float = 0.0,
    lambda_adv: float = 0.0,
    lambda_geo: float = 0.25,
    grl_lambda: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """MSE + InfoNCE + optional ATM distill + adv on h_sem + geometry NCE."""
    stats: dict[str, float] = {}
    tgt = clip_img.float()
    mse = F.mse_loss(out["clip_raw"], tgt)
    logits = out["clip_emb"] @ F.normalize(tgt, dim=-1).T / temp
    labels = torch.arange(logits.size(0), device=logits.device)
    nce = F.cross_entropy(logits, labels)
    loss = mse + lambda_nce * nce
    stats["mse"] = float(mse.detach())
    stats["nce"] = float(nce.detach())

    if atm_emb is not None and lambda_atm > 0:
        la = F.mse_loss(out["clip_emb"], F.normalize(atm_emb.float(), dim=-1))
        loss = loss + lambda_atm * la
        stats["atm"] = float(la.detach())

    if disc is not None and subject_id is not None and lambda_adv > 0:
        # fool discriminator on semantic branch
        adv = F.cross_entropy(disc(grad_reverse(out["h_sem"], grl_lambda)), subject_id.long())
        # encourage residual to keep subject signal
        subj_ce = F.cross_entropy(disc(out["h_subj"].detach()), subject_id.long())
        loss = loss + lambda_adv * (adv + 0.1 * subj_ce)
        stats["adv"] = float(adv.detach())

    if lambda_geo > 0:
        g = gallery if gallery is not None else F.normalize(tgt, dim=-1)
        g = F.normalize(g.float(), dim=-1)
        tau = out["log_tau"].exp()
        geo_logits = soft_csls_logits(
            out["clip_emb"], g, out["k_row"], out["k_col"], out["csls_w"], tau
        )
        # if gallery is batch targets, labels = arange; if external gallery, skip geo nce
        if g.size(0) == out["clip_emb"].size(0):
            geo_nce = F.cross_entropy(geo_logits, labels)
            loss = loss + lambda_geo * geo_nce
            stats["geo"] = float(geo_nce.detach())

    return loss, stats
