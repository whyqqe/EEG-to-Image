"""Retrieval–Generation Transport (RGT) via subject-conditioned Flow Matching.

Transports retrieval-optimal z_ret (SSP-512) onto generation manifold z_gen (ViT-H-1024),
with neighborhood-preservation so retrieval identity is not destroyed.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_interp(z0: torch.Tensor, z1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    if t.ndim == 1:
        t = t.view(-1, 1)
    alpha = torch.cos(0.5 * math.pi * t) ** 2
    sigma = torch.sin(0.5 * math.pi * t) ** 2
    return alpha * z0 + sigma * z1


def cosine_interp_velocity(z0: torch.Tensor, z1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    if t.ndim == 1:
        t = t.view(-1, 1)
    d_alpha = -math.pi * torch.cos(0.5 * math.pi * t) * torch.sin(0.5 * math.pi * t)
    d_sigma = math.pi * torch.sin(0.5 * math.pi * t) * torch.cos(0.5 * math.pi * t)
    return d_alpha * z0 + d_sigma * z1


class SinusoidalTimeEmbed(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 2:
            t = t.squeeze(-1)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / max(half - 1, 1)
        )
        args = t.unsqueeze(-1) * freqs.unsqueeze(0) * 1000.0
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return self.proj(emb)


class RGTVelocity(nn.Module):
    """Asymmetric CFM: state in gen_dim, condition = lift(z_ret) + subject FiLM."""

    def __init__(
        self,
        ret_dim: int = 512,
        gen_dim: int = 1024,
        hidden: int = 2048,
        time_dim: int = 256,
        n_subjects: int = 11,
        subj_dim: int = 64,
    ) -> None:
        super().__init__()
        self.ret_dim = ret_dim
        self.gen_dim = gen_dim
        self.lift = nn.Sequential(
            nn.Linear(ret_dim, gen_dim),
            nn.LayerNorm(gen_dim),
            nn.GELU(),
            nn.Linear(gen_dim, gen_dim),
        )
        self.subj_emb = nn.Embedding(n_subjects, subj_dim)
        self.subj_to_scale = nn.Linear(subj_dim, gen_dim)
        self.subj_to_shift = nn.Linear(subj_dim, gen_dim)
        self.time_embed = SinusoidalTimeEmbed(time_dim)
        self.net = nn.Sequential(
            nn.Linear(gen_dim * 2 + time_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, gen_dim),
        )
        self.res_scale = nn.Parameter(torch.tensor(0.0))
        # zero-init last velocity layer for stable start
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def condition(self, z_ret: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        c = self.lift(F.normalize(z_ret.float(), dim=-1))
        s = self.subj_emb(subject_ids.long().clamp(0, self.subj_emb.num_embeddings - 1))
        scale = torch.tanh(self.subj_to_scale(s))
        shift = self.subj_to_shift(s)
        return c * (1.0 + scale) + shift

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        z_ret: torch.Tensor,
        subject_ids: torch.Tensor,
    ) -> torch.Tensor:
        cond = self.condition(z_ret, subject_ids)
        te = self.time_embed(t)
        return self.net(torch.cat([z_t, cond, te], dim=-1))

    def decode(
        self,
        z_ret: torch.Tensor,
        subject_ids: torch.Tensor,
        *,
        steps: int = 20,
    ) -> torch.Tensor:
        cond0 = self.condition(z_ret, subject_ids)
        z = cond0.clone()
        dt = 1.0 / max(steps, 1)
        for i in range(steps):
            t_val = 1.0 - i * dt
            t = torch.full((z.shape[0],), t_val, device=z.device, dtype=z.dtype)
            v = self.forward(z, t, z_ret, subject_ids)
            z = z - dt * v
        gate = torch.tanh(self.res_scale)
        return F.normalize(cond0 + gate * (z - cond0), dim=-1)


def flow_matching_loss(
    model: RGTVelocity,
    z_gen: torch.Tensor,
    z_ret: torch.Tensor,
    subject_ids: torch.Tensor,
) -> torch.Tensor:
    z0 = F.normalize(z_gen.float(), dim=-1)
    # path endpoint at t=1 is lifted retrieval condition (not raw 512)
    z1 = F.normalize(model.condition(z_ret, subject_ids), dim=-1)
    t = torch.rand(z0.shape[0], device=z0.device, dtype=z0.dtype)
    z_t = cosine_interp(z0, z1, t)
    v_star = cosine_interp_velocity(z0, z1, t)
    v_pred = model(z_t, t, z_ret, subject_ids)
    return F.mse_loss(v_pred, v_star)


def neighborhood_preserve_loss(
    z_ret: torch.Tensor,
    z_gen_hat: torch.Tensor,
    *,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Match batch pairwise similarity structure (retrieval identity)."""
    a = F.normalize(z_ret.float(), dim=-1)
    b = F.normalize(z_gen_hat.float(), dim=-1)
    sa = a @ a.T / temperature
    sb = b @ b.T / temperature
    # stop-grad on retrieval structure
    pa = F.softmax(sa.detach(), dim=-1)
    log_pb = F.log_softmax(sb, dim=-1)
    return F.kl_div(log_pb, pa, reduction="batchmean")


def clip_info_nce(pred: torch.Tensor, target: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    p = F.normalize(pred.float(), dim=-1)
    t = F.normalize(target.float(), dim=-1)
    logits = (p @ t.T) / temperature
    labels = torch.arange(p.shape[0], device=p.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def clip_cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    p = F.normalize(pred.float(), dim=-1)
    t = F.normalize(target.float(), dim=-1)
    return (1.0 - (p * t).sum(-1)).mean()
