#!/usr/bin/env python3
"""MG-Flow modules: dual-granularity semantic heads + hierarchical CFM + gate."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def l2_t(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=-1)


def cosine_interp(z0: torch.Tensor, z1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    if t.ndim == 1:
        t = t.view(-1, 1)
    a = torch.cos(0.5 * math.pi * t) ** 2
    s = torch.sin(0.5 * math.pi * t) ** 2
    return a * z0 + s * z1


def cosine_interp_velocity(z0: torch.Tensor, z1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    if t.ndim == 1:
        t = t.view(-1, 1)
    da = -math.pi * torch.cos(0.5 * math.pi * t) * torch.sin(0.5 * math.pi * t)
    ds = math.pi * torch.sin(0.5 * math.pi * t) * torch.cos(0.5 * math.pi * t)
    return da * z0 + ds * z1


class SinusoidalTimeEmbed(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 2:
            t = t.squeeze(-1)
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / max(half - 1, 1))
        args = t.unsqueeze(-1) * freqs.unsqueeze(0) * 1000.0
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return self.proj(emb)


class DualSemanticHeads(nn.Module):
    """EEG(ret) → coarse / fine semantic embeddings (CLIP dim)."""

    def __init__(self, in_dim: int = 512, out_dim: int = 1024, hidden: int = 1024):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.head_c = nn.Linear(hidden, out_dim)
        self.head_f = nn.Linear(hidden, out_dim)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(l2_t(z))
        return l2_t(self.head_c(h)), l2_t(self.head_f(h))


class CondCFM(nn.Module):
    """Conditional CFM: transport noise/cond → target embedding."""

    def __init__(self, cond_dim: int, out_dim: int = 1024, hidden: int = 1536, time_dim: int = 128):
        super().__init__()
        self.lift = nn.Sequential(
            nn.Linear(cond_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
        self.time_embed = SinusoidalTimeEmbed(time_dim)
        self.net = nn.Sequential(
            nn.Linear(out_dim * 2 + time_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )
        self.ode_mix = nn.Parameter(torch.tensor(1.2))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def condition(self, c: torch.Tensor) -> torch.Tensor:
        return self.lift(l2_t(c))

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        cond = self.condition(c)
        te = self.time_embed(t)
        return self.net(torch.cat([z_t, cond, te], dim=-1))

    @torch.no_grad()
    def decode(self, c: torch.Tensor, steps: int = 16) -> torch.Tensor:
        cond0 = self.condition(c)
        z = cond0.clone()
        dt = 1.0 / max(steps, 1)
        for i in range(steps):
            t_val = 1.0 - i * dt
            t = torch.full((z.shape[0],), t_val, device=z.device, dtype=z.dtype)
            v = self.forward(z, t, c)
            z = z - dt * v
        keep = torch.sigmoid(self.ode_mix)
        return l2_t(keep * cond0 + (1.0 - keep) * z)


class MGFlowModel(nn.Module):
    def __init__(self, ret_dim: int = 512, clip_dim: int = 1024, hidden: int = 1024, cfm_hidden: int = 1536):
        super().__init__()
        self.heads = DualSemanticHeads(ret_dim, clip_dim, hidden)
        self.cfm_c = CondCFM(clip_dim, clip_dim, cfm_hidden)  # condition on z_s^c
        self.cfm_f = CondCFM(clip_dim, clip_dim, cfm_hidden)  # condition on z_s^f
        self.cfm_c2f = CondCFM(clip_dim, clip_dim, cfm_hidden)  # coarse→fine refine
        # map fine semantic to gen space residual (same dim as ViT-H)
        self.to_gen = nn.Sequential(
            nn.Linear(clip_dim, clip_dim),
            nn.GELU(),
            nn.Linear(clip_dim, clip_dim),
        )
        nn.init.zeros_(self.to_gen[-1].weight)
        nn.init.zeros_(self.to_gen[-1].bias)
        self.gate_mlp = nn.Sequential(
            nn.Linear(clip_dim * 2 + 1, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )

    def encode(self, z_ret: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.heads(z_ret)


def flow_matching_loss(model: CondCFM, target: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    z0 = l2_t(target)
    z1 = l2_t(model.condition(cond))
    t = torch.rand(z0.shape[0], device=z0.device, dtype=z0.dtype)
    z_t = cosine_interp(z0, z1, t)
    v_star = cosine_interp_velocity(z0, z1, t)
    v_pred = model(z_t, t, cond)
    return F.mse_loss(v_pred, v_star)


def clip_info_nce(pred: torch.Tensor, target: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    p, t = l2_t(pred), l2_t(target)
    logits = (p @ t.T) / temperature
    labels = torch.arange(p.shape[0], device=p.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def clip_cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (1.0 - (l2_t(pred) * l2_t(target)).sum(-1)).mean()


def orth_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # encourage some complementarity without forcing full orthogonality
    return ((l2_t(a) * l2_t(b)).sum(-1) ** 2).mean()
