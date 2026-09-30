#!/usr/bin/env python3
"""TCDA modules: Perception multi-granularity (Pc/Pf) + Relational coordinator R + gates."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def l2(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return F.normalize(x.float(), dim=dim)


def cosine_interp(z0: torch.Tensor, z1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    while t.ndim < z0.ndim:
        t = t.unsqueeze(-1)
    a = torch.cos(0.5 * math.pi * t) ** 2
    s = torch.sin(0.5 * math.pi * t) ** 2
    return a * z0 + s * z1


def cosine_interp_velocity(z0: torch.Tensor, z1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    while t.ndim < z0.ndim:
        t = t.unsqueeze(-1)
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


class UpsampleHead(nn.Module):
    """EEG vector → (C, 64, 64)."""

    def __init__(self, in_dim: int, out_ch: int, hidden: int = 1024, base_ch: int = 64):
        super().__init__()
        self.base = 8
        self.base_ch = base_ch
        self.out_ch = out_ch
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, base_ch * self.base * self.base),
        )
        self.up = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base_ch, base_ch // 2, 3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base_ch // 2, base_ch // 4, 3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base_ch // 4, out_ch, 3, padding=1),
        )
        nn.init.zeros_(self.up[-1].weight)
        nn.init.zeros_(self.up[-1].bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z).view(-1, self.base_ch, self.base, self.base)
        return self.up(x)


class SpatialCondCFM(nn.Module):
    def __init__(self, cond_dim: int, ch: int, hidden: int = 96, time_dim: int = 64):
        super().__init__()
        self.ch = ch
        self.lift = nn.Sequential(nn.Linear(cond_dim, 256), nn.GELU(), nn.Linear(256, ch * 8 * 8))
        self.lift_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(ch, ch, 3, padding=1),
        )
        self.time_embed = SinusoidalTimeEmbed(time_dim)
        self.time_to_ch = nn.Linear(time_dim, ch)
        self.net = nn.Sequential(
            nn.Conv2d(ch * 3, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, ch, 3, padding=1),
        )
        self.ode_mix = nn.Parameter(torch.tensor(1.0))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def condition(self, c: torch.Tensor) -> torch.Tensor:
        x = self.lift(l2(c)).view(-1, self.ch, 8, 8)
        return self.lift_up(x)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        cond = self.condition(c)
        te = self.time_to_ch(self.time_embed(t)).unsqueeze(-1).unsqueeze(-1)
        te = te.expand(-1, -1, z_t.shape[-2], z_t.shape[-1])
        return self.net(torch.cat([z_t, cond, te], dim=1))

    @torch.no_grad()
    def decode(self, c: torch.Tensor, steps: int = 10) -> torch.Tensor:
        cond0 = self.condition(c)
        z = cond0.clone()
        dt = 1.0 / max(steps, 1)
        for i in range(steps):
            t = torch.full((z.shape[0],), 1.0 - i * dt, device=z.device, dtype=z.dtype)
            z = z - dt * self.forward(z, t, c)
        keep = torch.sigmoid(self.ode_mix)
        return keep * cond0 + (1.0 - keep) * z


def spatial_cfm_loss(model: SpatialCondCFM, target: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    z0 = target.float()
    z1 = model.condition(cond)
    t = torch.rand(z0.shape[0], device=z0.device, dtype=z0.dtype)
    z_t = cosine_interp(z0, z1, t)
    v_star = cosine_interp_velocity(z0, z1, t)
    return F.mse_loss(model(z_t, t, cond), v_star)


class GateAssembler(nn.Module):
    """Predict img2img strength in [s_min, s_max] from pooled Pc/Pf/R + semantic conf proxy."""

    def __init__(self, s_min: float = 0.22, s_max: float = 0.42):
        super().__init__()
        self.s_min = s_min
        self.s_max = s_max
        self.mlp = nn.Sequential(
            nn.Linear(4, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: (B,4) — [pc_energy, pf_energy, r_mean, sem_norm]
        logit = self.mlp(feat).squeeze(-1)
        return self.s_min + (self.s_max - self.s_min) * torch.sigmoid(logit)


class TCDAModel(nn.Module):
    """Perception multi-granularity + relational saliency; semantic tower is frozen external."""

    def __init__(self, in_dim: int = 1024):
        super().__init__()
        self.head_pc = UpsampleHead(in_dim, out_ch=3, hidden=1024, base_ch=96)  # coarse blur RGB
        self.head_pf = UpsampleHead(in_dim, out_ch=1, hidden=768, base_ch=64)  # fine depth
        self.head_r = UpsampleHead(in_dim, out_ch=1, hidden=768, base_ch=64)  # saliency / layout
        self.cfm_pc = SpatialCondCFM(in_dim, ch=3, hidden=96)
        self.cfm_pf = SpatialCondCFM(in_dim, ch=1, hidden=80)
        self.cfm_r = SpatialCondCFM(in_dim, ch=1, hidden=80)
        # hierarchical: refine Pc features toward Pf-conditioned structure embedding
        self.cfm_c2f = SpatialCondCFM(in_dim, ch=1, hidden=80)
        self.gate = GateAssembler()
        # project Pc → single-channel energy for c2f target pairing
        self.pc_to_energy = nn.Conv2d(3, 1, 1)
        nn.init.constant_(self.pc_to_energy.weight, 1.0 / 3.0)
        nn.init.zeros_(self.pc_to_energy.bias)

    def forward_heads(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pc = torch.sigmoid(self.head_pc(z))  # [0,1] RGB
        pf = self.head_pf(z)
        r = torch.sigmoid(self.head_r(z))
        return pc, pf, r
