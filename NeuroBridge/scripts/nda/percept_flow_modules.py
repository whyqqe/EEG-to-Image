#!/usr/bin/env python3
"""PerceptFlow modules: VAE head + spatial CondCFM for structure latents/depth."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_interp(z0: torch.Tensor, z1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    # t: (B,) → broadcast to z shape
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


class VAEHead(nn.Module):
    """EEG embed → (4, 64, 64) SDXL VAE latent."""

    def __init__(self, in_dim: int, hidden: int = 1536, spatial: int = 64, ch: int = 4):
        super().__init__()
        self.spatial = spatial
        self.ch = ch
        self.base = 8
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 128 * self.base * self.base),
        )
        self.up = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, ch, 3, padding=1),
        )
        nn.init.zeros_(self.up[-1].weight)
        nn.init.zeros_(self.up[-1].bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z).view(-1, 128, self.base, self.base)
        return self.up(x)


class DepthHead(nn.Module):
    """EEG embed → (1, 64, 64) depth map."""

    def __init__(self, in_dim: int, hidden: int = 1024, spatial: int = 64):
        super().__init__()
        self.base = 8
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 64 * self.base * self.base),
        )
        self.up = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(16, 1, 3, padding=1),
        )
        nn.init.zeros_(self.up[-1].weight)
        nn.init.zeros_(self.up[-1].bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z).view(-1, 64, self.base, self.base)
        return self.up(x)


class SpatialCondCFM(nn.Module):
    """Conditional CFM over spatial tensors (B,C,H,W), cond from EEG vector."""

    def __init__(self, cond_dim: int, ch: int, hidden: int = 128, time_dim: int = 64):
        super().__init__()
        self.lift = nn.Sequential(
            nn.Linear(cond_dim, 256),
            nn.GELU(),
            nn.Linear(256, ch * 8 * 8),
        )
        self.lift_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 16
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 32
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 64
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
        self.ch = ch

    def condition(self, c: torch.Tensor) -> torch.Tensor:
        x = self.lift(F.normalize(c.float(), dim=-1)).view(-1, self.ch, 8, 8)
        return self.lift_up(x)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        cond = self.condition(c)
        te = self.time_to_ch(self.time_embed(t)).unsqueeze(-1).unsqueeze(-1)
        te = te.expand(-1, -1, z_t.shape[-2], z_t.shape[-1])
        return self.net(torch.cat([z_t, cond, te], dim=1))

    @torch.no_grad()
    def decode(self, c: torch.Tensor, steps: int = 12) -> torch.Tensor:
        cond0 = self.condition(c)
        z = cond0.clone()
        dt = 1.0 / max(steps, 1)
        for i in range(steps):
            t_val = 1.0 - i * dt
            t = torch.full((z.shape[0],), t_val, device=z.device, dtype=z.dtype)
            v = self.forward(z, t, c)
            z = z - dt * v
        keep = torch.sigmoid(self.ode_mix)
        return keep * cond0 + (1.0 - keep) * z


def spatial_cfm_loss(model: SpatialCondCFM, target: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    z0 = target.float()
    z1 = model.condition(cond)
    t = torch.rand(z0.shape[0], device=z0.device, dtype=z0.dtype)
    z_t = cosine_interp(z0, z1, t)
    v_star = cosine_interp_velocity(z0, z1, t)
    v_pred = model(z_t, t, cond)
    return F.mse_loss(v_pred, v_star)


class ResidualSpatialCondCFM(nn.Module):
    """I-CFM over residual latents: noise → (target − μ), conditioned on EEG + μ."""

    def __init__(self, cond_dim: int, ch: int, hidden: int = 128, time_dim: int = 64):
        super().__init__()
        self.lift = nn.Sequential(
            nn.Linear(cond_dim, 256),
            nn.GELU(),
            nn.Linear(256, ch * 8 * 8),
        )
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
        self.mu_proj = nn.Conv2d(ch, ch, 1)
        self.time_embed = SinusoidalTimeEmbed(time_dim)
        self.time_to_ch = nn.Linear(time_dim, ch)
        self.net = nn.Sequential(
            nn.Conv2d(ch * 4, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, ch, 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.ch = ch

    def condition(self, c: torch.Tensor) -> torch.Tensor:
        x = self.lift(F.normalize(c.float(), dim=-1)).view(-1, self.ch, 8, 8)
        return self.lift_up(x)

    def forward(
        self, z_t: torch.Tensor, t: torch.Tensor, c: torch.Tensor, mu: torch.Tensor
    ) -> torch.Tensor:
        cond = self.condition(c)
        mu_f = self.mu_proj(mu.float())
        te = self.time_to_ch(self.time_embed(t)).unsqueeze(-1).unsqueeze(-1)
        te = te.expand(-1, -1, z_t.shape[-2], z_t.shape[-1])
        return self.net(torch.cat([z_t, cond, mu_f, te], dim=1))

    @torch.no_grad()
    def decode(
        self,
        c: torch.Tensor,
        mu: torch.Tensor,
        steps: int = 12,
        n_avg: int = 1,
        deterministic: bool = True,
    ) -> torch.Tensor:
        outs = []
        for _ in range(max(n_avg, 1)):
            z = torch.zeros_like(mu) if deterministic else torch.randn_like(mu)
            dt = 1.0 / max(steps, 1)
            for i in range(steps):
                t_val = i * dt
                t = torch.full((z.shape[0],), t_val, device=z.device, dtype=z.dtype)
                v = self.forward(z, t, c, mu)
                z = z + dt * v
            outs.append(z)
        return torch.stack(outs, dim=0).mean(dim=0)


def residual_cfm_loss(
    model: ResidualSpatialCondCFM,
    residual: torch.Tensor,
    cond: torch.Tensor,
    mu: torch.Tensor,
) -> torch.Tensor:
    """Linear I-CFM: z0~N(0,I) → z1=residual, conditioned on EEG + stopgrad(μ)."""
    z1 = residual.float()
    z0 = torch.randn_like(z1)
    t = torch.rand(z1.shape[0], device=z1.device, dtype=z1.dtype)
    while t.ndim < z1.ndim:
        t = t.unsqueeze(-1)
    z_t = (1.0 - t) * z0 + t * z1
    v_star = z1 - z0
    t_flat = t.view(z1.shape[0])
    v_pred = model(z_t, t_flat, cond, mu.detach())
    return F.mse_loss(v_pred, v_star)


class RCFMLLModel(nn.Module):
    """L1 VAE mean head + residual Cond-CFM (structure-only; no depth)."""

    def __init__(self, in_dim: int = 1024, vae_ch: int = 4, spatial: int = 64):
        super().__init__()
        self.vae_head = VAEHead(in_dim, spatial=spatial, ch=vae_ch)
        self.cfm_res = ResidualSpatialCondCFM(in_dim, ch=vae_ch, hidden=160)

    def predict(
        self,
        z: torch.Tensor,
        alpha: float = 1.0,
        ode_steps: int = 12,
        n_avg: int = 1,
        deterministic: bool = True,
    ) -> torch.Tensor:
        mu = self.vae_head(z)
        if alpha <= 0:
            return mu
        res = self.cfm_res.decode(
            z, mu.detach(), steps=ode_steps, n_avg=n_avg, deterministic=deterministic
        )
        return mu + float(alpha) * res


class PerceptFlowModel(nn.Module):
    def __init__(self, in_dim: int = 1024, vae_ch: int = 4, spatial: int = 64):
        super().__init__()
        self.vae_head = VAEHead(in_dim, spatial=spatial, ch=vae_ch)
        self.depth_head = DepthHead(in_dim, spatial=spatial)
        self.cfm_vae = SpatialCondCFM(in_dim, ch=vae_ch, hidden=128)
        self.cfm_depth = SpatialCondCFM(in_dim, ch=1, hidden=96)
