"""RGT-v2 modules: cos-preserving transport + dual conditioning + encoder adapter."""

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


class RetEncoderAdapter(nn.Module):
    """Light residual adapter on z_ret (subject-aware), zero-init → identity."""

    def __init__(self, ret_dim: int = 512, bottleneck: int = 256, n_subjects: int = 11):
        super().__init__()
        self.subj = nn.Embedding(n_subjects, bottleneck)
        self.down = nn.Linear(ret_dim, bottleneck)
        self.up = nn.Linear(bottleneck, ret_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, z_ret: torch.Tensor, subject_ids: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.down(F.normalize(z_ret.float(), dim=-1)) + self.subj(subject_ids.long().clamp(0, self.subj.num_embeddings - 1)))
        return F.normalize(z_ret.float() + self.up(h), dim=-1)


class RGTVelocityV2(nn.Module):
    """Cos-preserving CFM: deeper velocity, dual cond (ret + optional nda), soft ODE mix."""

    def __init__(
        self,
        ret_dim: int = 512,
        gen_dim: int = 1024,
        hidden: int = 2560,
        time_dim: int = 256,
        n_subjects: int = 11,
        subj_dim: int = 64,
        use_nda_cond: bool = True,
    ) -> None:
        super().__init__()
        self.ret_dim = ret_dim
        self.gen_dim = gen_dim
        self.use_nda_cond = use_nda_cond
        self.lift = nn.Sequential(
            nn.Linear(ret_dim, gen_dim),
            nn.LayerNorm(gen_dim),
            nn.GELU(),
            nn.Linear(gen_dim, gen_dim),
            nn.LayerNorm(gen_dim),
        )
        self.subj_emb = nn.Embedding(n_subjects, subj_dim)
        self.subj_to_scale = nn.Linear(subj_dim, gen_dim)
        self.subj_to_shift = nn.Linear(subj_dim, gen_dim)
        if use_nda_cond:
            self.nda_proj = nn.Sequential(
                nn.Linear(gen_dim, gen_dim),
                nn.GELU(),
                nn.Linear(gen_dim, gen_dim),
            )
            self.nda_gate = nn.Parameter(torch.tensor(-2.0))  # start near 0 → mostly lift
        else:
            self.nda_proj = None
            self.nda_gate = None
        self.time_embed = SinusoidalTimeEmbed(time_dim)
        self.net = nn.Sequential(
            nn.Linear(gen_dim * 2 + time_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, gen_dim),
        )
        # decode mix: high → keep lift (high cos); low → trust ODE
        self.ode_mix = nn.Parameter(torch.tensor(1.5))  # sigmoid≈0.82 keep lift
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def condition(
        self,
        z_ret: torch.Tensor,
        subject_ids: torch.Tensor,
        z_nda: torch.Tensor | None = None,
    ) -> torch.Tensor:
        c = self.lift(F.normalize(z_ret.float(), dim=-1))
        s = self.subj_emb(subject_ids.long().clamp(0, self.subj_emb.num_embeddings - 1))
        c = c * (1.0 + torch.tanh(self.subj_to_scale(s))) + self.subj_to_shift(s)
        if self.use_nda_cond and z_nda is not None and self.nda_proj is not None:
            g = torch.sigmoid(self.nda_gate)
            c = (1.0 - g) * c + g * self.nda_proj(F.normalize(z_nda.float(), dim=-1))
        return c

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        z_ret: torch.Tensor,
        subject_ids: torch.Tensor,
        z_nda: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cond = self.condition(z_ret, subject_ids, z_nda)
        te = self.time_embed(t)
        return self.net(torch.cat([z_t, cond, te], dim=-1))

    def decode(
        self,
        z_ret: torch.Tensor,
        subject_ids: torch.Tensor,
        z_nda: torch.Tensor | None = None,
        *,
        steps: int = 20,
    ) -> torch.Tensor:
        cond0 = self.condition(z_ret, subject_ids, z_nda)
        z = cond0.clone()
        dt = 1.0 / max(steps, 1)
        for i in range(steps):
            t_val = 1.0 - i * dt
            t = torch.full((z.shape[0],), t_val, device=z.device, dtype=z.dtype)
            v = self.forward(z, t, z_ret, subject_ids, z_nda)
            z = z - dt * v
        keep = torch.sigmoid(self.ode_mix)
        # keep*cond0 + (1-keep)*ode  → preserve paired cos from lift
        return F.normalize(keep * cond0 + (1.0 - keep) * z, dim=-1)


def flow_matching_loss_v2(
    model: RGTVelocityV2,
    z_gen: torch.Tensor,
    z_ret: torch.Tensor,
    subject_ids: torch.Tensor,
    z_nda: torch.Tensor | None = None,
) -> torch.Tensor:
    z0 = F.normalize(z_gen.float(), dim=-1)
    z1 = F.normalize(model.condition(z_ret, subject_ids, z_nda), dim=-1)
    t = torch.rand(z0.shape[0], device=z0.device, dtype=z0.dtype)
    z_t = cosine_interp(z0, z1, t)
    v_star = cosine_interp_velocity(z0, z1, t)
    v_pred = model(z_t, t, z_ret, subject_ids, z_nda)
    return F.mse_loss(v_pred, v_star)


def neighborhood_preserve_loss(
    z_ret: torch.Tensor,
    z_gen_hat: torch.Tensor,
    *,
    temperature: float = 0.07,
) -> torch.Tensor:
    a = F.normalize(z_ret.float(), dim=-1)
    b = F.normalize(z_gen_hat.float(), dim=-1)
    sa = a @ a.T / temperature
    sb = b @ b.T / temperature
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
