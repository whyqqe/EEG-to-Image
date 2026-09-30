"""DA-Calibrator (SCC): Decode-Aware SDXL Condition Calibrator modules.

Evidence fusion (raw EEG + frozen NB) -> anchor-residual ViT-H 1024-d condition
for IP-Adapter, with differentiable soft memory and decode proxy.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def l2norm(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1)


class Freq2SemLite(nn.Module):
    """Lightweight frequency branch (D²-FOSA-inspired): rFFT log-power -> MLP."""

    def __init__(self, time_steps: int = 250, d_out: int = 256, fft_bins: int = 64):
        super().__init__()
        self.fft_bins = min(fft_bins, time_steps // 2 + 1)
        self.mlp = nn.Sequential(
            nn.Linear(self.fft_bins, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, d_out),
            nn.LayerNorm(d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) -> channel-mean waveform
        wave = x.mean(dim=1)
        spec = torch.fft.rfft(wave, dim=-1).abs()
        spec = spec[:, : self.fft_bins]
        spec = torch.log1p(spec)
        return self.mlp(spec)


class RawTemporalStream(nn.Module):
    """Shallow temporal Conv1D on multi-channel EEG."""

    def __init__(self, channels: int = 17, d_out: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, 48, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(48, 64, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Sequential(nn.Linear(64, d_out), nn.LayerNorm(d_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x).squeeze(-1)
        return self.proj(h)


class EvidenceEncoder(nn.Module):
    """Fuse raw EEG streams with frozen NB latent."""

    def __init__(self, nb_dim: int = 512, d_ctx: int = 512):
        super().__init__()
        self.raw_t = RawTemporalStream(d_out=d_ctx // 2)
        self.raw_f = Freq2SemLite(d_out=d_ctx // 2)
        self.fuse = nn.Sequential(
            nn.Linear(d_ctx + nb_dim, d_ctx),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_ctx, d_ctx),
            nn.LayerNorm(d_ctx),
        )

    def forward(self, eeg: torch.Tensor, z_nb: torch.Tensor) -> torch.Tensor:
        h_t = self.raw_t(eeg)
        h_f = self.raw_f(eeg)
        h_raw = torch.cat([h_t, h_f], dim=-1)
        return self.fuse(torch.cat([h_raw, z_nb], dim=-1))


class AnchorResidualCalibrator(nn.Module):
    """Predict sphere residual on retrieval anchor (+ optional memory correction)."""

    def __init__(self, d_ctx: int = 512, cond_dim: int = 1024, hidden_mult: float = 2.0):
        super().__init__()
        hid = int(cond_dim * hidden_mult)
        in_dim = d_ctx + cond_dim * 2
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hid, hid),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hid, cond_dim),
        )
        self.alpha_sem = nn.Parameter(torch.tensor(0.35))
        self.alpha_mem = nn.Parameter(torch.tensor(0.15))

    def forward(
        self,
        z_ctx: torch.Tensor,
        e_anchor: torch.Tensor,
        e_mem: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        a_n = l2norm(e_anchor)
        m_n = l2norm(e_mem)
        delta = self.mlp(torch.cat([z_ctx, a_n, m_n], dim=-1))
        delta = torch.tanh(delta)
        raw = a_n + self.alpha_sem * delta + self.alpha_mem * (m_n - a_n)
        e_cond = l2norm(raw)
        return e_cond, delta


class StrengthHead(nn.Module):
    """Predict img2img strength in [sigma_min, sigma_max]."""

    def __init__(self, d_in: int = 512, sigma_min: float = 0.35, sigma_max: float = 0.50):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.net = nn.Sequential(nn.Linear(d_in, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, z_ctx: torch.Tensor) -> torch.Tensor:
        s = torch.sigmoid(self.net(z_ctx))
        return self.sigma_min + (self.sigma_max - self.sigma_min) * s


class DifferentiableSoftMemory(nn.Module):
    """Soft attention over frozen CLIP gallery."""

    def __init__(
        self,
        gallery_clip: torch.Tensor,
        gallery_keys: torch.Tensor,
        soft_k: int = 5,
        tau: float = 0.07,
    ):
        super().__init__()
        self.soft_k = soft_k
        self.tau = tau
        self.register_buffer("gallery_clip", l2norm(gallery_clip))
        self.register_buffer("gallery_keys", l2norm(gallery_keys))

    def forward(self, query_keys: torch.Tensor) -> torch.Tensor:
        q = l2norm(query_keys)
        sim = q @ self.gallery_keys.T
        k = min(self.soft_k, sim.shape[1])
        topv, topi = torch.topk(sim, k=k, dim=1)
        w = torch.softmax(topv / max(self.tau, 1e-6), dim=1)
        gathered = self.gallery_clip[topi]
        out = torch.sum(gathered * w.unsqueeze(-1), dim=1)
        return l2norm(out)

    def top1_anchor(self, query_keys: torch.Tensor) -> torch.Tensor:
        q = l2norm(query_keys)
        sim = q @ self.gallery_keys.T
        idx = sim.argmax(dim=1)
        return l2norm(self.gallery_clip[idx])


class ProbeDecoder(nn.Module):
    """Predict post-decode CLIP from [condition, anchor]."""

    def __init__(self, dim: int = 1024, hidden_mult: float = 2.0, dropout: float = 0.1):
        super().__init__()
        hid = int(dim * hidden_mult)
        self.net = nn.Sequential(
            nn.Linear(dim * 2, hid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid, hid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid, dim),
        )

    def forward(self, e_cond: torch.Tensor, anchor_emb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([e_cond, anchor_emb], dim=-1)
        return self.net(x)


class ClipInfoNCE(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, eeg: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        eeg = l2norm(eeg)
        image = l2norm(image)
        logits = (eeg @ image.T) / self.temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss_e = F.cross_entropy(logits, labels)
        loss_i = F.cross_entropy(logits.T, labels)
        return (loss_e + loss_i) / 2


class HardNegativeNCE(nn.Module):
    """InfoNCE with optional hard negatives from precomputed neighbor indices."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        query: torch.Tensor,
        positive: torch.Tensor,
        hard_neg: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = l2norm(query)
        pos = l2norm(positive)
        pos_logits = (q * pos).sum(dim=-1) / self.temperature
        if hard_neg is not None and hard_neg.numel() > 0:
            neg = l2norm(hard_neg)
            neg_logits = (q.unsqueeze(1) * neg).sum(dim=-1) / self.temperature
            logits = torch.cat([pos_logits.unsqueeze(1), neg_logits], dim=1)
        else:
            logits = (q @ pos.T) / self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)


class SDXLConditionCalibrator(nn.Module):
    """Full SCC: evidence -> anchor-residual SDXL IP-Adapter condition."""

    def __init__(
        self,
        gallery_clip: torch.Tensor,
        gallery_keys: torch.Tensor,
        nb_dim: int = 512,
        cond_dim: int = 1024,
        d_ctx: int = 512,
        soft_k: int = 5,
        soft_tau: float = 0.07,
    ):
        super().__init__()
        self.evidence = EvidenceEncoder(nb_dim=nb_dim, d_ctx=d_ctx)
        self.memory = DifferentiableSoftMemory(gallery_clip, gallery_keys, soft_k, soft_tau)
        self.calibrator = AnchorResidualCalibrator(d_ctx=d_ctx, cond_dim=cond_dim)
        self.strength = StrengthHead(d_in=d_ctx)
        self.dino_head = nn.Sequential(
            nn.Linear(d_ctx, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.probe = ProbeDecoder(dim=cond_dim)

    def refresh_gallery_keys(self, keys: torch.Tensor) -> None:
        self.memory.gallery_keys.copy_(l2norm(keys))

    def forward(
        self,
        eeg: torch.Tensor,
        z_proj: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        z_ctx = self.evidence(eeg, z_proj)
        e_anchor = self.memory.top1_anchor(z_proj)
        e_mem = self.memory(z_proj)
        e_cond, delta = self.calibrator(z_ctx, e_anchor, e_mem)
        sigma = self.strength(z_ctx)
        dino_pred = l2norm(self.dino_head(z_ctx))
        probe_pred = l2norm(self.probe(e_cond, e_anchor))
        return {
            "z_ctx": z_ctx,
            "e_anchor": e_anchor,
            "e_mem": e_mem,
            "e_cond": e_cond,
            "delta": delta,
            "sigma": sigma,
            "dino_pred": dino_pred,
            "probe_pred": probe_pred,
        }
