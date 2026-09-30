"""R²-FOSA: Retrieval-anchored D²-FOSA — FSTDE + NB prior + Anchor-DDLG."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from decode_aligner_modules import DifferentiableSoftMemory, l2norm  # noqa: E402
from nmb_ddlem_train import DDLEM  # noqa: E402


class RawTemporalStream(nn.Module):
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
        return self.proj(self.net(x).squeeze(-1))


class FreqOrientedBranch(nn.Module):
    """D²-FOSA-inspired rFFT log-power branch (FOMamba-lite)."""

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
        wave = x.mean(dim=1)
        spec = torch.fft.rfft(wave, dim=-1).abs()[:, : self.fft_bins]
        return self.mlp(torch.log1p(spec))


class FSTDE(nn.Module):
    """Frequency-Spatio-Temporal Dynamics Encoder (lite FSTDE / FOMamba substitute)."""

    def __init__(self, channels: int = 17, d_ctx: int = 512, nb_dim: int = 512):
        super().__init__()
        half = d_ctx // 2
        self.raw_t = RawTemporalStream(channels, half)
        self.raw_f = FreqOrientedBranch(d_out=half)
        self.fuse = nn.Sequential(
            nn.Linear(d_ctx + nb_dim, d_ctx),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_ctx, d_ctx),
            nn.LayerNorm(d_ctx),
        )
        self.dino_head = nn.Sequential(
            nn.Linear(d_ctx, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
        )

    def forward(self, eeg: torch.Tensor, z_proj: torch.Tensor) -> torch.Tensor:
        h = torch.cat([self.raw_t(eeg), self.raw_f(eeg)], dim=-1)
        return self.fuse(torch.cat([h, z_proj], dim=-1))

    def predict_dino(self, ctx: torch.Tensor) -> torch.Tensor:
        return l2norm(self.dino_head(ctx))


class R2FOSAModel(nn.Module):
    """
    cond = [z_proj, e_anchor, fstde_ctx] -> Anchor-DDLG (bidirectional DDLEM).
    Inference: DDIM warm-start from e_anchor (retrieval anchor on CLIP sphere).
    """

    def __init__(
        self,
        gallery_clip: torch.Tensor,
        gallery_keys: torch.Tensor,
        proj_dim: int = 512,
        ctx_dim: int = 512,
        latent_dim: int = 1024,
        cond_dim: int = 512,
        hidden: int = 2048,
        soft_k: int = 5,
        soft_tau: float = 0.07,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.proj_dim = proj_dim
        self.ctx_dim = ctx_dim
        self.latent_dim = latent_dim
        self.fstde = FSTDE(channels=17, d_ctx=ctx_dim, nb_dim=proj_dim)
        self.memory = DifferentiableSoftMemory(gallery_clip, gallery_keys, soft_k, soft_tau)
        cond_in = proj_dim + latent_dim + ctx_dim
        self.ddlem = DDLEM(
            cond_in=cond_in,
            latent_dim=latent_dim,
            cond_dim=cond_dim,
            hidden=hidden,
            bidirectional=bidirectional,
        )

    def refresh_gallery_keys(self, keys: torch.Tensor) -> None:
        self.memory.gallery_keys.copy_(l2norm(keys))

    def build_cond(self, z_proj: torch.Tensor, e_anchor: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        return self.ddlem.encode_cond(torch.cat([z_proj, e_anchor, ctx], dim=-1))

    def align_predict(self, z_proj: torch.Tensor, e_anchor: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        return self.ddlem.align_predict(torch.cat([z_proj, e_anchor, ctx], dim=-1))

    def memory_read(self, z_proj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        e_anchor = self.memory.top1_anchor(z_proj)
        e_mem = self.memory(z_proj)
        return e_anchor, e_mem

    def top1_anchor(self, z_proj: torch.Tensor) -> torch.Tensor:
        return self.memory.top1_anchor(z_proj)
