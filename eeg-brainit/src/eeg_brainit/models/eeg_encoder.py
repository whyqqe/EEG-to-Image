"""MD-TF-CAE spectrogram encoder (Spec2VolCAMU-Net style).

Vendored lightly so we do not depend on Spec2Vol's VMUNet/diffusers import chain
at install time. When ``third_party/Spec2VolCAMU-Net`` is present, callers may
also load the full ``Spectrogram2fMRI`` decoder via ``virtual_fmri.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.branch_freq = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(3, 1), padding=(1, 0)),
            nn.SiLU(),
        )
        self.branch_time = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(1, 3), padding=(0, 1)),
            nn.SiLU(),
        )
        self.branch_joint = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.SiLU(),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(96, out_channels, kernel_size=1),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fusion(
            torch.cat([self.branch_freq(x), self.branch_time(x), self.branch_joint(x)], dim=1)
        )


class TimeSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        seq = x.permute(3, 0, 2, 1).reshape(w, n * h, c)
        out, _ = self.attn(seq, seq, seq)
        return out.reshape(w, n, h, c).permute(1, 3, 2, 0)


class EncoderBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int) -> None:
        super().__init__()
        self.multi_scale = MultiScaleConv(channels, channels)
        self.ln1 = nn.LayerNorm(channels)
        self.attn = TimeSelfAttention(channels, num_heads)
        self.ln2 = nn.LayerNorm(channels)
        self.downsample = nn.Conv2d(
            channels, channels, kernel_size=(1, 3), stride=(1, 2), padding=(0, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.multi_scale(x)
        x = self.ln1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = x + self.attn(x)
        x = self.ln2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.downsample(x)


class SpectrogramEncoder(nn.Module):
    def __init__(self, in_dim: int = 63, out_dim: int = 256, num_heads: int = 8) -> None:
        super().__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.block1 = EncoderBlock(out_dim, num_heads=num_heads)
        self.block2 = EncoderBlock(out_dim, num_heads=num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial_conv(x)
        x = self.block1(x)
        x = self.block2(x)
        pad_w = max(0, 64 - x.shape[-1])
        pad_h = max(0, 64 - x.shape[-2])
        if pad_w or pad_h:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        return x


class MDTFCAEEncoder(nn.Module):
    """Extract EEG latent features ``z_eeg`` from multi-channel spectrograms.

    Input:  (B, C, F, T) spectrogram
    Output: (B, d_eeg) pooled latent, and optionally token grid for projection.
    """

    def __init__(
        self,
        in_channels: int = 63,
        hidden_dim: int = 256,
        d_eeg: int = 512,
        num_heads: int = 8,
        grid_size: int = 8,
    ) -> None:
        super().__init__()
        self.encoder = SpectrogramEncoder(in_dim=in_channels, out_dim=hidden_dim, num_heads=num_heads)
        self.pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))
        self.token_proj = nn.Linear(hidden_dim, d_eeg)
        self.global_proj = nn.Linear(hidden_dim, d_eeg)
        self.d_eeg = d_eeg
        self.grid_size = grid_size

    def forward_features(self, spectrogram: torch.Tensor) -> torch.Tensor:
        x = spectrogram
        if x.ndim != 4:
            raise ValueError(f"Expected (B,C,F,T), got {tuple(x.shape)}")
        if x.shape[-2] < 8 or x.shape[-1] < 8:
            x = F.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False)
        return self.encoder(x)

    def forward(self, spectrogram: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.forward_features(spectrogram)  # (B, H, F', T')
        grid = self.pool(feat)  # (B, H, G, G)
        tokens = self.token_proj(grid.flatten(2).transpose(1, 2))  # (B, G*G, d)
        z_eeg = self.global_proj(feat.mean(dim=(-2, -1)))  # (B, d)
        return {"z_eeg": z_eeg, "eeg_tokens": tokens, "feat_map": feat}

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "MDTFCAEEncoder":
        return cls(
            in_channels=int(cfg.get("in_channels", 63)),
            hidden_dim=int(cfg.get("hidden_dim", 256)),
            d_eeg=int(cfg.get("d_eeg", 512)),
            num_heads=int(cfg.get("num_heads", 8)),
            grid_size=int(cfg.get("grid_size", 8)),
        )

    def load_pretrained(self, path: str | Path, strict: bool = False) -> None:
        ckpt = Path(path)
        if not ckpt.is_file():
            raise FileNotFoundError(
                f"Spec2Vol encoder checkpoint not found: {ckpt}. "
                "Official Spec2VolCAMU-Net does not publish weights; "
                "train locally or leave randomly initialized."
            )
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict) and "encoder" in state:
            state = state["encoder"]
        missing, unexpected = self.load_state_dict(state, strict=strict)
        print(f"[INFO] Loaded MD-TF-CAE from {ckpt} missing={len(missing)} unexpected={len(unexpected)}")
