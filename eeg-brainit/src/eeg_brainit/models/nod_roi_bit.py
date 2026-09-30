"""NOD ROI vector → Brain-IT-style tokens → CLIP (Phase-2 decoder)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from eeg_brainit.models.bit_cross_fusion import BITCrossFusion


class NodRoiBitDecoder(nn.Module):
    """Map NOD fMRI ROI vector (B, R) into BiT cross-transformer → CLIP embedding.

    Keeps the NeuroBOLT↔BiT contract as an fMRI interface: ROI activity in,
    CLIP semantics out. Optional partial init from Brain-IT decoder weights.
    """

    def __init__(
        self,
        roi_dim: int = 64,
        brain_dim: int = 512,
        num_brain_tokens: int = 64,
        num_query_tokens: int = 128,
        num_blocks: int = 2,
        num_heads: int = 8,
        clip_dim: int = 1024,
        dropout: float = 0.1,
        brainit_ckpt: str | None = None,
    ) -> None:
        super().__init__()
        self.roi_dim = int(roi_dim)
        self.brain_dim = int(brain_dim)
        self.num_brain_tokens = int(num_brain_tokens)
        self.clip_dim = int(clip_dim)

        self.roi_to_tokens = nn.Sequential(
            nn.LayerNorm(self.roi_dim),
            nn.Linear(self.roi_dim, self.brain_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.brain_dim, self.num_brain_tokens * self.brain_dim),
        )
        self.token_norm = nn.LayerNorm(self.brain_dim)
        self.bit = BITCrossFusion(
            brain_dim=self.brain_dim,
            num_brain_tokens=self.num_brain_tokens,
            num_query_tokens=int(num_query_tokens),
            num_blocks=int(num_blocks),
            num_heads=int(num_heads),
            clip_dim=self.clip_dim,
            vgg_dim=512,
            dropout=float(dropout),
            inject_eeg_as="none",
        )
        if brainit_ckpt:
            self.bit.load_pretrained_partial(brainit_ckpt)

    def forward(self, fmri_roi: torch.Tensor) -> dict[str, torch.Tensor]:
        b = fmri_roi.shape[0]
        tokens = self.roi_to_tokens(fmri_roi.float()).view(b, self.num_brain_tokens, self.brain_dim)
        tokens = self.token_norm(tokens)
        out = self.bit(tokens, eeg_tokens=None)
        clip_tokens = out["clip_tokens"]  # (B, Q, C)
        clip_emb = F.normalize(clip_tokens.mean(dim=1), dim=-1)
        return {
            "clip_emb": clip_emb,
            "clip_tokens": clip_tokens,
            "brain_tokens": tokens,
        }

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "NodRoiBitDecoder":
        return cls(
            roi_dim=int(cfg.get("roi_dim", 64)),
            brain_dim=int(cfg.get("brain_dim", 512)),
            num_brain_tokens=int(cfg.get("num_brain_tokens", 64)),
            num_query_tokens=int(cfg.get("num_query_tokens", 128)),
            num_blocks=int(cfg.get("num_blocks", 2)),
            num_heads=int(cfg.get("num_heads", 8)),
            clip_dim=int(cfg.get("clip_dim", 1024)),
            dropout=float(cfg.get("dropout", 0.1)),
            brainit_ckpt=cfg.get("brainit_ckpt"),
        )
