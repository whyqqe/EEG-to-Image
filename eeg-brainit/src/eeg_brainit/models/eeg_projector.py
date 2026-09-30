"""Project EEG latents into Brain-IT Brain-Token space."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class EEGTokenProjector(nn.Module):
    """Map ``z_eeg`` / EEG token grid to Brain-Token dimension for KV fusion.

    Mode ``global``: one token from pooled ``z_eeg``.
    Mode ``multiscale``: use the encoder token grid (k = grid*grid tokens).
    """

    def __init__(
        self,
        d_eeg: int = 512,
        brain_dim: int = 1024,
        num_tokens: int = 4,
        hidden_mult: int = 2,
        dropout: float = 0.1,
        mode: str = "multiscale",
    ) -> None:
        super().__init__()
        self.mode = mode
        self.num_tokens = num_tokens
        hidden = d_eeg * hidden_mult
        self.mlp = nn.Sequential(
            nn.Linear(d_eeg, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, brain_dim),
            nn.LayerNorm(brain_dim),
        )
        # Learnable expand for global mode: z_eeg -> k tokens.
        self.global_expand = nn.Linear(brain_dim, brain_dim * num_tokens)
        self.brain_dim = brain_dim

    def forward(
        self,
        z_eeg: torch.Tensor,
        eeg_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Returns:
            EEG tokens of shape (B, k, brain_dim) to concatenate with Brain Tokens.
        """
        if self.mode == "multiscale" and eeg_tokens is not None:
            return self.mlp(eeg_tokens)

        # Global path: one vector -> k tokens.
        base = self.mlp(z_eeg)  # (B, brain_dim)
        expanded = self.global_expand(base).view(z_eeg.shape[0], self.num_tokens, self.brain_dim)
        return expanded

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "EEGTokenProjector":
        return cls(
            d_eeg=int(cfg.get("d_eeg", 512)),
            brain_dim=int(cfg.get("brain_dim", 1024)),
            num_tokens=int(cfg.get("num_tokens", 4)),
            hidden_mult=int(cfg.get("hidden_mult", 2)),
            dropout=float(cfg.get("dropout", 0.1)),
            mode=str(cfg.get("mode", "multiscale")),
        )
