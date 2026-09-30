"""BIT Cross-Transformer with EEG tokens as extra Key/Value.

This module mirrors Brain-IT's Cross-Transformer interface:
  Query = learnable Query Tokens (image-feature questions)
  Key/Value = [Brain Tokens (128), EEG Tokens (k)]
  Output = localized semantic (CLIP) and structural (VGG-like) features
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class CrossBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm_kv = nn.LayerNorm(dim)
        self.norm_q = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_mlp = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, query: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(query)
        k = self.norm_kv(kv)
        attn_out, _ = self.attn(q, k, k, need_weights=False)
        x = query + attn_out
        x = x + self.mlp(self.norm_mlp(x))
        return x


class SelfBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class BITCrossFusion(nn.Module):
    """Brain Interaction Transformer Cross-Transformer with optional EEG KV injection."""

    def __init__(
        self,
        brain_dim: int = 1024,
        num_brain_tokens: int = 128,
        num_query_tokens: int = 256,
        num_blocks: int = 2,
        num_heads: int = 8,
        clip_dim: int = 1664,
        vgg_dim: int = 512,
        dropout: float = 0.0,
        inject_eeg_as: str = "kv",  # "kv" | "query" | "none"
    ) -> None:
        super().__init__()
        self.brain_dim = brain_dim
        self.num_brain_tokens = num_brain_tokens
        self.num_query_tokens = num_query_tokens
        self.inject_eeg_as = inject_eeg_as

        self.brain_centers = nn.Parameter(torch.randn(num_brain_tokens, brain_dim) * 0.02)
        self.query_tokens = nn.Parameter(torch.randn(num_query_tokens, brain_dim) * 0.02)
        self.norm_brain = nn.LayerNorm(brain_dim)

        self.self_blocks = nn.ModuleList(
            [SelfBlock(brain_dim, num_heads, dropout=dropout) for _ in range(num_blocks)]
        )
        self.cross_blocks = nn.ModuleList(
            [CrossBlock(brain_dim, num_heads, dropout=dropout) for _ in range(num_blocks + 1)]
        )

        self.clip_proj = nn.Linear(brain_dim, clip_dim)
        self.vgg_proj = nn.Linear(brain_dim, vgg_dim)
        self.clip_dim = clip_dim
        self.vgg_dim = vgg_dim

    def _expand(self, token: torch.Tensor, batch: int) -> torch.Tensor:
        return token.unsqueeze(0).expand(batch, -1, -1)

    def forward(
        self,
        brain_tokens: torch.Tensor,
        eeg_tokens: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            brain_tokens: (B, 128, D) from virtual-fMRI tokenizer
            eeg_tokens:   (B, k, D) projected EEG tokens (optional)
        """
        b = brain_tokens.shape[0]
        # Residual with shared learnable centers (Brain-IT style).
        centers = self.norm_brain(brain_tokens + self._expand(self.brain_centers, b))
        queries = self._expand(self.query_tokens, b)

        if self.inject_eeg_as == "query" and eeg_tokens is not None:
            queries = torch.cat([queries, eeg_tokens], dim=1)

        kv = centers
        if self.inject_eeg_as == "kv" and eeg_tokens is not None:
            kv = torch.cat([centers, eeg_tokens], dim=1)
        # inject_eeg_as == "none": ignore eeg_tokens (ablation)

        # Initial cross-attn, then alternate self on brain tokens + cross update.
        queries = self.cross_blocks[0](queries, kv)
        for i, self_blk in enumerate(self.self_blocks):
            centers = self_blk(centers)
            kv = centers
            if self.inject_eeg_as == "kv" and eeg_tokens is not None:
                kv = torch.cat([centers, eeg_tokens], dim=1)
            queries = queries + self.cross_blocks[i + 1](queries, kv)

        # Keep only the original query slots for image heads.
        image_queries = queries[:, : self.num_query_tokens]
        clip_tokens = self.clip_proj(image_queries)  # (B, Q, clip_dim)
        vgg_features = self.vgg_proj(image_queries)  # (B, Q, vgg_dim)
        return {
            "clip_tokens": clip_tokens,
            "vgg_features": vgg_features,
            "query_states": image_queries,
            "kv_len": kv.shape[1],
        }

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "BITCrossFusion":
        return cls(
            brain_dim=int(cfg.get("brain_dim", 1024)),
            num_brain_tokens=int(cfg.get("num_brain_tokens", 128)),
            num_query_tokens=int(cfg.get("num_query_tokens", 256)),
            num_blocks=int(cfg.get("num_blocks", 2)),
            num_heads=int(cfg.get("num_heads", 8)),
            clip_dim=int(cfg.get("clip_dim", 1664)),
            vgg_dim=int(cfg.get("vgg_dim", 512)),
            dropout=float(cfg.get("dropout", 0.0)),
            inject_eeg_as=str(cfg.get("inject_eeg_as", "kv")),
        )

    def load_pretrained_partial(self, path: str | Path) -> None:
        """Best-effort load of overlapping tensors from Brain-IT decoder pickles."""
        ckpt = Path(path)
        if not ckpt.is_file():
            # Prefer converted state_dict next to official pickle.
            alt = ckpt.with_name("decoder_clipg_state_dict.pt")
            if alt.is_file():
                ckpt = alt
            else:
                print(f"[WARN] Brain-IT checkpoint missing: {ckpt}")
                return
        # Ensure Brain-IT package is importable if the file is a pickled Module.
        brainit_root = Path(__file__).resolve().parents[3] / "third_party" / "brainit-fmri"
        if brainit_root.is_dir():
            import sys

            if str(brainit_root) not in sys.path:
                sys.path.insert(0, str(brainit_root))
        try:
            obj = torch.load(ckpt, map_location="cpu", weights_only=False)
        except ModuleNotFoundError as exc:
            print(f"[WARN] Could not unpickle {ckpt} ({exc}); skipping BIT preload")
            return
        if isinstance(obj, nn.Module):
            state = obj.state_dict()
        elif isinstance(obj, dict) and "state_dict" in obj:
            state = obj["state_dict"]
        else:
            state = obj if isinstance(obj, dict) else {}

        mapped: dict[str, torch.Tensor] = {}
        for key, value in state.items():
            if key in {"centers", "brain_centers"} and value.shape == self.brain_centers.shape:
                mapped["brain_centers"] = value
            elif key in {"pred_tokens", "query_tokens"} and value.shape == self.query_tokens.shape:
                mapped["query_tokens"] = value
            elif key.startswith("norm_centers.") and key.replace("norm_centers.", "norm_brain.") in dict(
                self.norm_brain.named_parameters()
            ):
                mapped[key.replace("norm_centers.", "norm_brain.")] = value
            elif key in {"proj.weight", "clip_proj.weight"} and value.shape == self.clip_proj.weight.shape:
                mapped["clip_proj.weight"] = value
            elif key in {"proj.bias", "clip_proj.bias"} and value.shape == self.clip_proj.bias.shape:
                mapped["clip_proj.bias"] = value
        missing, unexpected = self.load_state_dict(mapped, strict=False)
        print(
            f"[INFO] Partial Brain-IT load from {ckpt}: "
            f"mapped={len(mapped)} missing={len(missing)} unexpected={len(unexpected)}"
        )
