"""ATM-style EEG→CLIP backbone (NeurIPS 2024 EEG_Image_decode ATMS-inspired).

Compact, self-contained reimplementation of the ATMS retrieval encoder path:
  channel-wise attention over time → temporal-spatial PatchEmbedding → CLIP projector.
Used as frozen/pretrained f_θ for Physics-Prior LoRA adaptation experiments.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttentionEncoder(nn.Module):
    """Lightweight channel-as-token transformer (iTransformer-style)."""

    def __init__(self, n_channels: int = 63, seq_len: int = 250, d_model: int = 128, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(seq_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.out = nn.Linear(d_model, seq_len)
        self.n_channels = n_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) → (B, C, T)
        h = self.proj(x)
        h = self.encoder(h)
        return self.out(h)


class TemporalSpatialPatch(nn.Module):
    """ATM PatchEmbedding-style temporal-spatial conv stack."""

    def __init__(self, n_channels: int = 63, emb_size: int = 40):
        super().__init__()
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), stride=(1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (n_channels, 1), stride=(1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1)),
        )
        self.emb_size = emb_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        y = x.unsqueeze(1)
        y = self.tsconv(y)
        y = self.projection(y)
        return y.flatten(1)  # (B, emb*time)


class AtmStyleEEGEncoder(nn.Module):
    """f_θ: R^(C×T) → R^d  (ATM-style retrieval backbone)."""

    def __init__(
        self,
        n_channels: int = 63,
        seq_len: int = 250,
        clip_dim: int = 1024,
        d_model: int = 128,
        emb_size: int = 40,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.channel_encoder = ChannelAttentionEncoder(
            n_channels=n_channels, seq_len=seq_len, d_model=d_model, dropout=dropout
        )
        self.patch = TemporalSpatialPatch(n_channels=n_channels, emb_size=emb_size)
        # Infer flatten dim with a dry run buffer
        with torch.no_grad():
            dummy = torch.zeros(1, n_channels, seq_len)
            flat_dim = int(self.patch(dummy).shape[-1])
        self.flat_dim = flat_dim
        self.proj = nn.Sequential(
            nn.Linear(flat_dim, clip_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(clip_dim, clip_dim),
            nn.Dropout(dropout),
            nn.LayerNorm(clip_dim),
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))
        self.clip_dim = clip_dim

    def encode_hidden(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        h = self.channel_encoder(x)
        return self.patch(h)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        flat = self.encode_hidden(x)
        emb = F.normalize(self.proj(flat), dim=-1)
        return {"hidden": flat, "clip_emb": emb, "logit_scale": self.logit_scale.exp()}


class HierarchicalHeads(nn.Module):
    """Low / mid / high visual-cognitive heads on shared embedding."""

    def __init__(self, in_dim: int, clip_dim: int = 1024, n_classes: int = 1654):
        super().__init__()
        self.low = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, clip_dim))
        self.mid = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, clip_dim))
        self.high = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, n_classes))

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "low": F.normalize(self.low(h), dim=-1),
            "mid": F.normalize(self.mid(h), dim=-1),
            "high_logits": self.high(h),
        }


class PhysicsPriorAdaptModel(nn.Module):
    """Frozen/pretrained ATM backbone + physics-prior LoRA + hierarchical heads."""

    def __init__(
        self,
        backbone: AtmStyleEEGEncoder,
        physics_feat,
        physics_lora,
        hier_heads: HierarchicalHeads | None = None,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.physics_feat = physics_feat
        self.physics_lora = physics_lora
        self.hier = hier_heads
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        with torch.set_grad_enabled(any(p.requires_grad for p in self.backbone.parameters())):
            base = self.backbone(x)
        # Map hidden→clip space residual via physics LoRA in clip dim
        # Use clip emb as h, subspace feats projected to clip dim inside LoRA path
        sf = self.physics_feat(x)
        # project subspace feats to clip dim if needed
        if sf.shape[-1] != base["clip_emb"].shape[-1]:
            raise RuntimeError(
                f"subspace dim {sf.shape[-1]} != clip dim {base['clip_emb'].shape[-1]}; "
                "set PhysicsFeatureExtractor.out_dim == clip_dim"
            )
        adapted = self.physics_lora(base["clip_emb"], sf)
        adapted = F.normalize(adapted, dim=-1)
        out = {
            "clip_emb": adapted,
            "clip_base": base["clip_emb"],
            "logit_scale": base["logit_scale"],
            "gate": self.physics_lora.gated_weights(),
        }
        if self.hier is not None:
            out.update(self.hier(adapted))
        return out

    def adapter_parameters(self):
        for m in (self.physics_feat, self.physics_lora, self.hier):
            if m is None:
                continue
            yield from m.parameters()


def info_nce(pred: torch.Tensor, target: torch.Tensor, temp: float = 0.07) -> torch.Tensor:
    pn = F.normalize(pred.float(), dim=-1)
    tn = F.normalize(target.float(), dim=-1)
    logits = (pn @ tn.T) / temp
    labels = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


@torch.no_grad()
def retrieval_topk(pred: torch.Tensor, gallery: torch.Tensor, ks: tuple[int, ...] = (1, 5)) -> dict[str, float]:
    pn = F.normalize(pred.float(), dim=-1)
    gn = F.normalize(gallery.float(), dim=-1)
    sim = pn @ gn.T
    n = sim.size(0)
    ranks = sim.argsort(dim=-1, descending=True)
    gt = torch.arange(n, device=sim.device).unsqueeze(1)
    out = {"chance_top1": 1.0 / max(n, 1)}
    for k in ks:
        out[f"top{k}"] = (ranks[:, : min(k, n)] == gt).any(dim=1).float().mean().item()
    return out
