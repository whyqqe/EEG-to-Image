"""Physics residual adapter on frozen ATM embeddings (v2).

Design (success-oriented, literature-aligned):
  - Freeze authoritative ATM retrieval embeddings (NeurIPS'24) as f_θ
  - Build early EEG channel×band subspace features (physical prior)
  - Pretrain a structured residual adapter:  clip' = normalize(ATM + s * Δ_phys(EEG))
  - Few-shot personalize adapter (optionally gates) on a held-out subject

Refs: Subject-Conditioned LoRA (2025), FACE few-shot EEG adapter (2025),
      residual ATM distill practice in this repo.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from eeg_brainit.models.physics_prior_lora import (
    PhysicsFeatureExtractor,
    build_subspace_specs,
)


class PhysicsResidualAdapter(nn.Module):
    """Δ from raw EEG physical subspaces; residual on frozen ATM CLIP emb."""

    def __init__(
        self,
        clip_dim: int = 1024,
        rank: int = 8,
        feat_dim: int = 128,
        regions: list[str] | None = None,
        bands: list[str] | None = None,
        dropout: float = 0.1,
        init_scale: float = 0.1,
    ):
        super().__init__()
        self.specs = build_subspace_specs(regions=regions, bands=bands)
        self.feat = PhysicsFeatureExtractor(self.specs, out_dim=feat_dim, sfreq=250.0)
        # per-subspace low-rank map: feat_dim → clip_dim
        self.adapters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(feat_dim, rank, bias=False),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(rank, clip_dim, bias=False),
                )
                for _ in self.specs
            ]
        )
        for ad in self.adapters:
            nn.init.zeros_(ad[-1].weight)
        self.gate_logits = nn.Parameter(torch.zeros(len(self.specs)))
        self.res_scale = nn.Parameter(torch.tensor(float(init_scale)))
        self.clip_dim = clip_dim

    def gates(self) -> torch.Tensor:
        return torch.softmax(self.gate_logits, dim=0)

    def delta(self, eeg: torch.Tensor) -> torch.Tensor:
        sf = self.feat(eeg)  # (B, M, F)
        g = self.gates()
        out = 0.0
        for j, ad in enumerate(self.adapters):
            out = out + g[j] * ad(sf[:, j])
        return out

    def forward(self, eeg: torch.Tensor, atm_emb: torch.Tensor) -> dict[str, torch.Tensor]:
        atm = F.normalize(atm_emb.float(), dim=-1)
        d = self.delta(eeg)
        emb = F.normalize(atm + self.res_scale * d, dim=-1)
        return {
            "clip_emb": emb,
            "delta": d,
            "atm_emb": atm,
            "gates": self.gates(),
            "res_scale": self.res_scale.detach(),
        }

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class VanillaResidualAdapter(nn.Module):
    """Unstructured residual MLP ablation (parameter-matched ballpark)."""

    def __init__(
        self,
        n_channels: int = 63,
        seq_len: int = 250,
        clip_dim: int = 1024,
        hidden: int = 256,
        init_scale: float = 0.1,
    ):
        super().__init__()
        # pool time first to keep params comparable to physics adapter
        self.pool = nn.AdaptiveAvgPool1d(32)
        self.enc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_channels * 32, hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, clip_dim),
        )
        nn.init.zeros_(self.enc[-1].weight)
        nn.init.zeros_(self.enc[-1].bias)
        self.res_scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, eeg: torch.Tensor, atm_emb: torch.Tensor) -> dict[str, torch.Tensor]:
        atm = F.normalize(atm_emb.float(), dim=-1)
        h = self.pool(eeg.float())
        d = self.enc(h)
        emb = F.normalize(atm + self.res_scale * d, dim=-1)
        return {"clip_emb": emb, "delta": d, "atm_emb": atm}


def info_nce(pred: torch.Tensor, target: torch.Tensor, temp: float = 0.07) -> torch.Tensor:
    pn = F.normalize(pred.float(), dim=-1)
    tn = F.normalize(target.float(), dim=-1)
    logits = (pn @ tn.T) / temp
    labels = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def gate_entropy(gates: torch.Tensor) -> torch.Tensor:
    """Encourage peaked gates (lower entropy)."""
    g = gates.clamp_min(1e-8)
    return -(g * g.log()).sum()


@torch.no_grad()
def retrieval_topk(pred: torch.Tensor, gallery: torch.Tensor, ks=(1, 5)) -> dict[str, float]:
    pn = F.normalize(pred.float(), dim=-1)
    gn = F.normalize(gallery.float(), dim=-1)
    sim = pn @ gn.T
    n = sim.size(0)
    ranks = sim.argsort(dim=-1, descending=True)
    gt = torch.arange(n, device=sim.device).unsqueeze(1)
    out = {"chance_top1": 1.0 / max(n, 1)}
    for k in ks:
        out[f"top{k}"] = (ranks[:, : min(k, n)] == gt).any(1).float().mean().item()
    return out
