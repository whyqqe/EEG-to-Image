"""Physics-prior structured low-rank adapters for EEG→CLIP subject adaptation.

Core update (paper Eq.1):
  h' = h + Σ_j B_j A_j P_j h
where P_j masks a channel×frequency physical subspace.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# Official-style 63-ch BioSemi montage used by THINGS-EEG2 / ATM pipelines.
THINGS_EEG2_CHANNELS: list[str] = [
    "Fp1", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8",
    "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8",
    "P9", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8", "P10",
    "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2", "Iz",
]

REGION_DEFS: dict[str, set[str]] = {
    "frontal": {"Fp1", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8", "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8"},
    "central": {"FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "C5", "C3", "C1", "Cz", "C2", "C4", "C6"},
    "temporal": {"FT7", "FT8", "T7", "T8", "TP7", "TP8", "P9", "P10"},
    "parietal": {"CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8"},
    "occipital": {"PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2", "Iz"},
}

# Hz bands @ 250 Hz sampling (THINGS-EEG2).
BAND_DEFS: dict[str, tuple[float, float]] = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 50.0),
}


@dataclass(frozen=True)
class SubspaceSpec:
    name: str
    region: str
    band: str
    channel_idx: tuple[int, ...]
    band_hz: tuple[float, float]


def build_subspace_specs(
    ch_names: list[str] | None = None,
    regions: list[str] | None = None,
    bands: list[str] | None = None,
) -> list[SubspaceSpec]:
    ch_names = list(ch_names or THINGS_EEG2_CHANNELS)
    regions = regions or ["occipital", "parietal", "temporal", "frontal"]
    bands = bands or ["theta", "alpha", "beta", "gamma"]
    name_to_i = {c: i for i, c in enumerate(ch_names)}
    specs: list[SubspaceSpec] = []
    for region in regions:
        idxs = tuple(sorted(name_to_i[c] for c in REGION_DEFS[region] if c in name_to_i))
        if not idxs:
            continue
        for band in bands:
            specs.append(
                SubspaceSpec(
                    name=f"{region}_{band}",
                    region=region,
                    band=band,
                    channel_idx=idxs,
                    band_hz=BAND_DEFS[band],
                )
            )
    if not specs:
        raise RuntimeError("no physical subspaces constructed")
    return specs


class BandpassBank(nn.Module):
    """Differentiable depthwise FIR bandpass bank → (B, C, F, T')."""

    def __init__(self, bands: list[tuple[float, float]], sfreq: float = 250.0, kernel: int = 51):
        super().__init__()
        self.n_bands = len(bands)
        self.sfreq = float(sfreq)
        weight = torch.zeros(len(bands), 1, kernel)
        t = torch.arange(kernel, dtype=torch.float32) - float(kernel // 2)
        window = torch.hamming_window(kernel, periodic=False)
        for i, (lo, hi) in enumerate(bands):
            # Ideal bandpass via difference of sinc lowpasses.
            def _lp(fc: float) -> torch.Tensor:
                if fc <= 0:
                    return torch.zeros_like(t)
                # sinc(x) = sin(pi x)/(pi x); use torch.sinc if available
                arg = 2 * fc / sfreq * t
                if hasattr(torch, "sinc"):
                    s = torch.sinc(arg)
                else:
                    s = torch.where(arg == 0, torch.ones_like(arg), torch.sin(math.pi * arg) / (math.pi * arg))
                x = 2 * fc / sfreq * s
                x = x * window
                return x / (x.sum() + 1e-8)

            w = _lp(hi) - _lp(lo)
            w = w - w.mean()
            weight[i, 0] = w
        self.register_buffer("weight", weight, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) → (B, C, F, T)
        b, c, t = x.shape
        y = F.conv1d(x.reshape(b * c, 1, t), self.weight, padding=self.weight.shape[-1] // 2)
        y = y.view(b, c, self.n_bands, t)
        return y


class SubspaceLoRA(nn.Module):
    """One physical subspace adapter: ΔW_j = B_j A_j on masked features."""

    def __init__(self, dim: int, rank: int = 8, alpha: float = 16.0, dropout: float = 0.1):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / max(rank, 1)
        self.A = nn.Linear(dim, rank, bias=False)
        self.B = nn.Linear(rank, dim, bias=False)
        self.drop = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.A.weight, a=5**0.5)
        nn.init.zeros_(self.B.weight)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.B(self.A(self.drop(h))) * self.scaling


class PhysicsPriorLoRA(nn.Module):
    """Structured sum of subspace LoRAs with optional gate w ∈ R^M (Eq.1 + Eq.5)."""

    def __init__(
        self,
        dim: int,
        specs: list[SubspaceSpec],
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1,
        learnable_gate: bool = True,
    ):
        super().__init__()
        self.specs = specs
        self.adapters = nn.ModuleList(
            [SubspaceLoRA(dim, rank=rank, alpha=alpha, dropout=dropout) for _ in specs]
        )
        if learnable_gate:
            self.gate = nn.Parameter(torch.ones(len(specs)))
        else:
            self.register_buffer("gate", torch.ones(len(specs)), persistent=True)

    @property
    def num_subspaces(self) -> int:
        return len(self.specs)

    def gated_weights(self) -> torch.Tensor:
        return torch.sigmoid(self.gate)

    def forward(self, h: torch.Tensor, subspace_feats: torch.Tensor) -> torch.Tensor:
        """
        h: (B, D) backbone embedding
        subspace_feats: (B, M, D) per-subspace pooled features (already P_j-masked)
        """
        w = self.gated_weights()
        delta = 0.0
        for j, adapter in enumerate(self.adapters):
            delta = delta + w[j] * adapter(subspace_feats[:, j])
        return h + delta

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PhysicsFeatureExtractor(nn.Module):
    """Build per-subspace features P_j x from raw EEG (B,C,T)."""

    def __init__(self, specs: list[SubspaceSpec], out_dim: int = 256, sfreq: float = 250.0):
        super().__init__()
        self.specs = specs
        bands = [BAND_DEFS[s.band] for s in specs]
        # unique bands for bank
        uniq = sorted(set(BAND_DEFS[s.band] for s in specs))
        self.band_to_idx = {b: i for i, b in enumerate(uniq)}
        self.bank = BandpassBank(uniq, sfreq=sfreq)
        self.projs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(len(s.channel_idx), out_dim),
                    nn.GELU(),
                    nn.Linear(out_dim, out_dim),
                )
                for s in specs
            ]
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,T)
        xb = self.bank(x)  # (B,C,F,T)
        feats = []
        for j, spec in enumerate(self.specs):
            f_idx = self.band_to_idx[spec.band_hz]
            # masked channels × selected band → mean over time → (B, n_ch)
            sel = xb[:, list(spec.channel_idx), f_idx, :].mean(dim=-1)
            feats.append(self.projs[j](sel))
        return torch.stack(feats, dim=1)  # (B,M,D)


__all__ = [
    "THINGS_EEG2_CHANNELS",
    "REGION_DEFS",
    "BAND_DEFS",
    "SubspaceSpec",
    "build_subspace_specs",
    "PhysicsPriorLoRA",
    "PhysicsFeatureExtractor",
    "BandpassBank",
    "SubspaceLoRA",
]
