"""DecodeAligner: Probe-Decoder + differentiable soft memory."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def l2norm(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1)


class ProbeDecoder(nn.Module):
    """Predict post-decode CLIP embedding from [eeg_emb, anchor_emb]."""

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

    def forward(self, eeg_emb: torch.Tensor, anchor_emb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([eeg_emb, anchor_emb], dim=-1)
        return self.net(x)


class DifferentiableSoftMemory(nn.Module):
    """Soft attention over a frozen CLIP gallery (train-set ViT-H embeddings)."""

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
        return self.gallery_clip[sim.argmax(dim=1)]


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
