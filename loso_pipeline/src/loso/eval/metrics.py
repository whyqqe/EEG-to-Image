"""Reconstruction + retrieval metrics for Stage-4 evaluation.

Retrieval (on CLIP-image space, population-mean adapter, single-trial unless noted):
  top1 / top5   closed-set among 200 test concepts
  2-way         chance=50%, pairwise with a random foil
  40-way        chance=2.5%

Reconstruction (generated vs ground-truth 512px render):
  PixCorr, SSIM, PSNR, LPIPS, CLIP cosine, DINO cosine
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class RetrievalScores:
    top1: float
    top5: float
    two_way: float
    forty_way: float
    n: int


def retrieval_from_similarity(sim: torch.Tensor, labels: torch.Tensor | None = None,
                              n_two: int = 200, n_forty: int = 200,
                              seed: int = 0) -> RetrievalScores:
    """`sim` is (N_query, N_gallery); gallery index i is the match for query i."""
    n = sim.shape[0]
    if labels is None:
        labels = torch.arange(n, device=sim.device)
    # Closed-set top-k
    order = sim.argsort(dim=-1, descending=True)
    hits = order.eq(labels.unsqueeze(1))
    top1 = float(hits[:, :1].any(dim=1).float().mean())
    top5 = float(hits[:, :5].any(dim=1).float().mean())

    rng = np.random.default_rng(seed)
    # 2-way: for each query, pick one random foil and check if match scores higher
    two = 0.0
    for i in range(n):
        foil = int(rng.integers(0, n - 1))
        if foil >= i:
            foil += 1
        two += float(sim[i, i] > sim[i, foil])
    two /= max(1, n)

    # 40-way: match + 39 random foils
    forty = 0.0
    for i in range(n):
        foils = [j for j in range(n) if j != i]
        pick = rng.choice(foils, size=min(39, len(foils)), replace=False)
        cands = np.concatenate([[i], pick])
        forty += float(int(sim[i, cands].argmax().item()) == 0)
    forty /= max(1, n)

    return RetrievalScores(top1=top1, top5=top5, two_way=two, forty_way=forty, n=n)


def pixcorr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation over flattened pixels; a,b in [0,1] or uint8."""
    x = a.astype(np.float64).ravel()
    y = b.astype(np.float64).ravel()
    if x.std() < 1e-8 or y.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    from skimage.metrics import structural_similarity
    if a.dtype != np.float64:
        a = a.astype(np.float64)
        b = b.astype(np.float64)
        if a.max() > 1.5:
            a, b = a / 255.0, b / 255.0
    return float(structural_similarity(a, b, channel_axis=-1, data_range=1.0))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    if a.max() > 1.5:
        a, b = a / 255.0, b / 255.0
    mse = float(np.mean((a - b) ** 2))
    if mse < 1e-12:
        return 99.0
    return float(10.0 * math.log10(1.0 / mse))


class LPIPSMetric:
    def __init__(self, device: str = "cuda"):
        import lpips
        self.net = lpips.LPIPS(net="alex").to(device).eval()
        self.device = device

    @torch.inference_mode()
    def __call__(self, a: np.ndarray, b: np.ndarray) -> float:
        def to_t(x):
            if x.dtype == np.uint8:
                x = x.astype(np.float32) / 127.5 - 1.0
            else:
                x = x.astype(np.float32) * 2.0 - 1.0
            t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(self.device)
            return t
        return float(self.net(to_t(a), to_t(b)).item())


def cosine_np(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
