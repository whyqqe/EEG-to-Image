from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def pixel_correlation(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean Pearson correlation over flattened spatial dims. Shapes: (B, C, H, W)."""
    b = pred.shape[0]
    p = pred.reshape(b, -1)
    t = target.reshape(b, -1)
    p = p - p.mean(dim=1, keepdim=True)
    t = t - t.mean(dim=1, keepdim=True)
    num = (p * t).sum(dim=1)
    den = p.norm(dim=1) * t.norm(dim=1).clamp_min(1e-8)
    return (num / den).mean().item()


def ssim_simple(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """Lightweight luminance/contrast SSIM proxy for monitoring (not paper-grade)."""
    c1, c2 = 0.01**2, 0.03**2
    mu_x = pred.mean(dim=(-2, -1), keepdim=True)
    mu_y = target.mean(dim=(-2, -1), keepdim=True)
    sigma_x = ((pred - mu_x) ** 2).mean(dim=(-2, -1), keepdim=True)
    sigma_y = ((target - mu_y) ** 2).mean(dim=(-2, -1), keepdim=True)
    sigma_xy = ((pred - mu_x) * (target - mu_y)).mean(dim=(-2, -1), keepdim=True)
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2) + eps
    )
    return ssim.mean().item()


def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return F.mse_loss(pred, target).item()
