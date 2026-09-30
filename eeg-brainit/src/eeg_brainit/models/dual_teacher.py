"""Dual-Teacher CLIP Bridge: EEG → CLIP with image + frozen fMRI teachers."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from eeg_brainit.models.eeg2fmri import TemporalEEGEncoder


class DualTeacherEEG2CLIP(nn.Module):
    """ATM/NICE-style: temporal EEG encoder + MLP projector to CLIP space."""

    def __init__(
        self,
        n_channels: int = 26,
        clip_dim: int = 1024,
        d_model: int = 512,
        hidden: int = 1024,
        dropout: float = 0.2,
        depth: int = 4,
    ) -> None:
        super().__init__()
        self.encoder = TemporalEEGEncoder(
            n_channels=n_channels, d_model=d_model, dropout=dropout, depth=depth
        )
        self.proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, clip_dim),
        )
        self.clip_dim = clip_dim

    def forward(self, eeg: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encoder(eeg.float())
        emb = F.normalize(self.proj(z), dim=-1)
        return {"z_eeg": z, "clip_emb": emb}


class FmriClipTeacher(nn.Module):
    """Frozen GT fMRI ROI → CLIP teacher (loads Phase-2 fmri2clip.pt)."""

    def __init__(self, in_dim: int = 64, out_dim: int = 1024, hidden: int = 512, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, fmri: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(fmri.float()), dim=-1)

    @classmethod
    def from_checkpoint(cls, path: str, device: torch.device) -> "FmriClipTeacher":
        ck = torch.load(path, map_location="cpu", weights_only=False)
        in_dim = int(ck.get("in_dim", 64))
        model = cls(in_dim=in_dim)
        state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        # Phase-2 saves as FmriToClip with key prefix "net."
        if any(k.startswith("net.") for k in state):
            model.load_state_dict(state, strict=True)
        else:
            model.net.load_state_dict(state, strict=False)
        model.to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return model


def dual_teacher_losses(
    pred: torch.Tensor,
    z_img: torch.Tensor,
    z_fmri: torch.Tensor | None,
    *,
    temp: float = 0.07,
    lambda_nce: float = 1.0,
    lambda_mse_img: float = 1.0,
    lambda_mse_fmri: float = 0.25,
) -> dict[str, torch.Tensor]:
    pn = F.normalize(pred.float(), dim=-1)
    tn = F.normalize(z_img.float(), dim=-1)
    logits = (pn @ tn.T) / temp
    labels = torch.arange(logits.size(0), device=logits.device)
    nce = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
    mse_img = F.mse_loss(pn, tn)
    total = lambda_nce * nce + lambda_mse_img * mse_img
    out = {
        "total": total,
        "nce": nce.detach(),
        "mse_img": mse_img.detach(),
        "cos_img": (pn * tn).sum(-1).mean().detach(),
    }
    if z_fmri is not None and lambda_mse_fmri > 0:
        fn = F.normalize(z_fmri.float(), dim=-1)
        mse_f = F.mse_loss(pn, fn)
        total = total + lambda_mse_fmri * mse_f
        out["total"] = total
        out["mse_fmri"] = mse_f.detach()
        out["cos_fmri"] = (pn * fn).sum(-1).mean().detach()
    else:
        out["mse_fmri"] = torch.tensor(0.0)
        out["cos_fmri"] = torch.tensor(0.0)
    return out
