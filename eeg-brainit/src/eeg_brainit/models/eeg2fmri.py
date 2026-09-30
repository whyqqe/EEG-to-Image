"""EEG → fMRI (+ optional CLIP) for NOD Phase-1.

Mainline: fine-tune pretrained LaBraM → fMRI head (+ CLIP aux for Image path).
Probe TemporalEEGEncoder is retained only for ablation / smoke tests.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from eeg_brainit.models.labram_encoder import LaBraMEncoder


class TemporalEEGEncoder(nn.Module):
    """Scratch temporal CNN probe (ablation only — not formal training)."""

    def __init__(self, n_channels: int = 62, d_model: int = 768, dropout: float = 0.2, depth: int = 4) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(n_channels, n_channels, kernel_size=7, padding=3, groups=n_channels),
            nn.Conv1d(n_channels, d_model // 2, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.ms = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(d_model // 2, d_model // 2, kernel_size=k, padding=k // 2, groups=d_model // 2),
                    nn.Conv1d(d_model // 2, d_model // 2, kernel_size=1),
                    nn.GELU(),
                )
                for k in (3, 9, 25, 49)
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv1d(d_model // 2 * 4, d_model, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, groups=d_model),
                    nn.Conv1d(d_model, d_model, kernel_size=1),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for _ in range(max(1, depth))
            ]
        )
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, d_model),
            nn.Sigmoid(),
        )
        self.out_norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        h = self.stem(x)
        h = self.fuse(torch.cat([b(h) for b in self.ms], dim=1))
        for blk in self.blocks:
            h = h + blk(h)
        h = h * self.se(h).unsqueeze(-1)
        attn = torch.softmax(h.mean(dim=1, keepdim=True), dim=-1)
        return self.out_norm((h * attn).sum(dim=-1))


class EEG2fMRIModel(nn.Module):
    """Foundation (LaBraM) or probe encoder → fMRI head + optional CLIP head."""

    def __init__(
        self,
        n_channels: int = 62,
        num_rois: int = 128,
        clip_dim: int = 1024,
        d_model: int = 200,
        hidden: int = 1024,
        dropout: float = 0.15,
        use_clip_head: bool = True,
        backbone: str = "labram",
        ch_names: Sequence[str] | None = None,
        labram_ckpt: str | None = None,
        unfreeze_last_n_blocks: int = 4,
        train_patch_embed: bool = False,
        pool_mode: str = "mean",
        freeze_backbone_initially: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self._unfreeze_last_n_blocks = unfreeze_last_n_blocks
        self._train_patch_embed = train_patch_embed
        if backbone == "labram":
            if not ch_names:
                raise ValueError("labram backbone requires ch_names")
            enc = LaBraMEncoder(ch_names=ch_names, embed_dim=200, pool_mode=pool_mode)
            if labram_ckpt:
                stats = enc.load_pretrained(labram_ckpt)
                print(f"[INFO] LaBraM load {stats} pool={pool_mode}")
            if freeze_backbone_initially:
                enc.set_finetune_mode(unfreeze_last_n_blocks=0, train_patch_embed=False)
            else:
                enc.set_finetune_mode(
                    unfreeze_last_n_blocks=unfreeze_last_n_blocks,
                    train_patch_embed=train_patch_embed,
                )
            self.encoder = enc
            d_model = enc.embed_dim
        elif backbone == "temporal":
            self.encoder = TemporalEEGEncoder(n_channels=n_channels, d_model=d_model, dropout=dropout)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.fmri_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_rois),
        )
        self.use_clip_head = use_clip_head
        self.clip_head = None
        if use_clip_head:
            self.clip_head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, clip_dim),
            )

    def forward(self, eeg: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.encoder(eeg.float())
        fmri = self.fmri_head(feat)
        out: dict[str, torch.Tensor] = {"feat": feat, "fmri_pred": fmri, "fmri_roi": fmri}
        if self.clip_head is not None:
            out["clip_pred"] = self.clip_head(feat)
        return out

    def unfreeze_backbone(self) -> None:
        if self.backbone != "labram":
            for p in self.encoder.parameters():
                p.requires_grad = True
            return
        assert isinstance(self.encoder, LaBraMEncoder)
        self.encoder.set_finetune_mode(
            unfreeze_last_n_blocks=self._unfreeze_last_n_blocks,
            train_patch_embed=self._train_patch_embed,
        )

    def trainable_parameter_groups(
        self, backbone_lr: float, head_lr: float, weight_decay: float
    ) -> list[dict[str, Any]]:
        backbone_params = [p for p in self.encoder.parameters() if p.requires_grad]
        head_params = list(self.fmri_head.parameters())
        if self.clip_head is not None:
            head_params += list(self.clip_head.parameters())
        groups = []
        if backbone_params:
            groups.append(
                {"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay, "name": "backbone"}
            )
        groups.append({"params": head_params, "lr": head_lr, "weight_decay": weight_decay, "name": "heads"})
        return groups

    @classmethod
    def from_config(
        cls,
        cfg: dict[str, Any],
        ch_names: Sequence[str] | None = None,
    ) -> "EEG2fMRIModel":
        mcfg = cfg.get("eeg2fmri", {})
        tcfg = cfg.get("train", {})
        backbone = str(mcfg.get("backbone", "labram"))
        ckpt = mcfg.get("labram_ckpt", "checkpoints/neurobolt/labram-base.pth")
        heads_only = int(tcfg.get("heads_only_epochs", 0)) > 0
        return cls(
            n_channels=int(mcfg.get("in_dim", 62)),
            num_rois=int(mcfg.get("num_rois", 128)),
            clip_dim=int(mcfg.get("clip_dim", 1024)),
            d_model=int(mcfg.get("d_model", 200 if backbone == "labram" else 512)),
            hidden=int(mcfg.get("hidden", 1024)),
            dropout=float(mcfg.get("dropout", 0.15)),
            use_clip_head=bool(mcfg.get("use_clip_head", True)),
            backbone=backbone,
            ch_names=ch_names,
            labram_ckpt=str(ckpt) if ckpt else None,
            unfreeze_last_n_blocks=int(mcfg.get("unfreeze_last_n_blocks", 4)),
            train_patch_embed=bool(mcfg.get("train_patch_embed", False)),
            pool_mode=str(mcfg.get("pool_mode", "mean")),
            freeze_backbone_initially=heads_only,
        )


def pairwise_nce(pred: torch.Tensor, target: torch.Tensor, temp: float = 0.07) -> torch.Tensor:
    pn = F.normalize(pred.float(), dim=-1)
    tn = F.normalize(target.float(), dim=-1)
    logits = pn @ tn.transpose(0, 1) / max(temp, 1e-6)
    labels = torch.arange(pred.size(0), device=pred.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.transpose(0, 1), labels))


def fmri_losses(
    pred: torch.Tensor,
    target: torch.Tensor,
    lambda_mse: float = 1.0,
    lambda_corr: float = 1.0,
    lambda_cosine: float = 0.5,
    lambda_nce: float = 0.1,
    nce_temp: float = 0.1,
) -> dict[str, torch.Tensor]:
    pred = pred.float()
    target = target.float()
    mse = F.mse_loss(pred, target)
    p = pred - pred.mean(dim=-1, keepdim=True)
    t = target - target.mean(dim=-1, keepdim=True)
    denom = p.norm(dim=-1) * t.norm(dim=-1) + 1e-6
    corr = (p * t).sum(dim=-1) / denom
    loss_corr = (1.0 - corr).mean()
    pn = F.normalize(pred, dim=-1)
    tn = F.normalize(target, dim=-1)
    cos = (pn * tn).sum(dim=-1)
    loss_cos = (1.0 - cos).mean()
    loss_nce = pairwise_nce(pred, target, nce_temp)
    total = lambda_mse * mse + lambda_corr * loss_corr + lambda_cosine * loss_cos + lambda_nce * loss_nce
    return {
        "total": total,
        "mse": mse.detach(),
        "corr": corr.mean().detach(),
        "cosine": cos.mean().detach(),
        "loss_nce": loss_nce.detach(),
    }


def clip_losses(pred: torch.Tensor, target: torch.Tensor, temp: float = 0.07) -> dict[str, torch.Tensor]:
    """InfoNCE only (no cosine term — cosine rewards mean-collapse without ranking)."""
    pn = F.normalize(pred.float(), dim=-1)
    tn = F.normalize(target.float(), dim=-1)
    cos = (pn * tn).sum(dim=-1).mean()
    nce = pairwise_nce(pred, target, temp)
    return {"total": nce, "nce": nce.detach(), "cosine": cos.detach()}


@torch.no_grad()
def retrieval_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pn = F.normalize(pred.float(), dim=-1)
    tn = F.normalize(target.float(), dim=-1)
    sim = pn @ tn.transpose(0, 1)
    n = sim.size(0)
    ranks = sim.argsort(dim=-1, descending=True)
    gt = torch.arange(n, device=sim.device).unsqueeze(1)
    top1 = (ranks[:, :1] == gt).any(dim=1).float().mean().item()
    top5 = (ranks[:, : min(5, n)] == gt).any(dim=1).float().mean().item()
    return {"top1": top1, "top5": top5, "chance_top1": 1.0 / max(n, 1)}
