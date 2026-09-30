"""ATM pretrained embedding backbone + bridge into Brain-IT token space."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class AtmClipRefiner(nn.Module):
    """Light residual refiner in OpenCLIP ViT-H/14 space (1024-d)."""

    def __init__(self, dim: int = 1024, hidden: int = 1024, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + self.mlp(self.norm(x))
        return F.normalize(y, dim=-1)


class ClipReadout(nn.Module):
    """Pool tokens -> L2-normalized CLIP-H/14 embedding for alignment losses."""

    def __init__(self, in_dim: int = 1024, out_dim: int = 1024, hidden: int = 1024) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = tokens.mean(dim=1) if tokens.dim() == 3 else tokens
        return F.normalize(self.net(x), dim=-1)


class AtmToBitBridge(nn.Module):
    """Map ATM CLIP emb (B, 1024) -> brain tokens + EEG KV tokens for BIT."""

    def __init__(
        self,
        in_dim: int = 1024,
        brain_dim: int = 1024,
        num_brain_tokens: int = 128,
        num_eeg_tokens: int = 8,
        hidden: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_brain_tokens = num_brain_tokens
        self.num_eeg_tokens = num_eeg_tokens
        self.brain_dim = brain_dim
        self.in_norm = nn.LayerNorm(in_dim)
        self.brain_token_proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_brain_tokens * brain_dim),
        )
        self.eeg_token_proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_eeg_tokens * brain_dim),
        )
        self.brain_pos = nn.Parameter(torch.randn(num_brain_tokens, brain_dim) * 0.02)
        self.eeg_pos = nn.Parameter(torch.randn(num_eeg_tokens, brain_dim) * 0.02)

    def forward(self, atm_emb: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.in_norm(atm_emb)
        b = x.shape[0]
        brain = self.brain_token_proj(x).view(b, self.num_brain_tokens, self.brain_dim)
        eeg = self.eeg_token_proj(x).view(b, self.num_eeg_tokens, self.brain_dim)
        brain = brain + self.brain_pos.unsqueeze(0)
        eeg = eeg + self.eeg_pos.unsqueeze(0)
        return {
            "brain_tokens": brain,
            "eeg_kv_tokens": eeg,
            "atm_emb": F.normalize(atm_emb, dim=-1),
        }

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "AtmToBitBridge":
        return cls(
            in_dim=int(cfg.get("in_dim", 1024)),
            brain_dim=int(cfg.get("brain_dim", 1024)),
            num_brain_tokens=int(cfg.get("num_brain_tokens", 128)),
            num_eeg_tokens=int(cfg.get("num_eeg_tokens", 8)),
            hidden=int(cfg.get("hidden", 2048)),
            dropout=float(cfg.get("dropout", 0.1)),
        )


class AtmBrainITPipeline(nn.Module):
    """Pretrained-ATM-first architecture for EEG->Image.

    Recommended path (do NOT retrain ATM retrieval):
      frozen ATM emb
        -> clip_emb (= raw ATM; optional disabled/frozen refiner)
        -> AtmToBitBridge -> brain/eeg tokens -> bridge_clip (train target)
        -> BITCrossFusion -> clip_tokens/vgg -> bit_clip (generation path)
    """

    def __init__(
        self,
        bridge: AtmToBitBridge,
        bit: nn.Module,
        refiner: AtmClipRefiner | None = None,
        bridge_readout: ClipReadout | None = None,
        bit_readout: ClipReadout | None = None,
        use_bit: bool = True,
        freeze_atm_semantics: bool = True,
    ) -> None:
        super().__init__()
        self.bridge = bridge
        self.bit = bit
        self.refiner = refiner
        self.bridge_readout = bridge_readout or ClipReadout()
        self.bit_readout = bit_readout or ClipReadout(in_dim=getattr(bit, "clip_dim", 1664))
        # Residual scales (0 => exact ATM identity; preserves retrieval while learning tokens).
        self.bridge_res_scale = nn.Parameter(torch.zeros(()))
        self.bit_res_scale = nn.Parameter(torch.zeros(()))
        self.use_bit = use_bit
        self.freeze_atm_semantics = freeze_atm_semantics

    def forward(self, atm_emb: torch.Tensor) -> dict[str, torch.Tensor]:
        atm_emb = F.normalize(atm_emb.float(), dim=-1)
        if self.refiner is not None and not self.freeze_atm_semantics:
            clip_emb = self.refiner(atm_emb)
        else:
            clip_emb = atm_emb
        bridged = self.bridge(atm_emb)
        bridge_delta = self.bridge_readout(bridged["brain_tokens"])
        bridge_clip = F.normalize(atm_emb + self.bridge_res_scale * bridge_delta, dim=-1)
        out = {
            "clip_emb": clip_emb,
            "atm_emb": atm_emb,
            "brain_tokens": bridged["brain_tokens"],
            "eeg_kv_tokens": bridged["eeg_kv_tokens"],
            "bridge_clip": bridge_clip,
            "bridge_delta": bridge_delta,
        }
        if self.use_bit:
            bit_out = self.bit(bridged["brain_tokens"], bridged["eeg_kv_tokens"])
            out.update(bit_out)
            bit_delta = self.bit_readout(bit_out["clip_tokens"])
            out["bit_clip"] = F.normalize(atm_emb + self.bit_res_scale * bit_delta, dim=-1)
            out["bit_delta"] = bit_delta
        return out

    def apply_stage(self, stage: int) -> None:
        from eeg_brainit.utils.freeze import count_trainable, freeze, unfreeze

        freeze(self.bridge)
        freeze(self.bit)
        freeze(self.bridge_readout)
        freeze(self.bit_readout)
        self.bridge_res_scale.requires_grad_(False)
        self.bit_res_scale.requires_grad_(False)
        if self.refiner is not None:
            freeze(self.refiner)

        if stage == 0:
            if self.refiner is not None:
                unfreeze(self.refiner)
                self.freeze_atm_semantics = False
        elif stage == 1:
            unfreeze(self.bridge)
            unfreeze(self.bridge_readout)
            self.bridge_res_scale.requires_grad_(True)
            self.freeze_atm_semantics = True
        elif stage == 2:
            unfreeze(self.bridge)
            unfreeze(self.bridge_readout)
            unfreeze(self.bit)
            unfreeze(self.bit_readout)
            self.bridge_res_scale.requires_grad_(True)
            self.bit_res_scale.requires_grad_(True)
            self.freeze_atm_semantics = True
        elif stage == 3:
            unfreeze(self.bit)
            unfreeze(self.bit_readout)
            self.bit_res_scale.requires_grad_(True)
            self.freeze_atm_semantics = True
        elif stage == 4:
            # Pix-oriented S4: relax bridge+BIT readouts; allow embed to deviate from ATM.
            unfreeze(self.bridge_readout)
            unfreeze(self.bit)
            unfreeze(self.bit_readout)
            self.bridge_res_scale.requires_grad_(True)
            self.bit_res_scale.requires_grad_(True)
            self.freeze_atm_semantics = True
        else:
            raise ValueError(f"ATM pipeline supports stages 0..4, got {stage}")
        print(
            f"[INFO] ATM stage {stage}: trainable params = {count_trainable(self):,} "
            f"(freeze_atm_semantics={self.freeze_atm_semantics} "
            f"bridge_res={float(self.bridge_res_scale):.4f} bit_res={float(self.bit_res_scale):.4f})"
        )

    @classmethod
    def from_config(cls, cfg: dict[str, Any], project_root: str | None = None) -> "AtmBrainITPipeline":
        from pathlib import Path

        from eeg_brainit.models.bit_cross_fusion import BITCrossFusion

        atm_cfg = cfg.get("atm", {})
        bridge = AtmToBitBridge.from_config(atm_cfg.get("bridge", {}))
        bit = BITCrossFusion.from_config(cfg.get("bit", {}))
        refiner = None
        if bool(atm_cfg.get("refiner", {}).get("enabled", False)):
            refiner = AtmClipRefiner(
                dim=int(atm_cfg.get("dim", 1024)),
                hidden=int(atm_cfg.get("refiner", {}).get("hidden", 1024)),
                dropout=float(atm_cfg.get("refiner", {}).get("dropout", 0.1)),
            )
        dim = int(atm_cfg.get("dim", 1024))
        bridge_readout = ClipReadout(
            in_dim=int(atm_cfg.get("bridge", {}).get("brain_dim", 1024)),
            out_dim=dim,
            hidden=dim,
        )
        bit_readout = ClipReadout(
            in_dim=int(cfg.get("bit", {}).get("clip_dim", 1664)),
            out_dim=dim,
            hidden=dim,
        )
        pipe = cls(
            bridge=bridge,
            bit=bit,
            refiner=refiner,
            bridge_readout=bridge_readout,
            bit_readout=bit_readout,
            use_bit=bool(atm_cfg.get("use_bit", True)),
            freeze_atm_semantics=bool(atm_cfg.get("freeze_atm_semantics", True)),
        )
        bit_ckpt = cfg.get("bit", {}).get("checkpoint")
        if bit_ckpt:
            p = Path(bit_ckpt)
            if not p.is_absolute() and project_root:
                p = Path(project_root) / p
            if p.is_file():
                bit.load_pretrained_partial(p)
        return pipe
