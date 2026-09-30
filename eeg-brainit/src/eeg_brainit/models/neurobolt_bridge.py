"""NeuroBOLT fMRI ROI tokens -> Brain-IT interface + residual ATM distill."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from eeg_brainit.models.atm_bridge import ClipReadout


class RoiToBitBridge(nn.Module):
    """Map NeuroBOLT fMRI ROI tokens (B, T, 512) -> brain + eeg_kv tokens for BIT.

    Uses learnable queries with cross-attention over ROI tokens so the middle
    representation is genuinely EEG→fMRI-derived, not an ATM vector reshape.
    """

    def __init__(
        self,
        fmri_dim: int = 512,
        brain_dim: int = 1024,
        num_brain_tokens: int = 128,
        num_eeg_tokens: int = 8,
        num_heads: int = 8,
        hidden: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_brain_tokens = num_brain_tokens
        self.num_eeg_tokens = num_eeg_tokens
        self.brain_dim = brain_dim
        self.fmri_dim = fmri_dim

        self.ctx_proj = nn.Sequential(
            nn.LayerNorm(fmri_dim),
            nn.Linear(fmri_dim, brain_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.brain_queries = nn.Parameter(torch.randn(num_brain_tokens, brain_dim) * 0.02)
        self.eeg_queries = nn.Parameter(torch.randn(num_eeg_tokens, brain_dim) * 0.02)
        self.brain_attn = nn.MultiheadAttention(
            brain_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.eeg_attn = nn.MultiheadAttention(
            brain_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.brain_ff = nn.Sequential(
            nn.LayerNorm(brain_dim),
            nn.Linear(brain_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, brain_dim),
        )
        self.eeg_ff = nn.Sequential(
            nn.LayerNorm(brain_dim),
            nn.Linear(brain_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, brain_dim),
        )
        self.brain_out_norm = nn.LayerNorm(brain_dim)
        self.eeg_out_norm = nn.LayerNorm(brain_dim)

    def forward(self, fmri_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        # fmri_tokens: (B, T, 512) — already centered by dataset when enabled.
        ctx = self.ctx_proj(fmri_tokens.float())
        b = ctx.shape[0]
        bq = self.brain_queries.unsqueeze(0).expand(b, -1, -1)
        eq = self.eeg_queries.unsqueeze(0).expand(b, -1, -1)
        brain, _ = self.brain_attn(bq, ctx, ctx, need_weights=False)
        brain = self.brain_out_norm(bq + brain + self.brain_ff(brain))
        eeg, _ = self.eeg_attn(eq, ctx, ctx, need_weights=False)
        eeg = self.eeg_out_norm(eq + eeg + self.eeg_ff(eeg))
        return {
            "brain_tokens": brain,
            "eeg_kv_tokens": eeg,
            "fmri_ctx": ctx,
        }

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "RoiToBitBridge":
        return cls(
            fmri_dim=int(cfg.get("fmri_dim", 512)),
            brain_dim=int(cfg.get("brain_dim", 1024)),
            num_brain_tokens=int(cfg.get("num_brain_tokens", 128)),
            num_eeg_tokens=int(cfg.get("num_eeg_tokens", 8)),
            num_heads=int(cfg.get("num_heads", 8)),
            hidden=int(cfg.get("hidden", 1024)),
            dropout=float(cfg.get("dropout", 0.1)),
        )


class DirectRoiClipHead(nn.Module):
    """Ablation A: NeuroBOLT → mean-pool MLP → CLIP (no BIT)."""

    def __init__(self, fmri_dim: int = 512, clip_dim: int = 1024, hidden: int = 1024) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(fmri_dim),
            nn.Linear(fmri_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, clip_dim),
        )

    def forward(self, fmri_tokens: torch.Tensor) -> torch.Tensor:
        x = fmri_tokens.float().mean(dim=1)
        return F.normalize(self.net(x), dim=-1)


class NeuroBoltBrainITPipeline(nn.Module):
    """True EEG→fMRI(ROI)→Brain-IT path.

    residual_mode:
      - atm_skip: clip = normalize(atm + s * delta)  (safe floor; may hide NB signal)
      - none:     clip = normalize(delta)            (forces fMRI path to carry semantics)
    """

    def __init__(
        self,
        bridge: RoiToBitBridge,
        bit: nn.Module,
        bridge_readout: ClipReadout | None = None,
        bit_readout: ClipReadout | None = None,
        direct_head: DirectRoiClipHead | None = None,
        use_bit: bool = True,
        use_direct: bool = False,
        residual_mode: str = "atm_skip",
    ) -> None:
        super().__init__()
        self.bridge = bridge
        self.bit = bit
        self.bridge_readout = bridge_readout or ClipReadout()
        self.bit_readout = bit_readout or ClipReadout(in_dim=getattr(bit, "clip_dim", 1664))
        self.direct_head = direct_head
        self.bridge_res_scale = nn.Parameter(torch.zeros(()))
        self.bit_res_scale = nn.Parameter(torch.zeros(()))
        self.direct_res_scale = nn.Parameter(torch.zeros(()))
        self.use_bit = use_bit
        self.use_direct = use_direct
        if residual_mode not in ("atm_skip", "none"):
            raise ValueError(f"residual_mode must be atm_skip|none, got {residual_mode}")
        self.residual_mode = residual_mode

    def _compose(self, atm_emb: torch.Tensor, delta: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        if self.residual_mode == "none":
            return F.normalize(delta.float(), dim=-1)
        return F.normalize(atm_emb + scale * delta, dim=-1)

    def forward(
        self,
        fmri_tokens: torch.Tensor,
        atm_emb: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        atm_emb = F.normalize(atm_emb.float(), dim=-1)
        bridged = self.bridge(fmri_tokens)
        bridge_delta = self.bridge_readout(bridged["brain_tokens"])
        bridge_clip = self._compose(atm_emb, bridge_delta, self.bridge_res_scale)
        out: dict[str, torch.Tensor] = {
            "atm_emb": atm_emb,
            "fmri_tokens": fmri_tokens,
            "brain_tokens": bridged["brain_tokens"],
            "eeg_kv_tokens": bridged["eeg_kv_tokens"],
            "bridge_clip": bridge_clip,
            "bridge_delta": bridge_delta,
            "clip_emb": bridge_clip,
            "residual_mode": torch.tensor(0 if self.residual_mode == "none" else 1),
        }
        if self.use_direct and self.direct_head is not None:
            direct = self.direct_head(fmri_tokens)
            out["direct_raw"] = direct
            out["direct_clip"] = self._compose(atm_emb, direct, self.direct_res_scale)
        if self.use_bit:
            bit_out = self.bit(bridged["brain_tokens"], bridged["eeg_kv_tokens"])
            out.update(bit_out)
            bit_delta = self.bit_readout(bit_out["clip_tokens"])
            out["bit_delta"] = bit_delta
            out["bit_clip"] = self._compose(atm_emb, bit_delta, self.bit_res_scale)
        return out

    def apply_stage(self, stage: int) -> None:
        from eeg_brainit.utils.freeze import count_trainable, freeze, unfreeze

        freeze(self.bridge)
        freeze(self.bit)
        freeze(self.bridge_readout)
        freeze(self.bit_readout)
        if self.direct_head is not None:
            freeze(self.direct_head)
        self.bridge_res_scale.requires_grad_(False)
        self.bit_res_scale.requires_grad_(False)
        self.direct_res_scale.requires_grad_(False)

        if stage == 0:
            # Ablation A: direct MLP only.
            if self.direct_head is None:
                raise RuntimeError("stage 0 requires direct_head")
            unfreeze(self.direct_head)
            self.direct_res_scale.requires_grad_(True)
            self.use_direct = True
        elif stage == 1:
            unfreeze(self.bridge)
            unfreeze(self.bridge_readout)
            self.bridge_res_scale.requires_grad_(True)
        elif stage == 2:
            unfreeze(self.bridge)
            unfreeze(self.bridge_readout)
            unfreeze(self.bit)
            unfreeze(self.bit_readout)
            self.bridge_res_scale.requires_grad_(True)
            self.bit_res_scale.requires_grad_(True)
        elif stage == 3:
            unfreeze(self.bit)
            unfreeze(self.bit_readout)
            self.bit_res_scale.requires_grad_(True)
        else:
            raise ValueError(f"NeuroBolt pipeline supports stages 0..3, got {stage}")
        print(
            f"[INFO] NeuroBolt stage {stage}: trainable={count_trainable(self):,} "
            f"residual_mode={self.residual_mode} "
            f"bridge_res={float(self.bridge_res_scale):.4f} "
            f"bit_res={float(self.bit_res_scale):.4f} "
            f"direct_res={float(self.direct_res_scale):.4f}"
        )

    @classmethod
    def from_config(cls, cfg: dict[str, Any], project_root: str | None = None) -> "NeuroBoltBrainITPipeline":
        from pathlib import Path

        from eeg_brainit.models.bit_cross_fusion import BITCrossFusion

        nb_cfg = cfg.get("neurobolt", {})
        bridge = RoiToBitBridge.from_config(nb_cfg.get("bridge", {}))
        bit = BITCrossFusion.from_config(cfg.get("bit", {}))
        clip_dim = int(nb_cfg.get("clip_dim", 1024))
        brain_dim = int(nb_cfg.get("bridge", {}).get("brain_dim", 1024))
        bridge_readout = ClipReadout(in_dim=brain_dim, out_dim=clip_dim, hidden=clip_dim)
        bit_readout = ClipReadout(
            in_dim=int(cfg.get("bit", {}).get("clip_dim", 1664)),
            out_dim=clip_dim,
            hidden=clip_dim,
        )
        direct = None
        if bool(nb_cfg.get("direct", {}).get("enabled", False)) or int(cfg.get("train", {}).get("stage", 1)) == 0:
            direct = DirectRoiClipHead(
                fmri_dim=int(nb_cfg.get("bridge", {}).get("fmri_dim", 512)),
                clip_dim=clip_dim,
                hidden=int(nb_cfg.get("direct", {}).get("hidden", 1024)),
            )
        pipe = cls(
            bridge=bridge,
            bit=bit,
            bridge_readout=bridge_readout,
            bit_readout=bit_readout,
            direct_head=direct,
            use_bit=bool(nb_cfg.get("use_bit", True)),
            use_direct=direct is not None,
            residual_mode=str(nb_cfg.get("residual_mode", "atm_skip")),
        )
        bit_ckpt = cfg.get("bit", {}).get("checkpoint")
        if bit_ckpt:
            p = Path(bit_ckpt)
            if not p.is_absolute() and project_root:
                p = Path(project_root) / p
            if p.is_file():
                bit.load_pretrained_partial(p)
        return pipe
