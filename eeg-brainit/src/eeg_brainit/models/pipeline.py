"""End-to-end EEG -> Image pipeline with BIT Cross-Transformer fusion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from eeg_brainit.models.bit_cross_fusion import BITCrossFusion
from eeg_brainit.models.eeg_encoder import MDTFCAEEncoder
from eeg_brainit.models.eeg_projector import EEGTokenProjector
from eeg_brainit.models.virtual_fmri import VirtualFMRIBranch
from eeg_brainit.utils.freeze import freeze, unfreeze, count_trainable


class DirectCLIPHead(nn.Module):
    """NICE/ATM-style projector: EEG latent -> OpenCLIP image space."""

    def __init__(self, in_dim: int = 512, out_dim: int = 768, hidden: int = 1024) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z_eeg: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(z_eeg), dim=-1)


class ClipAlignHead(nn.Module):
    """Attention-pool BIT CLIP tokens and MLP-project to OpenCLIP space."""

    def __init__(self, in_dim: int = 1664, out_dim: int = 768, hidden_mult: int = 2) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.attn = nn.Linear(in_dim, 1)
        hidden = int(in_dim * hidden_mult)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, clip_tokens: torch.Tensor) -> torch.Tensor:
        x = self.norm(clip_tokens)
        w = torch.softmax(self.attn(x).squeeze(-1), dim=-1)
        pooled = torch.einsum("bt,btd->bd", w, x)
        return F.normalize(self.mlp(pooled), dim=-1)


class EEGBrainITPipeline(nn.Module):
    """
    Data flow:
      spectrogram
        -> MD-TF-CAE  -> z_eeg / eeg_tokens / feat_map
             |                 |
             |                 +-> EEG projector -> eeg_kv_tokens
             |                 +-> optional DirectCLIPHead -> clip_emb (NICE/ATM path)
             v
          virtual fMRI branch -> brain_tokens (128)
             |
             +-> BIT Cross-Transformer(KV = brain + eeg) -> clip_tokens, vgg_features
             +-> optional ClipAlignHead -> clip_emb (BIT path)
    """

    def __init__(
        self,
        encoder: MDTFCAEEncoder,
        projector: EEGTokenProjector,
        virtual_fmri: VirtualFMRIBranch,
        bit: BITCrossFusion,
        clip_align_head: ClipAlignHead | None = None,
        direct_clip_head: DirectCLIPHead | None = None,
        prefer_direct_clip: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.virtual_fmri = virtual_fmri
        self.bit = bit
        self.clip_align_head = clip_align_head
        self.direct_clip_head = direct_clip_head
        self.prefer_direct_clip = prefer_direct_clip

    def forward(self, spectrogram: torch.Tensor) -> dict[str, torch.Tensor]:
        enc = self.encoder(spectrogram)
        out: dict[str, torch.Tensor] = {**enc}

        # Lightweight NICE/ATM-style path (preferred for retrieval training).
        if self.direct_clip_head is not None and self.prefer_direct_clip:
            out["clip_emb"] = self.direct_clip_head(enc["z_eeg"])
            # Skip heavy BIT path during direct-clip training for speed/stability.
            return out

        virt = self.virtual_fmri(enc["feat_map"])
        eeg_kv = self.projector(enc["z_eeg"], enc["eeg_tokens"])
        bit_out = self.bit(virt["brain_tokens"], eeg_kv)
        out.update(virt)
        out.update(bit_out)
        out["eeg_kv_tokens"] = eeg_kv
        if self.clip_align_head is not None:
            out["clip_emb"] = self.clip_align_head(bit_out["clip_tokens"])
        elif self.direct_clip_head is not None:
            out["clip_emb"] = self.direct_clip_head(enc["z_eeg"])
        return out

    def apply_stage(self, stage: int) -> None:
        """Fine-tuning schedule. stage=0: NICE/ATM-style direct CLIP alignment."""
        freeze(self.encoder)
        freeze(self.virtual_fmri)
        freeze(self.bit)
        freeze(self.projector)
        if self.clip_align_head is not None:
            freeze(self.clip_align_head)
        if self.direct_clip_head is not None:
            freeze(self.direct_clip_head)

        if stage == 0:
            # Subject-dependent direct EEG->CLIP (literature default for retrieval).
            unfreeze(self.encoder)
            if self.direct_clip_head is not None:
                unfreeze(self.direct_clip_head)
        elif stage == 1:
            unfreeze(self.projector)
            if self.clip_align_head is not None:
                unfreeze(self.clip_align_head)
            if self.direct_clip_head is not None:
                unfreeze(self.direct_clip_head)
        elif stage == 2:
            unfreeze(self.projector)
            unfreeze(self.bit)
            if self.clip_align_head is not None:
                unfreeze(self.clip_align_head)
            if self.direct_clip_head is not None:
                unfreeze(self.direct_clip_head)
        elif stage == 3:
            unfreeze(self.projector)
            unfreeze(self.bit)
            unfreeze(self.virtual_fmri)
            if self.clip_align_head is not None:
                unfreeze(self.clip_align_head)
            if self.direct_clip_head is not None:
                unfreeze(self.direct_clip_head)
        elif stage == 4:
            unfreeze(self.encoder)
            unfreeze(self.projector)
            unfreeze(self.virtual_fmri)
            unfreeze(self.bit)
            if self.clip_align_head is not None:
                unfreeze(self.clip_align_head)
            if self.direct_clip_head is not None:
                unfreeze(self.direct_clip_head)
        else:
            raise ValueError(f"Unknown stage {stage}; expected 0..4")

        print(f"[INFO] Stage {stage}: trainable params = {count_trainable(self):,}")

    @classmethod
    def from_config(cls, cfg: dict[str, Any], project_root: str | Path | None = None) -> "EEGBrainITPipeline":
        root = str(project_root) if project_root is not None else None
        encoder = MDTFCAEEncoder.from_config(cfg.get("encoder", {}))
        projector = EEGTokenProjector.from_config(cfg.get("projector", {}))
        virtual_fmri = VirtualFMRIBranch.from_config(cfg.get("virtual_fmri", {}), project_root=root)
        bit = BITCrossFusion.from_config(cfg.get("bit", {}))

        align_cfg = cfg.get("clip_align", {})
        clip_align_head = None
        if bool(align_cfg.get("enabled", False)):
            clip_align_head = ClipAlignHead(
                in_dim=int(align_cfg.get("in_dim", cfg.get("bit", {}).get("clip_dim", 1664))),
                out_dim=int(align_cfg.get("out_dim", 768)),
            )

        direct_cfg = cfg.get("direct_clip", {})
        direct_clip_head = None
        if bool(direct_cfg.get("enabled", False)):
            direct_clip_head = DirectCLIPHead(
                in_dim=int(direct_cfg.get("in_dim", cfg.get("encoder", {}).get("d_eeg", 512))),
                out_dim=int(direct_cfg.get("out_dim", 768)),
                hidden=int(direct_cfg.get("hidden", 1024)),
            )

        pipe = cls(
            encoder,
            projector,
            virtual_fmri,
            bit,
            clip_align_head=clip_align_head,
            direct_clip_head=direct_clip_head,
            prefer_direct_clip=bool(direct_cfg.get("enabled", False)),
        )

        enc_ckpt = cfg.get("encoder", {}).get("checkpoint")
        if enc_ckpt and Path(enc_ckpt).is_file():
            encoder.load_pretrained(enc_ckpt, strict=False)

        bit_ckpt = cfg.get("bit", {}).get("checkpoint")
        if bit_ckpt and Path(bit_ckpt).is_file() and not bool(direct_cfg.get("enabled", False)):
            bit.load_pretrained_partial(bit_ckpt)

        return pipe


# Backward-compatible alias
CLIPAlignmentHead = ClipAlignHead
