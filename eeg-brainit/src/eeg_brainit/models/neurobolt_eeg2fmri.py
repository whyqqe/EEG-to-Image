"""NeuroBOLT backbone for NOD Phase-1 EEG→fMRI fine-tuning.

Loads ``glb.pth`` (NeurIPS'24 EEG→fMRI foundation), adapts MSS to short
visual epochs (T=200), and replaces the 1-ROI head with a NOD ROI head.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from eeg_brainit.models.labram_encoder import get_input_chans, normalize_ch_name

_NEUROBOLT_ROOT = Path("/project/peilab/why/eeg-to-image/third_party/NeuroBOLT")
if _NEUROBOLT_ROOT.is_dir() and str(_NEUROBOLT_ROOT) not in sys.path:
    sys.path.insert(0, str(_NEUROBOLT_ROOT))


class NeuroBoltEEG2fMRI(nn.Module):
    """NeuroBOLT init → NOD fMRI ROI regression (+ optional CLIP aux on fused feat)."""

    def __init__(
        self,
        ch_names: Sequence[str],
        num_rois: int = 200,
        clip_dim: int = 1024,
        hidden: int = 1024,
        dropout: float = 0.15,
        use_clip_head: bool = False,
        glb_ckpt: str | Path = "checkpoints/neurobolt/glb.pth",
        patch_size: int = 200,
        win_level: int = 1,
        unfreeze_last_n_blocks: int = 2,
        train_mss: bool = False,
        use_mss: bool = False,
        train_patch_embed: bool = False,
        heads_only: bool = False,
        head_depth: int = 1,
    ) -> None:
        super().__init__()
        # Avoid Brain-IT's top-level ``models`` package shadowing NeuroBOLT.
        nb_root = str(_NEUROBOLT_ROOT)
        sys.path = [p for p in sys.path if "brainit-fmri" not in p.replace("\\", "/")]
        while nb_root in sys.path:
            sys.path.remove(nb_root)
        sys.path.insert(0, nb_root)
        for key in list(sys.modules):
            if key == "models" or key.startswith("models."):
                mod = sys.modules.get(key)
                origin = getattr(mod, "__file__", "") or ""
                if "NeuroBOLT" not in origin.replace("\\", "/"):
                    del sys.modules[key]
        from models.model import neurobolt_default  # type: ignore

        self.patch_size = int(patch_size)
        self.ch_names = [normalize_ch_name(c) for c in ch_names]
        self.input_chans = get_input_chans(self.ch_names)
        self.num_channels = len(self.ch_names)
        self.use_clip_head = bool(use_clip_head)
        self.use_mss = bool(use_mss)
        self._unfreeze_last_n_blocks = int(unfreeze_last_n_blocks)
        self._train_mss = bool(train_mss) and self.use_mss
        self._train_patch_embed = bool(train_patch_embed)

        self.backbone = neurobolt_default(
            EEG_channel=self.num_channels,
            EEG_length=self.patch_size,
            num_roi=num_rois,
            win_level=int(win_level),
            init_values=0.1,
        )
        # Drop linear head; use compact MLP on TS(+optional MSS) features.
        self.backbone.head = nn.Identity()
        self.embed_dim = int(self.backbone.embed_dim)

        stats = self.load_glb(glb_ckpt)
        print(f"[INFO] NeuroBOLT glb load {stats} use_mss={self.use_mss}")

        layers: list[nn.Module] = [nn.LayerNorm(self.embed_dim), nn.Linear(self.embed_dim, hidden), nn.GELU(), nn.Dropout(dropout)]
        if int(head_depth) >= 2:
            layers.extend([nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(hidden, num_rois))
        self.fmri_head = nn.Sequential(*layers)
        self.clip_head = None
        if self.use_clip_head:
            self.clip_head = nn.Sequential(
                nn.LayerNorm(self.embed_dim),
                nn.Linear(self.embed_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, clip_dim),
            )

        if heads_only:
            self.set_finetune_mode(unfreeze_last_n_blocks=0, train_mss=False, train_patch_embed=False)
        else:
            self.set_finetune_mode(
                unfreeze_last_n_blocks=self._unfreeze_last_n_blocks,
                train_mss=self._train_mss,
                train_patch_embed=self._train_patch_embed,
            )

    def load_glb(self, path: str | Path) -> dict[str, Any]:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model_sd = self.backbone.state_dict()
        loadable = {
            k: v for k, v in sd.items() if k in model_sd and tuple(model_sd[k].shape) == tuple(v.shape)
        }
        missing, unexpected = self.backbone.load_state_dict(loadable, strict=False)
        # Partial copy of MSS channel tokens (glb has 26; NOD may have 62).
        if "mss_module.channel_tokens.weight" in sd:
            src = sd["mss_module.channel_tokens.weight"]
            dst = self.backbone.mss_module.channel_tokens.weight
            n = min(src.shape[0], dst.shape[0])
            with torch.no_grad():
                dst[:n].copy_(src[:n])
        return {
            "loaded": len(loadable),
            "ckpt": len(sd),
            "missing": len(missing),
            "shape_skipped": len(sd) - len(loadable),
            "channel_tokens_copied": int(
                min(sd["mss_module.channel_tokens.weight"].shape[0], self.num_channels)
                if "mss_module.channel_tokens.weight" in sd
                else 0
            ),
        }

    def set_finetune_mode(
        self,
        unfreeze_last_n_blocks: int = 4,
        train_mss: bool = True,
        train_patch_embed: bool = False,
    ) -> None:
        for p in self.backbone.parameters():
            if p.is_floating_point() or p.is_complex():
                p.requires_grad = False
        n = len(self.backbone.blocks)
        start = max(0, n - max(0, unfreeze_last_n_blocks))
        for i in range(start, n):
            for p in self.backbone.blocks[i].parameters():
                if p.is_floating_point() or p.is_complex():
                    p.requires_grad = True
        if unfreeze_last_n_blocks > 0:
            self.backbone.cls_token.requires_grad = True
            if self.backbone.pos_embed is not None:
                self.backbone.pos_embed.requires_grad = True
            if self.backbone.time_embed is not None:
                self.backbone.time_embed.requires_grad = True
            if self.backbone.fc_norm is not None:
                for p in self.backbone.fc_norm.parameters():
                    if p.is_floating_point() or p.is_complex():
                        p.requires_grad = True
        if train_patch_embed:
            for p in self.backbone.patch_embed.parameters():
                if p.is_floating_point() or p.is_complex():
                    p.requires_grad = True
        if train_mss:
            for p in self.backbone.mss_module.parameters():
                if p.is_floating_point() or p.is_complex():
                    p.requires_grad = True
        for p in self.fmri_head.parameters():
            p.requires_grad = True
        if self.clip_head is not None:
            for p in self.clip_head.parameters():
                p.requires_grad = True

    def unfreeze_backbone(self) -> None:
        self.set_finetune_mode(
            unfreeze_last_n_blocks=self._unfreeze_last_n_blocks,
            train_mss=self._train_mss,
            train_patch_embed=self._train_patch_embed,
        )

    def _prepare(self, eeg: torch.Tensor) -> torch.Tensor:
        x = eeg.float()
        if x.ndim != 3:
            raise ValueError(f"Expected (B,C,T), got {tuple(x.shape)}")
        if x.shape[1] != self.num_channels:
            raise ValueError(f"Expected {self.num_channels} channels, got {x.shape[1]}")
        t = x.shape[-1]
        if t >= self.patch_size:
            x = x[..., : self.patch_size]
        else:
            x = F.pad(x, (0, self.patch_size - t))
        return x.unsqueeze(2)  # (B, C, 1, P)

    def forward_features(self, eeg: torch.Tensor) -> torch.Tensor:
        from einops import rearrange

        x = self._prepare(eeg)
        x_tmp = self.backbone.forward_ts_features(x, input_chans=self.input_chans)
        if not self.use_mss:
            return x_tmp
        x_mss = self.backbone.mss_module(rearrange(x, "B N A T -> B N (A T)"), input_chans=None)
        return self.backbone.head_act(x_mss + x_tmp)

    def forward(self, eeg: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.forward_features(eeg)
        fmri = self.fmri_head(feat)
        out: dict[str, torch.Tensor] = {"feat": feat, "fmri_pred": fmri, "fmri_roi": fmri}
        if self.clip_head is not None:
            out["clip_pred"] = self.clip_head(feat)
        return out

    def trainable_parameter_groups(
        self, backbone_lr: float, head_lr: float, weight_decay: float
    ) -> list[dict[str, Any]]:
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
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
