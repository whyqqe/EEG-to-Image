"""LaBraM foundation backbone for ARIA — fast fine-tune path.

Speed changes vs v1 (3000-pad):
  • Crop EEG to patch_size=200 (no zero-pad to 3000) → ~15× fewer tokens
  • Keep 128-ch bank mapping (pretrained pos embed); only THINGS 63 are filled
  • Load braindecode/labram-pretrained with temporal_embedding slice adapt

REVE-base remains gated; LaBraM is the public FM used here.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import torch
import torch.nn as nn

from eeg_brainit.models.aria import l2_normalize
from eeg_brainit.models.physics_prior_lora import THINGS_EEG2_CHANNELS
from eeg_brainit.models.subject_align import MLPProjector

LABRAM_REPO = "braindecode/labram-pretrained"
LABRAM_N_TIMES = 200  # one patch; was 3000 (slow)
LABRAM_PATCH = 200


def _ensure_hf_env() -> None:
    cache = Path("/project/peilab/why/cache/eeg-brainit/hf")
    os.environ.setdefault("HF_HOME", str(cache))
    os.environ.setdefault("HF_HUB_CACHE", str(cache / "hub"))


def _load_ckpt_state_dict() -> dict[str, torch.Tensor]:
    from huggingface_hub import hf_hub_download

    # prefer safetensors
    try:
        path = hf_hub_download(LABRAM_REPO, "model.safetensors")
        from safetensors.torch import load_file

        return load_file(path)
    except Exception:
        path = hf_hub_download(LABRAM_REPO, "pytorch_model.bin")
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(obj, dict) and "state_dict" in obj:
            return obj["state_dict"]
        return obj


def load_labram_pretrained(device: torch.device | str = "cpu"):
    """Build LaBraM with n_times=200 and load adapted pretrained weights."""
    _ensure_hf_env()
    from huggingface_hub import hf_hub_download
    from braindecode.models import Labram

    cfg = json.loads(Path(hf_hub_download(LABRAM_REPO, "config.json")).read_text())
    chs_info = cfg["chs_info"]
    bank = [c["ch_name"].upper() for c in chs_info]
    things = [c.upper() for c in THINGS_EEG2_CHANNELS]
    missing = [c for c in things if c not in bank]
    if missing:
        raise RuntimeError(f"THINGS channels missing from LaBraM bank: {missing}")
    ch_index = torch.tensor([bank.index(c) for c in things], dtype=torch.long)

    model = Labram(
        n_outputs=0,
        n_chans=len(chs_info),
        n_times=LABRAM_N_TIMES,
        chs_info=chs_info,
        patch_size=LABRAM_PATCH,
    )
    ckpt = _load_ckpt_state_dict()
    model_sd = model.state_dict()
    adapted: dict[str, torch.Tensor] = {}
    skipped = []
    for k, v in ckpt.items():
        if k not in model_sd:
            skipped.append(k)
            continue
        tgt = model_sd[k]
        if tuple(v.shape) == tuple(tgt.shape):
            adapted[k] = v
        elif k == "temporal_embedding" and v.ndim == 3 and v.shape[-1] == tgt.shape[-1]:
            # ckpt often [1,16,D]; fast model [1,1,D] — take leading slices
            n_t = tgt.shape[1]
            adapted[k] = v[:, :n_t, :].contiguous()
        elif k == "position_embedding" and v.ndim == 3 and v.shape[-1] == tgt.shape[-1]:
            n_p = min(v.shape[1], tgt.shape[1])
            out = tgt.clone()
            out[:, :n_p, :] = v[:, :n_p, :]
            adapted[k] = out
        else:
            skipped.append(f"{k}:{tuple(v.shape)}!={tuple(tgt.shape)}")
    missing_keys, unexpected = model.load_state_dict(adapted, strict=False)
    print(
        f"[LaBraM] loaded={len(adapted)} missing={len(missing_keys)} "
        f"shape_skip={len(skipped)} n_times={LABRAM_N_TIMES}",
        flush=True,
    )
    return model.to(device), ch_index


class LabramARIAEncoder(nn.Module):
    """LaBraM FM → CLIP projector for absolute retrieval fine-tuning."""

    def __init__(
        self,
        n_subjects: int = 10,
        clip_dim: int = 1024,
        dropout: float = 0.25,
        unfreeze_last_n_blocks: int = 2,
        train_patch_embed: bool = False,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.labram, ch_index = load_labram_pretrained(device=device)
        self.register_buffer("ch_index", ch_index, persistent=True)
        self.n_times = LABRAM_N_TIMES
        self.embed_dim = int(getattr(self.labram, "embed_dim", 200) or 200)
        self.projector = MLPProjector(self.embed_dim, clip_dim=clip_dim, dropout=dropout)
        self.probe_id = nn.Sequential(
            nn.LayerNorm(clip_dim),
            nn.Linear(clip_dim, clip_dim // 2),
            nn.GELU(),
            nn.Linear(clip_dim // 2, n_subjects),
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))
        self.clip_dim = clip_dim
        self.n_subjects = n_subjects
        self.register_buffer("anchors", torch.zeros(1, clip_dim), persistent=True)
        self.register_buffer("anchors_ready", torch.zeros((), dtype=torch.bool))
        self.set_finetune_mode(unfreeze_last_n_blocks, train_patch_embed)

    def set_anchors(self, anchors: torch.Tensor) -> None:
        self.anchors = l2_normalize(anchors.float().detach())
        self.anchors_ready.fill_(True)

    def set_finetune_mode(self, unfreeze_last_n_blocks: int = 2, train_patch_embed: bool = False) -> None:
        for p in self.labram.parameters():
            p.requires_grad = False
        blocks = getattr(self.labram, "blocks", None)
        if blocks is not None and hasattr(blocks, "__len__"):
            n = len(blocks)
            start = max(0, n - max(0, unfreeze_last_n_blocks))
            for i in range(start, n):
                for p in blocks[i].parameters():
                    p.requires_grad = True
        if train_patch_embed and hasattr(self.labram, "patch_embed"):
            for p in self.labram.patch_embed.parameters():
                p.requires_grad = True
        for p in self.projector.parameters():
            p.requires_grad = True
        for p in self.probe_id.parameters():
            p.requires_grad = True
        self.logit_scale.requires_grad = True

    def expand_eeg(self, x: torch.Tensor) -> torch.Tensor:
        """(B,63,T) → (B, n_bank, n_times); linear resample full window to LaBraM patch."""
        b, c, t = x.shape
        if c != 63:
            raise ValueError(f"expected 63 channels, got {c}")
        if t != self.n_times:
            x = torch.nn.functional.interpolate(x, size=self.n_times, mode="linear", align_corners=False)
        n_bank = int(getattr(self.labram, "n_chans", 128) or 128)
        full = x.new_zeros(b, n_bank, self.n_times)
        full[:, self.ch_index, :] = x[:, :, : self.n_times]
        return full

    def encode_hidden(self, x: torch.Tensor) -> torch.Tensor:
        return self.labram(self.expand_eeg(x.float()))

    def encode_absolute(self, x: torch.Tensor, normalize: bool = True) -> dict[str, torch.Tensor]:
        h = self.encode_hidden(x)
        raw = self.projector(h)
        emb = l2_normalize(raw) if normalize else raw
        return {"latent": h, "clip_raw": raw, "clip_emb": l2_normalize(raw), "clip_out": emb}

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.encode_absolute(x, normalize=True)
        if bool(self.anchors_ready):
            from eeg_brainit.models.aria import relative_from_anchors

            out["rel_emb"] = relative_from_anchors(out["clip_emb"], self.anchors)
        out["logit_scale"] = self.logit_scale.exp()
        return out

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_groups(self, lr_backbone: float, lr_head: float, weight_decay: float):
        bb, head = [], []
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (bb if n.startswith("labram.") else head).append(p)
        groups = []
        if bb:
            groups.append({"params": bb, "lr": lr_backbone, "weight_decay": weight_decay})
        if head:
            groups.append({"params": head, "lr": lr_head, "weight_decay": weight_decay})
        return groups
