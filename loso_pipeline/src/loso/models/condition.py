"""Stage-3 conditioning: EEG -> IP-Adapter tokens for a frozen SDXL UNet.

Recipe mirrors CognitionCapturerPro / official IP-Adapter SDXL ViT-H:

  z_inv  --MLP-->  clip_img (1024)  --ImageProj-->  ip_tokens (N, 2048)
  text prompt tokens (empty / caption) are concatenated as the *first* half of
  `encoder_hidden_states`; IP tokens occupy the second half that the IP-Adapter
  attention processors read.

Only the EEG projector (and optional UNet LoRA) are trained.  The SDXL UNet,
VAE, text encoders and the IP-Adapter projection weights start from the official
checkpoint; freezing them keeps the diffusion prior intact and is what lets a
single A100 finish Stage 3 in hours rather than days.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from loso import paths


@dataclass
class ConditionConfig:
    d_inv: int = 512
    clip_dim: int = 1024          # CLIP ViT-H-14 image embedding
    cross_attention_dim: int = 2048  # SDXL UNet cross-attn width
    num_tokens: int = 4           # IP-Adapter SDXL default
    proj_hidden: int = 1024
    dropout: float = 0.0
    lora_rank: int = 16
    lora_alpha: int = 16
    use_lora: bool = True
    ip_scale: float = 1.0
    resolution: int = 512


class EEGImageProj(nn.Module):
    """Map EEG invariant features into CLIP image space, then into IP tokens.

    Two stages are kept separate on purpose:

    * `to_clip` is the *learnable* bridge from the Stage-2 representation into the
      space the official IP-Adapter was trained on; it is what Stage 3 actually
      optimises.
    * `image_proj` is the official IP-Adapter linear that expands a CLIP embedding
      into `num_tokens` cross-attention keys/values.  Loading its pretrained
      weights (and freezing them) reuses the visual prior; training it from
      scratch on EEG alone collapses the adapter into a free MLP with no diffusion
      grounding.
    """

    def __init__(self, cfg: ConditionConfig):
        super().__init__()
        self.cfg = cfg
        self.to_clip = nn.Sequential(
            nn.Linear(cfg.d_inv, cfg.proj_hidden),
            nn.LayerNorm(cfg.proj_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.proj_hidden, cfg.clip_dim),
        )
        self.image_proj = nn.Linear(
            cfg.clip_dim, cfg.cross_attention_dim * cfg.num_tokens, bias=True,
        )
        self.norm = nn.LayerNorm(cfg.cross_attention_dim)

    def forward(self, z_inv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (clip_img L2-normalised, ip_tokens of shape (B, N, D))."""
        clip = F.normalize(self.to_clip(z_inv).float(), dim=-1)
        tokens = self.image_proj(clip).reshape(
            clip.shape[0], self.cfg.num_tokens, self.cfg.cross_attention_dim,
        )
        return clip, self.norm(tokens)

    @torch.no_grad()
    def load_ip_adapter_proj(self, state: dict) -> int:
        """Load `image_proj` from an official IP-Adapter checkpoint.

        The vit-h SDXL file nests weights under `{"image_proj": {proj.*, norm.*},
        "ip_adapter": {...}}`.  Accept nested or flat keys.
        """
        src = state.get("image_proj", state) if isinstance(state, dict) else state
        if not isinstance(src, dict):
            raise TypeError("expected a state-dict-like mapping for image_proj")
        alias = {
            "image_proj.weight": ("proj.weight", "image_proj.proj.weight",
                                  "image_proj.weight"),
            "image_proj.bias": ("proj.bias", "image_proj.proj.bias",
                                "image_proj.bias"),
            "norm.weight": ("norm.weight", "image_proj.norm.weight"),
            "norm.bias": ("norm.bias", "image_proj.norm.bias"),
        }
        own = self.state_dict()
        loaded = 0
        for dst, candidates in alias.items():
            if dst not in own:
                continue
            for key in candidates:
                if key in src and hasattr(src[key], "shape") \
                        and tuple(src[key].shape) == tuple(own[dst].shape):
                    own[dst].copy_(src[key].to(dtype=own[dst].dtype))
                    loaded += 1
                    break
        self.load_state_dict(own, strict=True)
        return loaded


def resolve_ip_adapter_file() -> Path:
    """Locate `ip-adapter_sdxl_vit-h.bin` inside the HF hub cache."""
    hub = paths.HF_HUB / "models--h94--IP-Adapter" / "snapshots"
    if not hub.is_dir():
        raise FileNotFoundError(f"IP-Adapter cache missing under {hub}")
    matches = sorted(hub.glob("*/sdxl_models/ip-adapter_sdxl_vit-h.bin"))
    if not matches:
        raise FileNotFoundError(
            f"no ip-adapter_sdxl_vit-h.bin under {hub}; expected the vit-h SDXL variant"
        )
    return matches[0]


def resolve_ip_adapter_root() -> Path:
    return resolve_ip_adapter_file().parent.parent


def build_sdxl_components(device: str, dtype: torch.dtype):
    """Load frozen SDXL pieces used by Stage 3 training and inference."""
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

    sd = paths.SD_ID
    tokenizer = CLIPTokenizer.from_pretrained(sd, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(sd, subfolder="tokenizer_2")
    text_encoder = CLIPTextModel.from_pretrained(
        sd, subfolder="text_encoder", torch_dtype=dtype,
    ).to(device).eval()
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        sd, subfolder="text_encoder_2", torch_dtype=dtype,
    ).to(device).eval()
    vae = AutoencoderKL.from_pretrained(paths.VAE_ID, torch_dtype=dtype).to(device).eval()
    unet = UNet2DConditionModel.from_pretrained(
        sd, subfolder="unet", torch_dtype=dtype,
    ).to(device)
    scheduler = DDPMScheduler.from_pretrained(sd, subfolder="scheduler")

    for m in (text_encoder, text_encoder_2, vae):
        for p in m.parameters():
            p.requires_grad_(False)
    for p in unet.parameters():
        p.requires_grad_(False)

    return {
        "tokenizer": tokenizer,
        "tokenizer_2": tokenizer_2,
        "text_encoder": text_encoder,
        "text_encoder_2": text_encoder_2,
        "vae": vae,
        "unet": unet,
        "scheduler": scheduler,
        "vae_scale": float(getattr(vae.config, "scaling_factor", 0.13025)),
    }


def attach_lora(unet: nn.Module, rank: int = 16, alpha: int = 16) -> nn.Module:
    """LoRA on UNet attention projections; everything else stays frozen."""
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(
        r=rank, lora_alpha=alpha, init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    return get_peft_model(unet, cfg)


@torch.no_grad()
def encode_prompts(components: dict, prompts: list[str], device: str,
                   dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """SDXL dual-encoder prompt encoding -> (prompt_embeds, pooled_embeds)."""
    t1, t2 = components["tokenizer"], components["tokenizer_2"]
    tok = t1(
        prompts, padding="max_length", max_length=tok_max(t1),
        truncation=True, return_tensors="pt",
    )
    tok2 = t2(
        prompts, padding="max_length", max_length=tok_max(t2),
        truncation=True, return_tensors="pt",
    )
    e1 = components["text_encoder"](
        tok.input_ids.to(device), output_hidden_states=True,
    ).hidden_states[-2]
    out2 = components["text_encoder_2"](
        tok2.input_ids.to(device), output_hidden_states=True,
    )
    e2 = out2.hidden_states[-2]
    pooled = out2[0]
    prompt_embeds = torch.cat([e1, e2], dim=-1).to(dtype)
    return prompt_embeds, pooled.to(dtype)


def tok_max(tokenizer) -> int:
    return int(getattr(tokenizer, "model_max_length", 77))


def sdxl_time_ids(batch: int, resolution: int, device, dtype) -> torch.Tensor:
    """Original-size / crop / target-size conditioning required by SDXL."""
    # (orig_h, orig_w, crop_y, crop_x, target_h, target_w)
    row = torch.tensor(
        [resolution, resolution, 0, 0, resolution, resolution],
        device=device, dtype=dtype,
    )
    return row.unsqueeze(0).expand(batch, -1)


def inject_ip_tokens(prompt_embeds: torch.Tensor, ip_tokens: torch.Tensor) -> torch.Tensor:
    """Concatenate text tokens and IP tokens along the sequence axis.

    Diffusers' IP-Adapter attention processors split `encoder_hidden_states` into
    `[:, :text_len]` (text) and `[:, text_len:]` (image).  Concatenation is the
    contract those processors expect; replacing the text half would erase the
    prompt prior entirely.
    """
    return torch.cat([prompt_embeds, ip_tokens.to(prompt_embeds.dtype)], dim=1)
