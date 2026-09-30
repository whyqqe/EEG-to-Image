"""LaBraM temporal encoder (foundation backbone) for NOD fine-tuning.

Vendored from NeuroBOLT/LaBraM TemporalConv + Transformer blocks only —
no MSS spectral head / linear_attention_transformer dependency.

Input:  EEG (B, C, T)  — cropped/padded to patch_size=200 (800 ms @ 250 Hz)
Output: (B, embed_dim) CLS features
"""

from __future__ import annotations

import math
from functools import partial
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = x.new_empty(shape).bernoulli_(keep)
    return x.div(keep) * mask


def trunc_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0) -> torch.Tensor:
    with torch.no_grad():
        tensor.normal_(mean=mean, std=std)
        tensor.clamp_(-2 * std, 2 * std)
    return tensor



# 10-20 montage index table used by LaBraM / NeuroBOLT channel embeddings.
STANDARD_1020 = [
    "FP1", "FPZ", "FP2",
    "AF9", "AF7", "AF5", "AF3", "AF1", "AFZ", "AF2", "AF4", "AF6", "AF8", "AF10",
    "F9", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8", "F10",
    "FT9", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "FT10",
    "T9", "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8", "T10",
    "TP9", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "TP10",
    "P9", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "P10",
    "PO9", "PO7", "PO5", "PO3", "PO1", "POZ", "PO2", "PO4", "PO6", "PO8", "PO10",
    "O1", "OZ", "O2", "O9", "CB1", "CB2",
    "IZ", "O10", "T3", "T5", "T4", "T6", "M1", "M2", "A1", "A2",
]


def normalize_ch_name(name: str) -> str:
    return str(name).strip().upper()


def get_input_chans(ch_names: Sequence[str]) -> list[int]:
    """Map channel names → LaBraM pos_embed indices (0 = CLS)."""
    idxs = [0]
    missing = []
    for ch in ch_names:
        n = normalize_ch_name(ch)
        if n not in STANDARD_1020:
            missing.append(ch)
            continue
        idxs.append(STANDARD_1020.index(n) + 1)
    if missing:
        raise ValueError(f"channels not in 10-20 table: {missing}")
    return idxs


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int | None = None, drop: float = 0.0) -> None:
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 10,
        qkv_bias: bool = False,
        qk_norm: type[nn.Module] | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        all_head_dim = head_dim * num_heads
        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(all_head_dim)) if qkv_bias else None
        self.v_bias = nn.Parameter(torch.zeros(all_head_dim)) if qkv_bias else None
        self.q_norm = qk_norm(head_dim) if qk_norm is not None else None
        self.k_norm = qk_norm(head_dim) if qk_norm is not None else None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias)
            )
        qkv = F.linear(x, self.qkv.weight, qkv_bias)
        qkv = qkv.reshape(b, n, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.q_norm is not None:
            q = self.q_norm(q).type_as(v)
        if self.k_norm is not None:
            k = self.k_norm(k).type_as(v)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b, n, -1)
        x = self.proj(x)
        return self.proj_drop(x)


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: type[nn.Module] | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path_rate: float = 0.0,
        init_values: float = 0.1,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm, attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)
        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones(dim))
            self.gamma_2 = nn.Parameter(init_values * torch.ones(dim))
        else:
            self.gamma_1 = self.gamma_2 = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class TemporalConv(nn.Module):
    """LaBraM EEG → patch embedding (matches labram-base.pth)."""

    def __init__(self, in_chans: int = 1, out_chans: int = 8) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=(1, 15), stride=(1, 8), padding=(0, 7))
        self.gelu1 = nn.GELU()
        self.norm1 = nn.GroupNorm(4, out_chans)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.gelu2 = nn.GELU()
        self.norm2 = nn.GroupNorm(4, out_chans)
        self.conv3 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.norm3 = nn.GroupNorm(4, out_chans)
        self.gelu3 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, A, T)
        x = rearrange(x, "B N A T -> B (N A) T")
        x = x.unsqueeze(1)
        x = self.gelu1(self.norm1(self.conv1(x)))
        x = self.gelu2(self.norm2(self.conv2(x)))
        x = self.gelu3(self.norm3(self.conv3(x)))
        return rearrange(x, "B C NA T -> B NA (T C)")


class LaBraMEncoder(nn.Module):
    """Pretrained LaBraM temporal backbone → CLS feature (embed_dim=200)."""

    def __init__(
        self,
        ch_names: Sequence[str],
        embed_dim: int = 200,
        depth: int = 12,
        num_heads: int = 10,
        patch_size: int = 200,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        init_values: float = 0.1,
        pool_mode: str = "mean",
    ) -> None:
        super().__init__()
        if pool_mode not in ("cls", "mean"):
            raise ValueError(f"pool_mode must be cls|mean, got {pool_mode}")
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.pool_mode = pool_mode
        self.input_chans = get_input_chans(list(ch_names))
        self.ch_names = [normalize_ch_name(c) for c in ch_names]

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        qk_norm = partial(nn.LayerNorm, eps=1e-6)
        self.patch_embed = TemporalConv(out_chans=8)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 128 + 1, embed_dim))
        self.time_embed = nn.Parameter(torch.zeros(1, 16, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=4.0,
                    qkv_bias=False,
                    qk_norm=qk_norm,
                    drop=drop_rate,
                    drop_path_rate=dpr[i],
                    init_values=init_values,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        # labram-base.pth uses LayerNorm on tokens + CLS readout (not mean-pool fc_norm).
        self.norm = norm_layer(embed_dim)
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.time_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
        for layer_id, layer in enumerate(self.blocks):
            layer.attn.proj.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))
            layer.mlp.fc2.weight.data.div_(math.sqrt(2.0 * (layer_id + 1)))

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def load_pretrained(self, path: str | Path, strict: bool = False) -> dict[str, int]:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        cleaned = {}
        for k, v in sd.items():
            if k.startswith("student."):
                cleaned[k[len("student.") :]] = v
            elif not k.startswith(("teacher.", "logit_scale")):
                cleaned[k] = v
        model_sd = self.state_dict()
        loadable = {
            k: v for k, v in cleaned.items() if k in model_sd and tuple(model_sd[k].shape) == tuple(v.shape)
        }
        missing, unexpected = self.load_state_dict(loadable, strict=False)
        return {
            "loaded": len(loadable),
            "ckpt_student": sum(1 for k in sd if str(k).startswith("student.")),
            "missing": len(missing),
            "unexpected_skipped": len(cleaned) - len(loadable),
        }

    def set_finetune_mode(self, unfreeze_last_n_blocks: int = 2, train_patch_embed: bool = False) -> None:
        """Freeze backbone; optionally unfreeze last N blocks (+ norms / CLS)."""
        for p in self.parameters():
            p.requires_grad = False
        if train_patch_embed:
            for p in self.patch_embed.parameters():
                p.requires_grad = True
        n = len(self.blocks)
        start = max(0, n - max(0, unfreeze_last_n_blocks))
        for i in range(start, n):
            for p in self.blocks[i].parameters():
                p.requires_grad = True
        if unfreeze_last_n_blocks > 0:
            self.cls_token.requires_grad = True
            self.pos_embed.requires_grad = True
            self.time_embed.requires_grad = True
            for p in self.norm.parameters():
                p.requires_grad = True

    def _prepare_patches(self, eeg: torch.Tensor) -> torch.Tensor:
        """(B, C, T) → (B, C, 1, patch_size)."""
        x = eeg.float()
        if x.ndim != 3:
            raise ValueError(f"Expected (B,C,T), got {tuple(x.shape)}")
        t = x.shape[-1]
        if t >= self.patch_size:
            x = x[..., : self.patch_size]
        else:
            x = F.pad(x, (0, self.patch_size - t))
        return x.unsqueeze(2)  # (B, C, 1, P)

    def forward_features(self, eeg: torch.Tensor) -> torch.Tensor:
        x = self._prepare_patches(eeg)
        batch_size, n, a, t = x.shape
        input_time_window = a if t == self.patch_size else t
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        pos_embed_used = self.pos_embed[:, self.input_chans]
        pos_embed = (
            pos_embed_used[:, 1:, :]
            .unsqueeze(2)
            .expand(batch_size, -1, input_time_window, -1)
            .flatten(1, 2)
        )
        pos_embed = torch.cat((pos_embed_used[:, 0:1, :].expand(batch_size, -1, -1), pos_embed), dim=1)
        x = x + pos_embed

        nc = n if t == self.patch_size else a
        time_embed = (
            self.time_embed[:, 0:input_time_window, :]
            .unsqueeze(1)
            .expand(batch_size, nc, -1, -1)
            .flatten(1, 2)
        )
        x[:, 1:, :] = x[:, 1:, :] + time_embed

        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        if self.pool_mode == "mean":
            return x[:, 1:].mean(dim=1)
        return x[:, 0]

    def forward(self, eeg: torch.Tensor) -> torch.Tensor:
        return self.forward_features(eeg)
