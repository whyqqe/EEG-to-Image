"""POP-Former with Cross-Brain Orthogonal Attention (CBOA).

Mechanism
---------
1) Dual-stream tokens: periodic (stimulus-locked) vs aperiodic (identity-rich).
2) CBOA: single softmax over [self KV ⊕ cross-brain memory KV], where memory
   keys/values are linearly orthogonalized w.r.t. subject identity axes.
3) Aperiodic stream never enters CBOA / CLIP path; only drives a private adapter.

This is intentionally larger than prior ATM-style probes in this repo.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class StemEmbed(nn.Module):
    """Raw EEG (B,C,T) → channel tokens (B, C, D)."""

    def __init__(self, n_channels: int = 63, seq_len: int = 250, d_model: int = 512, dropout: float = 0.1):
        super().__init__()
        self.n_channels = n_channels
        self.d_model = d_model
        self.ch_proj = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.ch_mix = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) → tokens (B, C, D)
        tok = self.ch_proj(x.float())
        return self.ch_mix(tok)


class DualStreamSplit(nn.Module):
    """Soft split into periodic / aperiodic streams (learned gates)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.per = nn.Linear(d_model, d_model)
        self.ap = nn.Linear(d_model, d_model)

    def forward(self, tok: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        g = self.gate(tok)
        t_per = self.per(tok * g)
        t_ap = self.ap(tok * (1.0 - g))
        return t_per, t_ap


def orthogonalize(x: torch.Tensor, basis: torch.Tensor | None, eps: float = 1e-5) -> torch.Tensor:
    """Project out columns of basis (D, K) from x (..., D)."""
    if basis is None or basis.numel() == 0:
        return x
    # basis: (D, K), orthonormal preferred
    u = basis
    # (..., D) @ (D, K) → (..., K) @ (K, D)
    coef = x @ u
    return x - coef @ u.T


class CBOAAttention(nn.Module):
    """Cross-Brain Orthogonal Attention: joint softmax over self + memory KV."""

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.k_mem = nn.Linear(d_model, d_model)
        self.v_mem = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.scale = self.d_head**-0.5

    def _shape(self, t: torch.Tensor, b: int) -> torch.Tensor:
        # (B, N, D) → (B, H, N, Dh)
        return t.view(b, -1, self.n_heads, self.d_head).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        mem: torch.Tensor | None = None,
        id_basis: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x: (B, N, D) periodic tokens
        mem: (B, M, D) or (M, D) cross-brain memory tokens (already orth preferred)
        id_basis: (D, K) subject identity axes to project out of memory
        """
        b, n, _ = x.shape
        q = self._shape(self.q_proj(x), b)
        k_s = self._shape(self.k_proj(x), b)
        v_s = self._shape(self.v_proj(x), b)

        if mem is None or mem.numel() == 0:
            attn = torch.matmul(q, k_s.transpose(-2, -1)) * self.scale
            attn = self.drop(attn.softmax(dim=-1))
            out = torch.matmul(attn, v_s)
        else:
            if mem.dim() == 2:
                mem = mem.unsqueeze(0).expand(b, -1, -1)
            mem = orthogonalize(mem, id_basis)
            k_m = self._shape(self.k_mem(mem), b)
            v_m = self._shape(self.v_mem(mem), b)
            k = torch.cat([k_s, k_m], dim=2)  # (B,H,N+M,Dh)
            v = torch.cat([v_s, v_m], dim=2)
            attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn = self.drop(attn.softmax(dim=-1))
            out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(b, n, -1)
        return self.out(out)


class CBOABlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CBOAAttention(d_model, n_heads=n_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, mem=None, id_basis=None):
        x = x + self.attn(self.norm1(x), mem=mem, id_basis=id_basis)
        x = x + self.mlp(self.norm2(x))
        return x


class PrivateAdapter(nn.Module):
    """Aperiodic-conditioned FiLM on pooled periodic features (never in CBOA)."""

    def __init__(self, d_model: int, clip_dim: int):
        super().__init__()
        self.to_cond = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2 * clip_dim),
        )
        nn.init.zeros_(self.to_cond[-1].weight)
        nn.init.zeros_(self.to_cond[-1].bias)

    def forward(self, clip_raw: torch.Tensor, ap_pool: torch.Tensor) -> torch.Tensor:
        gb = self.to_cond(ap_pool)
        gamma, beta = gb.chunk(2, dim=-1)
        return clip_raw * (1.0 + gamma) + beta


class POPFormer(nn.Module):
    """Population-Orthogonal Periodic Transformer for EEG→CLIP."""

    def __init__(
        self,
        n_channels: int = 63,
        seq_len: int = 250,
        d_model: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        clip_dim: int = 1024,
        dropout: float = 0.1,
        n_subjects: int = 10,
        mem_tokens: int = 64,
        id_rank: int = 8,
    ):
        super().__init__()
        self.d_model = d_model
        self.clip_dim = clip_dim
        self.n_layers = n_layers
        self.mem_tokens = mem_tokens
        self.id_rank = id_rank

        self.stem = StemEmbed(n_channels, seq_len, d_model, dropout=dropout)
        self.split = DualStreamSplit(d_model)
        self.ch_pos = nn.Parameter(torch.zeros(1, n_channels, d_model))
        nn.init.normal_(self.ch_pos, std=0.02)

        self.blocks = nn.ModuleList(
            [CBOABlock(d_model, n_heads, dropout=dropout, mlp_ratio=4.0) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.pool_attn = nn.Linear(d_model, 1)

        self.projector = nn.Sequential(
            nn.Linear(d_model, clip_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(clip_dim, clip_dim),
            nn.LayerNorm(clip_dim),
        )
        self.private = PrivateAdapter(d_model, clip_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

        # subject probes (diagnostic + training constraints)
        self.probe_per = nn.Linear(d_model, n_subjects)
        self.probe_ap = nn.Linear(d_model, n_subjects)

        # learnable null memory (when bank empty / dropout)
        self.null_mem = nn.Parameter(torch.zeros(1, mem_tokens, d_model))
        nn.init.normal_(self.null_mem, std=0.02)

        # running identity basis buffer (D, K), updated outside / EMA
        self.register_buffer("id_basis", torch.zeros(d_model, id_rank), persistent=True)
        self.register_buffer("id_basis_ready", torch.zeros((), dtype=torch.bool))

    def encode_streams(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tok = self.stem(x) + self.ch_pos
        return self.split(tok)

    def encode_periodic(
        self,
        t_per: torch.Tensor,
        mem: torch.Tensor | None = None,
        use_memory: bool = True,
    ) -> torch.Tensor:
        basis = self.id_basis if bool(self.id_basis_ready) else None
        h = t_per
        for blk in self.blocks:
            m = mem if use_memory else None
            if m is None and use_memory:
                m = self.null_mem.expand(h.size(0), -1, -1)
            h = blk(h, mem=m, id_basis=basis)
        return self.norm(h)

    def pool(self, tokens: torch.Tensor) -> torch.Tensor:
        # attention pool over channels
        w = torch.softmax(self.pool_attn(tokens).squeeze(-1), dim=-1)  # (B, C)
        return torch.einsum("bc,bcd->bd", w, tokens)

    def forward(
        self,
        x: torch.Tensor,
        mem: torch.Tensor | None = None,
        use_memory: bool = True,
        use_private: bool = True,
        subject_id: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        t_per, t_ap = self.encode_streams(x)
        h_per = self.encode_periodic(t_per, mem=mem, use_memory=use_memory)
        pooled_per = self.pool(h_per)
        pooled_ap = self.pool(t_ap)

        raw = self.projector(pooled_per)
        if use_private:
            raw = self.private(raw, pooled_ap)
        emb = F.normalize(raw, dim=-1)

        out = {
            "clip_raw": raw,
            "clip_emb": emb,
            "pooled_per": pooled_per,
            "pooled_ap": pooled_ap,
            "t_per": t_per,
            "t_ap": t_ap,
            "logit_scale": self.logit_scale.exp(),
            "logits_per": self.probe_per(pooled_per),
            "logits_ap": self.probe_ap(pooled_ap.detach()),  # stop-grad into ap from probe_ap CE? keep both
        }
        # also probe ap without detach for encouraging identity in ap
        out["logits_ap_train"] = self.probe_ap(pooled_ap)
        if subject_id is not None:
            out["subject_id"] = subject_id
        return out

    @torch.no_grad()
    def update_id_basis(self, pooled_features: torch.Tensor, subject_ids: torch.Tensor, momentum: float = 0.1):
        """Estimate top subject-mean directions via class-conditional PCA on features."""
        # pooled_features: (N, D), subject_ids: (N,)
        device = pooled_features.device
        xs = []
        for s in subject_ids.unique():
            m = pooled_features[subject_ids == s].mean(0, keepdim=True)
            xs.append(m)
        means = torch.cat(xs, 0)  # (S, D)
        means = means - means.mean(0, keepdim=True)
        # PCA via SVD
        try:
            _, _, vh = torch.linalg.svd(means, full_matrices=False)
            k = min(self.id_rank, vh.size(0))
            basis = vh[:k].T  # (D, K)
            if basis.size(1) < self.id_rank:
                pad = torch.zeros(self.d_model, self.id_rank - basis.size(1), device=device)
                basis = torch.cat([basis, pad], 1)
            if bool(self.id_basis_ready):
                self.id_basis.mul_(1 - momentum).add_(basis, alpha=momentum)
            else:
                self.id_basis.copy_(basis)
                self.id_basis_ready.fill_(True)
        except Exception:
            pass

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CrossBrainMemoryBank:
    """Per-class memory of periodic pooled/channel tokens from training subjects."""

    def __init__(self, n_classes: int, d_model: int, mem_tokens: int = 64, device="cpu"):
        self.n_classes = n_classes
        self.d_model = d_model
        self.mem_tokens = mem_tokens
        self.device = device
        # store class prototypes as (n_classes, mem_tokens, d)
        self.slots = torch.zeros(n_classes, mem_tokens, d_model, device=device)
        self.counts = torch.zeros(n_classes, device=device)
        self.ready = False

    def to(self, device):
        self.device = device
        self.slots = self.slots.to(device)
        self.counts = self.counts.to(device)
        return self

    @torch.no_grad()
    def update(self, t_per: torch.Tensor, labels: torch.Tensor, momentum: float = 0.05):
        """t_per: (B, C, D) — subsample / pool channels to mem_tokens."""
        b, c, d = t_per.shape
        # pool channels to mem_tokens via adaptive average along channel dim
        x = t_per.transpose(1, 2)  # (B, D, C)
        x = F.adaptive_avg_pool1d(x, self.mem_tokens).transpose(1, 2)  # (B, M, D)
        for i in range(b):
            y = int(labels[i].item())
            if y < 0 or y >= self.n_classes:
                continue
            if self.counts[y] < 1:
                self.slots[y] = x[i]
            else:
                self.slots[y].mul_(1 - momentum).add_(x[i], alpha=momentum)
            self.counts[y] += 1
        self.ready = bool((self.counts > 0).any())

    @torch.no_grad()
    def lookup(self, labels: torch.Tensor, null: torch.Tensor) -> torch.Tensor:
        """Return (B, M, D) memory for each label; fallback to null."""
        b = labels.size(0)
        out = null.expand(b, -1, -1).clone()
        if not self.ready:
            return out
        for i in range(b):
            y = int(labels[i].item())
            if 0 <= y < self.n_classes and self.counts[y] > 0:
                out[i] = self.slots[y]
        return out

    @torch.no_grad()
    def lookup_clip_neighbors(
        self,
        clip_q: torch.Tensor,
        clip_gallery: torch.Tensor,
        topk: int = 8,
        null: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """For test / unknown labels: aggregate memory of CLIP-nearest train classes."""
        # clip_q: (B, Dclip), clip_gallery: (Ncls, Dclip) train image CLIP
        sim = F.normalize(clip_q, dim=-1) @ F.normalize(clip_gallery, dim=-1).T
        idx = sim.topk(k=min(topk, sim.size(1)), dim=1).indices  # (B, K)
        b = clip_q.size(0)
        if null is None:
            null = torch.zeros(1, self.mem_tokens, self.d_model, device=clip_q.device)
        acc = torch.zeros(b, self.mem_tokens, self.d_model, device=clip_q.device)
        for i in range(b):
            slots = []
            for j in idx[i].tolist():
                if self.counts[j] > 0:
                    slots.append(self.slots[j])
            if slots:
                acc[i] = torch.stack(slots, 0).mean(0)
            else:
                acc[i] = null[0]
        return acc
