"""Subject-Context EEG (SC-EEG) conditioning for cross-subject decoding.

Idea
----
At train / test time, feed M unlabeled background EEG trials from the *same*
subject as a free subject profile. A small encoder pools them into h_subj and
FiLM-modulates the query backbone features before the CLIP projector.

This is orthogonal to labeled K-shot personalization:
  - SC-EEG: label-free geometric / physiological subject prior
  - K-shot: optional semantic alignment with K EEG–image pairs
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from eeg_brainit.models.atm_backbone import AtmStyleEEGEncoder
from eeg_brainit.models.eeg2fmri import TemporalEEGEncoder


class FiLM(nn.Module):
    """Feature-wise linear modulation: h' = h * (1 + γ(c)) + β(c)."""

    def __init__(self, cond_dim: int, feat_dim: int):
        super().__init__()
        self.to_gamma = nn.Linear(cond_dim, feat_dim)
        self.to_beta = nn.Linear(cond_dim, feat_dim)
        nn.init.zeros_(self.to_gamma.weight)
        nn.init.zeros_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight)
        nn.init.zeros_(self.to_beta.bias)

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return h * (1.0 + self.to_gamma(c)) + self.to_beta(c)


class SubjectProfileEncoder(nn.Module):
    """Pool M context hidden vectors → subject profile."""

    def __init__(self, in_dim: int, profile_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, profile_dim),
            nn.GELU(),
            nn.Linear(profile_dim, profile_dim),
            nn.LayerNorm(profile_dim),
        )
        self.profile_dim = profile_dim

    def forward(self, ctx_hidden: torch.Tensor) -> torch.Tensor:
        # ctx_hidden: (B, M, D) or (M, D)
        if ctx_hidden.dim() == 2:
            pooled = ctx_hidden.mean(dim=0, keepdim=True)
        else:
            pooled = ctx_hidden.mean(dim=1)
        return self.net(pooled)


class TemporalCLIPEncoder(nn.Module):
    """Public-style temporal backbone (same as LOSO TTA script)."""

    def __init__(self, n_channels: int = 63, clip_dim: int = 1024):
        super().__init__()
        self.enc = TemporalEEGEncoder(n_channels=n_channels, d_model=512, dropout=0.2, depth=3)
        self.proj = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, clip_dim))
        self.flat_dim = 512
        self.clip_dim = clip_dim

    def encode_hidden(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x.float())

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encode_hidden(x)
        return {"hidden": h, "clip_emb": F.normalize(self.proj(h), dim=-1)}


class SCConditionedEEGEncoder(nn.Module):
    """Backbone + SC-EEG profile + FiLM + CLIP projector."""

    def __init__(
        self,
        backbone: str = "atm_style",
        n_channels: int = 63,
        seq_len: int = 250,
        clip_dim: int = 1024,
        profile_dim: int = 256,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.backbone_name = backbone
        if backbone == "atm_style":
            self.backbone = AtmStyleEEGEncoder(
                n_channels=n_channels, seq_len=seq_len, clip_dim=clip_dim, dropout=dropout
            )
            self.flat_dim = int(self.backbone.flat_dim)
            # Reuse ATM projector after FiLM on the same flat features.
            self.proj = self.backbone.proj
        elif backbone == "temporal":
            self.backbone = TemporalCLIPEncoder(n_channels=n_channels, clip_dim=clip_dim)
            self.flat_dim = int(self.backbone.flat_dim)
            self.proj = self.backbone.proj
        else:
            raise ValueError(backbone)

        self.profile_enc = SubjectProfileEncoder(self.flat_dim, profile_dim=profile_dim)
        self.film = FiLM(profile_dim, self.flat_dim)
        self.profile_dim = profile_dim
        self.clip_dim = clip_dim
        self.null_profile = nn.Parameter(torch.zeros(profile_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def encode_hidden(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.encode_hidden(x.float())

    def build_profile_from_eeg(
        self, ctx_eeg: torch.Tensor, detach_backbone: bool = True
    ) -> torch.Tensor:
        """Encode unlabeled context EEG → profile.

        ctx_eeg: (M, C, T) or (B, M, C, T)
        returns: (profile_dim,) or (B, profile_dim)

        Backbone features for context are detached by default (subject prior as
        conditioning signal); profile MLP still receives gradients.
        """
        single = ctx_eeg.dim() == 3
        if single:
            ctx_eeg = ctx_eeg.unsqueeze(0)
        b, m, c, t = ctx_eeg.shape
        with torch.no_grad() if detach_backbone else torch.enable_grad():
            flat = self.encode_hidden(ctx_eeg.reshape(b * m, c, t)).reshape(b, m, -1)
        if detach_backbone:
            flat = flat.detach()
        prof = self.profile_enc(flat)
        return prof[0] if single else prof

    def forward(
        self,
        x: torch.Tensor,
        profile: torch.Tensor | None = None,
        ctx_eeg: torch.Tensor | None = None,
        use_null_profile: bool = False,
    ) -> dict[str, torch.Tensor]:
        h = self.encode_hidden(x)
        if use_null_profile:
            c = self.null_profile.expand(h.size(0), -1)
        elif ctx_eeg is not None:
            c = self.build_profile_from_eeg(ctx_eeg)
            if c.dim() == 1:
                c = c.unsqueeze(0).expand(h.size(0), -1)
        elif profile is not None:
            c = profile
            if c.dim() == 1:
                c = c.unsqueeze(0).expand(h.size(0), -1)
        else:
            c = self.null_profile.expand(h.size(0), -1)
        h_cond = self.film(h, c)
        emb = F.normalize(self.proj(h_cond), dim=-1)
        return {
            "hidden": h,
            "hidden_cond": h_cond,
            "clip_emb": emb,
            "profile": c,
            "logit_scale": self.logit_scale.exp(),
        }

    def backbone_parameters(self):
        return self.backbone.parameters()

    def adapter_parameters(self):
        """Trainable units for K-shot personalization (backbone frozen)."""
        for p in self.profile_enc.parameters():
            yield p
        for p in self.film.parameters():
            yield p
        for p in self.proj.parameters():
            yield p
        yield self.null_profile
        yield self.logit_scale


class SubjectContextBank:
    """Per-subject mmap EEG bank for sampling unlabeled SC-EEG contexts."""

    def __init__(self, eeg_root, subjects: list[str], split: str = "train"):
        from pathlib import Path
        import numpy as np

        self.subjects = list(subjects)
        self.eeg = {}
        self.n = {}
        root = Path(eeg_root)
        fname = "train_eeg.npy" if split == "train" else "test_eeg.npy"
        for s in subjects:
            arr = np.load(root / s / fname, mmap_mode="r")
            self.eeg[s] = arr
            self.n[s] = int(arr.shape[0])
        self._id_to_name = {i: s for i, s in enumerate(self.subjects)}

    def sample_batch(
        self,
        subject_ids: torch.Tensor,
        m: int,
        rng: torch.Generator | None = None,
        exclude: dict[int, set[int]] | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Return (B, M, C, T) float32 contexts for each batch row's subject."""
        import numpy as np

        b = int(subject_ids.shape[0])
        out = np.zeros((b, m, 63, 250), dtype=np.float32)
        sid = subject_ids.detach().cpu().tolist()
        for i, raw_id in enumerate(sid):
            # ThingsEEG2SubjectDataset stores subject_id as 0..9 (= sub-01..)
            name = f"sub-{int(raw_id) + 1:02d}"
            if name not in self.eeg:
                # fallback: map by order if bank subjects are a subset
                name = self.subjects[int(raw_id) % len(self.subjects)]
            n = self.n[name]
            ban = exclude.get(int(raw_id), set()) if exclude else set()
            cand = [j for j in range(n) if j not in ban]
            if not cand:
                cand = list(range(n))
            replace = len(cand) < m
            if rng is not None:
                # use numpy with torch seed snapshot
                seed = int(torch.randint(0, 2**31 - 1, (1,), generator=rng).item())
                rs = np.random.RandomState(seed + i)
            else:
                rs = np.random.RandomState(None)
            idxs = rs.choice(cand, size=m, replace=replace)
            out[i] = np.asarray(self.eeg[name][idxs], dtype=np.float32)
        t = torch.from_numpy(out)
        return t.to(device) if device is not None else t

    def sample_subject(
        self,
        subject: str,
        m: int,
        seed: int = 0,
        exclude_idx: set[int] | None = None,
    ) -> torch.Tensor:
        import numpy as np

        n = self.n[subject]
        ban = exclude_idx or set()
        cand = [j for j in range(n) if j not in ban]
        if not cand:
            cand = list(range(n))
        rs = np.random.RandomState(seed)
        idxs = rs.choice(cand, size=m, replace=len(cand) < m)
        return torch.from_numpy(np.asarray(self.eeg[subject][idxs], dtype=np.float32))
