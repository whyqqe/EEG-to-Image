"""MindCross / MindBridge inspired shared + subject-specific EEG encoder.

References:
- MindCross: shared_embedder + per-subject embedder, ResFuse, diff-loss (s*r→0)
- MindBridge: ModuleDict subject Adapters + shared translator; reset-tuning for new subjects
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# CHANNEL SETS
#
# `posterior` is the 17 electrodes every historical run in this project used.  It
# is an INHERITED DEFAULT, not a measured choice: `grep` across the tree finds it
# described ("17 posterior channels, [0,250] samples", tdm_gate0.py:533) but never
# compared against the full montage, and no log or report records a channel
# ablation.  `brdt_probe.py:14` shows the author knew the released trials are
# 63-channel.  Meanwhile the published THINGS-EEG2 baselines keep ALL electrodes
# ("All electrodes were preserved for analysis", ICLR'24 data section), so this
# parameter is the one place where our preprocessing differs from the literature.
#
# `channels_num` is therefore a first-class experimental variable now, not a
# constant.  `all` returns an EMPTY list on purpose: EEGPreImageDataset only
# slices when `len(selected_channels) > 0`, so the empty list means "the whole
# montage in info.json order", which avoids hard-coding 63 names here.
# ---------------------------------------------------------------------------
POSTERIOR_17 = [
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2",
]
CHANNEL_SETS = {"posterior": POSTERIOR_17, "all": []}


def resolve_channels(name: str) -> list[str]:
    """`--channels` value -> the `selected_channels` argument for the dataset."""
    if name not in CHANNEL_SETS:
        raise KeyError(f"unknown channel set {name!r}; have {sorted(CHANNEL_SETS)}")
    return list(CHANNEL_SETS[name])


def channel_indices(old: list[str], new: list[str]) -> list[int]:
    """Positions of `old` inside `new`, so a checkpoint can be dilated exactly."""
    missing = [c for c in old if c not in new]
    if missing:
        raise ValueError(f"channel set is not a subset of the target montage: {missing}")
    return [new.index(c) for c in old]


def dilate_first_linear(
    weight: torch.Tensor,
    bias: torch.Tensor,
    idx: list[int],
    n_ch_new: int,
    n_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Widen a flattened-EEG first layer from `len(idx)` channels to `n_ch_new`.

    `EEGProjectWide.forward` flattens (channels, samples) row-major, so channel c
    owns columns ``[c*n_samples, (c+1)*n_samples)``.  The widened layer copies the
    old per-channel blocks into those columns at their positions in the NEW
    montage and leaves every other column at ZERO, which makes the widened model
    compute exactly the old model on its own channels -- an exact warm start
    rather than a re-initialisation, so "did the extra electrodes help?" is asked
    against a model that has lost nothing.
    """
    n_ch_old = len(idx)
    if weight.shape[1] != n_ch_old * n_samples:
        raise ValueError(
            f"checkpoint first layer is {weight.shape[1]} wide, expected "
            f"{n_ch_old * n_samples} for {n_ch_old} channels x {n_samples} samples"
        )
    out = torch.zeros(
        (weight.shape[0], n_ch_new * n_samples), dtype=weight.dtype, device=weight.device
    )
    for j, c in enumerate(idx):
        out[:, c * n_samples:(c + 1) * n_samples] = weight[
            :, j * n_samples:(j + 1) * n_samples
        ]
    return out, bias.clone()


class ResidualAdd(nn.Module):
    def __init__(self, f: nn.Module):
        super().__init__()
        self.f = f

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.f(x)


class EEGProjectWide(nn.Module):
    """Slightly enlarged EEGProject: one extra residual block (shared backbone)."""

    def __init__(
        self,
        feature_dim: int = 1024,
        eeg_sample_points: int = 250,
        channels_num: int = 17,
        dropout: float = 0.3,
        n_extra_blocks: int = 1,
    ):
        super().__init__()
        self.input_dim = eeg_sample_points * channels_num
        blocks: list[nn.Module] = [
            nn.Linear(self.input_dim, feature_dim),
            ResidualAdd(
                nn.Sequential(
                    nn.GELU(),
                    nn.Linear(feature_dim, feature_dim),
                    nn.Dropout(dropout),
                )
            ),
        ]
        for _ in range(n_extra_blocks):
            blocks.append(
                ResidualAdd(
                    nn.Sequential(
                        nn.GELU(),
                        nn.Linear(feature_dim, feature_dim),
                        nn.Dropout(dropout),
                    )
                )
            )
        blocks.append(nn.LayerNorm(feature_dim))
        self.model = nn.Sequential(*blocks)
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.shape[0], self.input_dim)
        return self.model(x)


class SubjectEmbedder(nn.Module):
    """Per-subject path from raw EEG (MindCross-style)."""

    def __init__(self, in_dim: int, h: int, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, h),
            nn.LayerNorm(h),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SubjectAdapter(nn.Module):
    """MindBridge-style bottleneck residual adapter on shared features."""

    def __init__(self, dim: int, bottleneck: int = 256, dropout: float = 0.1):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.norm = nn.LayerNorm(bottleneck)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.drop(self.act(self.norm(self.down(x)))))


class ResFuse(nn.Module):
    """MindCross ResFuse: mlp(cat(s,r)) + r."""

    def __init__(self, h: int, dropout: float = 0.15):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * h, h),
            nn.LayerNorm(h),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, s: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat((s, r), dim=-1)) + r


class SharedSpecificEncoder(nn.Module):
    """Shared wide backbone + per-subject embedder + ResFuse (+ optional adapter)."""

    def __init__(
        self,
        subject_ids: list[int],
        feature_dim: int = 1024,
        eeg_sample_points: int = 250,
        channels_num: int = 17,
        n_extra_blocks: int = 1,
        use_adapter: bool = True,
        adapter_bottleneck: int = 256,
    ):
        super().__init__()
        self.subject_ids = [int(s) for s in subject_ids]
        self.feature_dim = feature_dim
        self.input_dim = eeg_sample_points * channels_num
        self.shared = EEGProjectWide(
            feature_dim=feature_dim,
            eeg_sample_points=eeg_sample_points,
            channels_num=channels_num,
            n_extra_blocks=n_extra_blocks,
        )
        self.specific = nn.ModuleDict(
            {str(s): SubjectEmbedder(self.input_dim, feature_dim) for s in self.subject_ids}
        )
        self.fuse = ResFuse(feature_dim)
        self.use_adapter = use_adapter
        if use_adapter:
            self.adapters = nn.ModuleDict(
                {
                    str(s): SubjectAdapter(feature_dim, bottleneck=adapter_bottleneck)
                    for s in self.subject_ids
                }
            )
        else:
            self.adapters = None

    def _sid_key(self, sid: int | torch.Tensor) -> str:
        if isinstance(sid, torch.Tensor):
            sid = int(sid.item()) if sid.ndim == 0 else int(sid[0].item())
        return str(int(sid))

    def forward(
        self,
        x: torch.Tensor,
        subject_ids: torch.Tensor | int | None = None,
        return_parts: bool = False,
    ):
        b = x.shape[0]
        flat = x.view(b, self.input_dim)
        r = self.shared(x)

        if subject_ids is None:
            sid_t = torch.full((b,), self.subject_ids[0], device=x.device, dtype=torch.long)
        elif isinstance(subject_ids, int):
            sid_t = torch.full((b,), int(subject_ids), device=x.device, dtype=torch.long)
        else:
            sid_t = subject_ids.long().to(x.device)

        s = torch.zeros_like(r)
        for sid in torch.unique(sid_t):
            k = str(int(sid.item()))
            mask = sid_t == sid
            if k in self.specific:
                s[mask] = self.specific[k](flat[mask])
            else:
                s[mask] = r[mask]

        fused = self.fuse(s, r)
        if self.use_adapter and self.adapters is not None:
            out = fused.clone()
            for sid in torch.unique(sid_t):
                k = str(int(sid.item()))
                mask = sid_t == sid
                if k in self.adapters:
                    out[mask] = self.adapters[k](fused[mask])
        else:
            out = fused

        if return_parts:
            return out, s, r
        return out

    def freeze_shared(self) -> None:
        for p in self.shared.parameters():
            p.requires_grad = False
        for p in self.fuse.parameters():
            p.requires_grad = False

    def train_only_subject(self, sid: int) -> None:
        """MindCross/MindBridge calibrate: freeze shared, train one subject path."""
        self.freeze_shared()
        for k, m in self.specific.items():
            req = k == str(int(sid))
            for p in m.parameters():
                p.requires_grad = req
        if self.adapters is not None:
            for k, m in self.adapters.items():
                req = k == str(int(sid))
                for p in m.parameters():
                    p.requires_grad = req

    def add_subject(self, sid: int) -> None:
        k = str(int(sid))
        if k not in self.specific:
            self.specific[k] = SubjectEmbedder(self.input_dim, self.feature_dim)
            self.subject_ids.append(int(sid))
        if self.use_adapter and self.adapters is not None and k not in self.adapters:
            self.adapters[k] = SubjectAdapter(self.feature_dim)


def diff_loss(s: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """MindCross orthogonality: encourage s ⟂ r via MSE(s*r, 0)."""
    return F.mse_loss(s * r, torch.zeros_like(s))


def load_shared_from_eegproject(
    ss: SharedSpecificEncoder,
    eegproject_state: dict,
) -> int:
    """Partial-load a standard EEGProject checkpoint into the shared wide backbone."""
    shared_sd = ss.shared.state_dict()
    loaded = 0
    for k, v in eegproject_state.items():
        # map model.0 / model.1 / model.2 → same keys when shapes match
        if k in shared_sd and shared_sd[k].shape == v.shape:
            shared_sd[k] = v
            loaded += 1
    ss.shared.load_state_dict(shared_sd, strict=False)
    return loaded
