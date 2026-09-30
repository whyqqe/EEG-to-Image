"""EEG encoder: multi-scale temporal convolution, spatial/region modelling, a
lightweight Transformer, and the invariant/specific representation split.

Input geometry
--------------
A trial is ``(63, 250)``: 63 channels x 250 samples.  Reading the preprocessing
code that produced the release (`NeuroBridge/preprocess_eeg.py:137`,
``data[idx][:, :, int(baseline_duration * freq):]``) shows the first 0.2 s baseline
is *discarded*, so the 250 samples are exactly **0-1 s post-stimulus at 250 Hz**.
`info.json` ships a 300-entry ``times`` array because it records the array before
that slice -- it is a trap, and the offsets below are derived from the stripped
window, not from ``times``.

At 250 Hz the design's three temporal scales land on:

    25 ms  ->  6 taps
    50 ms  -> 12 taps
   100 ms  -> 25 taps

which covers the early visual components (P1 ~100 ms, N170 ~170 ms) that carry
most of the image-evoked signal, while the longer branch gathers the slower
category-level dynamics.

Region encoding without a montage
---------------------------------
The design offers "channel attention / spatial graph conv / functional region
encoding" as alternatives.  A spatial graph would need electrode coordinates, which
the release does not ship; hand-typing a montage table would introduce an
unverifiable error source into the model's input path.  Functional regions, by
contrast, are recoverable exactly from the channel names, which are the standard
10-10 labels (``Fp1 ... O2``): the alphabetic prefix *is* the region.  So the
channel module uses a learnable region embedding indexed by that prefix, plus an
SE-style channel attention -- both derived from shipped metadata, neither dependent
on a coordinate table.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

# Temporal branch widths, in samples at 250 Hz.
#
# The design's 25 / 50 / 100 ms correspond to 6.25 / 12.5 / 25 samples.  The kernels
# are the *odd* roundings (7 / 13 / 25 = 28 / 52 / 100 ms) because only an odd kernel
# admits an integer symmetric padding that preserves sequence length: with stride 1
# and kernel k you need 2p = k - 1, which has no integer solution for even k.  An even
# kernel leaves each branch one sample longer than its neighbours and the concat then
# fails.  `MultiScaleTemporal` asserts this rather than letting it surface as a shape
# error in `torch.cat`.
TEMPORAL_KERNELS: tuple[int, ...] = (7, 13, 25)

# Standard 10-10 labels, in the order the release stores them (info.json).
CHANNEL_NAMES: tuple[str, ...] = (
    "Fp1", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8",
    "F7", "F5", "F3", "F1", "F2", "F4", "F6", "F8",
    "FT9", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8", "FT10",
    "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8",
    "TP9", "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8", "TP10",
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8",
    "O1", "Oz", "O2",
)

# Ordered longest-first so that a two-letter prefix is never shadowed by a
# one-letter one: "Fp1" and "FC5" must not match the frontal rule for "F".
_PREFIX_TO_REGION: tuple[tuple[str, str], ...] = (
    ("Fp", "frontal_pole"),
    ("AF", "frontal_pole"),
    ("FC", "frontocentral"),
    ("FT", "frontotemporal"),
    ("CP", "centroparietal"),
    ("TP", "temporoparietal"),
    ("PO", "parieto_occipital"),
    ("F", "frontal"),
    ("T", "temporal"),
    ("C", "central"),
    ("P", "parietal"),
    ("O", "occipital"),
)

REGION_NAMES: tuple[str, ...] = (
    "frontal_pole", "frontal", "frontocentral", "frontotemporal",
    "temporal", "temporoparietal", "central",
    "centroparietal", "parietal", "parieto_occipital", "occipital",
)


def region_of(channel: str) -> str:
    for prefix, region in _PREFIX_TO_REGION:
        if channel.startswith(prefix):
            return region
    raise ValueError(f"cannot assign a functional region to channel {channel!r}")


def channel_regions(names: tuple[str, ...] = CHANNEL_NAMES) -> list[str]:
    return [region_of(n) for n in names]


def region_index(names: tuple[str, ...] = CHANNEL_NAMES) -> torch.Tensor:
    """Per-channel region id, as a long tensor of shape (C,)."""
    lookup = {name: i for i, name in enumerate(REGION_NAMES)}
    return torch.tensor([lookup[region_of(n)] for n in names], dtype=torch.long)


@dataclass
class EncoderConfig:
    n_channels: int = 63
    n_times: int = 250
    branch_width: int = 64          # features per temporal branch
    temporal_kernels: tuple[int, ...] = TEMPORAL_KERNELS
    pool_stride: int = 4            # time downsampling before the Transformer
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    ffn_mult: int = 4
    dropout: float = 0.1
    d_inv: int = 512                # subject-invariant embedding width
    d_sub: int = 128                # subject-specific embedding width
    adapter_blocks: int = 2         # how many *top* blocks get a subject adapter
    adapter_bottleneck: int = 32
    region_embed: bool = True
    channel_attention: bool = True
    pretrained_subjects: int = 0    # sized by the trainer from the training split

    @property
    def conv_out_channels(self) -> int:
        return self.branch_width * len(self.temporal_kernels)

    @property
    def n_tokens(self) -> int:
        return self.n_times // self.pool_stride


class SEChannelAttention(nn.Module):
    """Squeeze-excitation over the channel axis: which electrodes matter now."""

    def __init__(self, n_channels: int, n_features: int, reduction: int = 4):
        super().__init__()
        hidden = max(4, n_features // reduction)
        # Pool over time only, so the gate stays per (feature, channel) rather than
        # collapsing the channel axis it is supposed to weight.
        self.fc = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ELU(),
            nn.Linear(hidden, n_features), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, F, C, T)."""
        gate = x.mean(dim=-1)                    # (B, F, C)
        gate = self.fc(gate.transpose(1, 2))     # (B, C, F)
        return x * gate.transpose(1, 2).unsqueeze(-1)


class MultiScaleTemporal(nn.Module):
    """Parallel temporal convolutions at the three design scales.

    Each branch keeps the channel axis intact (kernel shape ``(1, k)`` on a
    single-channel "image"), because the channel axis is what the region
    embedding and channel attention act on.  Collapsing it happens afterwards,
    once per branch, with a depthwise spatial convolution.
    """

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        for k in cfg.temporal_kernels:
            if k % 2 == 0:
                raise ValueError(
                    f"temporal kernel {k} is even; an even kernel cannot preserve "
                    f"sequence length under symmetric padding, so branches would "
                    f"differ in length. Use an odd kernel (see TEMPORAL_KERNELS)."
                )
        self.branches = nn.ModuleList()
        for k in cfg.temporal_kernels:
            self.branches.append(nn.Sequential(
                nn.Conv2d(1, cfg.branch_width, (1, k), padding=(0, k // 2), bias=False),
                nn.BatchNorm2d(cfg.branch_width),
                nn.ELU(),
            ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) -> (B, F*Branches, C, T)."""
        x = x.unsqueeze(1)                       # (B, 1, C, T)
        branches = [b(x) for b in self.branches]
        lengths = {b.shape[-1] for b in branches}
        if len(lengths) != 1:
            raise RuntimeError(
                f"temporal branches disagree on length {sorted(lengths)}; "
                f"kernels {self.cfg_kernels()} must all preserve T"
            )
        return torch.cat(branches, dim=1)

    def cfg_kernels(self) -> tuple[int, ...]:
        return tuple(s[0].kernel_size[1] for s in self.branches)


class SpatialRegionBlock(nn.Module):
    """Region embedding + channel attention + depthwise spatial mixing.

    Region embeddings are added *before* the spatial collapse so that a channel's
    functional identity is available to whichever representation learns to use it;
    adding them afterwards would be vacuous, since the channel axis no longer
    exists.
    """

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        n_feat = cfg.conv_out_channels

        if cfg.region_embed:
            self.register_buffer("region_ids", region_index(), persistent=False)
            # One learned vector per (region, feature) pair, broadcast over time.
            self.region_embed = nn.Parameter(
                torch.zeros(len(REGION_NAMES), cfg.branch_width)
            )
            nn.init.normal_(self.region_embed, std=0.02)
            # Expand per-branch features onto the shared region table.
            self.region_proj = nn.Linear(cfg.branch_width, n_feat, bias=False)
        else:
            self.region_embed = None
            self.region_proj = None

        self.channel_attention = SEChannelAttention(cfg.n_channels, n_feat) \
            if cfg.channel_attention else None

        # Depthwise: one spatial filter per feature map, so each learned feature
        # chooses its own electrode weighting.
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, (cfg.n_channels, 1), groups=n_feat, bias=False),
            nn.BatchNorm2d(n_feat),
            nn.ELU(),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, F, C, T) -> (B, F, 1, T)."""
        if self.region_embed is not None:
            # (regions, branch_width) -> (regions, F) -> index by channel -> (C, F)
            table = self.region_proj(self.region_embed)          # (R, n_feat)
            per_channel = table[self.region_ids]                 # (C, n_feat)
            x = x + per_channel.t().unsqueeze(0).unsqueeze(-1)   # (1, n_feat, C, 1)
        if self.channel_attention is not None:
            x = self.channel_attention(x)
        return self.spatial_conv(x)


class SubjectAdapter(nn.Module):
    """Subject-specific bottleneck injected into the top Transformer blocks.

    A single shared encoder cannot represent inter-subject variability (skull
    thickness, impedance, cap placement); forcing it to would spend capacity on
    nuisance variation.  A per-subject low-rank residual absorbs that variation
    cheaply, and -- crucially for calibration -- it is the *only* part that
    few-shot adaptation needs to touch.
    """

    def __init__(self, d_model: int, n_subjects: int, bottleneck: int = 32,
                 dropout: float = 0.0):
        super().__init__()
        self.n_subjects = max(1, n_subjects)
        self.down = nn.Parameter(torch.empty(self.n_subjects, d_model, bottleneck))
        self.up = nn.Parameter(torch.empty(self.n_subjects, bottleneck, d_model))
        nn.init.normal_(self.down, std=1.0 / max(1, bottleneck) ** 0.5)
        nn.init.zeros_(self.up)      # start as an exact no-op
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, subject_id: torch.Tensor | None,
                mode: str = "subject") -> torch.Tensor:
        """h: (B, L, D); subject_id: (B,).

        `mode` selects how the per-subject parameters are resolved:

          "subject"  index by `subject_id` (training)
          "mean"     the average of every subject's adapter, i.e. the population-level
                     estimate, used for a subject the model has never seen
          "none"     skip the adapter entirely (zero residual)

        The return value is the *residual* for the caller to add
        (`h = h + adapter(...)`), not the adapted features.  "none" therefore has to
        return zeros, not `h`: returning `h` makes the caller compute `2 * h` and
        silently doubles every feature wherever the adapter is bypassed.

        "mean" rather than picking an arbitrary training subject, and rather than
        bypassing the adapter: the adapter is a residual that was *used* during
        training, so bypassing it at evaluation would change the function the model
        computes, while an arbitrary subject's adapter would inject a real,
        unrelated subject's statistics.  The mean is the natural prior and is also
        the correct initialization for few-shot calibration of a new subject.
        """
        if mode == "none":
            return torch.zeros_like(h)
        if mode == "mean":
            d = self.down.mean(dim=0, keepdim=True)
            u = self.up.mean(dim=0, keepdim=True)
            hidden = torch.einsum("bld,dr->blr", self.norm(h), d.squeeze(0))
            return self.drop(torch.einsum("blr,rd->bld", hidden, u.squeeze(0)))
        if subject_id is None:
            raise ValueError("subject_id is required when adapter mode is 'subject'")
        if subject_id.shape[0] != h.shape[0]:
            # Without this the failure surfaces as an einsum broadcast error that
            # names subscripts rather than the actual mismatch.
            raise ValueError(
                f"subject_id has batch {subject_id.shape[0]} but features have "
                f"batch {h.shape[0]}; one subject id is required per trial"
            )
        if self.n_subjects == 1:
            s = torch.zeros_like(subject_id)
        else:
            s = subject_id.clamp(0, self.n_subjects - 1)
        d = self.down[s]                                  # (B, D, R)
        u = self.up[s]                                    # (B, R, D)
        # (B, L, D) x (B, D, R) -> (B, L, R) x (B, R, D) -> (B, L, D)
        hidden = torch.einsum("bld,bdr->blr", self.norm(h), d)
        return self.drop(torch.einsum("blr,brd->bld", hidden, u))


class EEGEncoder(nn.Module):
    """Full EEG encoder producing the invariant/specific split."""

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg

        self.temporal = MultiScaleTemporal(cfg)
        self.spatial = SpatialRegionBlock(cfg)

        n_feat = cfg.conv_out_channels
        n_tokens = cfg.n_tokens
        self.pool = nn.AvgPool2d((1, cfg.pool_stride))
        self.to_tokens = nn.Sequential(
            nn.Linear(n_feat, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
        )
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, cfg.d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)

        n_adapter = min(cfg.adapter_blocks, cfg.n_layers)
        n_shared = cfg.n_layers - n_adapter
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * cfg.ffn_mult,
            dropout=cfg.dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.shared_blocks = nn.TransformerEncoder(
            layer, num_layers=max(1, n_shared), enable_nested_tensor=False,
        )
        self.adapted_blocks = nn.TransformerEncoder(
            layer, num_layers=n_adapter, enable_nested_tensor=False,
        ) if n_adapter > 0 else None
        self.adapters = nn.ModuleList([
            SubjectAdapter(cfg.d_model, max(1, cfg.pretrained_subjects),
                           cfg.adapter_bottleneck, cfg.dropout)
            for _ in range(n_adapter)
        ])
        self.final_norm = nn.LayerNorm(cfg.d_model)

        self.head_inv = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_inv),
            nn.LayerNorm(cfg.d_inv),
        )
        self.head_sub = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_sub),
            nn.LayerNorm(cfg.d_sub),
        )

    def resize_subjects(self, n_subjects: int) -> None:
        """Rebuild the adapters for a different subject count, preserving the logits.

        Called once by the trainer after it knows the LOSO split size.  The
        initialization is identical to construction (up-projection zeroed), so the
        adapters start as an exact no-op regardless of when this runs.
        """
        if n_subjects == self.cfg.pretrained_subjects:
            return
        device = self.pos.device
        self.cfg.pretrained_subjects = n_subjects
        self.adapters = nn.ModuleList([
            SubjectAdapter(self.cfg.d_model, n_subjects,
                           self.cfg.adapter_bottleneck, self.cfg.dropout)
            for _ in range(len(self.adapters))
        ]).to(device)

    def forward(self, x: torch.Tensor, subject_id: torch.Tensor | None = None,
                return_tokens: bool = False,
                adapter_mode: str = "subject") -> dict[str, torch.Tensor]:
        """x: (B, C, T). Returns a dict with z_inv, z_sub and tokens.

        `adapter_mode` selects how the subject adapters are resolved; see
        `SubjectAdapter.forward`.  Evaluation of a held-out subject must use "mean".
        """
        h = self.temporal(x)                 # (B, F, C, T)
        h = self.spatial(h)                  # (B, F, 1, T)
        h = self.pool(h)                     # (B, F, 1, T')
        h = h.squeeze(2).transpose(1, 2)     # (B, T', F)
        h = self.to_tokens(h) + self.pos[:, : h.shape[1]]

        h = self.shared_blocks(h)
        if self.adapted_blocks is not None and len(self.adapters) > 0:
            for block, adapter in zip(self.adapted_blocks.layers, self.adapters):
                h = block(h)
                h = h + adapter(h, subject_id, mode=adapter_mode)
        h = self.final_norm(h)

        pooled = h.mean(dim=1)
        out = {
            "z_inv": self.head_inv(pooled),
            "z_sub": self.head_sub(pooled),
        }
        if return_tokens:
            out["tokens"] = h
        return out
