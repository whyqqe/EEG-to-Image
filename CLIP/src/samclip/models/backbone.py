"""The shared EEG trunk ``f_theta`` -- ATM-S family, with NO subject conditioning.

Architecture (ATM-S / iTransformer-style, the backbone the plan doc §3 selects):

    (B, C, T)
      -> depthwise temporal conv          (local waveform features, per channel)
      -> channel token embedding          Linear(T -> d_model) applied per channel
      -> + channel positional embedding
      -> L x EncoderLayer                 (plain pre-norm transformer blocks)
      -> temporal-spatial aggregation     (collapse channel tokens, pool time tokens)
      -> MLP projector                    -> (B, d_model)

Why there is no conditioning seam any more
------------------------------------------
Through v1 this file exposed FiLM/LoRA seams inside every block so a subject vector
``z_s`` could modulate the trunk (`CondParams`, `apply_film`, `apply_lora`). That whole
paradigm was removed. The evidence, all of it measured in this project or in the
literature, is recorded in ``docs/eeg2image_v2_plan.md`` §2; the short version:

  * the learned ``z_s`` collapsed to a constant -- cosine 0.9999 between DIFFERENT
    subjects, separation -0.00000 (`scripts/diag_zcollapse.py`);
  * explicit conditioning produced no measurable gain over the id-table control
    (1-epoch sub-08 fold, job 637403);
  * an independent LOSO study measures that *suppressing* subject identity does not
    improve retrieval while pulling same-stimulus cross-subject representations
    together does (+2.23pp Top-1).

So the trunk is now a plain shared encoder and subject adaptation lives in the
objective (cross-subject consistency) and in label-free test-time geometry
(`samclip.calibration`). `ConditionedEncoderLayer` is renamed `EncoderLayer` and its
`CondParams` argument is gone; nothing downstream accepts a conditioning object.

A NOTE FOR ANYONE RE-INTRODUCING FREEZING
-----------------------------------------
The previous version of this stack carried a subtle and expensive lesson that was
implemented at the model level (`SAMCLIP.freeze_shared`). It is recorded here because
the code that enforced it is gone:

    ``requires_grad=False`` does NOT freeze a BatchNorm. With ``momentum > 0`` it keeps
    writing ``running_mean``/``running_var`` from every forward pass, even under
    ``torch.no_grad()``. A frozen trunk that keeps training therefore silently rewrites
    its own statistics; in this project that took ``agg.spatial``'s ``running_var`` from
    ~1.2e2 to ~1.4e23 over one episodic stage, and in eval mode the encoder divided by
    it, collapsing Top-1 from 9.5% to 0.5%.

If a freeze is introduced again, put every BatchNorm whose affine parameters are
frozen into ``.eval()`` and re-assert it after each ``.train()`` -- and verify by
snapshotting buffers before and after, not by inspection.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderLayer(nn.Module):
    """Pre-norm transformer block.

    Implemented in full rather than assembled from ``nn.TransformerEncoderLayer``
    because the trunk's channel-token layout needs the attention, the FFN and the
    residual structure to be explicitly readable -- the previous version's reason for
    hand-rolling it was that conditioning had to reach *inside* the block, and that
    seam is gone, but the explicit form is kept so the block stays a single place to
    read. It is also what makes the per-layer parameter count auditable against the
    ATM-S reference.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_ff: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, dim_ff)
        self.ff2 = nn.Linear(dim_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``(B, N, d_model)`` -> ``(B, N, d_model)``."""
        b, n, _ = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(b, n, self.d_model)
        x = x + self.drop(self.proj(attn))

        h = self.norm2(x)
        h = F.gelu(self.ff1(h))
        h = self.ff2(h)
        return x + self.drop(h)


class TemporalSpatialAggregator(nn.Module):
    """Collapse channel tokens to a single vector (ATM's temporal-spatial conv).

    Channel tokens `(B, C, d)` are treated as a `(B, 1, d, C)` map: axis 2 is the
    token-time axis `d`, axis 3 is the EEG-channel axis `C`. The first conv spans the
    whole `C` axis (kernel `(1, n_channels)`, "spatial": mixes channels), the second
    spans `d` (kernel `(temporal_kernel, 1)`, "temporal": mixes token-time), and the
    pool then reduces `d` to a fixed `pool_tokens` so the parameter count does not
    depend on `d_model`.

    Axis bookkeeping matters here, not just for readability: the *previous* version
    put `temporal_kernel` on axis 3 and pooled `(1, pool_tokens)`, i.e. it ran the
    "temporal" conv over an axis of size 1 (a scalar multiply) and then averaged over
    `d` -- the exact axis `EEGTrunk.norm` had just zero-meaned -- which left the head
    with a constant per-sample representation. See `EEGTrunk.forward`.
    """

    def __init__(self, d_model: int, n_channels: int, width: int = 16,
                 temporal_kernel: int = 16, pool_tokens: int = 8) -> None:
        super().__init__()
        self.width = width
        self.pool_tokens = pool_tokens
        self.spatial = nn.Sequential(
            nn.Conv2d(1, width, (1, n_channels), bias=False),
            nn.BatchNorm2d(width), nn.ELU(),
        )
        self.temporal = nn.Sequential(
            nn.Conv2d(width, width, (temporal_kernel, 1),
                      padding=(temporal_kernel // 2, 0), bias=False),
            nn.BatchNorm2d(width), nn.ELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((pool_tokens, 1))
        self.out_dim = width * pool_tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2).unsqueeze(1)            # (B,1,d,C)
        h = self.spatial(h)                           # (B,width,d,1)
        h = self.temporal(h)                          # (B,width,d,1)
        h = self.pool(h).flatten(1)                   # (B, width*pool_tokens)
        return h


class EEGTrunk(nn.Module):
    """Shared, subject-independent trunk: ``(B, C, T) -> (B, d_model)``.

    Every parameter in this module is shared across all subjects. That is the whole
    point of the v2 design: the encoder learns ONE representation function, and the
    subject differences are handled by the objective and by test-time geometry rather
    than by per-subject parameters.
    """

    def __init__(
        self,
        n_channels: int = 63,
        n_timepoints: int = 250,
        d_model: int = 200,
        n_heads: int = 4,
        n_blocks: int = 2,
        dim_ff: int = 512,
        dropout: float = 0.1,
        temporal_kernel: int = 25,
        channel_pos: bool = True,
        agg_width: int = 16,
        agg_pool: int = 8,
        front_end: str = "linear",
        front_pool: int = 5,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_timepoints = n_timepoints
        self.d_model = d_model

        pad = temporal_kernel // 2
        # ---------------------------------------------------------------- front end
        # `linear` is the historical path and the default, so every recorded run stays
        # bit-identical. `conv` is D2 (docs/eeg2image_v5_master_plan.md §3.3 P2).
        #
        # WHY `conv` EXISTS. The linear path is `nn.Linear(n_timepoints, d_model)`: a dense
        # map from all 250 timepoints to every one of the `d_model` features, with an
        # independent weight per (input timepoint, output feature) pair and NO locality or
        # translation prior. It is also where most of the trunk's parameters live (250x200
        # = 50k, against 1575 for a depthwise conv). A dense temporal map of that shape is
        # free to fit each subject's own temporal idiosyncrasies, which is exactly the
        # failure the subject gap measures (+31.22pp between a seen and an unseen subject
        # after v5's pillar A). The alternative is not "more capacity" but a different
        # HYPOTHESIS CLASS: an explicit low-pass, a wider receptive field via dilation, and
        # a FIXED average-pool downsample -- the reference's `Conv2d(1,40,(1,25)) +
        # AvgPool2d` idea, in the (B, C, T) layout this trunk uses (tokens are channels, so
        # the time axis is the one that gets convolved and pooled).
        #
        # This is the "constructive > soft penalty" principle (P3): a soft penalty
        # constrains the optimisation, a restricted hypothesis class constrains the
        # solution set. See docs/eeg2image_v5_master_plan.md §3.3.
        if front_end not in ("linear", "conv"):
            raise ValueError(f"front_end must be 'linear' or 'conv', got {front_end!r}")
        if front_end == "linear":
            self.temporal = nn.Sequential(
                nn.Conv1d(n_channels, n_channels, temporal_kernel, padding=pad,
                          groups=n_channels, bias=False),
                nn.BatchNorm1d(n_channels), nn.ELU(),
            )
            self.embed = nn.Linear(n_timepoints, d_model)
        else:
            if int(front_pool) < 1 or n_timepoints % int(front_pool) != 0:
                raise ValueError(
                    f"front_pool={front_pool} must divide n_timepoints={n_timepoints} "
                    f"(the pool is a FIXED downsample, so a ragged split would silently "
                    f"drop the tail of every trial)")
            self.temporal = nn.Sequential(
                # the same depthwise low-pass the linear path starts with
                nn.Conv1d(n_channels, n_channels, temporal_kernel, padding=pad,
                          groups=n_channels, bias=False),
                nn.BatchNorm1d(n_channels), nn.ELU(),
                # a DILATED depthwise conv: a receptive field of `temporal_kernel + 2*pad*3`
                # timesteps at ~8x lower parameter cost than widening a dense map, and with
                # the same translation prior. Locality is the point, so it stays depthwise.
                nn.Conv1d(n_channels, n_channels, temporal_kernel,
                          padding=pad * 3, dilation=3, groups=n_channels, bias=False),
                nn.BatchNorm1d(n_channels), nn.ELU(),
                # EXPLICIT low-pass + downsampling, then a small dense map on the pooled
                # series. `n_timepoints // front_pool` inputs instead of `n_timepoints` is
                # the parameter saving AND the constraint: the dense map can no longer
                # address individual raw timepoints.
                nn.AvgPool1d(int(front_pool), stride=int(front_pool)),
            )
            self.embed = nn.Linear(n_timepoints // int(front_pool), d_model)
        self.channel_pos = (nn.Parameter(torch.zeros(1, n_channels, d_model))
                            if channel_pos else None)
        if self.channel_pos is not None:
            nn.init.normal_(self.channel_pos, std=0.02)

        self.blocks = nn.ModuleList([
            EncoderLayer(d_model, n_heads, dim_ff, dropout) for _ in range(n_blocks)
        ])
        self.agg = TemporalSpatialAggregator(d_model, n_channels, width=agg_width,
                                             pool_tokens=agg_pool)
        # The final norm must sit on the POOLED feature, not on the `(B, C, d)` tokens.
        # `nn.LayerNorm` zero-means every token along `d_model`; `agg` then pools over
        # exactly that axis, so a token-level norm leaves the head with only its own
        # rounding error -- the encoder becomes a constant function of the input.
        self.norm = nn.LayerNorm(self.agg.out_dim)
        self.projector = nn.Sequential(
            nn.Linear(self.agg.out_dim, d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, d_model),
        )
        self.out_dim = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``(B, C, T)`` -> ``(B, d_model)``."""
        if x.dim() != 3:
            raise ValueError(f"EEG batch must be (B, C, T), got {tuple(x.shape)}")
        if x.shape[1] != self.n_channels:
            raise ValueError(
                f"trunk expects {self.n_channels} channels, got {x.shape[1]}. "
                f"The 17-vs-63 channel choice is a protocol decision (inter-subject "
                f"uses 63); a mismatch here means the fold was built with the wrong "
                f"channel set.")
        if x.shape[2] != self.n_timepoints:
            raise ValueError(f"trunk expects {self.n_timepoints} timepoints, "
                             f"got {x.shape[2]}")

        h = self.temporal(x)                        # (B, C, T)
        h = self.embed(h)                           # (B, C, d_model)
        if self.channel_pos is not None:
            h = h + self.channel_pos
        for blk in self.blocks:
            h = blk(h)
        h = self.agg(h)
        h = self.norm(h)
        return self.projector(h)
