"""EEG tokenization: turn 63 x 250 sensor-space EEG into tokens on a 2D grid.

Why this exists
---------------
EEGiT's ablation (THINGS-EEG, intra-subject Top-1) isolates two factors:

    pretrained ViT weights   +6.8   (63.6 -> 70.4)
    EEG patch representation +16.4  (54.0 -> 70.4)

So the *tokenization interface* is worth more than the choice of pretrained
backbone. A pretrained ViT carries strong priors about how tokens on a 2D grid
relate to each other; feeding it raw flattened EEG destroys that structure.
This module reconstructs it:

  1. Interpolate the 63 electrodes onto a regular 2D scalp grid using their real
     montage positions (inverse-distance weighting). This preserves electrode
     topology, which EEGiT also does by anatomical grouping + spatial interp.
  2. Split the 0-1000 ms window into T contiguous time windows.
  3. Each (grid cell, time window) becomes one token whose raw feature is the
     window's temporal waveform at that grid cell.

The resulting token grid is (G_h, G_w * T). Spatial adjacency is preserved in
both axes and temporal adjacency runs along the second axis, so the pretrained
positional embedding can be resampled onto it meaningfully.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------- montage
# Normalised 2D head coordinates: x = right(+)/left(-), y = anterior(+)/posterior(-).
# Standard 10-20 layout, matching the channel order in preprocessed_eeg/info.json.
_MONTAGE_XY: dict[str, tuple[float, float]] = {
    "Fp1": (-0.30, 0.95), "Fp2": (0.30, 0.95),
    "AF7": (-0.55, 0.80), "AF3": (-0.28, 0.82), "AFz": (0.00, 0.83),
    "AF4": (0.28, 0.82), "AF8": (0.55, 0.80),
    "F7": (-0.74, 0.60), "F5": (-0.52, 0.62), "F3": (-0.28, 0.63), "F1": (-0.10, 0.64),
    "F2": (0.10, 0.64), "F4": (0.28, 0.63), "F6": (0.52, 0.62), "F8": (0.74, 0.60),
    "FT9": (-0.88, 0.33), "FT7": (-0.80, 0.38),
    "FC5": (-0.55, 0.42), "FC3": (-0.30, 0.44), "FC1": (-0.10, 0.45), "FCz": (0.00, 0.45),
    "FC2": (0.10, 0.45), "FC4": (0.30, 0.44), "FC6": (0.55, 0.42),
    "FT8": (0.80, 0.38), "FT10": (0.88, 0.33),
    "T7": (-0.95, 0.05),
    "C5": (-0.58, 0.20), "C3": (-0.31, 0.22), "C1": (-0.10, 0.23), "Cz": (0.00, 0.23),
    "C2": (0.10, 0.23), "C4": (0.31, 0.22), "C6": (0.58, 0.20),
    "T8": (0.95, 0.05),
    "TP9": (-0.88, -0.28), "TP7": (-0.80, -0.30),
    "CP5": (-0.57, -0.05), "CP3": (-0.30, -0.03), "CP1": (-0.10, -0.02), "CPz": (0.00, -0.02),
    "CP2": (0.10, -0.02), "CP4": (0.30, -0.03), "CP6": (0.57, -0.05),
    "TP8": (0.80, -0.30), "TP10": (0.88, -0.28),
    "P7": (-0.75, -0.52), "P5": (-0.52, -0.52), "P3": (-0.28, -0.51), "P1": (-0.10, -0.50),
    "Pz": (0.00, -0.50), "P2": (0.10, -0.50), "P4": (0.28, -0.51), "P6": (0.52, -0.52),
    "P8": (0.75, -0.52),
    "PO7": (-0.55, -0.75), "PO3": (-0.26, -0.74), "POz": (0.00, -0.74),
    "PO4": (0.26, -0.74), "PO8": (0.55, -0.75),
    "O1": (-0.28, -0.93), "Oz": (0.00, -0.95), "O2": (0.28, -0.93),
}


def build_interpolation_matrix(
    channel_names: list[str], grid_h: int, grid_w: int, power: float = 2.0
) -> np.ndarray:
    """Inverse-distance-weight matrix mapping (n_channels,) -> (grid_h*grid_w,).

    Row i of the result sums to 1 and reads the scalp voltage at grid cell i as a
    weighted average of the recorded electrodes, weighted by 1/d^power.
    """
    missing = [c for c in channel_names if c not in _MONTAGE_XY]
    if missing:
        raise KeyError(f"no montage coordinates for channels: {missing}")

    src = np.array([_MONTAGE_XY[c] for c in channel_names], dtype=np.float64)

    # Target grid spans a square that comfortably encloses the montage.
    lim = 1.05
    ys = np.linspace(-lim, lim, grid_h)
    xs = np.linspace(-lim, lim, grid_w)
    gx, gy = np.meshgrid(xs, ys)                       # (grid_h, grid_w)
    tgt = np.stack([gx.ravel(), gy.ravel()], axis=1)   # (grid_h*grid_w, 2)

    d = np.linalg.norm(tgt[:, None, :] - src[None, :, :], axis=-1)   # (G, C)
    d = np.maximum(d, 1e-6)
    w = 1.0 / np.power(d, power)

    # Exact hits: an electrode sitting on a grid cell takes that cell alone.
    exact = d < 1e-5
    if exact.any():
        rows = np.where(exact.any(axis=1))[0]
        w[rows] = exact[rows].astype(np.float64)

    w /= w.sum(axis=1, keepdims=True)
    return w.astype(np.float32)


class EEGTokenizer(nn.Module):
    """(B, C, T) sensor EEG -> (B, N, D) tokens laid out on a (G_h, G_w*T_win) grid.

    Parameters
    ----------
    channel_names : channels present in the input, in order.
    d_model       : output token width (must match the backbone).
    grid_h, grid_w: spatial scalp grid resolution.
    n_time_windows: number of contiguous windows the 0-1000 ms span is cut into.
    """

    def __init__(
        self,
        channel_names: list[str],
        d_model: int,
        grid_h: int = 7,
        grid_w: int = 7,
        n_time_windows: int = 4,
        n_timepoints: int = 250,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.channel_names = list(channel_names)
        self.grid_h, self.grid_w = grid_h, grid_w
        self.n_time_windows = n_time_windows
        self.n_cells = grid_h * grid_w

        # Crop to a length divisible by the number of windows.
        self.win_len = n_timepoints // n_time_windows
        self.crop_len = self.win_len * n_time_windows

        interp = build_interpolation_matrix(self.channel_names, grid_h, grid_w)
        self.register_buffer("interp", torch.from_numpy(interp))   # (n_cells, C)

        # Per-token projection: window waveform (win_len) -> d_model.
        self.proj = nn.Sequential(
            nn.Linear(self.win_len, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

        # Token bookkeeping: index (cell, window) -> flat position.
        # Grid layout is (G_h, G_w * n_time_windows): token (r, c*T+t).
        self.n_tokens = self.n_cells * n_time_windows
        rows = torch.arange(grid_h).repeat_interleave(grid_w * n_time_windows)
        cols = torch.arange(grid_w * n_time_windows).repeat(grid_h)
        self.register_buffer("token_grid", torch.stack([rows, cols], dim=1).float())
        self.pos_lim = torch.tensor([float(grid_h), float(grid_w * n_time_windows)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) -> (B, N, D)."""
        if x.ndim != 3:
            raise ValueError(f"expected (B, C, T), got {tuple(x.shape)}")
        if x.shape[1] != len(self.channel_names):
            raise ValueError(
                f"channel mismatch: input {x.shape[1]}, tokenizer {len(self.channel_names)}"
            )
        x = x[..., : self.crop_len]

        # 1. montage interpolation: (B, C, T) -> (B, n_cells, T)
        #    interp is (n_cells, C) so this is a matmul over the channel axis.
        g = torch.einsum("gc,bct->bgt", self.interp.to(x.dtype), x)

        # 2. cut into time windows: (B, n_cells, T) -> (B, n_cells, T_win, win_len)
        b, nc, _ = g.shape
        g = g.view(b, nc, self.n_time_windows, self.win_len)

        # 3. project each (cell, window) token: (B, n_cells*T_win, win_len) -> (B, N, D)
        tok = g.reshape(b, self.n_tokens, self.win_len)
        return self.norm(self.proj(tok))


def normalise_token_grid(token_grid: torch.Tensor, pos_lim: torch.Tensor) -> torch.Tensor:
    """Map integer grid coords to [-1, 1]^2, the convention timm uses for pos_embed."""
    return token_grid / (pos_lim - 1.0) * 2.0 - 1.0


# ---------------------------------------------------------------- EEGiT-style
# Anatomical grouping from EEGiT's supplemental (THINGS-EEG), listed posterior ->
# anterior. This order IS the vertical axis of the EEG image: one patch row per
# region, so adjacent patch rows are adjacent brain regions.
#
# EEGiT's rationale, quoted: electrodes are "grouped into five anatomically
# meaningful regions ... preserving neurophysiologically relevant spatial
# correlations", then "linear interpolation is used to expand each brain region
# to P channels" where P is the ViT patch size. So each region ends up exactly
# one patch tall.
EEGIT_REGIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("occipital", ("PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2")),
    ("parietal", ("P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8")),
    ("temporal", ("FT9", "FT7", "FT8", "FT10", "T7", "T8", "TP9", "TP7", "TP8", "TP10")),
    ("central", ("FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
                 "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
                 "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6")),
    ("frontal", ("Fp1", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8",
                 "F7", "F5", "F3", "F1", "F2", "F4", "F6", "F8")),
)


# ---------------------------------------------------------------------------
# The released EEGiT code's own numbers, transcribed from
# `third_party/EEGiT/base/data_eeg.py`. They are kept separate from the tuples
# above rather than replacing them, because the two are not the same interface:
#
#   * `EEGIT_REGIONS`      posterior -> anterior, and each region's electrodes are
#                          re-sorted by montage x before interpolation.
#   * `EEGIT_REGIONS_OFFICIAL` anterior -> posterior, electrodes left in the order
#                          of `OFFICIAL_CHANNEL_ORDER`, and the two axes of the
#                          EEG image are SWAPPED (time vertical, regions
#                          horizontal).
#
# `style="eegit_official"` reproduces the second exactly; `style="nw"` keeps the
# first, which is what every result produced before this existed was trained on.
# Mixing them silently would change the interface of a run without changing a
# single flag it recorded, which is the failure mode worth the duplication to
# avoid.
OFFICIAL_CHANNEL_ORDER: tuple[str, ...] = (
    "Fp1", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8", "F7", "F5", "F3",
    "F1", "F2", "F4", "F6", "F8", "FT9", "FT7", "FC5", "FC3", "FC1",
    "FCz", "FC2", "FC4", "FC6", "FT8", "FT10", "T7", "C5", "C3", "C1",
    "Cz", "C2", "C4", "C6", "T8", "TP9", "TP7", "CP5", "CP3", "CP1",
    "CPz", "CP2", "CP4", "CP6", "TP8", "TP10", "P7", "P5", "P3", "P1",
    "Pz", "P2", "P4", "P6", "P8", "PO7", "PO3", "POz", "PO4", "PO8",
    "O1", "Oz", "O2",
)

# `EEGDataset.space_segmentation`, in the order the list is built there. That
# order is the vertical (region) axis of the EEG image.
EEGIT_REGIONS_OFFICIAL: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("frontal", ("Fp1", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8", "F7", "F5", "F3",
                 "F1", "F2", "F4", "F6", "F8")),
    ("central", ("FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
                 "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
                 "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6")),
    ("temporal", ("FT9", "FT7", "FT8", "FT10", "T7", "T8", "TP9", "TP7", "TP8", "TP10")),
    ("parietal", ("P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8")),
    ("occipital", ("PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2")),
)


def region_interp_matrix(n_src: int, n_dst: int) -> np.ndarray:
    """EEGiT Eq.1-2, as a matrix: (n_src,) electrodes -> (n_dst,) positions.

    Eq.1 sets the sampling position  x_i = i * (N_r - 1) / (P - 1)  and splits it
    into an integer part `j` and a fraction `alpha_i`; Eq.2 then reads
    `(1 - alpha_i) y_j + alpha_i y_(j+1)`, with the last position clamped to
    `y_(N_r - 1)`. Written out, that is exactly linear interpolation between the
    neighbouring electrodes, so each row of the returned matrix has two non-zeros
    (one, at the clamped last position) and sums to 1.
    """
    if n_dst < 1:
        raise ValueError(f"n_dst must be >= 1, got {n_dst}")
    if n_src < 1:
        raise ValueError(f"n_src must be >= 1, got {n_src}")
    if n_src == 1:
        # Degenerate region: every position reads the single electrode.
        return np.ones((n_dst, 1), dtype=np.float32)
    m = np.zeros((n_dst, n_src), dtype=np.float32)
    for i in range(n_dst):
        x = i * (n_src - 1) / (n_dst - 1)
        j = int(np.floor(x))
        a = float(x - j)
        if j >= n_src - 1:
            m[i, n_src - 1] = 1.0          # Eq.2's `j = N_r - 1` branch
        else:
            m[i, j] = 1.0 - a
            m[i, j + 1] = a
    return m


class EEGPatchTokenizer(nn.Module):
    """EEGiT's EEG patch representation: (B, C, T) -> (B, 3, H, W) for patch_embed.

    Why this replaces the grid tokenizer
    ------------------------------------
    EEGiT's own ablation prices the patch representation at +16.4 intra-subject
    Top-1 (54.0 -> 70.4) while the pretrained weights are worth +6.8. Its mechanism
    is specific: the EEG "image" is patchified by the *pretrained* `Conv2d`, so
    each 16x16 patch is a local space-time tile and the conv's visual priors apply.
    The grid tokenizer instead learned a random MLP on per-cell waveforms and
    never called `patch_embed` at all -- which is why the same backbone landed at
    21.0 here against EEGiT's 70.4.

    Geometry, matching EEGiT's stated "14 x 5 = 70 spatiotemporal patches":
      * anatomical axis : R brain regions, each linearly interpolated to exactly
                         `patch_size` positions -> R * patch_size columns/rows.
                         With all 63 channels and the 5 THINGS-EEG regions that is
                         80 positions = 5 patches, so one patch per region.
      * time axis       : uniformly resampled to `n_patches_w * patch_size`
                         -> 14 patches at the 224 width EEGiT uses.
      * replicated over 3 channels, as "the raw EEG signals are replicated across
        three channels to align with the RGB format of visual data".

    Which axis is which is a `style`:

      * `style="eegit_official"` -- H = time (14 patches), W = regions (5). This is
        what the released code actually builds: `img_size=(224, patch_size * 5)`
        with `use_kinematic` producing a `(B, 3, 224, 80)` tensor. Note this is the
        TRANSPOSE of the paper's Figure 3, whose caption says "the horizontal axis
        denotes time steps, and the vertical axis represents brain regions"; the
        code's own visualisation transposes the tensor before plotting, which is how
        the two can disagree without either being a typo. Regions run anterior ->
        posterior and each region's electrodes keep the dataset's channel order.
      * `style="nw"` -- H = regions (5 patches), W = time (14), regions posterior ->
        anterior with each region re-sorted left-to-right by montage x. This is the
        paper-figure orientation, and the interface every result produced before
        `style` existed was trained on.

    Both are 70 tokens, so both are tileable by the same 16x16 conv; the difference
    is which of the pretrained `pos_embed`'s two axes survives interpolation
    untouched, and which token adjacency the attention sees.

    Two deliberate deviations, both documented rather than silent:
      * z-score is per channel (a scalar per channel) rather than per
        (channel, timepoint). The input is already MVNN-whitened, which equalises
        scale across time; z-scoring each timepoint again would divide by a noise
        estimate and amplify the low-SNR samples.
      * a region's channels are ordered by their montage x coordinate before
        interpolation. Eq.1-2 interpolates along the 1D order it is given, and a
        left-to-right order is what makes the interpolation spatially meaningful;
        EEGiT's paper lists regions in that order but does not state the rule.
    """

    def __init__(
        self,
        channel_names: list[str],
        patch_size: int = 16,
        n_patches_w: int = 14,
        n_timepoints: int = 250,
        zscore: bool = True,
        dropout: float = 0.0,
        style: str = "nw",
    ) -> None:
        super().__init__()
        if style not in ("nw", "eegit_official"):
            raise ValueError(f"style must be 'nw' or 'eegit_official', got {style!r}")
        self.style = style
        self.channel_names = list(channel_names)
        self.patch_size = int(patch_size)
        # `n_patches_w` is the number of patches along the TIME axis in both
        # layouts; only that axis' position in the image tensor differs. The name
        # is kept because it is in every saved config and result file.
        self.n_patches_w = int(n_patches_w)
        self.n_time_patches = self.n_patches_w
        self.time_len = self.n_patches_w * self.patch_size
        self.n_timepoints = int(n_timepoints)
        self.zscore = zscore

        # Resolve each region against the channels actually present. A region
        # absent from the montage is dropped, so a 17-channel occipito-parietal
        # montage yields 2 regions rather than silently interpolating from nothing.
        idx_of = {c: i for i, c in enumerate(self.channel_names)}
        self.region_specs: list[tuple[str, list[int], int]] = []
        used: set[int] = set()
        if style == "eegit_official":
            # Anterior -> posterior, electrodes in the released code's own channel
            # order (`EEGDataset.channels`). No geometric re-sorting: the official
            # code interpolates along whatever order the region list happens to be
            # in, and reproducing a different order would be a different interface.
            rank = {c: i for i, c in enumerate(OFFICIAL_CHANNEL_ORDER)}
            for name, members in EEGIT_REGIONS_OFFICIAL:
                present = [c for c in members if c in idx_of]
                if not present:
                    continue
                unknown = [c for c in present if c not in rank]
                if unknown:
                    raise KeyError(f"region {name}: {unknown} missing from "
                                   f"OFFICIAL_CHANNEL_ORDER")
                present.sort(key=lambda c: rank[c])
                self.region_specs.append((name, [idx_of[c] for c in present], len(present)))
                used.update(idx_of[c] for c in present)
        else:
            # Left to right by montage x, regions posterior -> anterior.
            for name, members in EEGIT_REGIONS:
                present = [c for c in members if c in idx_of]
                if not present:
                    continue
                missing_xy = [c for c in present if c not in _MONTAGE_XY]
                if missing_xy:
                    raise KeyError(f"region {name}: no montage coordinates for {missing_xy}")
                present.sort(key=lambda c: _MONTAGE_XY[c][0])     # left -> right
                self.region_specs.append((name, [idx_of[c] for c in present], len(present)))
                used.update(idx_of[c] for c in present)

        # Every supplied channel must belong to exactly one region. A channel that
        # falls through would be silently dropped from the input, which is the
        # class of failure HANDOFF section 6.3 warns about: dense numeric EEG means
        # a wrong subset produces plausible-looking garbage instead of an error.
        unassigned = [c for i, c in enumerate(self.channel_names) if i not in used]
        if unassigned:
            raise ValueError(
                f"channels not covered by any EEGiT region: {sorted(unassigned)}. "
                f"Add them to EEGIT_REGIONS or run with the channels EEGiT groups.")

        self.n_regions = len(self.region_specs)
        self.region_width = self.n_regions * self.patch_size
        if style == "eegit_official":
            # Released code: `img_size=(224, patch_size * 5)`, i.e. the tensor
            # handed to timm is (3, time, regions) with time resampled to 224 by a
            # 2D bilinear `F.interpolate` over the (time, electrode) plane. That is
            # why the region axis here is the WIDTH, the opposite of `EEGiT_REGIONS`'
            # documentation in the paper's Figure 3 -- see the class docstring.
            self.height = self.time_len
            self.width = self.region_width
            self.grid = (self.n_patches_w, self.n_regions)      # H=time, W=regions
        else:
            self.height = self.region_width
            self.width = self.time_len
            self.grid = (self.n_regions, self.n_patches_w)      # H=regions, W=time

        # The `nw` path interpolates each region with an explicit Eq.1-2 matrix
        # (channels only) and then resamples time; the official path cannot use it
        # because its interpolation is 2D over (time, electrode). Only build the
        # buffers the chosen path reads, so a checkpoint carries exactly the
        # interface its config names.
        if style == "nw":
            mats = [region_interp_matrix(n_src, self.patch_size)
                    for _, _, n_src in self.region_specs]
            self.n_src_max = max(m.shape[1] for m in mats)
            padded = np.zeros((self.n_regions, self.patch_size, self.n_src_max),
                              dtype=np.float32)
            for r, m in enumerate(mats):
                padded[r, :, : m.shape[1]] = m
            self.register_buffer("interp", torch.from_numpy(padded))   # (R, P, S_max)

            # Row-selector: one-hot over the padded source axis. Combined with the
            # padded interp matrix this turns the per-region gather + matmul into a
            # single einsum over the full channel axis.
            sel = np.zeros((self.n_regions, self.n_src_max, len(self.channel_names)),
                           dtype=np.float32)
            for r, (_, idxs, n_src) in enumerate(self.region_specs):
                for s, ci in enumerate(idxs):
                    sel[r, s, ci] = 1.0
            self.register_buffer("gather", torch.from_numpy(sel))      # (R, S_max, C)

        # Fit statistics, filled in by `set_norm_stats`. Registered unconditionally
        # so the buffers move with the model and land in the checkpoint even when
        # zscore is off -- otherwise `set_norm_stats` would raise on a legitimate
        # call, and a saved model would lose the scale it was trained under.
        self.register_buffer("eeg_mean", torch.zeros(len(self.channel_names)))
        self.register_buffer("eeg_std", torch.ones(len(self.channel_names)))
        # The "was it fitted?" flag is a BUFFER, not a plain attribute, because a
        # plain attribute does not survive state_dict(): reloading a checkpoint
        # into a fresh tokenizer would leave the flag False and make the model
        # refuse to run, or -- if the guard were relaxed -- silently z-score with
        # mean 0 / std 1. Both are worse than the flag being saved.
        self.register_buffer("stats_set", torch.zeros((), dtype=torch.bool))

        self.drop = nn.Dropout(dropout)

    @property
    def n_tokens(self) -> int:
        """Number of patch tokens this tokenizer emits (excluding prefix tokens).

        Exposed so callers never have to know which tokenizer is in use. The grid
        tokenizer already had this name, and a reporting line that read it
        unconditionally is how a missing attribute turned into a crash 20 minutes
        into an allocation instead of at config-validation time.
        """
        return self.n_regions * self.n_patches_w

    @torch.no_grad()
    def set_norm_stats(self, eeg: np.ndarray, max_rows: int = 4096) -> None:
        """Compute the per-channel z-score statistics from training EEG.

        `eeg` is (n_concepts, n_img, C, T) or (n, C, T). Statistics come from the
        FIT split only; using the test split here would leak its scale into
        training. Subsampled because a mean and a variance do not need 15040 rows
        and the full array is ~1 GB.

        `ddof=1` and `clamp_min(1e-8)` reproduce `EEGDataset.alldataset_mean_std`,
        which uses `torch.std` (unbiased, i.e. ddof=1) and clamps at 1e-8. The
        numeric effect is nil -- with n = 4096 x 250 the unbiased correction is a
        factor 1 + 4.9e-7, which is below float32 eps (1.2e-7), and the `max_rows`
        subsample below moves the scale by ~1% anyway. It is done so the interface
        can be checked against the reference EXACTLY (see
        `test_eegit_official_interface.py`) rather than with a tolerance that
        would also hide a real mistake.
        """
        x = np.asarray(eeg, dtype=np.float64)
        x = x.reshape(-1, x.shape[-2], x.shape[-1])
        if x.shape[0] > max_rows:
            pick = np.linspace(0, x.shape[0] - 1, max_rows).astype(int)
            x = x[pick]
        mean = x.mean(axis=(0, 2))                       # (C,)
        std = x.std(axis=(0, 2), ddof=1)
        std = np.maximum(std, 1e-8)
        self.eeg_mean.copy_(torch.from_numpy(mean.astype(np.float32)))
        self.eeg_std.copy_(torch.from_numpy(std.astype(np.float32)))
        self.stats_set.fill_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) -> (B, 3, H, W), ready for the pretrained patch_embed."""
        if x.ndim != 3:
            raise ValueError(f"expected (B, C, T), got {tuple(x.shape)}")
        if x.shape[1] != len(self.channel_names):
            raise ValueError(
                f"channel mismatch: input {x.shape[1]}, tokenizer {len(self.channel_names)}")
        if self.zscore and not bool(self.stats_set):
            raise RuntimeError(
                "EEGPatchTokenizer.set_norm_stats() was never called; without it the "
                "z-score is a no-op and the patch_embed sees a different value range "
                "than it was trained on. Call it on the fit split before training.")

        if self.zscore:
            x = (x - self.eeg_mean.to(x.dtype).view(1, -1, 1)) / self.eeg_std.to(x.dtype).view(1, -1, 1)

        if self.style == "eegit_official":
            return self.drop(self._official_image(x))

        # (B, C, T) -> (B, R, P, T): per-region interpolation to `patch_size` rows.
        il = self.interp.to(x.dtype)                     # (R, P, S_max)
        g = self.gather.to(x.dtype)                      # (R, S_max, C)
        rows = torch.einsum("rps,rsc,bct->brpt", il, g, x)          # (B, R, P, T)
        b, r, p, _ = rows.shape
        img = rows.reshape(b, 1, r * p, x.shape[-1])                # (B, 1, H, T)

        # Time: uniform resampling to the target width, then replicate to 3 planes.
        if img.shape[-1] != self.width:
            img = F.interpolate(img, size=(self.height, self.width),
                                mode="bilinear", align_corners=False)
        img = img.repeat(1, 3, 1, 1)
        return self.drop(img)

    def _official_image(self, x: torch.Tensor) -> torch.Tensor:
        """The released code's `spatial_interpolate`, transcribed.

        For each anatomical region it takes the block of that region's electrodes
        as (time, electrode) and runs ONE 2D bilinear `F.interpolate` to
        (time_patches * patch_size, patch_size), then concatenates the regions along
        the electrode axis. The regions are then the image's WIDTH and time its
        HEIGHT, replicated over 3 colour planes.

        Two things this does that the `nw` path does not, both of which change the
        numbers rather than only the layout:

          * the interpolation is 2D, so a target cell is a bilinear blend of
            neighbouring (time, electrode) samples, not a 1D blend along electrodes
            followed by a separate time resample;
          * time is resampled 250 -> 224 with `align_corners=False`, which is a
            *shrinking* resample (the official `Preprocessed_data_250Hz_whiten`
            window is 250 samples, `timesteps: [0,250]`), not a slice.
        """
        planes = []
        for _, idxs, _ in self.region_specs:
            blk = x[:, idxs, :].unsqueeze(1)              # (B, 1, n_ch, T)
            blk = blk.transpose(2, 3)                     # (B, 1, T, n_ch)
            blk = F.interpolate(blk, size=(self.time_len, self.patch_size),
                                mode="bilinear", align_corners=False)
            planes.append(blk)                            # (B, 1, T', patch_size)
        img = torch.cat(planes, dim=3)                    # (B, 1, T', R*patch_size)
        return img.repeat(1, 3, 1, 1)

