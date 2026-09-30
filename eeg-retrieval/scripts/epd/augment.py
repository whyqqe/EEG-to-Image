"""EEG augmentation, ported from SAMGA's `module/eeg_augmentation.py`.

Why this exists
---------------
The sub-08 run without augmentation memorised the training set: in-batch Top-1 hit
97.9% while validation Top-1 fell from 12.7% (epoch 8) to 10.7% (epoch 30). SAMGA
-- which reaches 91.3% on the same subject and channel subset -- enables
`--eeg_aug_type noise` by default and applies it to the training split only. We
had no augmentation at all, so this module closes that gap.

Fidelity to the reference
-------------------------
The four transforms and their semantics match SAMGA. Two deliberate deviations,
both documented at the call site:

  * magnitudes are in OUR amplitude units. Our preprocessed EEG has std ~0.50
    (train) while SAMGA's default is `RandomGaussianNoise(std=0.001)`, which at
    this scale would be numerically inert. Noise is therefore re-scaled.
  * SAMGA's `RandomChannelDropout` and `RandomSmooth` mutate their input in place.
    Both are pure here, because in-place mutation of a cached array would corrupt
    it across epochs. This is a bug fix, not a behavioural change.
"""
from __future__ import annotations

import numpy as np

# Our preprocessed EEG has per-channel std ~0.4-0.6 (measured on sub-08).
# Augmentation magnitudes are expressed as a fraction of that so they stay
# meaningful if the preprocessing scale ever changes.
_SIGNAL_STD = 0.5


class RandomTimeShift:
    """Roll the signal along time by a random number of samples.

    A jitter of a few samples models the latency jitter of evoked responses
    between trials: the same stimulus does not elicit the same waveform at the
    same millisecond every time.
    """

    def __init__(self, max_shift: int = 5) -> None:
        self.max_shift = max_shift

    def __call__(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        shift = int(rng.integers(-self.max_shift, self.max_shift + 1))
        return np.roll(x, shift, axis=-1) if shift else x


class RandomGaussianNoise:
    """Add white noise. `std` is a fraction of the signal std (default 10%)."""

    def __init__(self, std: float = 0.1) -> None:
        self.std = std

    def __call__(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return x + rng.normal(0.0, self.std * _SIGNAL_STD, size=x.shape).astype(x.dtype)


class RandomChannelDropout:
    """Zero a random subset of channels.

    Models bad electrodes and the fact that a fixed montage is only a sample of
    the underlying scalp field. Because the tokenizer interpolates channels onto a
    scalp grid by inverse distance, dropping a channel degrades a whole
    neighbourhood rather than a single coordinate -- so this is a spatially
    structured perturbation, not per-feature dropout.
    """

    def __init__(self, drop_prob: float = 0.1) -> None:
        self.drop_prob = drop_prob

    def __call__(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        keep = rng.random(x.shape[0]) >= self.drop_prob
        if keep.all():
            return x
        out = x.copy()
        out[~keep] = 0.0
        return out


class RandomSmooth:
    """Temporally smooth a random subset of channels with a moving average.

    Low-pass filtering suppresses high-frequency noise while preserving the
    low-frequency evoked structure. Vectorised as a cumulative-sum box filter
    rather than SAMGA's per-sample Python loop; the result is identical up to
    floating point, and this runs ~10^3x faster.
    """

    def __init__(self, kernel_size: int = 5, smooth_prob: float = 0.3) -> None:
        self.kernel_size = kernel_size
        self.smooth_prob = smooth_prob

    def __call__(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        pick = rng.random(x.shape[0]) < self.smooth_prob
        if not pick.any():
            return x
        k = self.kernel_size
        half = k // 2
        # Pad by replication so the moving average is defined at both edges.
        pad = np.pad(x[pick], ((0, 0), (half, half)), mode="edge")
        csum = np.cumsum(pad, axis=-1)
        csum = np.concatenate([np.zeros_like(csum[..., :1]), csum], axis=-1)
        smooth = (csum[..., k:] - csum[..., :-k]) / k
        out = x.copy()
        out[pick] = smooth
        return out


class Compose:
    """Apply several transforms in order.

    SAMGA uses exactly one augmentation type per run. We allow a stack because the
    sub-08 run showed the capacity/data ratio is the binding constraint, and
    single weak transforms are unlikely to close an 87-point generalisation gap.
    Every arm that uses this reports which transforms were active, so the
    comparison against SAMGA's single-transform protocol stays explicit.
    """

    def __init__(self, transforms: list) -> None:
        self.transforms = list(transforms)

    def __call__(self, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        for t in self.transforms:
            x = t(x, rng)
        return x

    def __repr__(self) -> str:
        return f"Compose({[type(t).__name__ for t in self.transforms]})"


# ---------------------------------------------------------------- registry
# Names are the CLI surface (`--aug`). "full" is our stacked variant; the rest
# mirror SAMGA's `--eeg_aug_type` values one-for-one.
def build_aug(name: str, seed: int | None = None):
    """Return a callable (x, rng) -> x, or None for "none"."""
    name = (name or "none").lower()
    if name in ("none", "off", ""):
        return None
    if name == "noise":
        return RandomGaussianNoise(std=0.1)
    if name == "time_shift":
        return RandomTimeShift(max_shift=5)
    if name == "channel_dropout":
        return RandomChannelDropout(drop_prob=0.1)
    if name == "smooth":
        return RandomSmooth(kernel_size=5, smooth_prob=0.3)
    if name == "full":
        # Order matters: perturb the signal first, then smooth, then drop
        # channels. Dropping last means smoothing never sees an artificial
        # step edge at a zeroed channel.
        return Compose([
            RandomTimeShift(max_shift=5),
            RandomGaussianNoise(std=0.1),
            RandomSmooth(kernel_size=5, smooth_prob=0.3),
            RandomChannelDropout(drop_prob=0.1),
        ])
    raise KeyError(f"unknown augmentation {name!r}; see build_aug")


AUG_NAMES = ("none", "noise", "time_shift", "channel_dropout", "smooth", "full")
