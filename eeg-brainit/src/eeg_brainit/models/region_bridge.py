"""Region-Bridge: EEG spatiotemporal heads aligned to fMRI EVC/Ventral CLIP teachers."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from eeg_brainit.models.dual_teacher import DualTeacherEEG2CLIP, FmriClipTeacher, dual_teacher_losses
from eeg_brainit.models.eeg2fmri import TemporalEEGEncoder


# Channel groups for NOD posterior 26-ch montage
OCC_CH = {"O1", "Oz", "O2", "PO3", "POz", "PO4", "PO5", "PO6", "PO7", "PO8"}
VENT_CH = {"P7", "P8", "P5", "P6", "PO7", "PO8", "P3", "P4"}


def channel_index(ch_names: list[str], wanted: set[str]) -> list[int]:
    idx = [i for i, c in enumerate(ch_names) if c in wanted]
    if not idx:
        raise RuntimeError(f"no channels matched {wanted}")
    return idx


def time_slice(n_times: int, sfreq: float, tmin: float, win: tuple[float, float]) -> slice:
    a = int(round((win[0] - tmin) * sfreq))
    b = int(round((win[1] - tmin) * sfreq))
    a = max(0, min(n_times - 1, a))
    b = max(a + 1, min(n_times, b))
    return slice(a, b)


class RegionBridgeEEG(nn.Module):
    """Shared encoder with EVC / Ventral / Image heads."""

    def __init__(
        self,
        n_channels: int = 26,
        clip_dim: int = 1024,
        d_model: int = 512,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = TemporalEEGEncoder(n_channels=n_channels, d_model=d_model, dropout=dropout, depth=4)
        self.head_img = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, clip_dim),
        )
        self.head_evc = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 512),
            nn.GELU(),
            nn.Linear(512, clip_dim),
        )
        self.head_vent = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 512),
            nn.GELU(),
            nn.Linear(512, clip_dim),
        )

    def encode(self, eeg: torch.Tensor) -> torch.Tensor:
        return self.encoder(eeg.float())

    def forward(self, eeg: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encode(eeg)
        return {
            "z_eeg": z,
            "clip_img": F.normalize(self.head_img(z), dim=-1),
            "clip_evc": F.normalize(self.head_evc(z), dim=-1),
            "clip_vent": F.normalize(self.head_vent(z), dim=-1),
        }

    def forward_views(
        self,
        eeg: torch.Tensor,
        *,
        idx_occ: list[int],
        idx_vent: list[int],
        sl_early: slice,
        sl_late: slice,
    ) -> dict[str, torch.Tensor]:
        """Spatiotemporal routed forward for region teachers."""
        # Full for image head
        full = self.forward(eeg)
        # Early occipital → EVC head
        e_early = torch.zeros_like(eeg)
        e_early[:, idx_occ, sl_early] = eeg[:, idx_occ, sl_early]
        z_e = self.encode(e_early)
        # Late lateral → Ventral head
        e_late = torch.zeros_like(eeg)
        e_late[:, idx_vent, sl_late] = eeg[:, idx_vent, sl_late]
        z_v = self.encode(e_late)
        full["clip_evc"] = F.normalize(self.head_evc(z_e), dim=-1)
        full["clip_vent"] = F.normalize(self.head_vent(z_v), dim=-1)
        return full


__all__ = [
    "OCC_CH",
    "VENT_CH",
    "RegionBridgeEEG",
    "DualTeacherEEG2CLIP",
    "FmriClipTeacher",
    "dual_teacher_losses",
    "channel_index",
    "time_slice",
]
