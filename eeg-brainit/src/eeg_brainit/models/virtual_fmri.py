"""Virtual fMRI generation + voxel-to-cluster Brain Tokenizer (Brain-IT style).

Path 1 in the proposal: z_eeg / spectrogram -> Spec2Vol decoder -> virtual
fMRI volume -> Voxel-to-Cluster mapping -> 128 Brain Tokens.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _maybe_add_spec2vol_path(root: Path) -> Path | None:
    # Prefer the experiment-local clone only. External trees may miss mamba/CUDA
    # kernels and would break forward even if import succeeds.
    candidates = [root / "third_party" / "Spec2VolCAMU-Net"]
    for c in candidates:
        if c.is_dir():
            if str(c) not in sys.path:
                sys.path.insert(0, str(c))
            return c
    return None


class LightweightVolumeDecoder(nn.Module):
    """Fallback 2D volume decoder when Spec2Vol VMUNet is unavailable.

    Produces a pseudo-fMRI volume (B, D, H, W) from the MD-TF-CAE feature map.
    This keeps the pipeline runnable for smoke tests and early fusion experiments.
    """

    def __init__(self, in_channels: int = 256, out_depth: int = 32, spatial: int = 64) -> None:
        super().__init__()
        self.spatial = spatial
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, out_depth, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, feat_map: torch.Tensor) -> torch.Tensor:
        x = feat_map
        if x.shape[-2] != self.spatial or x.shape[-1] != self.spatial:
            x = F.interpolate(x, size=(self.spatial, self.spatial), mode="bilinear", align_corners=False)
        return self.net(x)


class VoxelToClusterTokenizer(nn.Module):
    """Compress a virtual volume into ``num_clusters`` Brain Tokens.

    If a pretrained GMM voxel-to-cluster mapping (``v2c_*.npy``) is provided and
    the volume is flattened to match ``num_voxels``, soft assignment is used.
    Otherwise a learned 3D/2D adaptive pool + linear projection is used.
    """

    def __init__(
        self,
        num_clusters: int = 128,
        brain_dim: int = 1024,
        in_channels: int = 32,
        v2c_path: str | None = None,
    ) -> None:
        super().__init__()
        self.num_clusters = num_clusters
        self.brain_dim = brain_dim
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, brain_dim // 4, 3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((int(num_clusters**0.5), int(num_clusters**0.5))),
        )
        # If sqrt is not integer (e.g. 128), fall back to 1D adaptive pool.
        self.use_1d = int(num_clusters**0.5) ** 2 != num_clusters
        if self.use_1d:
            self.pool1d = nn.AdaptiveAvgPool1d(num_clusters)
        self.proj = nn.Linear(brain_dim // 4, brain_dim)
        self.norm = nn.LayerNorm(brain_dim)
        self.register_buffer("v2c_soft", torch.empty(0), persistent=False)
        if v2c_path:
            self.load_v2c(v2c_path)

    def load_v2c(self, path: str) -> None:
        import numpy as np

        p = Path(path)
        if not p.is_file():
            print(f"[WARN] v2c mapping not found: {p}; using learned pooling tokenizer")
            return
        arr = np.load(p, allow_pickle=True)
        # Accept either hard labels (N,) or soft assignment (N, K).
        t = torch.as_tensor(arr)
        if t.ndim == 1:
            k = int(t.max().item()) + 1
            soft = F.one_hot(t.long(), num_classes=max(k, self.num_clusters)).float()
            soft = soft[:, : self.num_clusters]
        else:
            soft = t.float()
            if soft.shape[1] != self.num_clusters:
                soft = soft[:, : self.num_clusters]
        self.v2c_soft = soft
        print(f"[INFO] Loaded voxel-to-cluster mapping from {p} shape={tuple(soft.shape)}")

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Args:
            volume: (B, D, H, W) virtual fMRI
        Returns:
            brain_tokens: (B, num_clusters, brain_dim)
        """
        b, d, h, w = volume.shape
        if self.v2c_soft.numel() > 0 and self.v2c_soft.shape[0] == d * h * w:
            flat = volume.reshape(b, d * h * w)  # (B, V)
            # soft: (V, K) -> cluster means (B, K)
            clustered = flat @ self.v2c_soft.to(flat.device)  # (B, K)
            clustered = clustered.unsqueeze(-1).expand(-1, -1, self.brain_dim // 4)
            # Still project through a tiny MLP for dimension match.
            tokens = self.proj(clustered)
            return self.norm(tokens)

        x = self.conv(volume)
        if self.use_1d:
            x = x.flatten(2)  # (B, C, S)
            x = self.pool1d(x)  # (B, C, K)
            x = x.transpose(1, 2)
        else:
            x = x.flatten(2).transpose(1, 2)  # (B, K, C)
            if x.shape[1] != self.num_clusters:
                x = F.adaptive_avg_pool1d(x.transpose(1, 2), self.num_clusters).transpose(1, 2)
        return self.norm(self.proj(x))


class VirtualFMRIBranch(nn.Module):
    """Spectrogram feature map -> virtual volume -> Brain Tokens."""

    def __init__(
        self,
        in_channels: int = 256,
        volume_depth: int = 32,
        num_clusters: int = 128,
        brain_dim: int = 1024,
        v2c_path: str | None = None,
        prefer_official_decoder: bool = True,
        project_root: str | None = None,
    ) -> None:
        super().__init__()
        self.decoder = LightweightVolumeDecoder(
            in_channels=in_channels, out_depth=volume_depth, spatial=64
        )
        self.tokenizer = VoxelToClusterTokenizer(
            num_clusters=num_clusters,
            brain_dim=brain_dim,
            in_channels=volume_depth,
            v2c_path=v2c_path,
        )
        self.official_decoder = None
        if prefer_official_decoder and project_root is not None:
            self._try_load_official(Path(project_root))

    def _try_load_official(self, root: Path) -> None:
        path = _maybe_add_spec2vol_path(root)
        if path is None:
            return
        try:
            from Spectrogram2fMRI import fMRIDecoder  # type: ignore

            self.official_decoder = fMRIDecoder(in_dim=256, out_dim=32, ablation=False)
            print(f"[INFO] Using official Spec2Vol VMUNet decoder from {path}")
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Spec2Vol official decoder unavailable ({exc}); using lightweight decoder")

    def forward(self, feat_map: torch.Tensor) -> dict[str, torch.Tensor]:
        volume = None
        if self.official_decoder is not None:
            try:
                volume = self.official_decoder(feat_map)
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] Official Spec2Vol decoder failed ({exc}); using lightweight decoder")
                self.official_decoder = None
        if volume is None:
            volume = self.decoder(feat_map)
        brain_tokens = self.tokenizer(volume)
        return {"virtual_fmri": volume, "brain_tokens": brain_tokens}

    @classmethod
    def from_config(cls, cfg: dict[str, Any], project_root: str | None = None) -> "VirtualFMRIBranch":
        return cls(
            in_channels=int(cfg.get("in_channels", 256)),
            volume_depth=int(cfg.get("volume_depth", 32)),
            num_clusters=int(cfg.get("num_clusters", 128)),
            brain_dim=int(cfg.get("brain_dim", 1024)),
            v2c_path=cfg.get("v2c_path"),
            prefer_official_decoder=bool(cfg.get("prefer_official_decoder", True)),
            project_root=project_root,
        )
