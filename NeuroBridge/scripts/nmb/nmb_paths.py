"""Shared paths for NeuroMem-Bridge (NMB) pipeline."""
from __future__ import annotations

import os
from pathlib import Path

NB_ROOT = Path(__file__).resolve().parents[2]
BRAIN_HIVE = Path(os.environ.get("BRAIN_HIVE", "/project/peilab/why/Brain-HIVE"))
FUSION_PRIOR = Path(
    os.environ.get("FUSION_PRIOR", "/project/peilab/why/cache/fusion_prior/H14_B32_VAE")
)
CACHE = Path(os.environ.get("NMB_CACHE", "/project/peilab/why/cache/eeg-brainit"))

PROJ_META = {
    "CLIP-ViT-H-14-laion2B-s32B-b79K": 1024,
    "CLIP-ViT-B-32-laion2B-s34B-b79K": 512,
    "vae": 1024,
}
