"""Central path resolution for the sub-08 LOSO pipeline.

Every module imports its locations from here rather than recomputing them, so the
on-disk layout has exactly one definition.  All paths come from the environment
variables set by `env.sh`; importing this module without sourcing `env.sh` raises
immediately instead of silently writing into the wrong tree.
"""
from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "PROJECT_ROOT",
    "LOSO_ROOT",
    "DATA_ROOT",
    "ASSET_ROOT",
    "OUT_ROOT",
    "LOG_ROOT",
    "EEG_SRC",
    "IMG_SRC",
    "HF_HUB",
    "CLIP_ID",
    "DINO_ID",
    "VAE_ID",
    "BLIP2_ID",
    "SD_ID",
    "IP_ADAPTER_ID",
    "THINGS_DIR",
    "CAPTION_DIR",
    "TARGET_DIR",
    "CKPT_DIR",
    "ALIGN_DIR",
    "DIFFUSION_DIR",
    "CALIB_DIR",
    "EVAL_DIR",
    "ensure_dirs",
]


def _req(name: str) -> Path:
    """Read a required environment variable as a Path.

    Failing loudly here is deliberate: these variables encode which tree is being
    written to, and a default would turn a missing `source env.sh` into data
    landing in an unrelated directory.
    """
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"{name} is unset; source the pipeline environment first:\n"
            f"  source {os.environ.get('LOSO_ROOT', 'third_party/loso_pipeline')}/env.sh"
        )
    return Path(value)


PROJECT_ROOT: Path = _req("PROJECT_ROOT")
LOSO_ROOT: Path = _req("LOSO_ROOT")

DATA_ROOT: Path = _req("LOSO_DATA")
ASSET_ROOT: Path = _req("LOSO_ASSETS")
OUT_ROOT: Path = _req("LOSO_OUT")
LOG_ROOT: Path = _req("LOSO_LOG")

# Frozen inputs owned by other projects in the workspace.
EEG_SRC: Path = _req("EEG_SRC")
IMG_SRC: Path = _req("IMG_SRC")
HF_HUB: Path = _req("HUGGINGFACE_HUB_CACHE")

# Pinned model ids (see env.sh).
CLIP_ID: str = os.environ.get("LOSO_CLIP_ID", "laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
DINO_ID: str = os.environ.get("LOSO_DINO_ID", "vit_large_patch14_dinov2.lvd142m")
VAE_ID: str = os.environ.get("LOSO_VAE_ID", "stabilityai/sdxl-vae")
BLIP2_ID: str = os.environ.get("LOSO_BLIP2_ID", "Salesforce/blip2-opt-2.7b")
SD_ID: str = os.environ.get("LOSO_SD_ID", "stabilityai/stable-diffusion-xl-base-1.0")
IP_ADAPTER_ID: str = os.environ.get("LOSO_IP_ADAPTER_ID", "h94/IP-Adapter")

# --- this pipeline's own sub-trees ------------------------------------------
THINGS_DIR: Path = DATA_ROOT / "things"
CAPTION_DIR: Path = DATA_ROOT / "captions"
TARGET_DIR: Path = DATA_ROOT / "targets"
CKPT_DIR: Path = OUT_ROOT / "ckpt"

# Stages write here; keeping them separate makes a partial rerun unambiguous.
ALIGN_DIR: Path = OUT_ROOT / "align"
DIFFUSION_DIR: Path = OUT_ROOT / "diffusion"
CALIB_DIR: Path = OUT_ROOT / "calib"
EVAL_DIR: Path = OUT_ROOT / "eval"


def ensure_dirs() -> None:
    """Create every output directory this pipeline writes into."""
    for path in (
        DATA_ROOT, ASSET_ROOT, THINGS_DIR, CAPTION_DIR, TARGET_DIR,
        OUT_ROOT, LOG_ROOT, CKPT_DIR, ALIGN_DIR, DIFFUSION_DIR, CALIB_DIR, EVAL_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


# --- THINGS-EEG geometry -----------------------------------------------------
# These are properties of the released dataset, not tunables.  They are asserted
# against the actual .npy headers at load time so a mismatched copy is caught
# before any features are written.
N_TRAIN_CONCEPTS = 1654
N_IMAGES_PER_CONCEPT = 10
N_TRAIN_REPS = 4
N_TEST_CONCEPTS = 200
N_TEST_REPS = 80
N_CHANNELS = 63
N_TIMES = 250          # 250 Hz over the 1 s post-stimulus window
SFREQ = 250.0
