"""POLARIS on CogCapPro -- shared configuration.

Design doc: eeg-retrieval/docs/DESIGN_POLARIS_inter_reconstruction.md

This package vendors CogCapPro's *model* code (pure torch/einops, no Lightning, no
diffusers) so that the architecture is reproduced faithfully while the training loop,
the inter-subject mechanisms and the deployment recovery are ours.

Why vendored rather than imported
---------------------------------
CogCapPro's own requirements pin diffusers==0.36.0 / transformers==4.57.6 / torch==2.5.0
(third_party/CognitionCapturerPro/requirements.txt), while the shared venv this project
runs on is diffusers==0.31.0 / transformers==4.46.3 / torch==2.13.0. Upgrading the shared
venv would risk the working eeg-brainit SDXL+IP-Adapter reconstruction stack. CogCapPro's
model files import only torch/einops/numpy/math, so they vendor cleanly and the version
conflict never arises.

The condition space
-------------------
`clip_h14_ip_adapter/{clip_h14_train,clip_h14_test}.npy` are
`open_clip.encode_image -> projected image_embeds` for ViT-H-14 laion2b_s32b_b79k,
i.e. exactly the 1024-d space `ip-adapter_sdxl_vit-h` conditions on. Its manifest records
a listing gate (16540/16540 compared, match) tying row order to `image_metadata.npy`.
That is why this file can treat them as ground-truth conditions rather than as
"features we hope are aligned".
"""
from __future__ import annotations

import os
from pathlib import Path


def redirect_caches() -> None:
    """Redirect HF/torch caches into the project tree. Must precede torch/open_clip import."""
    base = "/project/peilab/why/cache/eeg-brainit"
    os.environ.setdefault("HF_HOME", f"{base}/hf")
    os.environ.setdefault("HF_HUB_CACHE", f"{base}/hf/hub")
    os.environ.setdefault("OPENCLIP_CACHE_DIR", f"{base}/open_clip")
    os.environ.setdefault("TORCH_HOME", f"{base}/torch")
    os.environ.setdefault("XDG_CACHE_HOME", "/project/peilab/why/cache/xdg")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


redirect_caches()

ROOT = Path(__file__).resolve().parents[2]           # .../eeg-retrieval
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
COGCAP_OUT = OUTPUTS / "cogcap"

EEG_DIR = DATA / "preprocessed_eeg"
IMAGE_FEATURE_DIR = DATA / "image_feature"
IP_ADAPTER_FEATURE_DIR = IMAGE_FEATURE_DIR / "clip_h14_ip_adapter"
IMAGES_SET = DATA / "images_set"
IMAGE_METADATA = IMAGES_SET / "image_metadata.npy"

# ---------------------------------------------------------------- geometry
# preprocessed_eeg/sub-XX/train.npy  [1654, 10, 4, 63, 250]
# preprocessed_eeg/sub-XX/test.npy   [200,  1, 80, 63, 250]
N_TRAIN_CONCEPTS = 1654
N_TEST_CONCEPTS = 200
N_IMAGES_PER_CONCEPT = 10
N_TRAIN_REPS = 4
N_TEST_REPS = 80
N_CHANNELS = 63
N_TIMEPOINTS = 250
SFREQ = 250.0
TEST_WAY = N_TEST_CONCEPTS

# Number of distinct stimuli (image instances), not concepts. This is CogCapPro's
# `img_index` granularity and therefore the granularity of the multi-positive groups.
N_TRAIN_STIMULI = N_TRAIN_CONCEPTS * N_IMAGES_PER_CONCEPT      # 16540
N_TEST_STIMULI = N_TEST_CONCEPTS                                # 200 (1 image/concept)

# ---------------------------------------------------------------- from CogCapPro
# brain_backbone.py: EEGProjectLayer_multimodal_cogcap_list(timesteps=[0,250]) builds
# Cogcap(sequence_length=timesteps[1]=250); Proj_eeg flattens 36 (time) x 40 (ch) = 1440.
TIMESTEPS = (0, 250)
EMBED_DIM = 1440
Z_DIM = 1024
# CogCapPro's ClipLoss_Modified_DDP(top_k=10, cos_batch=512) -- training/module.py:129
TOP_K = 10
COS_BATCH = 512

SUBJECTS = list(range(1, 11))
CLIP_ARCH = "ViT-H-14"
CLIP_PRETRAINED = "laion2b_s32b_b79k"


def subject_dir(subject_id: int) -> Path:
    return EEG_DIR / f"sub-{subject_id:02d}"


def modality_feature_path(modality: str, split: str) -> Path:
    """Per-modality condition features, laid out like the IP-Adapter cache."""
    return IMAGE_FEATURE_DIR / f"cogcap_{modality}" / f"{modality}_{split}.npy"


def default_modalities() -> list[str]:
    """Modalities whose conditions exist as of this writing.

    `image` is the eeg-brainit IP-Adapter cache. `depth`/`edge` are produced by
    `cogcap.prep_features`. `text` is deliberately absent: CogCapPro's text target is a
    BLIP2 caption encoded by the same CLIP text tower, and this project's caption files
    are SDXL prompts keyed by concept, not the per-image BLIP2 texts the upstream loader
    expects (`data/eeg.py:264-269` reads weights/texts/eeg/texts_BLIP2_{mode}.npy). Using
    them would silently change what the text branch is aligned to, so it is left out
    rather than approximated.
    """
    return ["image", "depth", "edge"]
