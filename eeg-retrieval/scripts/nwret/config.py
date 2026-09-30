"""Shared configuration for the NW-Retrieval pipeline (zero-shot EEG-to-image retrieval).

Hard conventions (see HANDOFF.md):
  * Interpreter: /project/peilab/why/eeg-brainit/.venv/bin/python
  * All caches redirect into /project/peilab/why/cache -- never /home.
  * Login node has no GPU; training must go through sbatch.
"""
from __future__ import annotations

import os
from pathlib import Path


# ---------------------------------------------------------------- cache redirect
# Must run before importing torch/open_clip/timm so they pick these up.
def redirect_caches() -> None:
    base = "/project/peilab/why/cache/eeg-brainit"
    os.environ.setdefault("HF_HOME", f"{base}/hf")
    os.environ.setdefault("HF_HUB_CACHE", f"{base}/hf/hub")
    os.environ.setdefault("OPENCLIP_CACHE_DIR", f"{base}/open_clip")
    os.environ.setdefault("TORCH_HOME", f"{base}/torch")
    os.environ.setdefault("XDG_CACHE_HOME", "/project/peilab/why/cache/xdg")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


redirect_caches()

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[2]          # .../eeg-retrieval
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
THIRD_PARTY = ROOT / "third_party"

EEG_DIR = DATA / "preprocessed_eeg"
IMAGE_FEATURE_DIR = DATA / "image_feature" / "ViT-H-14"

# ---------------------------------------------------------------- data geometry
# train.npy [1654, 10, 4, 63, 250]   test.npy [200, 1, 80, 63, 250]
N_TRAIN_CONCEPTS = 1654
N_TEST_CONCEPTS = 200
N_IMAGES_PER_CONCEPT = 10
N_TRAIN_REPS = 4
N_TEST_REPS = 80
N_CHANNELS = 63
N_TIMEPOINTS = 250
SFREQ = 250.0

# ---------------------------------------------------------------- channels
# Occipito-parietal subset used by the retrieval SOTA (SAMGA/EEGiT-style).
# Reasoning: the visual evoked response is dominated by posterior electrodes,
# and both SAMGA and EEGiT report the occipital region carries most of the signal.
CHANNELS_OCCIPITO_PARIETAL = [
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8",
    "O1", "Oz", "O2",
]

# ---------------------------------------------------------------- protocol
# The zero-shot retrieval protocol follows SAMGA exactly (third_party/SAMGA):
#   repetitions averaged, test -> 200 trials, 200-way diagonal retrieval.
TEST_WAY = N_TEST_CONCEPTS


def subject_dir(subject_id: int) -> Path:
    return EEG_DIR / f"sub-{subject_id:02d}"
