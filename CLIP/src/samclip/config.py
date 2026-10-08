"""Paths, data geometry and cache redirection for SAM-CLIP.

`redirect_caches()` MUST be called before importing torch / transformers / open_clip.
`$HOME` is 100% full on this cluster (see AGENTS.md §2.4), so every tool that
defaults to ``~/.cache`` has to be pointed at ``/project/peilab/why/cache`` first.

Importing this module performs the redirection as a side effect, so the safe
pattern is simply::

    from samclip import config  # noqa: F401  (redirects caches on import)
    import torch

which is why every entry point starts with ``from samclip import config``.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------- caching
CACHE_ROOT = Path("/project/peilab/why/cache")

#: Project-local scratch for everything transient. Kept INSIDE the project tree rather
#: than in `$HOME` or `/tmp`: `$HOME` is 100% full on this cluster, and the workspace
#: rule is that only `CLIP/` and the shared model cache are writable (AGENTS.md §1.2),
#: so a scratch dir under `CLIP/` is the one location that is both writable and durable
#: across a job's lifetime.
SCRATCH = Path(__file__).resolve().parents[2] / "scratch"

#: Subdirectories of `SCRATCH` that some library will want to create for itself. Listed
#: explicitly so `redirect_caches` can pre-create them -- a library that cannot create
#: its own config dir fails at import with a confusing error rather than falling back.
_SCRATCH_DIRS = {
    "XDG_CONFIG_HOME": SCRATCH / "config",
    "MPLCONFIGDIR": SCRATCH / "mpl",
}

#: `TMPDIR` is the ONE cache variable that must NOT point at NFS, and this was learned
#: the expensive way rather than assumed. It was briefly pointed at `SCRATCH/tmp` and
#: every training job then ended with ~3600 lines of
#:
#:     OSError: [Errno 16] Device or resource busy: '.nfs000000089f450aef00003343'
#:     ... in multiprocessing/util.py::_remove_temp_dir -> shutil.rmtree
#:
#: The mechanism is an NFS/multiprocessing incompatibility, not a bug in our code:
#: `multiprocessing` creates a temp dir per worker (`get_temp_dir()` honours TMPDIR) and
#: `_remove_temp_dir` deletes it from an *exit finalizer*, while files inside may still be
#: open. On a local filesystem `unlink` on an open file just works; on NFS the file is
#: silly-renamed to `.nfs*` instead, which stays visible, so `rmtree` hits EBUSY and the
#: finalizer raises. It is raised during interpreter shutdown, so training results were
#: never wrong (rc=0, 60 epochs logged) -- but it left 960 stale `pymp-*` directories
#: behind and buried the real log in false tracebacks, which is the actual danger: a log
#: where every job emits 3600 lines of noise is a log where a real error cannot be seen.
#:
#: Note this is NOT `$HOME`, so it does not conflict with the reason caches are
#: redirected -- and it is job-scoped, so concurrent jobs cannot collide in it.
TMPDIR = Path("/tmp") / f"clip-{os.environ.get('SLURM_JOB_ID') or 'local'}"


def redirect_caches() -> None:
    """Point every downloader/cache at shared or project-local storage (idempotent)."""
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    os.environ.setdefault("HF_HOME", str(CACHE_ROOT / "huggingface"))
    os.environ.setdefault("HF_HUB_CACHE", str(CACHE_ROOT / "huggingface" / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(CACHE_ROOT / "huggingface"))
    os.environ.setdefault("OPENCLIP_CACHE_DIR", str(CACHE_ROOT / "open_clip"))
    os.environ.setdefault("TORCH_HOME", str(CACHE_ROOT / "torch"))
    os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT / "xdg"))
    os.environ.setdefault("PIP_CACHE_DIR", str(CACHE_ROOT / "pip"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    # The rest of the "defaults to $HOME" family. These are cheap insurance: on a volume with
    # ZERO bytes free, the failure mode is not a slow run, it is a hard `OSError: [Errno 28]`
    # mid-job (or an import that dies before any of our code runs). Each entry is a library that
    # is either already installed or plausibly added later, and each would otherwise write into
    # `~/.cache` / `~/.config` / `~/.triton` / `~/.cache/huggingface`.
    for var, path in {
        "HF_DATASETS_CACHE": CACHE_ROOT / "huggingface" / "datasets",
        "HF_ASSETS_CACHE": CACHE_ROOT / "huggingface" / "assets",
        "HF_MODULES_CACHE": CACHE_ROOT / "huggingface" / "modules",
        "TORCH_EXTENSIONS_DIR": CACHE_ROOT / "torch" / "extensions",
        "CUDA_CACHE_PATH": CACHE_ROOT / "cuda",
        "TRITON_CACHE_DIR": CACHE_ROOT / "triton",
        "NUMBA_CACHE_DIR": CACHE_ROOT / "numba",
        "XDG_DATA_HOME": CACHE_ROOT / "xdg" / "data",
        "JOBLIB_TEMP_FOLDER": SCRATCH / "joblib",
        "WANDB_DIR": SCRATCH / "wandb",
        "WANDB_CACHE_DIR": SCRATCH / "wandb" / "cache",
    }.items():
        os.environ.setdefault(var, str(path))
        path.mkdir(parents=True, exist_ok=True)
    # `XDG_CONFIG_HOME`/`MPLCONFIGDIR` are closed here and not only in the sbatch headers
    # because those headers do not exist for a login-node or interactive run. They are the
    # two that bite LATER rather than now: nothing in this project imports `matplotlib` or
    # a logging dashboard today, but the moment one is added it writes its font/state
    # cache to `~/.config` -- the same full volume -- and fails at import.
    for var, path in _SCRATCH_DIRS.items():
        os.environ.setdefault(var, str(path))
        path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TMPDIR", str(TMPDIR))
    TMPDIR.mkdir(parents=True, exist_ok=True)


redirect_caches()

# ----------------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[2]          # .../CLIP
SRC = ROOT / "src"
OUTPUTS = ROOT / "outputs"
CONFIGS = ROOT / "configs"
LOGS = ROOT / "logs"

# THINGS-EEG2 (shared, read-only). The EEG lives under `NeuroBridge`; `/project/.../data`
# holds only images and captions -- see the plan doc §3.
SHARED = Path("/project/peilab/why")
DATA_ROOT = SHARED / "NeuroBridge" / "data" / "things_eeg"
EEG_DIR = DATA_ROOT / "preprocessed_eeg"
IMAGE_FEATURE_ROOT = DATA_ROOT / "image_feature"

IMAGES_ROOT = SHARED / "data" / "images_set"
CAPTIONS_DIR = SHARED / "data" / "captions"

# Local scratch for SAM-CLIP's own processed caches (inside CLIP/, per AGENTS.md §1.2).
CACHE_DIR = ROOT / "data" / "cache"

#: v6 route features are written HERE, not into `IMAGE_FEATURE_ROOT`. The shared image
#: feature tree lives under another project's working tree (`NeuroBridge/`), which
#: AGENTS.md §1.1 marks read-only: reading someone else's cache is fine, adding files to
#: it is not. Keeping our derived routes under `CLIP/` also makes them deletable without
#: touching anything another project depends on.
ROUTE_FEATURE_ROOT = ROOT / "data" / "routes"

# The eeg-retrieval project already implements MVNN + coordinate recovery. We
# import (never modify) its `epd` package; this is the path to add to sys.path.
EPD_ROOT = SHARED / "eeg-retrieval" / "scripts"

# --------------------------------------------------------------- data geometry
N_TRAIN_CONCEPTS = 1654
N_TEST_CONCEPTS = 200
N_IMAGES_PER_CONCEPT = 10
N_TRAIN_REPS = 4
N_TEST_REPS = 80
N_CHANNELS = 63
N_TIMEPOINTS = 250
SFREQ = 250.0

# Occipito-parietal subset (used for intra-subject; inter-subject uses all 63).
CHANNELS_OCCIPITO_PARIETAL = [
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8",
    "O1", "Oz", "O2",
]

# ------------------------------------------------------------- image feature sets
# Multi-layer caches available on disk (see plan doc §3.2).
IMAGE_FEATURE_SETS = {
    "clip_h14_multilevel": {
        "dir": IMAGE_FEATURE_ROOT / "clip_h14_multilevel",
        "layers": [20, 24, 28, 32, 36],
        "dim": 1280,
        "pattern": "image_{split}_layer{layer}.npy",   # (1654,10,dim) / (200,1,dim)
    },
    "internvit_multilevel": {
        "dir": IMAGE_FEATURE_ROOT / "internvit_multilevel_20_24_28_32_36",
        "layers": [20, 24, 28, 32, 36],
        "dim": 3200,
        "pattern": "image_{split}_layer{layer}.npy",
    },
    # ---------------------------------------------------------------- v6 routes
    # `docs/eeg2image_v6_architecture.md` §2.2.  These are DIFFERENT frozen visual
    # GEOMETRIES, not more layers of the same one -- the distinction is load-bearing,
    # because our own records show adding a 5th *semantic* layer bought almost nothing
    # while HVF reports that adding a low-level view bought a lot.
    #
    # `dinov2_l14`  -- self-supervised (timm ViT-L/14, DINOv2 weights). A similarity
    #                  geometry trained WITHOUT captions, so it is not collinear with the
    #                  InternViT/CLIP family (CORTIVA's CVR route uses SynCLR for the same
    #                  reason; SynCLR is not in the shared cache, DINOv2 is).
    # `pixel_ll`    -- the LOW-LEVEL route: 24x24 RGB pixels, per-dimension z-scored on
    #                  the train split. This is the slot HVF fills with SDXL-VAE latents
    #                  ("stacking semantic encoders helps little; a pixel/VAE latent
    #                  helps a lot"). A VAE needs `diffusers` (absent) or a new download;
    #                  raw downsampled pixels are the same KIND of low-level signal
    #                  (colour, layout, texture) at zero dependency cost. The honest
    #                  caveat: a VAE latent is a learned compression, this is not.
    "dinov2_l14": {
        "dir": ROUTE_FEATURE_ROOT / "dinov2_l14",
        "layers": [0],
        "dim": 1024,
        "pattern": "image_{split}_layer{layer}.npy",
    },
    "pixel_ll": {
        "dir": ROUTE_FEATURE_ROOT / "pixel_ll_24",
        "layers": [0],
        "dim": 1728,
        "pattern": "image_{split}_layer{layer}.npy",
    },
}

#: v6's heterogeneous routes, in fusion order. Each entry gets its OWN EEG projection,
#: `img_pre` and shared head; scores are fused (see `train.Trainer.assemble`), never the
#: embeddings -- merging embeddings early imposes one similarity geometry and discards
#: the disagreement between the routes, which is the thing being tested.
DEFAULT_ROUTES = [
    {"name": "alpha", "feature_set": "internvit_multilevel",
     "layers": [20, 24, 28, 32, 36], "dim": 3200},
    {"name": "gamma", "feature_set": "dinov2_l14", "layers": [0], "dim": 1024},
    {"name": "beta", "feature_set": "pixel_ll", "layers": [0], "dim": 1728},
]

#: Uniform by default, for a measured reason rather than laziness: CORTIVA's four weight
#: controls put UNIFORM highest (74.22 vs query-aligned 73.5, interval crossing zero) and
#: the Kittler sum-rule is first-order insensitive to the weights' variance.
DEFAULT_FUSION_WEIGHTS = "uniform"


def subject_dir(subject_id: int) -> Path:
    return EEG_DIR / f"sub-{subject_id:02d}"


def all_subjects() -> list[int]:
    return list(range(1, 11))
