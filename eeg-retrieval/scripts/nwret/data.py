"""Data loading and the zero-shot retrieval protocol.

Protocol (mirrors third_party/SAMGA so numbers are comparable)
-------------------------------------------------------------
  repetitions are averaged, so
      train (1654, 10, 4, 63, 250) -> (1654, 10, 63, 250)
      test  (200,  1, 80, 63, 250) -> (200,  1, 63, 250)
  Test therefore yields one trial per concept and retrieval is 200-way with the
  correct pairing on the diagonal.

Leak-free model selection
-------------------------
SAMGA selects its checkpoint by **test** Top-1. That is a leak, so we do not do
it. Instead a set of training *concepts* is held out as validation, and test is
scored exactly once for the reported number:

    fit  : 1654 - n_val training concepts
    val  : n_val training concepts        (checkpoint selection, layer scans, ...)
    test : 200 test concepts              (scored once, never for selection)

The split is concept-level, so no image of a validation concept is ever trained on.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from . import config


# ---------------------------------------------------------------- raw loading
def _average_reps(arr: np.ndarray) -> np.ndarray:
    """(C, I, R, Ch, T) -> (C, I, Ch, T), averaging repetitions."""
    return arr.mean(axis=2).astype(np.float32)


def load_subject(
    subject_id: int,
    channels: list[str] | None = None,
    cache_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one subject's averaged train/test EEG, restricted to `channels`.

    Returns (train (1654, 10, Ch, T), test (200, 1, Ch, T)).
    """
    cache_dir = cache_dir or (config.OUTPUTS / "cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = f"sub{subject_id:02d}_{'all63' if channels is None else f'{len(channels)}ch'}"
    tr_path = cache_dir / f"eeg_train_{tag}.npy"
    te_path = cache_dir / f"eeg_test_{tag}.npy"

    if tr_path.is_file() and te_path.is_file():
        return np.load(tr_path), np.load(te_path)

    sdir = config.subject_dir(subject_id)
    info = json.loads((config.EEG_DIR / "info.json").read_text())
    all_ch = info["ch_names"]

    idx = None
    if channels is not None:
        missing = [c for c in channels if c not in all_ch]
        if missing:
            raise KeyError(f"channels not in montage: {missing}")
        idx = [all_ch.index(c) for c in channels]

    out = []
    for fn, path in (("train", sdir / "train.npy"), ("test", sdir / "test.npy")):
        arr = _average_reps(np.load(path))
        if idx is not None:
            arr = arr[..., idx, :]
        out.append(arr)

    tr, te = out
    np.save(tr_path, tr)
    np.save(te_path, te)
    return tr, te


# ---------------------------------------------------------------- splits
@dataclass
class Split:
    fit_concepts: np.ndarray
    val_concepts: np.ndarray


def concept_split(n_val: int = 150, seed: int = 2025) -> Split:
    """Concept-level split of the 1654 training concepts."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(config.N_TRAIN_CONCEPTS)
    val = np.sort(perm[:n_val])
    fit = np.sort(perm[n_val:])
    assert not (set(val.tolist()) & set(fit.tolist()))
    return Split(fit_concepts=fit, val_concepts=val)


# ---------------------------------------------------------------- datasets
class TrainDataset(Dataset):
    """Rows are (concept, image) pairs; targets are frozen image features.

    The image features come from the ViT-H-14 cache and already live in the
    joint image-text space (post visual.proj), which is exactly where the
    semantic branch is asked to land.

    Augmentation is applied per draw, so the same trial presents differently each
    epoch. It is scoped to this dataset only -- `TestDataset` and the validation
    path never augment, which is what keeps the selection signal honest.
    """

    def __init__(
        self,
        eeg: np.ndarray,                 # (C, I, Ch, T)
        image_feat: np.ndarray,          # (C, I, D)
        concepts: np.ndarray,
        l2norm: bool = True,
        augment=None,                    # callable (x, rng) -> x, or None
        seed: int = 2025,
        slots: list[int] | None = None,
    ) -> None:
        self.eeg = eeg
        self.feat = image_feat
        self.concepts = np.asarray(concepts)
        if l2norm:
            f = self.feat / np.maximum(np.linalg.norm(self.feat, axis=-1, keepdims=True), 1e-8)
            self.feat = f.astype(np.float32)
        # `n_slots_total` is the ON-DISK layout (10 images per concept); `slots` is
        # the subset actually trained on and `n_img` its size. The two have to be
        # kept apart because the structural target caches are addressed by the
        # on-disk layout (`concept * 10 + slot`), so narrowing the training set must
        # not change how a row is computed.
        self.n_slots_total = eeg.shape[1]
        if slots is None:
            self.slots = list(range(self.n_slots_total))
        else:
            bad = [s for s in slots if not (0 <= s < self.n_slots_total)]
            if bad:
                raise ValueError(f"slots {bad} outside 0..{self.n_slots_total - 1}")
            self.slots = sorted(int(s) for s in slots)
            if not self.slots:
                raise ValueError("slots is empty")
        self.n_img = len(self.slots)
        self.augment = augment
        self.seed = seed
        self._rng: np.random.Generator | None = None

    def _generator(self) -> np.random.Generator:
        """One independent stream per DataLoader worker.

        Workers are forked, so a generator built in __init__ would be duplicated
        across workers and every worker would draw the identical sequence. Seeding
        lazily from the worker id keeps the augmentation streams distinct.
        """
        if self._rng is None:
            wi = torch.utils.data.get_worker_info()
            wid = wi.id if wi is not None else 0
            self._rng = np.random.default_rng(self.seed + 7919 * wid)
        return self._rng

    def __len__(self) -> int:
        return len(self.concepts) * self.n_img

    def __getitem__(self, i: int):
        c = int(self.concepts[i // self.n_img])
        j = self.slots[i % self.n_img]
        x = self.eeg[c, j]
        if self.augment is not None:
            x = self.augment(x, self._generator())
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(self.feat[c, j]), c

    def concept_ids(self) -> np.ndarray:
        return np.repeat(self.concepts, self.n_img)


class TestDataset(Dataset):
    """One averaged trial per concept -> square n_way retrieval.

    `slot` selects which image of each concept to pair against. The test split has
    exactly one image per concept (shape (200, 1, Ch, T)), so slot 0 is the whole
    task. The validation split is (n_conc, 10, Ch, T) -- ten images per concept --
    and slot 0 alone would throw away 90% of the holdout, so the selection path
    sweeps all ten slots and averages. See train.py:evaluate_selection.
    """

    def __init__(self, eeg: np.ndarray, image_feat: np.ndarray,
                 l2norm: bool = True, slot: int = 0) -> None:
        # eeg: (n, n_img, Ch, T) -> (n, Ch, T) for the chosen slot
        self.eeg = eeg[:, slot]
        f = image_feat[:, slot]
        if l2norm:
            f = f / np.maximum(np.linalg.norm(f, axis=-1, keepdims=True), 1e-8)
        self.feat = f.astype(np.float32)

    def __len__(self) -> int:
        return self.eeg.shape[0]

    def __getitem__(self, i: int):
        return torch.from_numpy(self.eeg[i]), torch.from_numpy(self.feat[i]), i


class AuxTargetDataset(TrainDataset):
    """`TrainDataset` plus the structure tower's per-image targets.

    Same rows, same order, same augmentation stream -- the structure targets are
    additional views of the *same* (concept, image) pair, not a second dataset.

    Rows are addressed as `concept_idx * n_img + slot`, which is the layout the
    caches were written in: `build_gt_vae_latents.py` and `build_gt_depth_cache.py`
    both walk `sorted(concept_dirs)` and then `sorted(images_in_dir)`, the same
    order `load_subject()` produces. That correspondence is the whole reason a flat
    memmap can stand in for a keyed lookup, so it is asserted at construction
    rather than trusted (see the `n_rows` check below): a silent row misalignment
    here would pair every image with another image's latent and still train, still
    produce a decreasing loss, and still decode into plausible-looking pictures.
    """

    def __init__(
        self,
        eeg: np.ndarray,
        image_feat: np.ndarray,
        concepts: np.ndarray,
        aux_vae: np.ndarray | None = None,     # (n_concept*n_slots_total, 4, 64, 64)
        aux_depth: np.ndarray | None = None,   # (n_concept*n_slots_total, 64, 64)
        vae_mean: np.ndarray | None = None,    # (4,)  fit-split statistics
        vae_std: np.ndarray | None = None,
        l2norm: bool = True,
        augment=None,
        seed: int = 2025,
        slots: list[int] | None = None,
    ) -> None:
        super().__init__(eeg, image_feat, concepts, l2norm=l2norm, augment=augment,
                         seed=seed, slots=slots)
        n_total_concepts = eeg.shape[0]
        self.aux_vae = aux_vae
        self.aux_depth = aux_depth
        if aux_vae is None and aux_depth is None:
            raise ValueError("AuxTargetDataset needs at least one of aux_vae / aux_depth")

        for name, arr in (("vae", aux_vae), ("depth", aux_depth)):
            if arr is None:
                continue
            # `n_slots_total`, not `n_img`: the caches are keyed by the on-disk
            # layout, which is 10 images per concept even when training uses one.
            # Checking against the narrowed `n_img` would reject a correctly built
            # cache, and -- worse -- using it to compute `row` below would silently
            # read another image's latent.
            if arr.shape[0] != n_total_concepts * self.n_slots_total:
                raise ValueError(
                    f"{name} target has {arr.shape[0]} rows but the EEG layout "
                    f"({n_total_concepts} concepts x {self.n_slots_total} images) implies "
                    f"{n_total_concepts * self.n_slots_total}. The caches are positional, so a "
                    f"row-count mismatch means the pairing is wrong -- refusing to train.")

        if aux_vae is not None:
            if vae_mean is None or vae_std is None:
                raise ValueError("aux_vae requires vae_mean / vae_std from the fit split")
            self.vae_mean = np.asarray(vae_mean, dtype=np.float32).reshape(-1, 1, 1)
            self.vae_std = np.asarray(vae_std, dtype=np.float32).reshape(-1, 1, 1)

    def __getitem__(self, i: int):
        x, f, c = super().__getitem__(i)
        # The on-disk slot, not the position within `slots`, so the row stays the
        # same identity it had before the subsetting.
        slot = self.slots[i % self.n_img]
        row = int(self.concepts[i // self.n_img]) * self.n_slots_total + slot
        out = [x, f, c]
        if self.aux_vae is not None:
            v = np.asarray(self.aux_vae[row], dtype=np.float32)
            v = (v - self.vae_mean) / self.vae_std
            out.append(torch.from_numpy(np.ascontiguousarray(v)))
        if self.aux_depth is not None:
            d = np.asarray(self.aux_depth[row], dtype=np.float32)
            # `.copy()` because the cache is a read-only memmap: torch warns on
            # non-writable arrays (writing would be undefined behaviour) and the
            # warning fires once per DataLoader worker, which is pure log noise.
            # 16 KB per sample against a 64x64 map, so the copy is free here.
            d = np.ascontiguousarray(d).copy()
            out.append(torch.from_numpy(d)[None])                        # (1, 64, 64)
        return tuple(out)
