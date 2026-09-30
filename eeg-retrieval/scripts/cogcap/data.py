"""LOSO data for POLARIS on CogCapPro.

Layout conventions, all of which matter downstream:

* Rows are **subject-major**: row `r` of the training tensor belongs to source subject
  `r // n_stimuli` and carries stimulus id `r % n_stimuli`. Condition targets are therefore
  expanded with `np.tile`, not `np.repeat`. (`np.repeat` silently pairs every stimulus with
  the wrong subject; the multi-positive loss would then be grouping the wrong rows and every
  inter-subject number would be meaningless. EPD's `test_epd_loso.py` guards the same trap.)
* `train_avg` / `test_avg` are True, matching `configs/cogcappro.yaml:35-36`, so the 4 train
  repetitions and 80 test repetitions are averaged away *before* the model sees them. This
  is why the per-stimulus row count is the number of subjects, not 4x that.
* Per-subject standardisation is computed from that subject's own *train* split and applied
  to both splits. For the held-out subject this uses no labels, so it is a legitimate
  transductive step and is the same thing the dataset's offline whitening does.

The 4 reps x 16540 stimuli x 63 ch x 250 t of raw EEG is ~4.2 GB per subject on disk
(ten subjects => ~42 GB read per run). `ensure_eeg_cache` averages once into
`outputs/cogcap/cache/`, after which a run reads ~1 GB per subject.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from cogcap import config


# ------------------------------------------------------------------ EEG cache
def cache_path(subject: int, split: str, concepts: int = 0) -> Path:
    tag = f"_c{concepts}" if concepts else ""
    return config.COGCAP_OUT / "cache" / f"sub-{subject:02d}_{split}_avg{tag}.npy"


def ensure_eeg_cache(subject: int, split: str, force: bool = False, concepts: int = 0) -> Path:
    """Average the repetitions once and store the result.

    Also standardises per channel. The statistics come from the subject's own train split
    for both splits, so train/test see one common scale; using per-split statistics would
    leak the test set's own mean into the test representation and would make the deployment
    recovery (which must be fit on train and applied to test) inconsistent with training.

    `concepts` truncates the *concept axis of the memmap before any averaging*, so a smoke
    run reads a few MB instead of 4.2 GB per subject. Slicing after reshaping would
    materialise the whole array again and make the flag pointless.
    """
    out = cache_path(subject, split, concepts)
    meta = out.with_suffix(".json")
    if out.exists() and meta.exists() and not force:
        return out

    src = config.subject_dir(subject) / f"{split}.npy"
    arr = np.load(src, mmap_mode="r")
    if concepts:
        arr = arr[:concepts]
    # train [1654, 10, 4, C, T] -> [16540, C, T];  test [200, 1, 80, C, T] -> [200, C, T]
    flat = arr.reshape(-1, arr.shape[-3], arr.shape[-2], arr.shape[-1])
    averaged = flat.mean(axis=1).astype(np.float32)              # [n, C, T]

    stats_src = config.subject_dir(subject) / "train.npy"
    if split == "train":
        s_arr = arr
    else:
        s_arr = np.load(stats_src, mmap_mode="r")
        if concepts:
            s_arr = s_arr[:concepts]
    s_flat = s_arr.reshape(-1, s_arr.shape[-3], s_arr.shape[-2], s_arr.shape[-1])
    s_avg = s_flat.mean(axis=1)
    mean = s_avg.mean(axis=(0, 2)).astype(np.float32)            # per channel
    std = s_avg.std(axis=(0, 2)).astype(np.float32)
    std = np.maximum(std, 1e-6)

    averaged = (averaged - mean[None, :, None]) / std[None, :, None]

    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, averaged)
    meta.write_text(json.dumps({
        "subject": subject, "split": split, "shape": list(averaged.shape),
        "source": str(src), "averaged_reps": True, "concepts": concepts or "all",
        "standardize": "per-channel, statistics from the subject's own train split",
        "channel_mean": mean.tolist(), "channel_std": std.tolist(),
    }, indent=2))
    return out


# ------------------------------------------------------------------ features
def load_modality_features(modality: str, split: str, suffix: str = "") -> torch.Tensor:
    """[n_concepts, n_images, 1024] ground-truth conditions for one modality.

    `suffix` is applied to the *generated* modalities only. `image` is the upstream
    eeg-brainit IP-Adapter cache (`clip_h14_ip_adapter`, 67 MB, produced and listing-gated
    by `recon/extract_clip_h14.py`); it is not ours to truncate, and a smoke run does not
    need to, since `[.. : n_stim]` on a concept-major array already yields exactly the same
    leading stimuli the truncated depth/edge arrays hold.
    """
    if modality == "image":
        path = config.IP_ADAPTER_FEATURE_DIR / f"clip_h14_{split}.npy"
    else:
        path = config.IMAGE_FEATURE_DIR / f"cogcap_{modality}" / f"{modality}_{split}{suffix}.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"missing {modality}/{split} conditions at {path}. "
            f"Run `python -m cogcap.prep_features --modalities {modality}` first."
        )
    arr = np.load(path)
    return torch.from_numpy(np.ascontiguousarray(arr)).float()


class LOSOData:
    """One LOSO fold: S source subjects train, the held-out subject is evaluated."""

    def __init__(self, sources, target, modalities, limit: int = 0, seed: int = 0,
                 target_recovery_rows: int = 0, verbose: bool = True,
                 feature_suffix: str = ""):
        assert target not in sources, "target subject must not appear in sources"
        self.sources = list(sources)
        self.target = int(target)
        self.modalities = list(modalities)
        self.feature_suffix = feature_suffix
        self.S = len(self.sources)
        self.rng = np.random.default_rng(seed)

        if verbose:
            print(f"[data ] fold {self.sources} -> target {self.target} | modalities {self.modalities}")

        # A smoke run must not pay the full ~42 GB cache read. Truncating the concept axis
        # is the only place that can be done without materialising the whole array.
        self.cache_concepts = (int(np.ceil(limit / config.N_IMAGES_PER_CONCEPT))
                               if limit else 0)

        # ---- source training split
        tr = []
        for s in self.sources:
            tr.append(np.load(ensure_eeg_cache(s, "train", concepts=self.cache_concepts),
                              mmap_mode="r"))
        n_stim = tr[0].shape[0]
        assert all(a.shape[0] == n_stim for a in tr), "source subjects differ in stimulus count"
        if limit:
            n_stim = min(n_stim, limit)
        # subject-major: [S, n_stim, C, T] -> [S*n_stim, C, T]
        self.train_eeg = torch.from_numpy(
            np.concatenate([np.asarray(a[:n_stim]) for a in tr], axis=0)
        ).float()
        self.n_stim = n_stim
        self.n_source_rows = self.S * n_stim
        self.train_subj = torch.arange(self.S).repeat_interleave(n_stim)
        stim = torch.arange(n_stim).repeat(self.S)
        self.train_stim = stim
        self.train_concept = torch.div(stim, config.N_IMAGES_PER_CONCEPT, rounding_mode="floor")

        feats = {m: load_modality_features(m, "train", self.feature_suffix)
                 for m in self.modalities}
        # Flatten to stimuli BEFORE slicing. Slicing the [concept, image, 1024] array first
        # would cut the concept axis, which is the same thing only when the slice covers
        # every concept -- i.e. it would work in a full run and corrupt a truncated one.
        seq = {m: f.reshape(-1, f.shape[-1]) for m, f in feats.items()}
        # subject-major expansion. `torch.tile` not `repeat`: repeat would pair stimulus `s`
        # with subject `s`, which is a silently wrong multi-positive grouping.
        self.train_mod = {
            m: torch.tile(seq[m][:n_stim], (self.S, 1)) for m in self.modalities
        }

        # stimulus ids must be global so that the concatenated [subject, stimulus] identity
        # is what the multi-positive loss groups on; the stimulus index alone is already
        # unique per image here, so no concept offset is needed.
        assert self.train_stim.numel() == self.train_eeg.shape[0]

        # ---- source validation split (their own test split, 200-way)
        v = [np.load(ensure_eeg_cache(s, "test"), mmap_mode="r") for s in self.sources]
        n_val = v[0].shape[0]
        self.val_eeg = torch.from_numpy(np.concatenate([np.asarray(a) for a in v], 0)).float()
        self.val_stim = torch.arange(n_val).repeat(self.S)
        self.val_subj = torch.arange(self.S).repeat_interleave(n_val)
        vfeat = {m: load_modality_features(m, "test", self.feature_suffix)
                 for m in self.modalities}
        vseq = {m: f.reshape(-1, f.shape[-1]) for m, f in vfeat.items()}
        self.val_mod = {
            m: torch.tile(vseq[m][:n_val], (self.S, 1)) for m in self.modalities
        }

        # ---- held-out subject
        self.test_eeg = torch.from_numpy(
            np.array(np.load(ensure_eeg_cache(self.target, "test"), mmap_mode="r"),
                     dtype=np.float32)
        ).float()
        self.test_stim = torch.arange(self.test_eeg.shape[0])
        self.test_subj = torch.full((self.test_eeg.shape[0],), -1, dtype=torch.long)
        self.test_mod = {m: vseq[m][:n_val] for m in self.modalities}
        assert self.test_eeg.shape[0] == self.test_mod[self.modalities[0]].shape[0]

        # ---- held-out subject's *unlabeled* train split, for fitting the recovery
        ttr = np.load(ensure_eeg_cache(self.target, "train",
                                       concepts=self.cache_concepts), mmap_mode="r")
        n_rec = ttr.shape[0] if not target_recovery_rows else min(target_recovery_rows, ttr.shape[0])
        self.recovery_eeg = torch.from_numpy(
            np.array(ttr[:n_rec], dtype=np.float32)
        ).float()
        self.recovery_stim = torch.arange(n_rec)
        # These are the *labels* for the recovery rows. The label-free protocol never reads
        # them; they exist for the `--pair-mode oracle` ceiling arm only.
        self.recovery_mod = {m: seq[m][:n_rec] for m in self.modalities}
        if verbose:
            print(f"[data ] train rows {self.train_eeg.shape} | val {self.val_eeg.shape} "
                  f"| target test {self.test_eeg.shape} | recovery fit {self.recovery_eeg.shape}")

    def batches(self, split: str, batch_size: int, shuffle: bool, seed: int = 0):
        n = self.train_eeg.shape[0] if split == "train" else self.val_eeg.shape[0]
        idx = np.arange(n)
        if shuffle:
            np.random.default_rng(seed).shuffle(idx)
        for i in range(0, n, batch_size):
            sel = idx[i:i + batch_size]
            yield self._gather(split, sel)

    def _gather(self, split: str, sel):
        t = torch.as_tensor(sel, dtype=torch.long)
        if split == "train":
            return {
                "eeg": self.train_eeg[t],
                "stim": self.train_stim[t],
                "subj": self.train_subj[t],
                "concept": self.train_concept[t],
                "mod": {m: self.train_mod[m][t] for m in self.modalities},
            }
        return {
            "eeg": self.val_eeg[t],
            "stim": self.val_stim[t],
            "subj": self.val_subj[t],
            "concept": torch.div(self.val_stim[t], 1, rounding_mode="floor"),
            "mod": {m: self.val_mod[m][t] for m in self.modalities},
        }

    def diagnostics(self) -> dict:
        """The numbers design doc §7.2 asks to report rather than assume."""
        return {
            "n_sources": self.S,
            "n_stimuli": int(self.n_stim),
            "n_source_rows": int(self.n_source_rows),
            "rows_per_stimulus": self.S,
            "recovery_fit_rows": int(self.recovery_eeg.shape[0]),
            "modalities": self.modalities,
        }


def load_subject_raw(subject: int, split: str) -> np.ndarray:
    """Unaveraged [n_img, n_rep, C, T] -- needed only for noise-covariance estimation."""
    arr = np.load(config.subject_dir(subject) / f"{split}.npy", mmap_mode="r")
    return arr.reshape(-1, arr.shape[-3], arr.shape[-2], arr.shape[-1])
