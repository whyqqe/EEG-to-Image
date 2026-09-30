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
from .mvnn import Whitener, apply as mvnn_apply, fit_from_blocks


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
    """Concept-level split of the 1654 training concepts.

    `n_val == 0` is a legitimate request, not a degenerate one: it is the
    SOTA protocol. Shallow Alignment states it outright -- "the main experiments
    follow the standard protocol and train on the full training set without a
    validation split" -- and SCORE and Shallow Alignment both report the FINAL
    epoch rather than a selected one. So "no holdout" and "last-epoch
    checkpoint" are one decision, and the empty val set is its representation.
    Callers must then not ask for a selection signal; `evaluate_selection` on an
    empty array raises rather than silently scoring zero concepts.
    """
    if n_val < 0:
        raise ValueError(f"n_val must be >= 0, got {n_val}")
    if n_val == 0:
        return Split(fit_concepts=np.arange(config.N_TRAIN_CONCEPTS),
                     val_concepts=np.empty(0, dtype=int))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(config.N_TRAIN_CONCEPTS)
    val = np.sort(perm[:n_val])
    fit = np.sort(perm[n_val:])
    assert not (set(val.tolist()) & set(fit.tolist()))
    return Split(fit_concepts=fit, val_concepts=val)


# ---------------------------------------------------------------- LOSO
def _channel_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std over trials and time, matching the tokenizer's axis.

    `EEGPatchTokenizer.set_norm_stats` reduces `(n_rows, C, T)` as
    `mean(axis=(0, 2))` -- per channel, with time and trial both marginalised.
    Standardising here with a different axis convention would leave a residual
    scale the tokenizer then could not undo, so the convention is copied rather
    than re-derived.
    """
    x = np.asarray(x, dtype=np.float64)
    flat = x.reshape(-1, x.shape[-2], x.shape[-1])
    mean = flat.mean(axis=(0, 2))
    std = np.maximum(flat.std(axis=(0, 2), ddof=1), 1e-8)
    return mean.astype(np.float32), std.astype(np.float32)


# ---------------------------------------------------------------- MVNN
def _raw_blocks(
    subject_id: int,
    split: str,
    channels: list[str] | None = None,
) -> np.ndarray:
    """One subject's UN-AVERAGED trials as `(n_cond, n_rep, C, T)`.

    The averaged caches cannot feed MVNN: the residual about each condition's mean
    is the only thing in the array that estimates noise, and averaging repetitions
    is exactly the operation that removes it. So this reads the raw file, whose
    layout is documented in `config`: `(1654, 10, 4, 63, 250)` in train and
    `(200, 1, 80, 63, 250)` in test.

    The leading `(concept, image)` axes are collapsed into a single condition axis
    because they are interchangeable here -- a repetition is a repeat of the same
    (concept, image) stimulus -- and the balanced block layout is what
    `mvnn.fit_from_blocks` consumes.
    """
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    arr = np.load(config.subject_dir(subject_id) / f"{split}.npy")
    if channels is not None:
        info = json.loads((config.EEG_DIR / "info.json").read_text())
        all_ch = info["ch_names"]
        missing = [c for c in channels if c not in all_ch]
        if missing:
            raise KeyError(f"channels not in montage: {missing}")
        arr = arr[..., [all_ch.index(c) for c in channels], :]
    return arr.reshape(-1, arr.shape[2], arr.shape[3], arr.shape[4])


def mvnn_whitener(
    subject_id: int,
    split: str,
    channels: list[str] | None = None,
    cache_dir: Path | None = None,
    shrinkage: str = "lw",
    fixed: float = 0.1,
    max_cond: int = 0,
    verbose: bool = True,
) -> Whitener:
    """Fit -- or reload -- one subject's MVNN whitener from one split's residuals.

    Which split is a protocol decision, not an implementation detail, and it is the
    caller's to make (see `load_subject_std`). This function only does what it is
    told and records the provenance in the sidecar.

    Cached as a `(C, C)` matrix plus a JSON sidecar of the diagnostics, because a
    whitener is a preprocessing statistic exactly like a z-score table: it must be
    the SAME matrix that reaches training and evaluation, and re-fitting it per run
    would silently make two arms incomparable if Ledoit-Wolf happened to pick a
    different intensity. The `max_cond` smoke override is part of the cache key for
    the same reason.
    """
    cache_dir = cache_dir or (config.OUTPUTS / "cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = f"sub{subject_id:02d}_{'all63' if channels is None else f'{len(channels)}ch'}"
    key = f"{tag}_{split}_{shrinkage}" + (f"_c{max_cond}" if max_cond else "")
    w_path = cache_dir / f"mvnn_W_{key}.npy"
    j_path = cache_dir / f"mvnn_W_{key}.json"

    if w_path.is_file() and j_path.is_file():
        meta = json.loads(j_path.read_text())
        wh = Whitener(
            w=np.load(w_path), sigma=np.zeros((0, 0), dtype=np.float64),
            lam=float(meta.get("lam", 0.0)), lam_min=float(meta.get("lam_min", 0.0)),
            lam_max=float(meta.get("lam_max", 0.0)), cond=float(meta.get("cond", 0.0)),
            n_trials=int(meta.get("n_trials", 0)), n_cond=int(meta.get("n_cond", 0)),
            n_rep=int(meta.get("n_rep", 0)), n_time=int(meta.get("n_time", 0)),
            shrinkage=str(meta.get("shrinkage", shrinkage)), extra=meta)
        if verbose:
            print(f"[mvnn ] sub-{subject_id:02d} {split}: cached {key} "
                  f"(lam {wh.lam:.4f}, cond {wh.cond:.1f})")
        return wh

    blocks = _raw_blocks(subject_id, split, channels)
    wh = fit_from_blocks(blocks, shrinkage=shrinkage, fixed=fixed,
                         max_cond=max_cond, verbose=False)
    del blocks
    wh.extra = {**wh.as_dict(), "subject": subject_id, "split": split,
                "channels": "all63" if channels is None else len(channels)}
    np.save(w_path, wh.w)
    j_path.write_text(json.dumps(wh.extra, indent=2))
    if verbose:
        print(f"[mvnn ] sub-{subject_id:02d} {split}: {wh.describe()} -> {key}")
    return wh


def load_subject_std(
    subject_id: int,
    channels: list[str] | None = None,
    cache_dir: Path | None = None,
    mvnn: str = "off",
    mvnn_shrinkage: str = "lw",
    mvnn_max_cond: int = 0,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """One subject's EEG: MVNN-whitened (optional), then z-scored per subject.

    SAMGA's preprocessing: "Channel-wise z-score normalization is performed using
    the mean and standard deviation computed from the training split, and the
    same statistics are applied to the test data." The word doing the work is
    *per subject* -- nine subjects concatenated under one global mean/std would
    let the highest-amplitude subject own the scale, which is precisely the
    failure mode cross-subject training has to avoid.

    The statistics are computed from the training split, never the test split, so
    no test-set scale leaks in. They are label-free: they use the EEG, not the
    concept identities, which is why applying them to a held-out subject is
    legitimate under a strict LOSO protocol rather than a hidden calibration.

    Order: MVNN first, then the z-score
    -----------------------------------
    MVNN is a *noise* operation and the z-score is a *scale* operation, and the
    order is not free: the z-score computed after whitening sees already-equalised
    channels and is therefore nearly the identity, which is the intent. Doing it the
    other way round would rescale the channels before the covariance is estimated
    and change what the whitener is equalising.

    `mvnn` selects which split's residuals the whitener is fitted from, and the
    three values are three different protocols, not three spellings of one:

      * `"off"` -- no MVNN. The pipeline's historical behaviour, kept so the
        ablation is one flag and not one code path.
      * `"train"` -- fitted on this subject's own labelled training trials. The
        literature's phrasing ("MVNN is applied to the training data") means this,
        and it is the right choice for the NINE SOURCE subjects.
      * `"test"` -- fitted on this subject's test-trial residuals. Only
        `within-condition` grouping is used (which trials repeated the same
        stimulus), never which stimulus it was, and the standard protocol already
        requires that grouping in order to average repetitions. Required for the
        HELD-OUT subject: fitting on its training split would use data that a strict
        LOSO fold excludes from training.

    Whichever is chosen, the same matrix whitens both returned arrays, so a cache
    entry is a `(subject, channel set, fit split)` and the file names carry all three.
    """
    if mvnn not in ("off", "train", "test"):
        raise ValueError(f"mvnn must be 'off', 'train' or 'test', got {mvnn!r}")
    cache_dir = cache_dir or (config.OUTPUTS / "cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = f"sub{subject_id:02d}_{'all63' if channels is None else f'{len(channels)}ch'}"
    suffix = "_std" if mvnn == "off" else f"_mvnn{mvnn}_std"
    tr_path = cache_dir / f"eeg_train_{tag}{suffix}.npy"
    te_path = cache_dir / f"eeg_test_{tag}{suffix}.npy"

    if tr_path.is_file() and te_path.is_file():
        return np.load(tr_path), np.load(te_path)

    tr, te = load_subject(subject_id, channels, cache_dir)
    if mvnn != "off":
        wh = mvnn_whitener(subject_id, mvnn, channels, cache_dir,
                           shrinkage=mvnn_shrinkage, max_cond=mvnn_max_cond,
                           verbose=verbose)
        tr, te = mvnn_apply(tr, wh), mvnn_apply(te, wh)
    mean, std = _channel_stats(tr)
    shape = (1, 1, -1, 1)
    tr = ((tr - mean.reshape(shape)) / std.reshape(shape)).astype(np.float32)
    te = ((te - mean.reshape(shape)) / std.reshape(shape)).astype(np.float32)
    np.save(tr_path, tr)
    np.save(te_path, te)
    return tr, te


@dataclass
class LosoData:
    """One leave-one-subject-out fold, flattened so that row = (subject, concept).

    Only the EEG lives here. The image side is expanded by the caller
    (`expand_loso_images`) from whatever alignment target that run selected, so a
    LOSO fold can be trained against a single cached layer, a blended multi-layer
    stack, or any `--target-features` directory without this loader having to know
    which. Baking one particular feature file in here is how the intra-subject and
    inter-subject runs would silently end up on different alignment targets.

    Row layout, with `C = 1654` concepts and `S` source subjects:

        row r = subject i = r // C, concept c = r % C

    `subject_of_row` indexes into `source_subjects` (0..S-1), NOT into the
    dataset's absolute subject numbering. A checkpoint trained on one fold must
    not need to know which fold it was, or the 1..10 embedding table would be
    fold-dependent and the arms would not be comparable.
    """

    tr_eeg: np.ndarray             # (S*C, I, Ch, T)
    tr_subject_of_row: np.ndarray  # (S*C,) int64 in 0..S-1
    te_eeg: np.ndarray             # (200, 1, Ch, T)
    source_subjects: list[int]
    target_subject: int

    @property
    def n_subjects(self) -> int:
        return len(self.source_subjects)

    @property
    def n_concepts(self) -> int:
        return self.tr_eeg.shape[0] // self.n_subjects


def expand_loso_images(img_tr: np.ndarray, n_subjects: int) -> np.ndarray:
    """Repeat the concept-major image features once per source subject.

    Image `j` of concept `c` is ONE stimulus, but nine subjects each produced a
    response to it. Since `load_loso` stacks subjects subject-major, the matching
    feature array is the original tiled along the concept axis -- not
    `np.repeat`, which would give `c,c,c,...` rather than `c,c,c` blocks per
    subject.

    The repetition is what makes the cross-subject positives real: with all nine
    copies present, row (subject a, concept c) and row (subject b, concept c) share
    an image feature, so a cross-subject consistency term has something to match on.

    But note what "share an image feature" cannot buy, because it is easy to assume
    the opposite here. Tiling makes the copies BIT-IDENTICAL, so under a pairwise
    InfoNCE the duplicate at column `c` is a copy of row i's own positive at column
    `i`, valued exactly equal to it. Grouping co-stimulus rows is therefore
    necessary but NOT sufficient for multi-positive alignment to do anything: with a
    subject-independent target the multi-positive loss is algebraically EQUAL to the
    pairwise one -- same value, same gradients -- because every duplicate row has
    identical logits and the average over positives collapses to a single term.
    `test_epd_multipos.py` measures this as an exact zero difference.

    Multi-positive alignment only becomes a real objective when the image side
    differs between subjects, which is what SAMGA's subject-aware layer router
    (`--target-fusion routed_sr`) provides and a plain tiled target does not. The
    two switches are coupled; `--multipos` alone is inert.
    """
    if n_subjects < 1:
        raise ValueError(f"n_subjects must be >= 1, got {n_subjects}")
    if n_subjects == 1:
        return img_tr
    reps = (n_subjects,) + (1,) * (img_tr.ndim - 1)
    return np.tile(img_tr, reps)


def load_loso(
    source_subjects: list[int],
    target_subject: int,
    channels: list[str] | None = None,
    cache_dir: Path | None = None,
    mvnn: str = "off",
    verbose: bool = True,
) -> LosoData:
    """Load a LOSO fold's EEG: 9 labelled source subjects, 1 unlabelled target.

    The held-out subject's TRAINING split is deliberately not returned. Strict
    cross-subject means the target subject's labelled pairs never enter training
    (SCORE: "Our strict cross-subject protocol exposes no target subject data
    during training"), and the only thing read from it is the per-channel scale of
    that subject's own signal, which carries no concept information.

    Under `mvnn != "off"` the two roles use different fit splits, and that asymmetry
    is the point rather than an inconsistency: a source subject is whitened by its
    own labelled training residuals, while the held-out subject is whitened by its
    own test residuals, because its training split is exactly what the fold holds
    out. Whitening the target with a SOURCE whitener would be the alternative, and
    it is the wrong one to default to: the noise covariance is a property of the
    electrode impedances of one head, and the inter-subject literature's per-subject
    normalisation exists precisely because that property does not transfer.

    Requires every subject to be standardised (`load_subject_std`), because the
    concatenation is only meaningful if the nine blocks share a scale. The
    assertion is not decoration: nine subjects at their native amplitudes produce a
    pool whose global statistics are dominated by whichever subject recorded
    loudest, and the resulting run trains fine and scores badly for a reason no log
    line mentions.
    """
    if target_subject in source_subjects:
        raise SystemExit(
            f"target subject {target_subject} also appears in --source-subjects "
            f"{source_subjects}; that is not a holdout")
    if not source_subjects:
        raise SystemExit("--source-subjects is empty; nothing to train on")
    if list(dict.fromkeys(source_subjects)) != list(source_subjects):
        raise SystemExit(f"--source-subjects {source_subjects} has duplicates; subject "
                         f"index i maps to a different subject's block, so the "
                         f"per-subject residual would be split across two identities")
    if mvnn not in ("off", "train", "test"):
        raise SystemExit(f"--mvnn must be off/train/test, got {mvnn!r}")

    n_conc = config.N_TRAIN_CONCEPTS
    tr_blocks, subj_blocks = [], []
    for i, s in enumerate(source_subjects):
        tr_s, _ = load_subject_std(s, channels, cache_dir,
                                   mvnn=("train" if mvnn != "off" else "off"))
        if tr_s.shape[0] != n_conc:
            raise SystemExit(
                f"sub-{s:02d} train has {tr_s.shape[0]} concepts, expected {n_conc}")
        tr_blocks.append(tr_s)
        subj_blocks.append(np.full(n_conc, i, dtype=np.int64))

    tr_eeg = np.concatenate(tr_blocks, axis=0)
    tr_sub = np.concatenate(subj_blocks, axis=0)
    del tr_blocks

    _, te_eeg = load_subject_std(target_subject, channels, cache_dir,
                                 mvnn=("test" if mvnn != "off" else "off"))
    if te_eeg.shape[0] != config.N_TEST_CONCEPTS:
        raise SystemExit(
            f"sub-{target_subject:02d} test has {te_eeg.shape[0]} concepts, "
            f"expected {config.N_TEST_CONCEPTS}")

    data = LosoData(tr_eeg=tr_eeg, tr_subject_of_row=tr_sub,
                    te_eeg=te_eeg, source_subjects=list(source_subjects),
                    target_subject=int(target_subject))
    assert_standardised(data)
    if verbose:
        mb = tr_eeg.nbytes / 2 ** 20
        print(f"[loso ] {len(source_subjects)} source subjects "
              f"{source_subjects} -> {tr_eeg.shape} ({mb:.0f} MiB); "
              f"hold out sub-{target_subject:02d} -> {te_eeg.shape} "
              f"({data.n_subjects} subject ids, 1..{data.n_subjects} rows each)")
    return data


def assert_standardised(data: LosoData, tol: float = 5e-3) -> None:
    """Verify the per-subject standardisation actually reached the array.

    Silent failure modes this catches, both of which train fine and report
    plausible numbers: a stale `_std.npy` written before a preprocessing change,
    and the cache being hit for a subject whose `_std` file was never created.
    """
    n_conc = data.n_concepts
    for i, s in enumerate(data.source_subjects):
        blk = data.tr_eeg[i * n_conc:(i + 1) * n_conc]
        got = float(blk.mean())
        sd = float(blk.std())
        if abs(got) > tol or abs(sd - 1.0) > tol:
            raise SystemExit(
                f"sub-{s:02d} is not standardised: mean {got:+.4f} std {sd:.4f} "
                f"(want 0/1 within {tol}). A stale *_std.npy is the usual cause; "
                f"delete outputs/cache/eeg_train_sub{s:02d}_*.npy and re-run.")


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
        subject_of_row: np.ndarray | None = None,   # (C,) int64, LOSO only
        stimulus_of_row: np.ndarray | None = None,  # (C,) int64, LOSO + --multipos
    ) -> None:
        self.eeg = eeg
        self.feat = image_feat
        self.concepts = np.asarray(concepts)
        # The per-row subject index, present only in LOSO. Opt-in rather than
        # always-returned because `__getitem__`'s tuple arity is read positionally
        # by the training loop (`x, f, c = batch[0], batch[1], batch[2]`) and by
        # `AuxTargetDataset`, so widening the tuple unconditionally would shift
        # every existing caller's structural targets by one slot.
        if subject_of_row is not None:
            subject_of_row = np.asarray(subject_of_row)
            if subject_of_row.shape[0] != self.concepts.shape[0]:
                raise ValueError(
                    f"subject_of_row has {subject_of_row.shape[0]} entries but there "
                    f"are {self.concepts.shape[0]} rows; the subject of a row would "
                    f"be read from another row's slot -- silent, and exactly the kind "
                    f"of misalignment that still trains")
        self.subject_of_row = subject_of_row
        # The stimulus index, i.e. WHICH PICTURE a row's image is, in the concept
        # space of the un-tiled feature cache. It cannot be recovered from the image
        # features when the target is subject-dependent (each subject's routed
        # target for one stimulus is a different vector, so "identical features" no
        # longer means "same picture"), which is why it is carried explicitly rather
        # than inferred. See `expand_loso_images`.
        if stimulus_of_row is not None:
            stimulus_of_row = np.asarray(stimulus_of_row)
            if stimulus_of_row.shape[0] != self.concepts.shape[0]:
                raise ValueError(
                    f"stimulus_of_row has {stimulus_of_row.shape[0]} entries but there "
                    f"are {self.concepts.shape[0]} rows; rows would be grouped with "
                    f"another picture's rows, which turns distinct stimuli into "
                    f"positives of each other and is satisfied by collapsing them")
        self.stimulus_of_row = stimulus_of_row
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

    @property
    def n_meta(self) -> int:
        """How many metadata fields `__getitem__` appends after `(x, f, c)`.

        The order is fixed -- subject, then stimulus. Callers that need to slice off
        the structural targets (`AuxTargetDataset`, the training loop) should ask the
        dataset instead of testing `subject_of_row is not None`, because that test is
        a second copy of the arity rule in a second file, and adding the stimulus
        field to only one of the two copies is how a batch silently shifts by one.
        """
        return ((0 if self.subject_of_row is None else 1)
                + (0 if self.stimulus_of_row is None else 1))

    def __getitem__(self, i: int):
        c = int(self.concepts[i // self.n_img])
        j = self.slots[i % self.n_img]
        x = self.eeg[c, j]
        if self.augment is not None:
            x = self.augment(x, self._generator())
        x = torch.from_numpy(np.ascontiguousarray(x))
        f = torch.from_numpy(self.feat[c, j])
        out = [x, f, c]
        if self.subject_of_row is not None:
            out.append(int(self.subject_of_row[c]))
        if self.stimulus_of_row is not None:
            # The GLOBAL stimulus id: concept in the un-tiled space times the on-disk
            # slot count, plus the on-disk slot. On-disk rather than the position
            # within `slots` so that the id stays the same picture's id when the
            # training set is narrowed to a subset of slots -- two rows must be
            # grouped by what they are, not by where they sit in this run.
            out.append(int(self.stimulus_of_row[c]) * self.n_slots_total + j)
        return tuple(out)

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
        subject_of_row: np.ndarray | None = None,
        stimulus_of_row: np.ndarray | None = None,
    ) -> None:
        # `subject_of_row` / `stimulus_of_row` are forwarded, not dropped. Leaving
        # them out was a latent LOSO bug: the training loop slices the structural
        # targets off with `batch[3 + n_meta:]`, so a dataset that silently declined
        # to emit the metadata would put a VAE latent where the loop was reading a
        # subject id -- no exception, just the wrong tensor in the subject embedding.
        super().__init__(eeg, image_feat, concepts, l2norm=l2norm, augment=augment,
                         seed=seed, slots=slots, subject_of_row=subject_of_row,
                         stimulus_of_row=stimulus_of_row)
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
        got = super().__getitem__(i)
        # `TrainDataset` returns (x, f, c) or, in LOSO, (x, f, c, subject). The
        # structural targets are appended after whatever it produced, so this must
        # stay a prefix-unpack rather than an arity-checked unpack.
        x, f, c = got[0], got[1], got[2]
        # The on-disk slot, not the position within `slots`, so the row stays the
        # same identity it had before the subsetting.
        slot = self.slots[i % self.n_img]
        row = int(self.concepts[i // self.n_img]) * self.n_slots_total + slot
        out = list(got)
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
