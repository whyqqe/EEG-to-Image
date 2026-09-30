"""A batch sampler that puts whole repeated-trial groups in one batch.

Why this exists
---------------
`ClipLossModified` derives its pseudo-label from the frozen targets: the k most
similar targets in the batch, intersected with the same image.  That only carries
information if a batch actually contains several trials of one image.  The official
dataloader cannot arrange that (subject-major indexing plus 1024 batches inside
1654 rows per subject), so its soft label silently reduces to a one-hot and the
recipe degenerates to plain InfoNCE.

This sampler closes that gap by construction: it is given the dataset's grouping
(image slot -> the indices of its trials) and emits batches made of whole groups.
A batch of `batch_size` is then `batch_size // group_size` complete groups, so

  * every sample has `group_size - 1` same-image positives, all with target
    similarity exactly 1.0, so the top-k step puts the whole group in the label;
  * the number of distinct images per batch is `batch_size // group_size`, and
    every one of them appears as a negative for every other -- negative diversity
    is reduced but not destroyed.

For the averaged protocol `group_size` is the number of training subjects (9), so
a 1024 batch holds 113 distinct images.  For single-trial it is
`subjects * reps` (36), so 28 distinct images -- a real narrowing of the negative
set, and therefore of the effective task difficulty.  That trade is deliberate:
the previous run's collapse came in part from a pseudo-label that was one-hot for
75.8% of the batch, and a gradient that says "this trial's own image is the only
positive in a pool of unrelated photographs" is a much weaker signal than "these
36 trials of one photograph should agree".

`--sampler shuffled` restores the literal official behaviour for comparison.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterator, Sequence

import numpy as np
from torch.utils.data import Sampler


def group_indices(groups: Sequence[int]) -> dict[int, list[int]]:
    """Flat per-sample group ids -> {group_id: [sample indices, in order]}."""
    out: dict[int, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        out[int(group)].append(index)
    return dict(out)


class GroupedBatchSampler(Sampler[list[int]]):
    """Yields batches of complete groups.

    `drop_last` is applied implicitly: any trailing partial batch is discarded,
    because a short batch would give its samples a different (smaller) negative
    set and a different soft-label normalisation than the rest.
    """

    def __init__(self, groups: Sequence[int], batch_size: int, *,
                 shuffle: bool = True, seed: int = 0, drop_last: bool = True):
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self._by_group = group_indices(groups)

        sizes = {len(v) for v in self._by_group.values()}
        if len(sizes) != 1:
            # Mixed group sizes would make the batch-to-group arithmetic below
            # depend on which groups happened to be drawn, so the loss would see a
            # batch whose positive count varies between steps.
            raise ValueError(
                f"group sizes are not uniform: saw {sorted(sizes)}. "
                "Every image must contribute the same number of trials for a "
                "grouped batch to have a fixed positive count.")
        self.group_size = sizes.pop() if sizes else 1

        per_batch = self.batch_size // self.group_size
        if per_batch < 2:
            raise ValueError(
                f"batch_size {self.batch_size} / group_size {self.group_size} "
                f"= {per_batch} groups per batch; need at least 2 distinct images "
                "or every batch is a single group and all other rows are positives.")
        self.groups_per_batch = per_batch
        self.n_batches = len(self._by_group) // per_batch if drop_last \
            else -(-len(self._by_group) // per_batch)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self) -> Iterator[list[int]]:
        keys = sorted(self._by_group)
        rng = np.random.default_rng((self.seed, self.epoch))
        if self.shuffle:
            keys = list(rng.permutation(keys))

        usable = self.groups_per_batch * self.n_batches
        keys = keys[:usable]

        for start in range(0, usable, self.groups_per_batch):
            batch: list[int] = []
            for key in keys[start:start + self.groups_per_batch]:
                members = self._by_group[key]
                if self.shuffle:
                    # Shuffling inside the group stops position-in-batch from
                    # becoming a proxy for subject, which matters because the
                    # subject-wise linear is the only per-subject parameter.
                    members = [members[i] for i in rng.permutation(len(members))]
                batch.extend(members)
            yield batch
