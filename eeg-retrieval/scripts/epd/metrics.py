"""Retrieval metrics.

`retrieve_all` is byte-for-byte the SAMGA definition (third_party/SAMGA/module/util.py)
so our numbers are directly comparable to the published SOTA. Do not "improve" it
-- a different tie-breaking or normalisation convention would make the comparison
meaningless.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def topk(matrix: np.ndarray, k: int) -> tuple[int, int]:
    sorted_indices = np.argsort(-matrix, axis=1)
    rankings = np.argsort(sorted_indices, axis=1)
    diagonal_ranks = np.diag(rankings) + 1

    count_k = 0
    count_1 = 0
    for i in range(sorted_indices.shape[0]):
        if diagonal_ranks[i] <= k:
            count_k += 1
        if diagonal_ranks[i] == 1:
            count_1 += 1
    return count_k, count_1


def retrieve_all(
    eeg_features: np.ndarray, image_features: np.ndarray, average: bool = True
) -> tuple[int, int, int]:
    """Returns (top5_count, top1_count, n). Identical to SAMGA's implementation."""
    similarity_matrix = cosine_similarity(eeg_features, image_features)
    count_5, count_1 = topk(similarity_matrix, 5)
    return count_5, count_1, eeg_features.shape[0]


def retrieval_report(eeg_features: np.ndarray, image_features: np.ndarray) -> dict:
    c5, c1, n = retrieve_all(eeg_features, image_features)
    return {
        "top1": 100.0 * c1 / n,
        "top5": 100.0 * c5 / n,
        "n": int(n),
    }


def mean_rank(eeg_features: np.ndarray, image_features: np.ndarray) -> float:
    """Mean rank of the correct item; lower is better. More sensitive than Top-1
    when the task is far from saturated, which is the case for inter-subject runs."""
    sim = cosine_similarity(eeg_features, image_features)
    order = np.argsort(-sim, axis=1)
    ranks = np.argsort(order, axis=1)
    return float(np.diag(ranks).mean() + 1.0)


def rank_vector(sim: np.ndarray) -> np.ndarray:
    """Rank of the diagonal entry in each row of a similarity matrix, 1-based.

    Deliberately mirrors `topk`'s convention (`argsort` is stable, and a hit is
    `rank <= k`) rather than replacing it: `retrieve_all` above is pinned to SAMGA's
    implementation and must not be refactored, so the two are kept in agreement by an
    assertion at the call site instead of by shared code.
    """
    order = np.argsort(-sim, axis=1)
    return np.diag(np.argsort(order, axis=1)) + 1


def retrieval_per_concept(eeg_features: np.ndarray, image_features: np.ndarray,
                          ks: tuple[int, ...] = (1, 5)) -> dict:
    """Per-concept Top-k indicators over exactly the matrix the aggregate uses.

    Why this exists
    ---------------
    A 200-way Top-1 carries a standard error of ~3.5 points, so every arm comparison
    on this task was reported with a `min_detectable_diff` of ~9.8 points and every
    pair of arms that differed by less was "indistinguishable". Most of that error is
    NOT noise: concepts differ enormously in how decodable they are, and that
    difficulty is a property of the STIMULUS, so it is shared by every arm. Comparing
    two arms' aggregate Top-1 treats that shared difficulty as if it were independent
    error in each arm, which is what makes the intervals ~14x wider than they need to
    be (the same defect that made the generation arms look identical until the
    per-concept `q_i` decomposition was added).

    Writing the per-concept indicators out lets a later comparison pair them over the
    same 200 concepts and cancel the concept-difficulty term. Pairing does not create
    evidence that is not in the data; it stops the dominant nuisance term from being
    charged to the effect.

    `top{k}` is a list of 0/1 in the same order as the input concepts (which is
    `list_test_images()` order for the test split), and `mean(top1)` reproduces the
    aggregate `top1` exactly -- asserted at the call site rather than assumed.
    """
    sim = cosine_similarity(eeg_features, image_features)
    rk = rank_vector(sim)
    out: dict = {"n": int(sim.shape[0]), "rank": [int(r) for r in rk]}
    for k in ks:
        out[f"top{k}"] = [int(r <= k) for r in rk]
    return out
