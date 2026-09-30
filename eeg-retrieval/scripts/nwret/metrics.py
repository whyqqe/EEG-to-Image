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
