#!/usr/bin/env python3
"""Held-in validation split that is leak-free by construction.

WHY
---
Audit finding P1: five selection sites chose checkpoints / hyper-parameters by
looking at the 200-concept TEST set:
    train.py:331             save_by_top1 / test loss
    nda_nvol_scan.py:86      sort by test top1
    nda_dual_train.py:396    score = rn50_top1 + 50*fuse_cos + 20*txt_cos, on test_loader
    train_eeg_vae_head.py:173    best MAE on the TEST VAE latents
    train_eeg_depth_head.py:150  best Pearson on the TEST depth maps
Measured cost of that bias at the root encoder: best-on-test 73.0% vs
last-epoch 68.5% (std over the latter half 1.13) -> 3.3-4.5pp of pure inflation.

THIS MODULE
-----------
Carves the 1654 TRAINING concepts into three disjoint sets:
    valA : root-stage selection   (NB encoder, NVOL layer scan)
    valB : downstream selection   (semantic tower, VAE head, depth head)
    fit  : actual gradient updates
Two val sets rather than one because otherwise NVOL would select its layer on a
set the NB encoder had already been selected on, which biases the estimate
(not a test leak, but an avoidable upstream bias).

CRITICAL: THINGS-EEG2 concepts are stored in ALPHABETICAL order, so a
contiguous slice would put "aardvark..axe" in valA and "baboon.." in fit --
semantically clustered and useless as a validation set. We therefore permute.

The TEST set (200 concepts) is never referenced here and stays untouched until
the single final evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

N_CONCEPTS = 1654
REPS = 10  # 1654 concepts x 10 images = 16540 rows


def concept_split(
    n_concepts: int = N_CONCEPTS,
    reps: int = REPS,
    val_a_concepts: int = 83,
    val_b_concepts: int = 82,
    seed: int = 20260910,
) -> dict:
    """Return row-index split (concept-disjoint, alphabetically shuffled)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_concepts)
    a = perm[:val_a_concepts]
    b = perm[val_a_concepts : val_a_concepts + val_b_concepts]
    f = perm[val_a_concepts + val_b_concepts :]

    def rows(c: np.ndarray) -> np.ndarray:
        # concept i occupies rows [reps*i, reps*(i+1))
        return (c[:, None] * reps + np.arange(reps)[None, :]).reshape(-1)

    fit_rows, a_rows, b_rows = np.sort(rows(f)), np.sort(rows(a)), np.sort(rows(b))
    return {
        "seed": seed,
        "n_concepts": n_concepts,
        "reps": reps,
        "fit_concepts": sorted(int(x) for x in f),
        "val_a_concepts": sorted(int(x) for x in a),
        "val_b_concepts": sorted(int(x) for x in b),
        "fit_rows": fit_rows.tolist(),
        "val_a_rows": a_rows.tolist(),
        "val_b_rows": b_rows.tolist(),
        "counts": {"fit": len(f), "val_a": len(a), "val_b": len(b), "total": n_concepts},
        "note": "concept-disjoint, permuted (THINGS concepts are alphabetical); test set never used",
    }


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows_for(split: dict, which: str, n_total: int | None = None) -> np.ndarray:
    """Row indices for 'fit' | 'val_a' | 'val_b'.

    If n_total is given and differs from the split's row count, we scale by
    concepts (some pipelines drop a few rows), keeping the same concept ids.
    """
    key = {"fit": "fit_rows", "val_a": "val_a_rows", "val_b": "val_b_rows"}[which]
    idx = np.asarray(split[key], dtype=np.int64)
    n_split = split["counts"]["total"] * split["reps"]
    if n_total is None or n_total == n_split:
        return idx
    # fall back to concept ids, rescaled by the observed reps
    ckey = {"fit": "fit_concepts", "val_a": "val_a_concepts", "val_b": "val_b_concepts"}[which]
    concepts = np.asarray(split[ckey], dtype=np.int64)
    reps = n_total // split["counts"]["total"]
    assert reps * split["counts"]["total"] == n_total, (reps, n_total)
    return (concepts[:, None] * reps + np.arange(reps)[None, :]).reshape(-1)


def mask_for(split: dict, which: str, n_total: int) -> np.ndarray:
    m = np.zeros(n_total, dtype=bool)
    m[rows_for(split, which, n_total)] = True
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260910)
    a = ap.parse_args()
    s = concept_split(seed=a.seed)
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(s, indent=2), encoding="utf-8")
    c = s["counts"]
    print(f"[OK] held-in split -> {p}")
    print(f"     fit   {c['fit']:>5} concepts ({c['fit']/c['total']:.1%})")
    print(f"     valA  {c['val_a']:>5} concepts  <- root stage (NB encoder, NVOL scan)")
    print(f"     valB  {c['val_b']:>5} concepts  <- downstream (tower, VAE head, depth head)")
    print(f"     TEST  {200:>5} concepts  <- UNTOUCHED, single final eval")
    print("     concept-disjoint + permuted; test set not referenced")


if __name__ == "__main__":
    main()
