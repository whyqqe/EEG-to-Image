#!/usr/bin/env python3
"""200-way ViT-H gallery retrieval for saved test embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)


def retrieval_metrics(query: np.ndarray, gallery: np.ndarray) -> dict:
    q, g = l2(query.astype(np.float32)), l2(gallery.astype(np.float32))
    sim = q @ g.T
    n = sim.shape[0]
    ranks = []
    hits1 = hits5 = 0
    for i in range(n):
        order = np.argsort(-sim[i])
        rank = int(np.where(order == i)[0][0]) + 1
        ranks.append(rank)
        hits1 += int(rank == 1)
        hits5 += int(rank <= 5)
    paired = float(np.mean([sim[i, i] for i in range(n)]))
    return {
        "n": n,
        "top1": hits1 / n,
        "top5": hits5 / n,
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(np.mean(ranks)),
        "paired_cos": paired,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-npy", type=str, required=True)
    ap.add_argument("--gallery", type=str, required=True)
    ap.add_argument("--output-json", type=str, default="")
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    emb = np.load(args.embed_npy)
    gallery = np.load(args.gallery)
    m = retrieval_metrics(emb, gallery)
    m["tag"] = args.tag or Path(args.embed_npy).stem
    m["embed_npy"] = str(args.embed_npy)
    print(json.dumps(m, indent=2))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(m, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
