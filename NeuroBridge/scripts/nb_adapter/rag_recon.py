#!/usr/bin/env python3
"""Phase 0: RAG-Recon — NB RN50 embedding retrieval → ViT-H CLIP gallery embeds."""

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
    ranks, hits1, hits5 = [], 0, 0
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


def rag_top1(query: np.ndarray, gallery: np.ndarray, vith: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sim = query @ gallery.T
    idx = np.argmax(sim, axis=1)
    return l2(vith[idx].astype(np.float32)), idx


def rag_soft_k(
    query: np.ndarray,
    gallery: np.ndarray,
    vith: np.ndarray,
    k: int,
    tau: float,
) -> tuple[np.ndarray, np.ndarray]:
    sim = query @ gallery.T
    n = sim.shape[0]
    out = np.zeros((n, vith.shape[1]), dtype=np.float32)
    top_idx = np.zeros((n, k), dtype=np.int64)
    for i in range(n):
        order = np.argsort(-sim[i])[:k]
        top_idx[i] = order
        w = sim[i, order]
        w = np.exp((w - w.max()) / max(tau, 1e-6))
        w = w / w.sum()
        out[i] = w @ vith[order]
    return l2(out), top_idx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-dir", type=str, required=True)
    ap.add_argument("--clip-train", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--input-key", type=str, default="proj", choices=["proj", "raw"])
    ap.add_argument("--soft-k", type=int, default=5)
    ap.add_argument("--soft-tau", type=float, default=0.07)
    args = ap.parse_args()

    embed_dir = Path(args.embed_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    q_train = l2(np.load(embed_dir / f"z_eeg_{args.input_key}_train.npy"))
    q_test = l2(np.load(embed_dir / f"z_eeg_{args.input_key}_test.npy"))
    vith_train = np.load(args.clip_train).astype(np.float32)
    vith_test_gallery = l2(vith_train)  # gallery for NB-space retrieval

    top1_emb, top1_idx = rag_top1(q_test, q_train, vith_train)
    soft_emb, soft_idx = rag_soft_k(q_test, q_train, vith_train, args.soft_k, args.soft_tau)

    np.save(out_dir / "rag_top1_test_clip_1024.npy", top1_emb)
    np.save(out_dir / "rag_soft5_test_clip_1024.npy", soft_emb)
    np.save(out_dir / "rag_top1_neighbor_idx.npy", top1_idx)
    np.save(out_dir / "rag_soft5_neighbor_idx.npy", soft_idx)

    # NB retrieval in RN50-aligned space (train gallery self-retrieval on test queries)
    nb_retrieval = retrieval_metrics(q_test, q_train)

    report = {
        "phase": 0,
        "input_key": args.input_key,
        "soft_k": args.soft_k,
        "soft_tau": args.soft_tau,
        "nb_space_retrieval_train_gallery": nb_retrieval,
    }
    clip_test_path = Path(args.clip_train).parent / "clip_img_test_1024.npy"
    if clip_test_path.is_file():
        gt = l2(np.load(clip_test_path))
        report["rag_top1_cos_to_gt_vith"] = float(np.mean(np.sum(top1_emb * gt, axis=1)))
        report["rag_soft5_cos_to_gt_vith"] = float(np.mean(np.sum(soft_emb * gt, axis=1)))

    (out_dir / "rag_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"[OK] {out_dir}")


if __name__ == "__main__":
    main()
