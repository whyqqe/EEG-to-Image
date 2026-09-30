#!/usr/bin/env python3
"""NB memory router: train/test RAG soft-k -> ViT-H embeds for CFT conditioning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)


def rag_soft_k(query: np.ndarray, gallery: np.ndarray, vith: np.ndarray, k: int, tau: float) -> tuple[np.ndarray, np.ndarray]:
    q = l2(query.astype(np.float32))
    g = l2(gallery.astype(np.float32))
    sim = q @ g.T
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
    ap.add_argument("--clip-test", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--input-key", type=str, default="proj")
    ap.add_argument("--soft-k", type=int, default=5)
    ap.add_argument("--soft-tau", type=float, default=0.07)
    args = ap.parse_args()

    embed_dir = Path(args.embed_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    q_tr = np.load(embed_dir / f"z_eeg_{args.input_key}_train.npy")
    q_te = np.load(embed_dir / f"z_eeg_{args.input_key}_test.npy")
    vith_tr = np.load(args.clip_train).astype(np.float32)
    vith_te_target = np.load(args.clip_test).astype(np.float32)

    mem_tr, idx_tr = rag_soft_k(q_tr, q_tr, vith_tr, args.soft_k, args.soft_tau)
    mem_te, idx_te = rag_soft_k(q_te, q_tr, vith_tr, args.soft_k, args.soft_tau)

    np.save(out_dir / "rag_soft5_train_clip_1024.npy", mem_tr)
    np.save(out_dir / "rag_soft5_test_clip_1024.npy", mem_te)
    np.save(out_dir / "rag_soft5_neighbor_idx_test.npy", idx_te)

    report = {
        "train_cos_to_gt_vith": float(np.mean(np.sum(mem_tr * l2(vith_tr), axis=1))),
        "test_cos_to_gt_vith": float(np.mean(np.sum(mem_te * l2(vith_te_target), axis=1))),
        "soft_k": args.soft_k,
    }
    (out_dir / "memory_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
