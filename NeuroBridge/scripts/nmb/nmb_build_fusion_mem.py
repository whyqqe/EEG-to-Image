#!/usr/bin/env python3
"""Build Fusion-space memory vectors from NB proj RAG indices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)


def rag_soft_k(query: np.ndarray, gallery_q: np.ndarray, gallery_y: np.ndarray, k: int, tau: float) -> np.ndarray:
    q = l2(query.astype(np.float32))
    gq = l2(gallery_q.astype(np.float32))
    sim = q @ gq.T
    n = sim.shape[0]
    out = np.zeros((n, gallery_y.shape[1]), dtype=np.float32)
    for i in range(n):
        order = np.argsort(-sim[i])[:k]
        w = sim[i, order]
        w = np.exp((w - w.max()) / max(tau, 1e-6))
        w = w / w.sum()
        out[i] = w @ gallery_y[order]
    return l2(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-dir", type=str, required=True)
    ap.add_argument("--fusion-train", type=str, required=True)
    ap.add_argument("--fusion-test", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--soft-k", type=int, default=5)
    ap.add_argument("--soft-tau", type=float, default=0.07)
    args = ap.parse_args()

    embed_dir = Path(args.embed_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    q_tr = np.load(embed_dir / "z_eeg_proj_train.npy")
    q_te = np.load(embed_dir / "z_eeg_proj_test.npy")
    fusion_tr = np.load(args.fusion_train).astype(np.float32)
    fusion_te = np.load(args.fusion_test).astype(np.float32)
    n_tr = min(len(q_tr), len(fusion_tr))
    n_te = min(len(q_te), len(fusion_te))
    q_tr, fusion_tr = q_tr[:n_tr], fusion_tr[:n_tr]
    q_te, fusion_te = q_te[:n_te], fusion_te[:n_te]

    mem_tr = rag_soft_k(q_tr, q_tr, fusion_tr, args.soft_k, args.soft_tau)
    mem_te = rag_soft_k(q_te, q_tr, fusion_tr, args.soft_k, args.soft_tau)

    np.save(out_dir / "fusion_mem_train.npy", mem_tr)
    np.save(out_dir / "fusion_mem_test.npy", mem_te)

    report = {
        "train_cos_to_gt_fusion": float(np.mean(np.sum(mem_tr * l2(fusion_tr), axis=1))),
        "test_cos_to_gt_fusion": float(np.mean(np.sum(mem_te * l2(fusion_te), axis=1))),
        "soft_k": args.soft_k,
        "soft_tau": args.soft_tau,
    }
    (out_dir / "fusion_mem_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
