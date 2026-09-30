#!/usr/bin/env python3
"""Phase 3: ensemble RAG + prior embeddings; export combined CLIP conditions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)


def blend(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    out = alpha * a + (1.0 - alpha) * b
    return l2(out.astype(np.float32))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rag-npy", type=str, required=True)
    ap.add_argument("--prior-npy", type=str, required=True)
    ap.add_argument("--output-npy", type=str, required=True)
    ap.add_argument("--alpha", type=float, default=0.5, help="weight on RAG embed")
    ap.add_argument("--report-json", type=str, default="")
    args = ap.parse_args()

    rag = l2(np.load(args.rag_npy).astype(np.float32))
    prior = l2(np.load(args.prior_npy).astype(np.float32))
    n = min(len(rag), len(prior))
    combo = blend(rag[:n], prior[:n], args.alpha)
    out = Path(args.output_npy)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, combo)

    report = {
        "phase": 3,
        "alpha_rag": args.alpha,
        "n": n,
        "output": str(out),
        "mean_cos_rag_prior": float(np.mean(np.sum(rag[:n] * prior[:n], axis=1))),
    }
    if args.report_json:
        Path(args.report_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
