#!/usr/bin/env python3
"""Blend CFT Fusion prediction with Fusion-space RAG memory."""

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
    ap.add_argument("--cft-npy", type=str, required=True)
    ap.add_argument("--mem-npy", type=str, required=True)
    ap.add_argument("--gt-npy", type=str, default="", help="optional fusion GT for reporting")
    ap.add_argument("--output-npy", type=str, required=True)
    ap.add_argument("--alpha", type=float, default=0.5, help="weight on CFT embed")
    ap.add_argument("--report-json", type=str, default="")
    args = ap.parse_args()

    cft = l2(np.load(args.cft_npy).astype(np.float32))
    mem = l2(np.load(args.mem_npy).astype(np.float32))
    n = min(len(cft), len(mem))
    combo = blend(cft[:n], mem[:n], args.alpha)
    out = Path(args.output_npy)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, combo)

    report = {
        "alpha_cft": args.alpha,
        "n": n,
        "output": str(out),
        "mean_cos_cft_mem": float(np.mean(np.sum(cft[:n] * mem[:n], axis=1))),
    }
    if args.gt_npy:
        gt = l2(np.load(args.gt_npy).astype(np.float32))[:n]
        report["cos_to_gt"] = float(np.mean(np.sum(combo * gt, axis=1)))
        report["cos_cft_gt"] = float(np.mean(np.sum(cft[:n] * gt, axis=1)))
        report["cos_mem_gt"] = float(np.mean(np.sum(mem[:n] * gt, axis=1)))
    if args.report_json:
        Path(args.report_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
