#!/usr/bin/env python3
"""Phase0: NVOL scan — ridge probe from frozen NB EEG embeds to each CLIP layer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-8)


def align_rows(eeg: np.ndarray, img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if eeg.shape[0] == img.shape[0]:
        return eeg, img
    if img.shape[0] > eeg.shape[0] and img.shape[0] % eeg.shape[0] == 0:
        r = img.shape[0] // eeg.shape[0]
        return eeg, img.reshape(eeg.shape[0], r, -1).mean(axis=1)
    n = min(eeg.shape[0], img.shape[0])
    print(f"[WARN] truncate align to n={n} (eeg={eeg.shape[0]}, img={img.shape[0]})")
    return eeg[:n], img[:n]


def retrieval(q: np.ndarray, g: np.ndarray) -> dict:
    q, g = l2(q), l2(g)
    sim = q @ g.T
    n = sim.shape[0]
    ranks = np.argsort(-sim, axis=1)
    hit = ranks == np.arange(n)[:, None]
    return {
        "top1": float(hit[:, :1].any(axis=1).mean()),
        "top5": float(hit[:, :5].any(axis=1).mean()),
        "paired_cos": float(np.mean(np.sum(q * g, axis=1))),
        "n": int(n),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eeg-train", type=str, required=True)
    ap.add_argument("--eeg-test", type=str, required=True)
    ap.add_argument(
        "--select-by",
        choices=["val", "test"],
        default="val",
        help="leak-free default: choose the layer on held-in train concepts. "
        "'test' reproduces the old (contaminated) behaviour for audit only.",
    )
    ap.add_argument("--clip-layers-dir", type=str, required=True)
    ap.add_argument("--output-json", type=str, required=True)
    ap.add_argument("--ridge-alpha", type=float, default=1.0)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top-k-layers", type=int, default=3)
    args = ap.parse_args()

    eeg_tr = np.load(args.eeg_train).astype(np.float32)
    eeg_te = np.load(args.eeg_test).astype(np.float32)
    root = Path(args.clip_layers_dir)
    train_dir, test_dir = root / "train", root / "test"
    layer_files = sorted(train_dir.glob("layer_*.npy"))
    if not layer_files:
        raise FileNotFoundError(f"no layer_*.npy under {train_dir}")

    results = []
    for lf in layer_files:
        li = int(lf.stem.split("_")[1])
        y_tr = np.load(lf).astype(np.float32)
        y_te = np.load(test_dir / lf.name).astype(np.float32)
        x_tr, y_tr = align_rows(eeg_tr, y_tr)
        x_te, y_te = align_rows(eeg_te, y_te)

        x_a, x_va, y_a, y_va = train_test_split(
            x_tr, y_tr, test_size=args.val_ratio, random_state=args.seed
        )
        reg = Ridge(alpha=args.ridge_alpha, fit_intercept=True)
        reg.fit(x_a, y_a)
        va = retrieval(reg.predict(x_va), y_va)
        te = retrieval(reg.predict(x_te), y_te)
        row = {"layer": li, "val": va, "test": te, "dim": int(y_tr.shape[1])}
        results.append(row)
        print(
            f"layer {li:02d}: val_top1={va['top1']:.3f} test_top1={te['top1']:.3f} "
            f"test_cos={te['paired_cos']:.3f}"
        )

    results = sorted(results, key=lambda r: r[args.select_by]["top1"], reverse=True)
    topk = [r["layer"] for r in results[: args.top_k_layers]]
    out = {
        "protocol": "ridge_probe_NB_eeg_to_clip_layer",
        "eeg_train": args.eeg_train,
        "eeg_test": args.eeg_test,
        "top_k_layers": topk,
        "best_layer": topk[0] if topk else None,
        "results": results,
        "design_note": {
            "semantic_retrieval_space": "RN50/SSP-512 (NeuroBridge empirical SOTA)",
            "nvol_role": "Perception teacher + decode bridge, not primary gallery space",
            "generation_space": "ViT-H-14 1024 via decode head / IP-Adapter",
        },
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"best_layer": out["best_layer"], "top_k_layers": topk}, indent=2))


if __name__ == "__main__":
    main()
