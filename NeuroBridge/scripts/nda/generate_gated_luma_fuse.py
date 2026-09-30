#!/usr/bin/env python3
"""Confidence-gated luma fuse (ATM/MLSP-style strength control).

High structure confidence (u_str) → more structure weight (lower sem_alpha).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def load_rgb(path: Path, size: int) -> np.ndarray:
    return np.asarray(
        Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BICUBIC),
        dtype=np.float32,
    )


def luma(x: np.ndarray) -> np.ndarray:
    return 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]


def match_luma_to_ref(img: np.ndarray, ref: np.ndarray) -> np.ndarray:
    y = luma(img)
    yr = luma(ref)
    mean_y = float(y.mean()) + 1e-6
    mean_r = float(yr.mean()) + 1e-6
    scale = mean_r / mean_y
    out = img * scale
    std_y = float(y.std()) + 1e-6
    std_r = float(yr.std()) + 1e-6
    out = (out - out.mean()) * (0.65 + 0.35 * (std_r / std_y)) + mean_r
    return np.clip(out, 0, 255)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--struct-dir", type=str, required=True)
    ap.add_argument("--semantic-dir", type=str, required=True)
    ap.add_argument("--u-str-npy", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--tag", type=str, default="gated_luma")
    ap.add_argument("--alpha-min", type=float, default=0.45)
    ap.add_argument("--alpha-max", type=float, default=0.80)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--max-images", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.output_dir)
    gen = out / "generated"
    gen.mkdir(parents=True, exist_ok=True)
    sdir, gdir = Path(args.struct_dir), Path(args.semantic_dir)
    u = np.load(args.u_str_npy).astype(np.float32).reshape(-1)
    order = np.argsort(u)
    ranks = np.empty_like(u, dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, num=len(u), dtype=np.float32)

    n = len(sorted(gdir.glob("*.png")))
    if args.max_images > 0:
        n = min(n, args.max_images)
    alphas = []
    for i in tqdm(range(n), desc=args.tag):
        dst = gen / f"{i:03d}.png"
        a = float(args.alpha_max - ranks[i] * (args.alpha_max - args.alpha_min))
        alphas.append(a)
        if dst.is_file():
            continue
        sem = load_rgb(gdir / f"{i:03d}.png", args.size)
        struct = load_rgb(sdir / f"{i:03d}.png", args.size)
        blend = a * sem + (1.0 - a) * struct
        blend = match_luma_to_ref(blend, sem)
        Image.fromarray(blend.astype(np.uint8)).save(dst)

    report = {
        "tag": args.tag,
        "mode": "confidence_gated_luma_fuse",
        "alpha_min": args.alpha_min,
        "alpha_max": args.alpha_max,
        "alpha_mean": float(np.mean(alphas)),
        "n": n,
        "note": "ATM/MLSP low-level strength control + CogCapPro-style structure prior",
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.save(out / "sem_alpha_per_sample.npy", np.asarray(alphas, dtype=np.float32))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
