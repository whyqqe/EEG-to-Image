#!/usr/bin/env python3
"""Luma-matched semantic↔structure fuse (fixes brightness drop of plain RGB blend)."""

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
    """Scale blended image so mean luminance matches semantic ref (per-image)."""
    y = luma(img)
    yr = luma(ref)
    mean_y = float(y.mean()) + 1e-6
    mean_r = float(yr.mean()) + 1e-6
    scale = mean_r / mean_y
    # gentle: also lightly match std
    std_y = float(y.std()) + 1e-6
    std_r = float(yr.std()) + 1e-6
    # apply on RGB uniformly then clip
    out = img * scale
    # residual contrast toward ref
    out = (out - out.mean()) * (0.65 + 0.35 * (std_r / std_y)) + mean_r
    return np.clip(out, 0, 255)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--struct-dir", type=str, required=True)
    ap.add_argument("--semantic-dir", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--tag", type=str, default="luma_fuse")
    ap.add_argument("--sem-alpha", type=float, default=0.55, help="weight on semantic (rest=structure)")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--no-luma-match", action="store_true")
    ap.add_argument("--max-images", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.output_dir)
    gen = out / "generated"
    gen.mkdir(parents=True, exist_ok=True)
    sdir, gdir = Path(args.struct_dir), Path(args.semantic_dir)
    n = len(sorted(gdir.glob("*.png")))
    if args.max_images > 0:
        n = min(n, args.max_images)
    a = float(np.clip(args.sem_alpha, 0.0, 1.0))

    for i in tqdm(range(n), desc=args.tag):
        dst = gen / f"{i:03d}.png"
        if dst.is_file():
            continue
        sem = load_rgb(gdir / f"{i:03d}.png", args.size)
        struct = load_rgb(sdir / f"{i:03d}.png", args.size)
        blend = a * sem + (1.0 - a) * struct
        if not args.no_luma_match:
            blend = match_luma_to_ref(blend, sem)
        Image.fromarray(blend.astype(np.uint8)).save(dst)

    report = {
        "tag": args.tag,
        "mode": "luma_matched_fuse",
        "sem_alpha": a,
        "luma_match": not args.no_luma_match,
        "n": n,
        "struct_dir": args.struct_dir,
        "semantic_dir": args.semantic_dir,
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
