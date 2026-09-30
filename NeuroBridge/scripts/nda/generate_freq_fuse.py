#!/usr/bin/env python3
"""Frequency-domain fusion: low-freq from structure map, high-freq from semantic gen.

Preserves layout/color (SSIM) while keeping photorealistic textures (FID/CLIP).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def gaussian_blur_np(img: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur via PIL for simplicity/stability."""
    from PIL import ImageFilter

    radius = max(1, int(round(sigma * 2)))
    pil = Image.fromarray(img.astype(np.uint8))
    # approximate: repeated box ≈ gaussian; use GaussianBlur
    return np.asarray(pil.filter(ImageFilter.GaussianBlur(radius=radius)), dtype=np.float32)


def freq_fuse(struct: np.ndarray, semantic: np.ndarray, sigma: float) -> np.ndarray:
    """out = low(struct) + high(semantic)."""
    s = struct.astype(np.float32)
    g = semantic.astype(np.float32)
    low_s = gaussian_blur_np(s, sigma)
    low_g = gaussian_blur_np(g, sigma)
    high_g = g - low_g
    out = low_s + high_g
    return np.clip(out, 0, 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--struct-dir", type=str, required=True, help="low-level / structured RGB")
    ap.add_argument("--semantic-dir", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--sigma", type=float, default=8.0, help="Gaussian cutoff (px at 512)")
    ap.add_argument("--tag", type=str, default="freq_fuse")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--max-images", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.output_dir)
    gen = out / "generated"
    gen.mkdir(parents=True, exist_ok=True)
    sdir, gdir = Path(args.struct_dir), Path(args.semantic_dir)
    n = len(sorted(sdir.glob("*.png")))
    if args.max_images > 0:
        n = min(n, args.max_images)

    for i in tqdm(range(n), desc=f"freq-fuse σ={args.sigma}"):
        dst = gen / f"{i:03d}.png"
        if dst.is_file():
            continue
        s = np.asarray(
            Image.open(sdir / f"{i:03d}.png").convert("RGB").resize((args.size, args.size), Image.Resampling.BICUBIC)
        )
        g = np.asarray(
            Image.open(gdir / f"{i:03d}.png").convert("RGB").resize((args.size, args.size), Image.Resampling.BICUBIC)
        )
        Image.fromarray(freq_fuse(s, g, args.sigma)).save(dst)

    report = {"tag": args.tag, "mode": "freq_fuse", "sigma": args.sigma, "n": n,
              "struct_dir": args.struct_dir, "semantic_dir": args.semantic_dir}
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
