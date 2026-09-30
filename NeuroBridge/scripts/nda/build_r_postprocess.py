#!/usr/bin/env python3
"""Post-process predicted saliency with Pc energy (no retrain)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sal-dir", type=str, required=True)
    ap.add_argument("--pc-dir", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--pc-power", type=float, default=0.85)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sdir, pdir = Path(args.sal_dir), Path(args.pc_dir)
    n = len(sorted(sdir.glob("*.png")))
    for i in tqdm(range(n), desc="r-post"):
        sal = np.asarray(
            Image.open(sdir / f"{i:03d}.png").convert("L").resize((args.size, args.size), Image.Resampling.BICUBIC),
            dtype=np.float32,
        ) / 255.0
        pc = np.asarray(
            Image.open(pdir / f"{i:03d}.png").convert("RGB").resize((args.size, args.size), Image.Resampling.BICUBIC),
            dtype=np.float32,
        ) / 255.0
        # local contrast of Pc as objectness proxy
        gray = 0.299 * pc[..., 0] + 0.587 * pc[..., 1] + 0.114 * pc[..., 2]
        # simple local energy via box subtract
        from PIL import ImageFilter

        g_pil = Image.fromarray((gray * 255).astype(np.uint8))
        blur = np.asarray(g_pil.filter(ImageFilter.GaussianBlur(radius=8)), dtype=np.float32) / 255.0
        energy = np.abs(gray - blur)
        energy = energy / (energy.max() + 1e-8)
        mix = np.clip(sal * (0.35 + 0.65 * (energy ** args.pc_power)), 0, 1)
        mix = mix / (mix.max() + 1e-8)
        Image.fromarray((mix * 255).astype(np.uint8)).convert("RGB").save(out / f"{i:03d}.png")
    print(f"[OK] wrote {n} maps -> {out}")


if __name__ == "__main__":
    main()
