#!/usr/bin/env python3
"""TCDA injection: saliency-weighted fuse of structure (Pc) and semantic gen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def load(path: Path, size: int) -> np.ndarray:
    return np.asarray(
        Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BICUBIC),
        dtype=np.float32,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--struct-dir", type=str, required=True, help="Pc blur RGB")
    ap.add_argument("--semantic-dir", type=str, required=True)
    ap.add_argument("--saliency-dir", type=str, required=True, help="R saliency RGB (gray)")
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--tag", type=str, default="tcda_sal_fuse")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument(
        "--mode",
        type=str,
        default="sal_fuse",
        choices=["sal_fuse", "sal_protect", "sal_floor"],
        help="sal_fuse: (1-M)*struct+M*sem; sal_floor: mix toward that blend but keep semantic floor",
    )
    ap.add_argument("--sal-gamma", type=float, default=1.0, help=">1 sharpens saliency mask")
    ap.add_argument(
        "--sem-floor",
        type=float,
        default=0.0,
        help="for sal_floor: out = floor*sem + (1-floor)*sal_fuse (protects 2-way)",
    )
    ap.add_argument("--m-min", type=float, default=0.0, help="clip mask lower bound")
    ap.add_argument("--m-max", type=float, default=1.0, help="clip mask upper bound")
    ap.add_argument("--max-images", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.output_dir)
    gen = out / "generated"
    gen.mkdir(parents=True, exist_ok=True)
    sdir, gdir, rdir = Path(args.struct_dir), Path(args.semantic_dir), Path(args.saliency_dir)
    n = len(sorted(sdir.glob("*.png")))
    if args.max_images > 0:
        n = min(n, args.max_images)

    for i in tqdm(range(n), desc=args.tag):
        dst = gen / f"{i:03d}.png"
        if dst.is_file():
            continue
        s = load(sdir / f"{i:03d}.png", args.size)
        g = load(gdir / f"{i:03d}.png", args.size)
        m = load(rdir / f"{i:03d}.png", args.size).mean(axis=-1, keepdims=True) / 255.0
        m = np.clip(m, 0, 1) ** float(args.sal_gamma)
        m = np.clip(m, float(args.m_min), float(args.m_max))
        if args.mode == "sal_fuse":
            blend = (1.0 - m) * s + m * g
            out_img = blend
        elif args.mode == "sal_protect":
            out_img = (1.0 - 0.65 * m) * s + (0.65 * m) * g
        else:
            blend = (1.0 - m) * s + m * g
            fl = float(np.clip(args.sem_floor, 0.0, 1.0))
            out_img = fl * g + (1.0 - fl) * blend
        Image.fromarray(out_img.clip(0, 255).astype(np.uint8)).save(dst)

    report = {
        "tag": args.tag,
        "mode": args.mode,
        "sal_gamma": args.sal_gamma,
        "sem_floor": args.sem_floor,
        "m_min": args.m_min,
        "m_max": args.m_max,
        "n": n,
        "struct_dir": args.struct_dir,
        "semantic_dir": args.semantic_dir,
        "saliency_dir": args.saliency_dir,
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
