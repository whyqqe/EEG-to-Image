#!/usr/bin/env python3
"""Build a qualitative comparison panel for ERDC (B0 / random / brain / GT).

Saves side-by-side JPEGs under --output-dir and a manifest JSON.
No GT leakage into selection — GT is display-only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from eval_atm_pipeline import list_test_images


def _open(p: Path, size: int) -> Image.Image:
    return Image.open(p).convert("RGB").resize((size, size), Image.Resampling.BICUBIC)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b0-dir", type=str, required=True)
    parser.add_argument("--random-dir", type=str, required=True)
    parser.add_argument("--brain-dir", type=str, required=True)
    parser.add_argument("--images-root", type=str, default="/project/peilab/why/data/images_set")
    parser.add_argument("--scores-npy", type=str, default="", help="optional (n,k) brain scores")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--n-panel", type=int, default=24)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.output_dir)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    gt = list_test_images(Path(args.images_root))

    def _abs(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else ROOT / path

    b0 = _abs(args.b0_dir)
    rd = _abs(args.random_dir)
    br = _abs(args.brain_dir)

    n = min(len(gt), len(list(br.glob("*.png"))))
    rng = np.random.RandomState(args.seed)
    # prefer high brain-margin examples if scores available
    if args.scores_npy:
        sp = Path(args.scores_npy)
        if not sp.is_absolute():
            sp = ROOT / sp
        scores = np.load(sp)
        margin = scores.max(1) - scores.mean(1)
        idx = np.argsort(-margin)[: args.n_panel]
    else:
        idx = rng.choice(n, size=min(args.n_panel, n), replace=False)

    labels = ["B0", "random", "brain", "GT"]
    cell = args.size
    pad = 8
    header = 28
    W = cell * 4 + pad * 5
    H = cell + pad * 2 + header
    manifest = []
    for j, i in enumerate(idx):
        i = int(i)
        imgs = [
            _open(b0 / f"{i:03d}.png", cell),
            _open(rd / f"{i:03d}.png", cell),
            _open(br / f"{i:03d}.png", cell),
            _open(gt[i], cell),
        ]
        canvas = Image.new("RGB", (W, H), (245, 245, 245))
        draw = ImageDraw.Draw(canvas)
        for c, lab in enumerate(labels):
            draw.text((pad + c * (cell + pad), 6), f"{lab}", fill=(20, 20, 20))
            canvas.paste(imgs[c], (pad + c * (cell + pad), header))
        path = out / f"panel_{j:02d}_img{i:03d}.jpg"
        canvas.save(path, quality=92)
        manifest.append({"panel": path.name, "index": i})

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[OK] wrote {len(manifest)} panels -> {out}")


if __name__ == "__main__":
    main()
