#!/usr/bin/env python3
"""sdedit_ll comparison grid: rows = top-k semantic (CLIP cosine gen vs GT).

Uses already-cached per-image CLIP features (cache/gen_*.npz, cache/gt_feats.npz)
so no GPU inference is needed. Columns: GT | HCMA(sub-08) | sdedit_ll(sub-08) | ATM(official sub-08).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

CACHE = Path("/project/peilab/why/NeuroBridge/outputs/standard7_protocol/cache")
GT_ROOT = Path("/project/peilab/why/data/images_set/test_images")


def list_test_images(images_root: Path) -> list[Path]:
    paths: list[Path] = []
    for d in sorted([p for p in images_root.iterdir() if p.is_dir()]):
        imgs = sorted(list(d.glob("*.jpg")) + list(d.glob("*.JPEG")) + list(d.glob("*.png")))
        paths.extend(imgs)
    return paths


def load_rgb(p: Path, size: int) -> Image.Image:
    return Image.open(p).convert("RGB").resize((size, size), Image.Resampling.BICUBIC)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--select-tag", type=str, default="sdedit_ll_sub08")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--cell", type=int, default=224)
    ap.add_argument("--cols", type=str, required=True,
                    help="name=gen_tag,name=gen_dir; special: name=gt")
    args = ap.parse_args()

    gt = list_test_images(GT_ROOT)
    n = len(gt)
    z = np.load(CACHE / "gt_feats.npz", allow_pickle=False)
    gt_clip = z["clip"]
    sel = np.load(CACHE / f"gen_{args.select_tag}.npz", allow_pickle=False)["clip"]
    # cosine top-k (select tag = reference for 'semantic best')
    c = (gt_clip * sel).sum(1) / (
        np.linalg.norm(gt_clip, axis=1) * np.linalg.norm(sel, axis=1) + 1e-8
    )
    order = np.argsort(-c)[: args.k]
    scores = {int(i): float(c[i]) for i in order}

    # build column list
    cols = []  # (name, Path or None-for-gt)
    for item in args.cols.split(","):
        name, ref = item.split("=", 1)
        if ref == "gt":
            cols.append((name, None))
        else:
            p = Path(ref)
            cols.append((name, p))

    cell = args.cell
    header_h = 30
    W = cell * len(cols)
    H = header_h + cell * len(order)
    canvas = Image.new("RGB", (W, H), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    for j, (name, _p) in enumerate(cols):
        draw.text((j * cell + 6, 5), name[:16], fill=(20, 20, 20), font=font)

    for r, i in enumerate(order):
        y = header_h + r * cell
        for j, (_name, p) in enumerate(cols):
            src = gt[i] if p is None else p / f"{i:03d}.png"
            x = j * cell
            if src.is_file():
                canvas.paste(load_rgb(src, cell), (x, y))
            else:
                draw.rectangle([x, y, x + cell - 1, y + cell - 1], fill=(200, 80, 80))
        draw.text((4, y + 4), f"#{i:03d} cos={scores[i]:.3f}", fill=(255, 255, 0), font=font)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    grid_path = out / "sdedit_ll_compare_semantic_top.png"
    canvas.save(grid_path)

    meta = {
        "indices": [int(i) for i in order],
        "scores": scores,
        "select": f"CLIP cosine top-{args.k} on tag {args.select_tag}",
        "columns": [name for name, _ in cols],
        "grid": str(grid_path),
    }
    (out / "sdedit_ll_compare_report.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
