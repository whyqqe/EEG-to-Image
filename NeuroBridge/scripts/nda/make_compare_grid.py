#!/usr/bin/env python3
"""Build side-by-side comparison grids: GT | methods...

Optionally auto-select rows by highest CLIP cosine(gen, GT) from a reference gen dir.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


def list_test_images(images_root: Path) -> list[Path]:
    root = images_root / "test_images"
    paths: list[Path] = []
    for d in sorted([p for p in root.iterdir() if p.is_dir()]):
        imgs = sorted(list(d.glob("*.jpg")) + list(d.glob("*.JPEG")) + list(d.glob("*.png")))
        paths.extend(imgs)
    return paths


def load_rgb(path: Path, size: int) -> Image.Image:
    return Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BICUBIC)


@torch.no_grad()
def select_best_semantic_indices(
    gen_dir: Path,
    gt_paths: list[Path],
    k: int,
    device: torch.device,
    batch_size: int = 16,
) -> tuple[list[int], list[float]]:
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai", device=device
    )
    model.eval()

    def encode(paths: list[Path]) -> np.ndarray:
        embs = []
        for i in tqdm(range(0, len(paths), batch_size), desc="clip-select"):
            batch = paths[i : i + batch_size]
            imgs = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in batch]).to(device)
            z = F.normalize(model.encode_image(imgs).float(), dim=-1)
            embs.append(z.cpu().numpy())
        return np.concatenate(embs, 0)

    gens = [gen_dir / f"{i:03d}.png" for i in range(len(gt_paths))]
    gens = [p for p in gens if p.is_file()]
    n = min(len(gens), len(gt_paths))
    gt_emb = encode(gt_paths[:n])
    gen_emb = encode(gens[:n])
    cos = (gt_emb * gen_emb).sum(1)
    order = np.argsort(-cos)[:k]
    return [int(i) for i in order], [float(cos[i]) for i in order]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-root", type=str, default="/project/peilab/why/data/images_set")
    ap.add_argument("--cols", type=str, required=True, help="name=dir,name=dir,...")
    ap.add_argument("--indices", type=str, default="")
    ap.add_argument(
        "--auto-select-gen-dir",
        type=str,
        default="",
        help="if set, pick top-k indices by CLIP cosine(gen, GT)",
    )
    ap.add_argument("--auto-select-k", type=int, default=12)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--cell", type=int, default=160)
    ap.add_argument("--metrics-json", type=str, default="", help="optional summary to caption")
    args = ap.parse_args()

    cache = Path("/project/peilab/why/cache/eeg-brainit")
    os.environ.setdefault("HF_HOME", str(cache / "hf"))
    os.environ.setdefault("HF_HUB_CACHE", str(cache / "hf" / "hub"))
    os.environ.setdefault("OPENCLIP_CACHE_DIR", str(cache / "open_clip"))

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    gt = list_test_images(Path(args.images_root))
    cols = []
    for item in args.cols.split(","):
        name, path = item.split("=", 1)
        cols.append((name.strip(), Path(path.strip())))

    scores = None
    if args.auto_select_gen_dir:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        idxs, scores = select_best_semantic_indices(
            Path(args.auto_select_gen_dir), gt, args.auto_select_k, device
        )
    elif args.indices.strip():
        idxs = [int(x) for x in args.indices.split(",") if x.strip()]
    else:
        idxs = [0, 7, 15, 31, 63, 90, 120, 150, 175, 199]

    cell = args.cell
    header_h = 28
    W = cell * (1 + len(cols))
    H = header_h + cell * len(idxs)
    canvas = Image.new("RGB", (W, H), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    labels = ["GT"] + [n for n, _ in cols]
    for j, lab in enumerate(labels):
        draw.text((j * cell + 6, 6), lab[:18], fill=(20, 20, 20), font=font)

    for r, i in enumerate(idxs):
        y = header_h + r * cell
        canvas.paste(load_rgb(gt[i], cell), (0, y))
        for j, (_, d) in enumerate(cols):
            p = d / f"{i:03d}.png"
            if p.is_file():
                canvas.paste(load_rgb(p, cell), ((j + 1) * cell, y))
            else:
                draw.rectangle([((j + 1) * cell, y), ((j + 2) * cell - 1, y + cell - 1)], fill=(200, 80, 80))
        label = f"#{i:03d}"
        if scores is not None:
            label += f" {scores[r]:.2f}"
        draw.text((4, y + 4), label, fill=(255, 255, 0), font=font)

    grid_path = out / "compare_grid.png"
    canvas.save(grid_path)

    rows_dir = out / "rows"
    rows_dir.mkdir(exist_ok=True)
    for r, i in enumerate(idxs):
        row = canvas.crop((0, header_h + r * cell, W, header_h + (r + 1) * cell))
        row.save(rows_dir / f"row_{i:03d}.png")

    meta = {
        "indices": idxs,
        "scores": scores,
        "select_mode": "clip_cosine_top" if scores is not None else "manual",
        "columns": ["GT"] + [n for n, _ in cols],
        "grid": str(grid_path),
        "n_rows": len(idxs),
    }
    if args.metrics_json and Path(args.metrics_json).is_file():
        meta["metrics_ref"] = args.metrics_json
    (out / "compare_report.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
