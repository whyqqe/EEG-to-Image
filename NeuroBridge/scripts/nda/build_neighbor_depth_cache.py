#!/usr/bin/env python3
"""Precompute DepthAnything maps for RAG neighbors (train images)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def list_train_images(images_root: Path) -> list[Path]:
    root = images_root / "training_images"
    paths: list[Path] = []
    for d in sorted([p for p in root.iterdir() if p.is_dir()]):
        imgs = sorted(list(d.glob("*.jpg")) + list(d.glob("*.JPEG")) + list(d.glob("*.png")))
        paths.extend(imgs)
    return paths


def depth_to_rgb(depth: np.ndarray) -> Image.Image:
    d = depth.astype(np.float32)
    d = (d - d.min()) / (d.max() - d.min() + 1e-8)
    u8 = (d * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(np.stack([u8, u8, u8], axis=-1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--neighbor-idx-npy", type=str, required=True)
    ap.add_argument("--images-root", type=str, default="/project/peilab/why/data/images_set")
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--model-id", type=str, default="depth-anything/Depth-Anything-V2-Small-hf")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--max-images", type=int, default=0, help="limit unique neighbors for debug")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    neigh = np.load(args.neighbor_idx_npy)
    if neigh.ndim > 1:
        idxs = set(int(x) for x in neigh[:, 0].tolist())
    else:
        idxs = set(int(x) for x in neigh.tolist())
    # also cache top-5 if present for future
    if neigh.ndim > 1 and neigh.shape[1] > 1:
        for r in range(neigh.shape[1]):
            idxs.update(int(x) for x in neigh[:, r].tolist())
    idxs = sorted(idxs)
    if args.max_images > 0:
        idxs = idxs[: args.max_images]

    train_paths = list_train_images(Path(args.images_root))
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    processor = AutoImageProcessor.from_pretrained(args.model_id)
    model = AutoModelForDepthEstimation.from_pretrained(args.model_id).to(device)
    model.eval()

    done = 0
    for i in tqdm(idxs, desc="depth-cache"):
        path = out / f"{i:06d}.png"
        if path.is_file():
            done += 1
            continue
        img = Image.open(train_paths[i]).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            pred = model(**inputs).predicted_depth
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=(args.size, args.size),
            mode="bicubic",
            align_corners=False,
        ).squeeze().float().cpu().numpy()
        depth_to_rgb(pred).save(path)
        done += 1

    report = {
        "model": args.model_id,
        "n_unique_neighbors": len(idxs),
        "n_cached": done,
        "size": args.size,
        "output_dir": str(out),
    }
    (out / "depth_cache_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
