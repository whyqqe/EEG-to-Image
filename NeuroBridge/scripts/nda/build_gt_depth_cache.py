#!/usr/bin/env python3
"""Build GT DepthAnything maps for train/test stimuli (not neighbors).

Saves:
  train_depth_64.npy  (N_train, 64, 64) float32 in [0,1]
  test_depth_64.npy   (N_test, 64, 64)
  test_rgb_512/{i:03d}.png  — oracle depth RGB for ControlNet upper-bound
  train_index.json / test_index.json — image path mapping
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def list_split_images(images_root: Path, split: str) -> list[Path]:
    root = images_root / ("training_images" if split == "train" else "test_images")
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
    ap.add_argument("--images-root", type=str, default="/project/peilab/why/data/images_set")
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--model-id", type=str, default="depth-anything/Depth-Anything-V2-Small-hf")
    ap.add_argument("--low-res", type=int, default=64)
    ap.add_argument("--rgb-size", type=int, default=512)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--splits", type=str, default="train,test")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    processor = AutoImageProcessor.from_pretrained(args.model_id)
    model = AutoModelForDepthEstimation.from_pretrained(args.model_id).to(device)
    model.eval()

    report = {"model": args.model_id, "low_res": args.low_res, "rgb_size": args.rgb_size, "splits": {}}

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        paths = list_split_images(Path(args.images_root), split)
        npy_path = out / f"{split}_depth_{args.low_res}.npy"
        index_path = out / f"{split}_index.json"
        rgb_dir = out / f"{split}_rgb_{args.rgb_size}"
        if split == "test":
            rgb_dir.mkdir(parents=True, exist_ok=True)

        if npy_path.is_file() and npy_path.stat().st_size > 1000:
            arr = np.load(npy_path)
            need_rgb = split == "test" and (
                not rgb_dir.is_dir() or len(list(rgb_dir.glob("*.png"))) < len(paths)
            )
            if not need_rgb:
                print(f"[SKIP] {split} depth exists {arr.shape}")
                report["splits"][split] = {"n": int(arr.shape[0]), "skipped": True}
                continue
            print(f"[INFO] {split} npy exists; regenerating missing RGB only")
            # fall through would recompute all — cheaper to just warn and recompute test
            print(f"[INFO] recomputing {split} to refresh RGB")

        depths = np.zeros((len(paths), args.low_res, args.low_res), dtype=np.float32)
        index = []
        bs = max(1, int(args.batch_size))
        for start in tqdm(range(0, len(paths), bs), desc=f"gt-depth-{split}"):
            batch_paths = paths[start : start + bs]
            imgs = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = processor(images=imgs, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                pred_full = model(**inputs).predicted_depth  # (B,H,W) native res
            pred = torch.nn.functional.interpolate(
                pred_full.unsqueeze(1),
                size=(args.low_res, args.low_res),
                mode="bicubic",
                align_corners=False,
            ).squeeze(1)
            for j, p in enumerate(batch_paths):
                d = pred[j].float().cpu().numpy()
                d = (d - d.min()) / (d.max() - d.min() + 1e-8)
                depths[start + j] = d
                index.append({"idx": start + j, "path": str(p)})
                if split == "test":
                    # oracle ControlNet RGB from full-res depth → rgb_size
                    pred_hi = torch.nn.functional.interpolate(
                        pred_full[j : j + 1].unsqueeze(1),
                        size=(args.rgb_size, args.rgb_size),
                        mode="bicubic",
                        align_corners=False,
                    ).squeeze().float().cpu().numpy()
                    depth_to_rgb(pred_hi).save(rgb_dir / f"{start + j:03d}.png")

        np.save(npy_path, depths)
        index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
        report["splits"][split] = {"n": len(paths), "npy": str(npy_path), "rgb_dir": str(rgb_dir) if split == "test" else None}
        print(f"[OK] {split} {depths.shape} → {npy_path}")

    (out / "gt_depth_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
