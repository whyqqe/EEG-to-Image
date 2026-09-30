#!/usr/bin/env python3
"""Train-split GT depth at 64x64 only. No RGB, no test recompute.

The shared HCMA-S cache already has test_depth_64.npy. What is missing is the
16540-row train target needed to fit the unified spatial field. Writing RGB for
train would be several GB; this file writes one float32 npy (~270MB) and stops.
"""

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
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        paths.extend(sorted(list(d.glob("*.jpg")) + list(d.glob("*.JPEG")) + list(d.glob("*.png"))))
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-root", type=str, default="/project/peilab/why/data/images_set")
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--model-id", type=str, default="depth-anything/Depth-Anything-V2-Small-hf")
    ap.add_argument("--low-res", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    npy = out / f"train_depth_{args.low_res}.npy"
    if npy.is_file() and npy.stat().st_size > 1_000_000:
        arr = np.load(npy, mmap_mode="r")
        print(f"[SKIP] {npy} {arr.shape}")
        return

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    processor = AutoImageProcessor.from_pretrained(args.model_id)
    model = AutoModelForDepthEstimation.from_pretrained(args.model_id).to(device).eval()
    paths = list_train_images(Path(args.images_root))
    depths = np.zeros((len(paths), args.low_res, args.low_res), dtype=np.float32)
    index = []
    bs = max(1, int(args.batch_size))
    for start in tqdm(range(0, len(paths), bs), desc="gt-depth-train"):
        batch = paths[start:start + bs]
        imgs = [Image.open(p).convert("RGB") for p in batch]
        inputs = processor(images=imgs, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            pred = model(**inputs).predicted_depth
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1), size=(args.low_res, args.low_res),
            mode="bicubic", align_corners=False,
        ).squeeze(1)
        for j, p in enumerate(batch):
            d = pred[j].float().cpu().numpy()
            depths[start + j] = (d - d.min()) / (d.max() - d.min() + 1e-8)
            index.append({"idx": start + j, "path": str(p)})
    np.save(npy, depths)
    (out / "train_index.json").write_text(json.dumps(index), encoding="utf-8")
    (out / "gt_depth_train_report.json").write_text(json.dumps({
        "n": len(paths), "shape": list(depths.shape), "rgb": False,
        "note": "train npy only; test depth is reused from the existing HCMA-S cache",
    }, indent=2), encoding="utf-8")
    print(f"[OK] {depths.shape} -> {npy} ({npy.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
