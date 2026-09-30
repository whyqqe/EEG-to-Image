#!/usr/bin/env python3
"""S1 offline targets: DINOv2 [CLS] embeddings for THINGS train/test images."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


def list_split_images(images_root: Path, split: str) -> list[Path]:
    root = images_root / f"{split}_images"
    paths: list[Path] = []
    for d in sorted([p for p in root.iterdir() if p.is_dir()]):
        imgs = sorted(list(d.glob("*.jpg")) + list(d.glob("*.JPEG")) + list(d.glob("*.png")))
        if not imgs:
            raise FileNotFoundError(f"no images in {d}")
        paths.extend(imgs)
    return paths


@torch.no_grad()
def encode_dinov2(paths: list[Path], device: torch.device, batch_size: int = 32) -> np.ndarray:
    import timm

    model = timm.create_model(
        "vit_large_patch14_reg4_dinov2.lvd142m",
        pretrained=True,
        num_classes=0,
    )
    model.eval().to(device)
    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_config, is_training=False)

    feats = []
    for i in tqdm(range(0, len(paths), batch_size), desc="dinov2"):
        batch_paths = paths[i : i + batch_size]
        xs = torch.stack([transform(Image.open(p).convert("RGB")) for p in batch_paths]).to(device)
        emb = model(xs)
        emb = F.normalize(emb.float(), dim=-1)
        feats.append(emb.cpu().numpy())
    return np.concatenate(feats, axis=0).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-root", type=str, default="/project/peilab/why/data/images_set")
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    cache = Path(os.environ.get("HF_HUB_CACHE", "/project/peilab/why/cache/eeg-brainit/hf/hub"))
    os.environ.setdefault("TORCH_HOME", str(cache.parent / "torch"))

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    images_root = Path(args.images_root)

    report = {"device": str(device), "splits": {}}
    for split in ("training", "test"):
        paths = list_split_images(images_root, split)
        feats = encode_dinov2(paths, device, args.batch_size)
        name = "train" if split == "training" else "test"
        out_path = out / f"dinov2_{name}.npy"
        np.save(out_path, feats)
        report["splits"][name] = {"n": len(paths), "dim": feats.shape[1], "path": str(out_path)}
        print(f"[OK] {out_path} shape={feats.shape}")

    (out / "offline_targets_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
