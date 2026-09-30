#!/usr/bin/env python3
"""Build LOCAL semantic targets from DINOv2 patch tokens.

WHY (theory)
------------
HCMA's alignment targets are ALL global pooled vectors:
    RN50 (1024), ViT-H-14 (1024), DINOv2 (1024)  -- via timm num_classes=0, which
    returns only the CLS token, so the 37x37=1369 patch tokens are DISCARDED.

Written as a chain rule over the target image y:

    I(e; y) = I(e; y_glob) + I(e; y_loc | y_glob)
              \__________/     \____________________/
                supervised        NEVER supervised

The second term is not zero -- it has simply never been asked for. That is the
"granularity axis" gap: HCMA has two DOWNSTREAM USES (semantics vs structure) but
only ONE GRANULARITY (global).

This script supplies the missing supervision: spatially resolved DINOv2 patch
tokens, adaptive-pooled to a G x G grid so each cell is a local semantic unit.

NOTE on novelty boundary: Brain-HIVE (ICLR'26) uses MULTI-ENCODER fusion into one
fused token; this is a GRANULARITY axis (spatial support), not an encoder-fusion
axis, and it is not what the (already reused) hierarchical CLIP layers provide --
those are pooled too.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


def list_split_images(images_root: Path, split: str) -> list[Path]:
    root = images_root / ("training_images" if split == "train" else "test_images")
    paths: list[Path] = []
    for d in sorted([p for p in root.iterdir() if p.is_dir()]):
        imgs = sorted(list(d.glob("*.jpg")) + list(d.glob("*.JPEG")) + list(d.glob("*.png")))
        paths.extend(imgs)
    return paths


@torch.no_grad()
def extract(model, transform, paths: list[Path], device, batch_size: int, grid: int):
    """Return (global (N,D), local (N, G*G, D)) in float32."""
    globs, locs = [], []
    n_prefix = int(getattr(model, "num_prefix_tokens", 1))
    for i in tqdm(range(0, len(paths), batch_size), desc="dino-patch"):
        batch = paths[i : i + batch_size]
        x = torch.stack([transform(Image.open(p).convert("RGB")) for p in batch]).to(device)
        tokens = model.forward_features(x)          # (B, prefix+patches, D)
        if tokens.ndim == 4:                        # some timm models return a map
            tokens = tokens.flatten(2).transpose(1, 2)
        patches = tokens[:, n_prefix:, :]           # drop CLS + register tokens
        D = patches.shape[-1]
        n = patches.shape[1]
        s = int(round(n ** 0.5))
        if s * s != n:                              # non-square grid -> pad
            s = int(np.ceil(np.sqrt(n)))
            pad = s * s - n
            patches = torch.cat([patches, patches[:, -1:, :].expand(-1, pad, -1)], dim=1)
        pm = patches.reshape(patches.shape[0], s, s, D).permute(0, 3, 1, 2)   # B,D,s,s
        pooled = F.adaptive_avg_pool2d(pm, (grid, grid))                      # B,D,G,G
        local = pooled.flatten(2).transpose(1, 2)                             # B,G*G,D
        locs.append(F.normalize(local.float(), dim=-1).cpu().numpy())
        globs.append(F.normalize(patches.mean(dim=1).float(), dim=-1).cpu().numpy())
    return (np.concatenate(globs, 0).astype(np.float32),
            np.concatenate(locs, 0).astype(np.float32))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-root", default="/project/peilab/why/data/images_set")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--grid", type=int, default=6, help="GxG local cells (theory: one local unit per cell)")
    ap.add_argument("--input-size", type=int, default=224,
                    help="override DINOv2 resolution (native 518 is ~5x slower; 224 gives a 16x16 patch grid)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    import timm

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = timm.create_model("vit_large_patch14_reg4_dinov2.lvd142m",
                              pretrained=True, num_classes=0).eval().to(device)
    cfg = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**cfg, is_training=False)
    print(f"[INFO] dino input {cfg.get('input_size')} patch={model.patch_embed.patch_size} "
          f"prefix_tokens={getattr(model, 'num_prefix_tokens', None)} grid={args.grid}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = {"model": "vit_large_patch14_reg4_dinov2.lvd142m", "grid": args.grid,
              "input_size": list(cfg.get("input_size", [])), "splits": {}}

    for split in args.splits:
        paths = list_split_images(Path(args.images_root), split)
        if not paths:
            print(f"[WARN] no images for split={split}, skip")
            continue
        g, l = extract(model, transform, paths, device, args.batch_size, args.grid)
        np.save(out / f"dinov2_global_{split}.npy", g)
        np.save(out / f"dinov2_local_{split}.npy", l)
        report["splits"][split] = {"n": len(paths), "global": list(g.shape), "local": list(l.shape)}
        print(f"[OK] {split}: n={len(paths)} global={g.shape} local={l.shape}")

    (out / "dinov2_local_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
