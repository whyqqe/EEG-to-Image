#!/usr/bin/env python3
"""Extract OpenCLIP intermediate-layer pooled features for NVOL / HCF."""

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


def parse_layers(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


@torch.no_grad()
def extract_layers(
    paths: list[Path],
    layers: list[int],
    device: torch.device,
    batch_size: int,
    model_name: str,
    pretrained: str,
) -> dict[int, np.ndarray]:
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    model.eval()
    visual = model.visual
    n_blocks = len(visual.transformer.resblocks)
    for li in layers:
        if li < 0 or li >= n_blocks:
            raise ValueError(f"layer {li} out of range [0, {n_blocks})")

    hooks: dict[int, list[torch.Tensor]] = {li: [] for li in layers}

    def make_hook(idx: int):
        def _hook(_m, _inp, out):
            # OpenCLIP ViT block out is usually [seq, batch, dim]; some builds use [batch, seq, dim].
            x = out
            if x.ndim != 3:
                raise RuntimeError(f"unexpected block out shape {tuple(x.shape)}")
            if x.shape[0] > x.shape[1]:
                # [seq, B, C] -> CLS at token 0
                cls = x[0]
            else:
                # [B, seq, C]
                cls = x[:, 0]
            hooks[idx].append(F.normalize(cls.float(), dim=-1).cpu())

        return _hook

    handles = [
        visual.transformer.resblocks[li].register_forward_hook(make_hook(li)) for li in layers
    ]

    for i in tqdm(range(0, len(paths), batch_size), desc="clip-layers"):
        batch_paths = paths[i : i + batch_size]
        xs = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in batch_paths]).to(device)
        _ = model.encode_image(xs)

    for h in handles:
        h.remove()

    out: dict[int, np.ndarray] = {}
    for li in layers:
        out[li] = torch.cat(hooks[li], dim=0).numpy().astype(np.float32)
        if out[li].shape[0] != len(paths):
            raise RuntimeError(f"layer {li}: got {out[li].shape[0]} != {len(paths)}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-root", type=str, default="/project/peilab/why/data/images_set")
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--layers", type=str, default="8,10,12,14,16,18,20,22,24,28")
    ap.add_argument("--model", type=str, default="ViT-H-14")
    ap.add_argument("--pretrained", type=str, default="laion2b_s32b_b79k")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--splits", type=str, default="training,test")
    args = ap.parse_args()

    cache = Path(os.environ.get("OPENCLIP_CACHE_DIR", "/project/peilab/why/cache/eeg-brainit/open_clip"))
    os.environ.setdefault("OPENCLIP_CACHE_DIR", str(cache))
    os.environ.setdefault("HF_HUB_CACHE", os.environ.get("HF_HUB_CACHE", "/project/peilab/why/cache/eeg-brainit/hf/hub"))

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    layers = parse_layers(args.layers)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    report = {
        "model": args.model,
        "pretrained": args.pretrained,
        "layers": layers,
        "device": str(device),
        "splits": {},
    }

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        paths = list_split_images(Path(args.images_root), split)
        feats = extract_layers(paths, layers, device, args.batch_size, args.model, args.pretrained)
        name = "train" if split == "training" else split
        split_dir = out / name
        split_dir.mkdir(parents=True, exist_ok=True)
        meta = {"n": len(paths), "layers": {}}
        for li, arr in feats.items():
            p = split_dir / f"layer_{li:02d}.npy"
            np.save(p, arr)
            meta["layers"][str(li)] = {"path": str(p), "shape": list(arr.shape)}
            print(f"[OK] {p} {arr.shape}")
        report["splits"][name] = meta
        (split_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    (out / "clip_layers_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
