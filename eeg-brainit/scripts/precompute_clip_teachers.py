#!/usr/bin/env python3
"""Precompute OpenCLIP image embeddings for THINGS-EEG2 teacher alignment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def path_key(p: str) -> str:
    return str(Path(p).resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default="data/processed/manifest.jsonl")
    parser.add_argument("--output-dir", type=str, default="data/processed/teachers/clip_ViT-L-14_openai")
    parser.add_argument("--clip-model", type=str, default="ViT-L-14")
    parser.add_argument("--clip-pretrained", type=str, default="openai")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-images", type=int, default=0, help="0 = all unique images")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    root = ROOT
    manifest = root / args.manifest
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_root = Path("/project/peilab/why/cache/eeg-brainit")
    os.environ.setdefault("HF_HOME", str(cache_root / "hf"))
    os.environ.setdefault("OPENCLIP_CACHE_DIR", str(cache_root / "open_clip"))
    os.environ.setdefault("TORCH_HOME", str(cache_root / "torch"))

    paths = []
    seen = set()
    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            p = path_key(rec["image"])
            if p in seen:
                continue
            if not Path(p).is_file():
                raise FileNotFoundError(p)
            seen.add(p)
            paths.append(p)
    if args.smoke:
        args.max_images = min(128, len(paths))
    if args.max_images > 0:
        paths = paths[: args.max_images]
    print(f"[INFO] unique images={len(paths)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        args.clip_model, pretrained=args.clip_pretrained, device=device
    )
    model.eval()

    embs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(paths), args.batch_size), desc="clip-teacher"):
            batch_paths = paths[i : i + args.batch_size]
            imgs = torch.stack(
                [preprocess(Image.open(p).convert("RGB")) for p in batch_paths], dim=0
            ).to(device)
            feat = model.encode_image(imgs)
            feat = F.normalize(feat.float(), dim=-1)
            embs.append(feat.cpu().numpy().astype(np.float32))
    emb = np.concatenate(embs, axis=0)
    assert emb.shape[0] == len(paths)

    np.save(out_dir / "embeddings.npy", emb)
    with (out_dir / "paths.json").open("w", encoding="utf-8") as f:
        json.dump(paths, f)
    index = {p: i for i, p in enumerate(paths)}
    with (out_dir / "index.json").open("w", encoding="utf-8") as f:
        json.dump(index, f)
    meta = {
        "clip_model": args.clip_model,
        "clip_pretrained": args.clip_pretrained,
        "dim": int(emb.shape[1]),
        "n": int(emb.shape[0]),
        "fingerprint": hashlib.sha1("\n".join(paths).encode()).hexdigest()[:12],
    }
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[OK] wrote {out_dir} shape={emb.shape} meta={meta}")


if __name__ == "__main__":
    main()
