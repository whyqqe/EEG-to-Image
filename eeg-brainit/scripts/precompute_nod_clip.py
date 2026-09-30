#!/usr/bin/env python3
"""Precompute OpenCLIP embeddings for NOD ImageNet stimuli (by image_id)."""

from __future__ import annotations

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stimuli-dir",
        type=str,
        default="data/nod/raw/ds005811/stimuli/ImageNet",
    )
    parser.add_argument(
        "--events-glob",
        type=str,
        default="data/nod/raw/ds005811/derivatives/detailed_events/sub-*_events.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/nod/processed/clip_vit_h14",
    )
    parser.add_argument("--clip-model", type=str, default="ViT-H-14")
    parser.add_argument("--clip-pretrained", type=str, default="laion2b_s32b_b79k")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-images", type=int, default=0)
    args = parser.parse_args()

    cache_root = Path("/project/peilab/why/cache/eeg-brainit")
    os.environ.setdefault("HF_HOME", str(cache_root / "hf"))
    os.environ.setdefault("OPENCLIP_CACHE_DIR", str(cache_root / "open_clip"))
    os.environ.setdefault("TORCH_HOME", str(cache_root / "torch"))

    stim = ROOT / args.stimuli_dir
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    # Collect image_ids from events ONLY (paired trials). Do NOT scan all stimuli JPEGs
    # (NOD stimuli folder can contain ~80k ImageNet files unrelated to EEG trials).
    ids: set[str] = set()
    event_paths = sorted(ROOT.glob(args.events_glob))
    direct = ROOT / args.events_glob
    if not event_paths and direct.is_file():
        event_paths = [direct]
    if not event_paths:
        raise FileNotFoundError(f"No events matched: {args.events_glob}")
    for p in event_paths:
        with p.open("r", encoding="utf-8") as f:
            header = f.readline()
            cols = header.strip().split(",")
            idx = cols.index("image_id")
            for line in f:
                parts = line.strip().split(",")
                if len(parts) > idx:
                    ids.add(parts[idx].lower())
    image_ids = sorted(ids)
    if args.max_images > 0:
        image_ids = image_ids[: args.max_images]
    print(f"[INFO] image_ids={len(image_ids)} from {len(event_paths)} event files")

    paths = []
    missing = 0
    for iid in image_ids:
        p = stim / f"{iid}.JPEG"
        if not p.is_file():
            # try lowercase/uppercase variants
            alts = list(stim.glob(f"{iid}.*"))
            if not alts:
                missing += 1
                paths.append(None)
                continue
            p = alts[0]
        paths.append(p)
    print(f"[INFO] missing files={missing}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import open_clip

    tags = [args.clip_pretrained, "laion2b_s32b_b79k", "laion2b_s32b_b55k", "openai"]
    model = preprocess = None
    used_tag = None
    for tag in tags:
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                args.clip_model, pretrained=tag, device=device
            )
            used_tag = tag
            break
        except Exception as e:
            print(f"[WARN] clip tag {tag} failed: {e}")
    if model is None:
        raise RuntimeError("Failed to load OpenCLIP")
    model.eval()
    print(f"[INFO] device={device} clip={args.clip_model}/{used_tag}")

    dim = None
    embs = np.zeros((len(image_ids), 1024), dtype=np.float32)  # resized after first batch
    valid = np.zeros((len(image_ids),), dtype=np.bool_)
    with torch.no_grad():
        batch_imgs = []
        batch_idx = []
        for i, p in enumerate(tqdm(paths, desc="nod-clip")):
            if p is None:
                continue
            batch_imgs.append(preprocess(Image.open(p).convert("RGB")))
            batch_idx.append(i)
            if len(batch_imgs) >= args.batch_size or i == len(paths) - 1:
                if not batch_imgs:
                    continue
                x = torch.stack(batch_imgs, dim=0).to(device)
                feat = F.normalize(model.encode_image(x).float(), dim=-1).cpu().numpy().astype(np.float32)
                if dim is None:
                    dim = feat.shape[1]
                    embs = np.zeros((len(image_ids), dim), dtype=np.float32)
                for j, row in zip(batch_idx, feat):
                    embs[j] = row
                    valid[j] = True
                batch_imgs, batch_idx = [], []

    keep = [i for i, v in enumerate(valid) if v]
    image_ids_k = [image_ids[i] for i in keep]
    embs_k = embs[keep]
    np.save(out / "embeddings.npy", embs_k)
    (out / "image_ids.json").write_text(json.dumps(image_ids_k), encoding="utf-8")
    index = {iid: i for i, iid in enumerate(image_ids_k)}
    (out / "index.json").write_text(json.dumps(index), encoding="utf-8")
    meta = {
        "clip_model": args.clip_model,
        "clip_pretrained": used_tag,
        "dim": int(embs_k.shape[1]),
        "n": int(embs_k.shape[0]),
        "missing": missing,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[OK] wrote {out} n={meta['n']} dim={meta['dim']}")


if __name__ == "__main__":
    main()
