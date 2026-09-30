#!/usr/bin/env python3
"""Extract CLIP ViT-H-14 SPATIAL patch tokens for the TRAIN rows (iREPA targets).

WHY THIS EXISTS
---------------
Every target in this project is a pooled vector: `sem_image_*.npy` is
(1280,) per image, and the cached `clip_layers/layer_*.npy` are pooled too.  So
nothing in the project could align EEG to a SPATIAL feature map, which is what
REPA/iREPA (reference-based representation alignment) do for diffusion training.
This script produces that target: the 8x8 pooled patch tokens of the SAME CLIP
model (`ViT-H-14` / `laion2b_s32b_b79k`) whose pooled embedding the semantic
tower already regresses to, so both live in one geometry.

TRAIN ONLY, ON PURPOSE
  iREPA is a TRAINING-TIME auxiliary objective, so it needs targets for the rows
  the model is fitted on and nothing else.  The 200 test images are NEVER encoded
  here -- that is what keeps the alignment out of the leak-free boundary.  (The
  row ORDER is inherited from `captions_train.jsonl`, which is the same file the
  target builder walks, so row i of this cache is row i of `sem_*_train.npy`;
  `tdm_train.py` asserts the length matches and refuses to align otherwise.)

OUTPUT
  (N, 64, 1280) float16 = the 256 patch tokens of a 16x16 grid, spatially
  average-pooled 2x2 and flattened row-major, so token 8*r+c corresponds to the
  (r, c) cell of an 8x8 grid.  That is the exact layout the structural head uses
  to reshape its 64 tokens into the (4, 64, 64) latent, and the 64x64 latent grid
  divides evenly into 8x8 cells, so the correspondence is exact and not a
  convention that could silently drift.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

NB_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions-jsonl", type=str,
                    default=str(NB_ROOT / "outputs/g2/captions/captions_train.jsonl"))
    ap.add_argument("--out", type=str,
                    default=str(NB_ROOT / "outputs/tdm/clip_patch/train_patch_f16.npy"))
    ap.add_argument("--model", type=str, default="ViT-H-14")
    ap.add_argument("--pretrained", type=str, default="laion2b_s32b_b79k")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--pool", type=int, default=2, help="spatial pooling factor of the "
                    "16x16 patch grid; 2 -> 8x8 = 64 tokens")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    if out.is_file():
        a = np.load(out, mmap_mode="r")
        print(f"[patch] exists {out} {a.shape}; nothing to do")
        return
    out.parent.mkdir(parents=True, exist_ok=True)

    paths: list[str] = []
    for line in Path(args.captions_jsonl).read_text(encoding="utf-8").splitlines():
        if line.strip():
            paths.append(json.loads(line)["path"])
    if args.limit > 0:
        paths = paths[:args.limit]
    n = len(paths)
    print(f"[patch] {n} TRAIN images from {args.captions_jsonl}")

    import open_clip
    from PIL import Image

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained, device=dev)
    model.eval()
    d = model.visual.width if hasattr(model.visual, "width") else 1280
    print(f"[patch] {args.model}/{args.pretrained} on {dev} (dim {d})")

    P = args.pool
    # (N, 64, 1280) float16: 16540 * 64 * 1280 * 2 B ~= 2.7 GB, which is why the
    # full 256-token grid is pooled rather than stored raw (10.9 GB).
    arr = np.lib.format.open_memmap(out.with_suffix(".tmp.npy"), mode="w+",
                                    dtype=np.float16, shape=(n, 256 // (P * P), d))
    with torch.no_grad():
        for i in range(0, n, args.batch_size):
            chunk = paths[i:i + args.batch_size]
            imgs = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in chunk]).to(dev)
            # `forward_intermediates(..., output_fmt="NLC")` is the only form that
            # returns the SPATIAL token grid: the default output_fmt gives
            # (B, C, H, W) and the plain encode_image gives the pooled vector.
            # `indices=-1` keeps only the last block, because materialising all 32
            # blocks would cost 32x the memory for no information we use.
            # `indices=1` means "the LAST block" (this API counts from the end but
            # rejects -1), which is what we want: materialising all 32 blocks would
            # cost 32x the memory for no information we use.
            try:
                out_d = model.visual.forward_intermediates(imgs, indices=1, output_fmt="NLC")
            except TypeError:
                out_d = model.visual.forward_intermediates(imgs, output_fmt="NLC")
            tok = out_d["image_intermediates"]
            tok = tok[-1] if isinstance(tok, (list, tuple)) else tok
            if tok.dim() == 4:                                   # (B, C, H, W)
                tok = tok.flatten(2).transpose(1, 2)
            if tok.shape[1] == 257:                              # drop the CLS token
                tok = tok[:, 1:]
            g = int(round(tok.shape[1] ** 0.5))
            if g * g != tok.shape[1]:
                raise SystemExit(f"[FATAL] {tok.shape[1]} patch tokens is not a square grid")
            if tok.shape[-1] != d:
                raise SystemExit(f"[FATAL] token dim {tok.shape[-1]} != expected {d}")
            # spatial 2x2 average pooling: (B, g, g, d) -> (B, g/P, P, g/P, P, d)
            t = tok.reshape(len(chunk), g // P, P, g // P, P, -1).mean(dim=(2, 4))
            t = t.reshape(len(chunk), -1, d)
            arr[i:i + len(chunk)] = t.half().cpu().numpy()
            if i % (args.batch_size * 40) == 0:
                print(f"[patch] {i}/{n}", flush=True)
    arr.flush()
    del arr
    out.with_suffix(".tmp.npy").rename(out)
    print(f"[patch] wrote {out} {np.load(out, mmap_mode='r').shape}")


if __name__ == "__main__":
    main()
