#!/usr/bin/env python
"""Extract per-layer CLIP ViT-H-14 image features, for the alignment-layer scan.

Why this exists
---------------
The design doc says the single largest lever is *which layer you align to*
(§5.3 lists CLIP candidates {12, 20, 28, 32}, centred on 20-24; §7 phase 1 says
to sweep all 32 CLIP blocks). The implementation instead swept the *EEG pathway's*
depth, which is a different axis entirely, and so the sweep came back flat.

The cached features we had been aligning to are the *final* layer only:
`data/image_feature/ViT-H-14/image_train.npy` is 1024-d, which is exactly
`visual.proj`'s output width, i.e. the last block projected into the joint space.
So every arm so far asked the EEG encoder to hit CLIP's most semantic, most
abstraction-heavy representation, and there was no way to test any other depth.

This dumps the CLS token after every one of the 32 blocks (1280-d residual stream,
pre-projection) plus the final projected `_pooled` (1024-d, the shipped space).
One forward pass per image yields all 33 arrays, because the hooks are attached to
every block at once -- so scanning 32 layers costs the same as scanning one.

Verification is a hard gate, not a report
-----------------------------------------
`_pooled` must reproduce the shipped features. `test_fixes`-style discipline
applies: the shipped arrays define the concept order *and* the image-slot order,
and if our extraction disagrees on either, then every per-layer result is silently
misaligned and the layer comparison is meaningless. So a mismatch aborts the run
rather than emitting features that look fine.

Usage
-----
    python scripts/nwret/extract_layers.py --split train --batch 64 --workers 8
    python scripts/nwret/extract_layers.py --split test  --batch 64 --workers 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nwret import config

IMG_ROOT = config.DATA / "images_set"
SPLIT_DIR = {"train": "training_images", "test": "test_images"}
EXT = (".jpg", ".jpeg", ".png")


def list_concepts(split: str, limit: int = 0) -> list[tuple[str, list[str]]]:
    """[(concept_dir, [img_path, ...]), ...] in lexicographic (canonical) order.

    Lexicographic order on the zero-padded concept id is the canonical THINGS
    order, and it is the order the EEG arrays use. Assume nothing: the caller
    verifies it against the shipped feature arrays.
    """
    root = IMG_ROOT / SPLIT_DIR[split]
    out: list[tuple[str, list[str]]] = []
    for cid in sorted(d.name for d in root.iterdir() if d.is_dir()):
        imgs = sorted(f.name for f in (root / cid).iterdir()
                      if f.name.lower().endswith(EXT))
        if imgs:
            out.append((cid, [str(root / cid / f) for f in imgs]))
    return out[:limit] if limit else out


class ConceptImages(Dataset):
    """Flat list of (concept_index, image_path) so batching can be worker-parallel.

    Decoding is the bottleneck here (16540 JPEGs), not the GPU, so this yields raw
    PIL images and lets the collate function apply the transform on the worker.
    """

    def __init__(self, concepts: list[tuple[str, list[str]]], preprocess) -> None:
        self.items = [(ci, p) for ci, (_cid, paths) in enumerate(concepts) for p in paths]
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        ci, path = self.items[i]
        from PIL import Image
        with Image.open(path) as im:
            t = self.preprocess(im.convert("RGB"))
        return t, ci


def _as_rows(a: np.ndarray) -> np.ndarray:
    """(..., D) -> (n_rows, D), preserving row order."""
    return a.reshape(-1, a.shape[-1])


def collate(batch):
    xs = torch.stack([b[0] for b in batch])
    idx = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return xs, idx


@torch.no_grad()
def extract(split: str, batch: int, workers: int, limit: int, device: str) -> tuple[dict, list]:
    import open_clip

    concepts = list_concepts(split, limit)
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14", pretrained="laion2b_s32b_b79K")
    model = model.to(device).eval()
    visual = model.visual
    blocks = visual.transformer.resblocks
    n_blocks = len(blocks)
    d_model = int(visual.class_embedding.shape[0])

    n_img = np.array([len(p) for _c, p in concepts])
    print(f"[extract] split={split} concepts={len(concepts)} "
          f"imgs/concept={n_img.min()}..{n_img.max()} total={int(n_img.sum())}")
    print(f"[extract] blocks={n_blocks} d_model={d_model} "
          f"proj={None if visual.proj is None else tuple(visual.proj.shape)}")

    # Hook every block at once: one forward pass feeds all 33 outputs.
    acts: dict[str, torch.Tensor] = {}
    handles = []
    for i, blk in enumerate(blocks):
        handles.append(blk.register_forward_hook(
            lambda _m, _i, o, i=i: acts.__setitem__(
                f"block{i + 1:02d}", (o[0] if isinstance(o, (tuple, list)) else o))))

    ds = ConceptImages(concepts, preprocess)
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers,
                    collate_fn=collate, pin_memory=(device == "cuda"))

    # Accumulate on CPU as float32, one chunk list per key.
    acc: dict[str, list[np.ndarray]] = {}
    pos: dict[str, list[np.ndarray]] = {}
    t0 = time.time()
    done = 0
    for x, ci in dl:
        x = x.to(device, non_blocking=True)
        acts.clear()
        pooled = visual(x)
        pooled = pooled if torch.is_tensor(pooled) else pooled[0]

        for k, v in acts.items():
            # Block output is (B, L, D) here (open_clip batch_first=True); the
            # CLS token is index 0 and matches ln_post(cls) @ proj after projection.
            if v.shape[0] != x.shape[0]:
                v = v.permute(1, 0, 2)
            acc.setdefault(k, []).append(v[:, 0].float().cpu().numpy())
            pos.setdefault(k, []).append(ci.numpy())
        acc.setdefault("_pooled", []).append(pooled.float().cpu().numpy())
        pos.setdefault("_pooled", []).append(ci.numpy())

        done += x.shape[0]
        if done % (batch * 25) < batch:
            print(f"    {done}/{len(ds)} imgs  {time.time() - t0:.0f}s", flush=True)

    for h in handles:
        h.remove()

    feats: dict[str, np.ndarray] = {}
    for k, chunks in acc.items():
        v = np.concatenate(chunks, 0)
        idx = np.concatenate(pos[k], 0)
        # Sort back into concept-major order: DataLoader preserves order here, but
        # relying on that would make the alignment depend on an implementation
        # detail of shuffle=False plus worker prefetching.
        order = np.argsort(idx, kind="stable")
        feats[k] = v[order]
    return feats, [c for c, _ in concepts], n_img


def verify_against_shipped(feats: dict, split: str) -> dict:
    """Hard gate: `_pooled` must reproduce the shipped ViT-H-14 features.

    These arrays define both the concept order and the image-slot order used by
    every EEG array. If they disagree, per-layer features are misaligned with the
    EEG and the scan would compare layers using scrambled pairs.
    """
    shipped_p = config.IMAGE_FEATURE_DIR / f"image_{split}.npy"
    if not shipped_p.is_file():
        return {"checked": False, "reason": f"missing {shipped_p}"}
    shipped = np.load(shipped_p)
    ours = feats["_pooled"]

    # Compare as flat rows, not raw shapes. The shipped arrays carry an explicit
    # image axis -- (200, 1, 1024) on the test split, which has 1 image per concept
    # -- while ours is (200, 1024) when flattened. Comparing shapes directly made
    # this gate refuse to run on the test split, and a gate that cannot fire is
    # worse than no gate: it reads as verified in the manifest.
    A, B = _as_rows(ours), _as_rows(shipped)
    if A.shape != B.shape:
        return {"checked": False, "reason": f"rows {A.shape} vs shipped {B.shape}"}
    A = A.astype(np.float64)
    B = B.astype(np.float64)
    Ac = A - A.mean(0, keepdims=True)
    Bc = B - B.mean(0, keepdims=True)
    cos = (Ac * Bc).sum(1) / np.maximum(
        np.linalg.norm(Ac, axis=1) * np.linalg.norm(Bc, axis=1), 1e-12)
    return {"checked": True, "n_rows": int(A.shape[0]),
            "mean_cosine": float(cos.mean()), "min_cosine": float(cos.min()),
            "p01_cosine": float(np.percentile(cos, 1)),
            "frac_below_0.99": float((cos < 0.99).mean())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(config.OUTPUTS / "features" / "clip_h14_layers"))
    ap.add_argument("--split", default="train", choices=["train", "test", "both"])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="first N concepts (smoke only)")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    if a.device == "cuda" and not torch.cuda.is_available():
        a.device = "cpu"
        print("[warn] CUDA unavailable; running on CPU (smoke test only)")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    splits = ["train", "test"] if a.split == "both" else [a.split]

    t_all = time.time()
    for split in splits:
        print("=" * 88)
        feats, concepts, n_img = extract(split, a.batch, a.workers, a.limit, a.device)
        d = out / split
        d.mkdir(parents=True, exist_ok=True)

        manifest = {"split": split, "n_concepts": len(concepts),
                    "imgs_per_concept": int(n_img[0]) if n_img.min() == n_img.max() else "ragged",
                    "source": "open_clip ViT-H-14 laion2b_s32b_b79K, CLS token per block",
                    "layers": {}}
        for k, v in sorted(feats.items()):
            # Save with the image axis explicit, matching the shipped arrays:
            # (n_concepts, imgs_per_concept, D). Flattening the test split to
            # (200, D) looks harmless but breaks two downstream guards -- train.py
            # compares shape[:2] against the EEG's (200, 1, Ch, T), and the probe
            # indexes Yte[:, 0]. Both then either refuse to run or silently pick the
            # wrong axis, so keep the axis.
            arr = _as_rows(v)
            if n_img.min() == n_img.max():
                arr = arr.reshape(len(concepts), int(n_img[0]), v.shape[-1])
            elif arr.shape[0] != int(n_img.sum()):
                raise SystemExit(f"{k}: {arr.shape[0]} rows != {int(n_img.sum())} expected")
            manifest["layers"][k] = list(arr.shape)
            np.save(d / f"{k}.npy", arr.astype(np.float32))

        vr = verify_against_shipped(feats, split)
        manifest["verify_vs_shipped"] = vr
        (d / "manifest.json").write_text(json.dumps(manifest, indent=2))
        (d / "concepts.json").write_text(json.dumps({"split": split, "ids": concepts}, indent=2))

        print(f"[extract] {split}: {len(feats)} arrays -> {d}")
        for probe_key in ("block01", "block16", "block32", "_pooled"):
            if probe_key in manifest["layers"]:
                print(f"[extract]   {probe_key:>8} {manifest['layers'][probe_key]}")
        print(f"[verify ] vs shipped ViT-H-14: {json.dumps(vr)}")
        if not vr.get("checked"):
            print(f"[FATAL] could not verify against the shipped features: {vr.get('reason')}")
            print("        Without this the per-layer features may not be row-aligned with the EEG.")
            return 3
        if vr["mean_cosine"] < 0.99 or vr["min_cosine"] < 0.9:
            print(f"[FATAL] our _pooled does not reproduce the shipped features "
                  f"(mean cos {vr['mean_cosine']:.4f}, min {vr['min_cosine']:.4f}).")
            print("        Concept order or preprocessing differs -> layer features would be")
            print("        compared against the wrong EEG rows. Fix before proceeding.")
            return 4
        print("[verify ] OK: _pooled reproduces the shipped features; rows are aligned")

    print(f"\n[extract] done in {time.time() - t_all:.0f}s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
