#!/usr/bin/env python
"""Validate the extracted multimodal targets before any training consumes them.

Two classes of check, because they fail differently:

1. *Completeness.* Targets are written into a preallocated memmap, so a killed
   job leaves valid-looking `zeros` in the unfinished rows.  A missing row is
   therefore indistinguishable from a real value by inspection, and the only
   trustworthy completeness signal is the atomic progress sidecar plus the
   manifest.  This script fails loudly if any target is short.

2. *Semantic alignment.* Shapes matching proves nothing about whether image `i`
   was paired with the right concept.  The end-to-end check is CLIP zero-shot
   retrieval over the test split: take each test image's CLIP image embedding,
   rank it against all 200 concept text embeddings, and require the correct
   concept to come first.  CLIP is a strong model on THINGS stimuli, so a healthy
   pipeline scores near-perfectly; a misordered index, a wrong axis, or a
   mismatched preprocessing chain collapses this to near chance (0.5%).  This is
   the single most valuable check in the stage.

Run this as the gate at the end of the data-prep job.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from loso import paths
from loso.data import things

# Minimum acceptable CLIP zero-shot top-1 over the 200 test concepts.  The true
# value is ~0.95-1.0; anything below this floor means the pairing is broken, not
# that CLIP is having a bad day.
MIN_CLIP_TOPK1 = 0.60


def check_progress(name: str, split: str, n_rows: int) -> tuple[bool, str]:
    arr_path = paths.TARGET_DIR / f"{name}_{split}.npy"
    prog_path = arr_path.with_suffix(arr_path.suffix + ".progress.json")
    if not arr_path.is_file():
        return False, f"missing {arr_path}"
    if not prog_path.is_file():
        return False, f"missing progress sidecar {prog_path} (completeness unverifiable)"
    payload = json.loads(prog_path.read_text())
    if payload.get("n_rows") != n_rows:
        return False, f"progress n_rows={payload.get('n_rows')} != {n_rows}"
    done = len(payload.get("done", []))
    if done < n_rows:
        return False, f"{done}/{n_rows} rows written"
    return True, f"{n_rows} rows"


def find_nonfinite_rows(arr: np.ndarray, chunk: int = 512) -> int:
    """Count rows containing any NaN/Inf -- the fp16-VAE failure mode."""
    if not np.issubdtype(arr.dtype, np.floating):
        return 0
    total = 0
    for start in range(0, arr.shape[0], chunk):
        block = np.asarray(arr[start:start + chunk])
        flat = block.reshape(block.shape[0], -1)
        total += int((~np.isfinite(flat)).any(axis=1).sum())
    return total


def find_all_zero_rows(arr: np.ndarray, chunk: int = 512) -> int:
    """Count rows that are exactly zero -- a sentinel for never-written rows."""
    total = 0
    for start in range(0, arr.shape[0], chunk):
        block = np.asarray(arr[start:start + chunk])
        flat = block.reshape(block.shape[0], -1)
        total += int((np.abs(flat).sum(axis=1) == 0).sum())
    return total


def clip_zeroshot_check(split: str, n_concepts: int) -> tuple[bool, str]:
    """CLIP image -> concept text top-1 over the split's unique concepts."""
    img = np.load(paths.TARGET_DIR / f"clip_image_{split}.npy", mmap_mode="r")
    txt = np.load(paths.TARGET_DIR / f"clip_text_concept_{split}.npy", mmap_mode="r")

    records = things.build_index(split)
    # One representative image per concept: the first slot.  For the test split
    # that is the only slot; for train it is image_index 0 of each concept.
    k = paths.N_IMAGES_PER_CONCEPT if split == "train" else 1
    probe_idx = [c * k for c in range(n_concepts)]

    e_img = torch.from_numpy(np.asarray(img[probe_idx], dtype=np.float32))
    e_txt = torch.from_numpy(np.asarray(txt[:n_concepts], dtype=np.float32))
    e_img = torch.nn.functional.normalize(e_img, dim=-1)
    e_txt = torch.nn.functional.normalize(e_txt, dim=-1)

    sim = e_img @ e_txt.T
    ranks = sim.argsort(dim=-1, descending=True)
    labels = torch.arange(n_concepts)
    top1 = (ranks[:, 0] == labels).float().mean().item()
    top5 = (ranks[:, :5] == labels[:, None]).any(dim=-1).float().mean().item()

    ok = top1 >= MIN_CLIP_TOPK1
    msg = (f"CLIP zero-shot on {split}: top1={top1:.3f} top5={top5:.3f} "
           f"over {n_concepts} concepts (floor {MIN_CLIP_TOPK1})")
    if not ok:
        msg += ("\n        -> image/concept pairing or CLIP preprocessing is broken;"
                " the index order and the image_processor must both be checked")
    return ok, msg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--skip-semantic", action="store_true",
                    help="only check completeness (semantic check needs clip targets)")
    args = ap.parse_args()

    manifest_path = paths.TARGET_DIR / "manifest.json"
    if not manifest_path.is_file():
        print(f"[FATAL] no manifest at {manifest_path}; run prep_multimodal.py first")
        return 2
    manifest = json.loads(manifest_path.read_text())

    failures: list[str] = []
    print("=== completeness ===")
    for key, meta in sorted(manifest.items()):
        if key == "config" or not isinstance(meta, dict):
            continue
        name, _, split = key.rpartition("_")
        if split not in args.splits:
            continue
        n_rows = meta["shape"][0]
        ok, msg = check_progress(name, split, n_rows)

        # Cross-check the declared shape against the file on disk, so a stale
        # manifest cannot vouch for a replaced array.
        arr_path = paths.TARGET_DIR / f"{name}_{split}.npy"
        if arr_path.is_file():
            arr = np.load(arr_path, mmap_mode="r")
            if list(arr.shape) != list(meta["shape"]):
                ok, msg = False, f"on-disk shape {arr.shape} != manifest {meta['shape']}"
            zeros = find_all_zero_rows(arr)
            if zeros:
                ok, msg = False, f"{zeros} all-zero rows present"
            bad = find_nonfinite_rows(arr)
            if bad:
                ok, msg = False, f"{bad} rows contain NaN/Inf"
        status = "OK  " if ok else "FAIL"
        print(f"  [{status}] {key:28s} {msg}")
        if not ok:
            failures.append(key)

    if not args.skip_semantic and not failures:
        print("=== semantic alignment ===")
        for split, n_concepts in (("test", paths.N_TEST_CONCEPTS),
                                  ("train", paths.N_TRAIN_CONCEPTS)):
            if split not in args.splits:
                continue
            need = [paths.TARGET_DIR / f"clip_image_{split}.npy",
                    paths.TARGET_DIR / f"clip_text_concept_{split}.npy"]
            if not all(p.is_file() for p in need):
                print(f"  [skip] {split}: clip targets absent")
                continue
            ok, msg = clip_zeroshot_check(split, n_concepts)
            print(f"  [{'OK  ' if ok else 'FAIL'}] {msg}")
            if not ok:
                failures.append(f"zeroshot_{split}")

    if failures:
        print(f"\n[FATAL] {len(failures)} check(s) failed: {failures}", file=sys.stderr)
        return 1
    print("\n[OK] all target checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
