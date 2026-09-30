#!/usr/bin/env python3
"""Merge multiple ERDC candidate banks into one mega-bank for fused reselect."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def resolve_cand(path: Path) -> Path:
    if (path / "candidates").is_dir():
        return path / "candidates"
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-dirs", type=str, nargs="+", required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-images", type=int, default=0)
    args = parser.parse_args()

    banks: list[Path] = []
    for d in args.bank_dirs:
        p = Path(d)
        if not p.is_absolute():
            p = ROOT / p
        banks.append(resolve_cand(p))

    out_cand = Path(args.output_dir)
    if not out_cand.is_absolute():
        out_cand = ROOT / out_cand
    out_cand.mkdir(parents=True, exist_ok=True)
    out_root = out_cand.parent if out_cand.name == "candidates" else out_cand
    if out_cand.name != "candidates":
        out_cand = out_root / "candidates"
        out_cand.mkdir(parents=True, exist_ok=True)

    nb_src = banks[0].parent / "neighbor_idx.npy"
    if nb_src.is_file():
        shutil.copy2(nb_src, out_root / "neighbor_idx.npy")

    merged_specs: list[dict] = []
    feat_chunks: list[np.ndarray] = []
    bank_meta: list[tuple[Path, list[dict]]] = []
    n: int | None = None
    k_off = 0

    for cand in banks:
        specs = json.loads((cand / "specs.json").read_text())
        feats = np.load(cand / "clip_feats_flat.npy")
        k = len(specs)
        feats2 = feats.reshape(-1, k, feats.shape[-1])
        if args.max_images > 0:
            feats2 = feats2[: args.max_images]
        if n is None:
            n = feats2.shape[0]
        elif feats2.shape[0] != n:
            raise RuntimeError(f"bank {cand} n={feats2.shape[0]} != {n}")
        bank_meta.append((cand, specs))
        for sp in specs:
            merged_specs.append(
                {
                    **sp,
                    "k": k_off + int(sp["k"]),
                    "_src_bank": str(cand),
                    "_src_k": int(sp["k"]),
                }
            )
        feat_chunks.append(feats2)
        k_off += k

    merged_feats = np.concatenate(feat_chunks, axis=1)
    np.save(out_cand / "clip_feats_flat.npy", merged_feats.reshape(n, -1, merged_feats.shape[-1]))

    for i in range(n):
        for sp in merged_specs:
            src = Path(sp["_src_bank"]) / f"{i:03d}_k{sp['_src_k']}.png"
            dst = out_cand / f"{i:03d}_k{sp['k']}.png"
            if not dst.is_file():
                shutil.copy2(src, dst)

    clean_specs = [{k: v for k, v in sp.items() if not k.startswith("_")} for sp in merged_specs]
    (out_cand / "specs.json").write_text(json.dumps(clean_specs, indent=2), encoding="utf-8")
    print(f"[OK] merged {len(banks)} banks -> {out_cand} n={n} k={len(clean_specs)}")


if __name__ == "__main__":
    main()
