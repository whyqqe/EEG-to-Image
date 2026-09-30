#!/usr/bin/env python3
"""Random-sample reconstruction comparison grid for ViT-H NB experiments."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image

BRAINIT = Path("/project/peilab/why/eeg-brainit")
sys.path.insert(0, str(BRAINIT / "scripts"))
from eval_atm_pipeline import list_test_images  # type: ignore


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", type=str, required=True)
    ap.add_argument("--images-root", type=str, default="/project/peilab/why/data/images_set")
    ap.add_argument("--num-samples", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--columns", type=str, default="gt,mlp,teacher,atm_prior")
    ap.add_argument("--titles", type=str, default="GT,ViT-H NB+MLP,Teacher,ATM prior")
    args = ap.parse_args()

    base = Path(args.base_dir)
    gen_root = base / "generation"
    cols = [c.strip() for c in args.columns.split(",")]
    titles = [t.strip() for t in args.titles.split(",")]
    if len(cols) != len(titles):
        raise ValueError("columns and titles length mismatch")

    gt_paths = list_test_images(Path(args.images_root))
    n_gt = len(gt_paths)

    # Only use indices where every column has a generated image (handles partial runs).
    valid: list[int] = []
    for idx in range(n_gt):
        ok = True
        for col in cols:
            if col == "gt":
                continue
            p = gen_root / col / "generated" / f"{idx:03d}.png"
            if not p.is_file():
                ok = False
                break
        if ok:
            valid.append(idx)
    if not valid:
        raise RuntimeError(f"no complete rows found under {gen_root}")

    rng = random.Random(args.seed)
    k = min(args.num_samples, len(valid))
    indices = sorted(rng.sample(valid, k))

    rows = len(indices)
    fig = plt.figure(figsize=(2.2 * len(cols), 2.2 * rows))
    gs = GridSpec(rows, len(cols) + 1, width_ratios=[1] * len(cols) + [0.08], wspace=0.02, hspace=0.08)

    for r, idx in enumerate(indices):
        for c, col in enumerate(cols):
            ax = fig.add_subplot(gs[r, c])
            if col == "gt":
                img_path = gt_paths[idx]
            else:
                gen_dir = gen_root / col / "generated"
                img_path = gen_dir / f"{idx:03d}.png"
                if not img_path.is_file():
                    raise FileNotFoundError(img_path)
            ax.imshow(load_rgb(img_path))
            ax.axis("off")
            if r == 0:
                ax.set_title(titles[c], fontsize=11, pad=4)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    meta = {
        "seed": args.seed,
        "indices": indices,
        "columns": cols,
        "titles": titles,
        "output": str(out),
        "gt_paths": [str(gt_paths[i]) for i in indices],
    }
    meta_path = out.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[OK] saved {out}")
    print(f"[OK] meta {meta_path}")


if __name__ == "__main__":
    main()
