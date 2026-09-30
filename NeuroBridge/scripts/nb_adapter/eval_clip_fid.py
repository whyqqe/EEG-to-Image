#!/usr/bin/env python3
"""CLIP (ViT-H image↔GT) + FID for NB-generated image folders."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

BRAINIT = Path("/project/peilab/why/eeg-brainit")
sys.path.insert(0, str(BRAINIT / "scripts"))
sys.path.insert(0, str(BRAINIT / "src"))

from eval_atm_pipeline import image_metrics, list_test_images  # type: ignore
from erdc_fid_metrics import compute_fid  # type: ignore


def list_gen_images(gen_dir: Path) -> list[Path]:
    paths = sorted(gen_dir.glob("*.png"))
    if not paths:
        paths = sorted(gen_dir.glob("*.jpg"))
    return paths


def eval_one(
    gen_dir: Path,
    gt_paths: list[Path],
    device: torch.device,
    tag: str,
    batch_size: int,
) -> dict:
    gen_paths = list_gen_images(gen_dir)
    if not gen_paths:
        raise FileNotFoundError(f"no images in {gen_dir}")
    n = min(len(gen_paths), len(gt_paths))
    gen_paths = gen_paths[:n]
    gt = gt_paths[:n]
    clip = image_metrics(gen_paths, gt, device)
    fid = compute_fid(gen_dir, gt, device, batch_size=batch_size)
    return {
        "tag": tag,
        "gen_dir": str(gen_dir),
        "n": n,
        "clip_cosine": clip["clip_cosine"],
        "ssim": clip["ssim"],
        "pixcorr": clip["pixcorr"],
        "fid": fid["fid"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-root", type=str, required=True, help="parent with subdirs containing generated/")
    ap.add_argument("--tags", type=str, default="", help="comma-separated subdir names; default=all")
    ap.add_argument("--images-root", type=str, default="/project/peilab/why/data/images_set")
    ap.add_argument("--output-json", type=str, required=True)
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    cache = Path("/project/peilab/why/cache/eeg-brainit")
    os.environ.setdefault("HF_HOME", str(cache / "hf"))
    os.environ.setdefault("HF_HUB_CACHE", str(cache / "hf" / "hub"))
    os.environ.setdefault("OPENCLIP_CACHE_DIR", str(cache / "open_clip"))
    os.environ.setdefault("TORCH_HOME", str(cache / "torch"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    gen_root = Path(args.gen_root)
    gt_all = list_test_images(Path(args.images_root))
    if args.max_images > 0:
        gt_all = gt_all[: args.max_images]

    if args.tags.strip():
        tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    else:
        tags = sorted([p.name for p in gen_root.iterdir() if p.is_dir()])

    results = []
    for tag in tags:
        gen_dir = gen_root / tag / "generated"
        if not gen_dir.is_dir():
            print(f"[WARN] skip {tag}: no {gen_dir}")
            continue
        print(f"[INFO] eval {tag} n={len(list_gen_images(gen_dir))}")
        results.append(eval_one(gen_dir, gt_all, device, tag, args.batch_size))

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {"n_gt": len(gt_all), "results": results}
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[OK] {out}")


if __name__ == "__main__":
    main()
