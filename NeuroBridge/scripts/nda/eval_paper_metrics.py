#!/usr/bin/env python3
"""Paper-grade metrics: official PixCorr/SSIM (gray@425), CLIP cosine, FID."""

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

BRAINIT = Path("/project/peilab/why/eeg-brainit")
sys.path.insert(0, str(BRAINIT / "scripts"))
sys.path.insert(0, str(BRAINIT / "src"))

from erdc_fid_metrics import compute_fid  # type: ignore
from eval_atm_pipeline import list_test_images  # type: ignore


def list_gen(gen_dir: Path) -> list[Path]:
    paths = sorted(gen_dir.glob("*.png"))
    return paths or sorted(gen_dir.glob("*.jpg"))


def eval_folder(gen_dir: Path, gt_paths: list[Path], device: torch.device, tag: str, batch_size: int) -> dict:
    import open_clip

    gens = list_gen(gen_dir)
    n = min(len(gens), len(gt_paths))
    gens, gts = gens[:n], gt_paths[:n]
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14", pretrained="laion2b_s32b_b79k", device=device
    )
    model.eval()

    clips = []
    for i in tqdm(range(n), desc=f"paper-metrics[{tag}]"):
        g = Image.open(gens[i]).convert("RGB")
        t = Image.open(gts[i]).convert("RGB")
        with torch.no_grad():
            ge = F.normalize(model.encode_image(preprocess(g).unsqueeze(0).to(device)).float(), dim=-1)
            te = F.normalize(model.encode_image(preprocess(t).unsqueeze(0).to(device)).float(), dim=-1)
            clips.append(float((ge * te).sum()))

    # Official Ozcelik/ATM/CogCap low-level. RGB@256 / ssim_simple are not this protocol.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from eval_standard7 import lowlevel
    ll = lowlevel(gens, gts)

    fid = compute_fid(gen_dir, gts, device, batch_size=batch_size)
    return {
        "tag": tag,
        "gen_dir": str(gen_dir),
        "n": n,
        "clip_cosine": float(np.mean(clips)),
        "pixcorr": float(ll["pixcorr"]),
        "ssim": float(ll["ssim"]),
        "ssim_skimage": float(ll["ssim"]),
        "fid": fid["fid"],
        "ssim_protocol": "skimage gray@425 gaussian σ=1.5 use_sample_covariance=False data_range=1.0",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-root", type=str, required=True)
    ap.add_argument("--tags", type=str, default="")
    ap.add_argument("--images-root", type=str, default="/project/peilab/why/data/images_set")
    ap.add_argument("--output-json", type=str, required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    cache = Path("/project/peilab/why/cache/eeg-brainit")
    os.environ.setdefault("HF_HOME", str(cache / "hf"))
    os.environ.setdefault("HF_HUB_CACHE", str(cache / "hf" / "hub"))
    os.environ.setdefault("OPENCLIP_CACHE_DIR", str(cache / "open_clip"))
    os.environ.setdefault("TORCH_HOME", str(cache / "torch"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen_root = Path(args.gen_root)
    gt = list_test_images(Path(args.images_root))
    tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags.strip() else sorted(
        p.name for p in gen_root.iterdir() if p.is_dir()
    )
    results = []
    for tag in tags:
        gdir = gen_root / tag / "generated"
        if not gdir.is_dir() or not list_gen(gdir):
            print(f"[WARN] skip {tag}")
            continue
        results.append(eval_folder(gdir, gt, device, tag, args.batch_size))

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_gt": len(gt),
        "ssim_note": "SSIM/PixCorr from eval_standard7.lowlevel (official gray@425 gaussian)",
        "results": results,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"[OK] {out}")


if __name__ == "__main__":
    main()
