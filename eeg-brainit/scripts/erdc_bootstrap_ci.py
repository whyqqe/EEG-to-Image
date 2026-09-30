#!/usr/bin/env python3
"""Bootstrap 95% CI for PixCorr and CLIP cosine (per-image paired metrics)."""

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
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from eeg_brainit.utils.metrics import pixel_correlation, ssim_simple
from eval_atm_pipeline import list_test_images


def list_gen(gen_dir: Path, n: int) -> list[Path]:
    paths = []
    for i in range(n):
        p = gen_dir / f"{i:03d}.png"
        if not p.is_file():
            p = gen_dir / f"{i:03d}.jpg"
        paths.append(p)
    return paths


@torch.no_grad()
def per_image_metrics(gen_dir: Path, gt_paths: list[Path], device: torch.device) -> dict:
    import open_clip
    from torchvision import transforms

    n = len(gt_paths)
    gen_paths = list_gen(gen_dir, n)
    to_tensor = transforms.Compose(
        [
            transforms.Resize((256, 256), antialias=True),
            transforms.ToTensor(),
        ]
    )
    clip_model, _, clip_pre = open_clip.create_model_and_transforms(
        "ViT-H-14", pretrained="laion2b_s32b_b79k", device=device
    )
    clip_model.eval()

    pix, clip, ssim = [], [], []
    for i in range(n):
        g = Image.open(gen_paths[i]).convert("RGB")
        t = Image.open(gt_paths[i]).convert("RGB")
        gp = to_tensor(g).unsqueeze(0).to(device)
        gt = to_tensor(t).unsqueeze(0).to(device)
        pix.append(pixel_correlation(gp, gt))
        ssim.append(ssim_simple(gp, gt))
        ge = F.normalize(
            clip_model.encode_image(clip_pre(g).unsqueeze(0).to(device)).float(), dim=-1
        )
        te = F.normalize(
            clip_model.encode_image(clip_pre(t).unsqueeze(0).to(device)).float(), dim=-1
        )
        clip.append(float((ge * te).sum()))
    del clip_model
    return {
        "pixcorr": np.array(pix, dtype=np.float64),
        "clip_cosine": np.array(clip, dtype=np.float64),
        "ssim": np.array(ssim, dtype=np.float64),
    }


def bootstrap_ci(arr: np.ndarray, n_boot: int = 2000, seed: int = 42) -> dict:
    rng = np.random.RandomState(seed)
    n = len(arr)
    means = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        means.append(float(arr[idx].mean()))
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot) - 1]
    return {"mean": float(arr.mean()), "ci95_lo": lo, "ci95_hi": hi}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen-dir", type=str, required=True)
    parser.add_argument("--images-root", type=str, default="/project/peilab/why/data/images_set")
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()

    cache_root = Path("/project/peilab/why/cache/eeg-brainit")
    os.environ.setdefault("TORCH_HOME", str(cache_root / "torch"))
    os.environ.setdefault("OPENCLIP_CACHE_DIR", str(cache_root / "open_clip"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen_dir = Path(args.gen_dir)
    if not gen_dir.is_absolute():
        gen_dir = ROOT / gen_dir
    gt = list_test_images(Path(args.images_root))
    per = per_image_metrics(gen_dir, gt, device)

    report = {
        "n": len(gt),
        "gen_dir": str(gen_dir),
        "tag": args.tag,
        "pixcorr": bootstrap_ci(per["pixcorr"], args.n_boot),
        "clip_cosine": bootstrap_ci(per["clip_cosine"], args.n_boot),
        "ssim": bootstrap_ci(per["ssim"], args.n_boot),
    }

    out = Path(args.output_json)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    p = report["pixcorr"]
    c = report["clip_cosine"]
    print(
        f"[OK] {args.tag}: Pix={p['mean']:.4f} [{p['ci95_lo']:.4f},{p['ci95_hi']:.4f}] "
        f"CLIP={c['mean']:.4f} [{c['ci95_lo']:.4f},{c['ci95_hi']:.4f}]"
    )
    print(f"[OK] wrote {out}")


if __name__ == "__main__":
    main()
