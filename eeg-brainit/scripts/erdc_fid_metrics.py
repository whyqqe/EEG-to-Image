#!/usr/bin/env python3
"""FID between generated images and THINGS-EEG2 test GT (distribution-level).

Uses torchmetrics FrechetInceptionDistance (Inception-v3 features).
Writes JSON under --output-json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from eval_atm_pipeline import list_test_images


def list_gen(gen_dir: Path, n: int) -> list[Path]:
    out = []
    for i in range(n):
        p = gen_dir / f"{i:03d}.png"
        if not p.is_file():
            p = gen_dir / f"{i:03d}.jpg"
        if not p.is_file():
            raise FileNotFoundError(p)
        out.append(p)
    return out


def load_tensor(path: Path, tfm: transforms.Compose, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return tfm(img).unsqueeze(0).to(device)


@torch.no_grad()
def compute_fid(
    gen_dir: Path,
    gt_paths: list[Path],
    device: torch.device,
    batch_size: int = 16,
) -> dict:
    n = len(gt_paths)
    gen_paths = list_gen(gen_dir, n)
    fid = FrechetInceptionDistance(normalize=True).to(device)
    tfm = transforms.Compose(
        [
            transforms.Resize((299, 299), antialias=True),
            transforms.ToTensor(),
        ]
    )

    for start in tqdm(range(0, n, batch_size), desc=f"FID-real[{gen_dir.name}]"):
        end = min(start + batch_size, n)
        batch = torch.cat([load_tensor(gt_paths[i], tfm, device) for i in range(start, end)], dim=0)
        fid.update(batch, real=True)

    for start in tqdm(range(0, n, batch_size), desc=f"FID-fake[{gen_dir.name}]"):
        end = min(start + batch_size, n)
        batch = torch.cat([load_tensor(gen_paths[i], tfm, device) for i in range(start, end)], dim=0)
        fid.update(batch, real=False)

    score = float(fid.compute().item())
    return {"n": n, "fid": score, "gen_dir": str(gen_dir)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen-dir", type=str, required=True)
    parser.add_argument("--images-root", type=str, default="/project/peilab/why/data/images_set")
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-images", type=int, default=0)
    args = parser.parse_args()

    cache_root = Path("/project/peilab/why/cache/eeg-brainit")
    os.environ.setdefault("TORCH_HOME", str(cache_root / "torch"))
    os.environ.setdefault("OPENCLIP_CACHE_DIR", str(cache_root / "open_clip"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen_dir = Path(args.gen_dir)
    if not gen_dir.is_absolute():
        gen_dir = ROOT / gen_dir
    gt = list_test_images(Path(args.images_root))
    if args.max_images > 0:
        gt = gt[: args.max_images]

    report = compute_fid(gen_dir, gt, device, batch_size=args.batch_size)
    if args.tag:
        report["tag"] = args.tag

    out = Path(args.output_json)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[OK] {args.tag or gen_dir.name}: FID={report['fid']:.2f}")
    print(f"[OK] wrote {out}")


if __name__ == "__main__":
    main()
