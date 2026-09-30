#!/usr/bin/env python3
"""Cache SDXL VAE latents (float16 on disk). Encode in float32 to avoid NaNs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms


def list_split_images(images_root: Path, split: str) -> list[Path]:
    root = images_root / ("training_images" if split == "train" else "test_images")
    paths: list[Path] = []
    for d in sorted([p for p in root.iterdir() if p.is_dir()]):
        imgs = sorted(list(d.glob("*.jpg")) + list(d.glob("*.JPEG")) + list(d.glob("*.png")))
        paths.extend(imgs)
    return paths


def resolve_vae(hub: Path):
    from diffusers import AutoencoderKL

    sdxl_root = hub / "models--stabilityai--stable-diffusion-xl-base-1.0" / "snapshots"
    if sdxl_root.is_dir():
        for snap in sorted(sdxl_root.iterdir(), reverse=True):
            vae_dir = snap / "vae"
            if (vae_dir / "config.json").is_file():
                # force float32 encode — fp16 VAE encode produced all-NaN on H800
                return AutoencoderKL.from_pretrained(str(vae_dir), torch_dtype=torch.float32)
    return AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", subfolder="vae", torch_dtype=torch.float32
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-root", type=str, default="/project/peilab/why/data/images_set")
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--splits", type=str, default="train,test")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    hub = Path(os.environ.get("HF_HUB_CACHE", "/project/peilab/why/cache/eeg-brainit/hf/hub"))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    vae = resolve_vae(hub).to(device)
    vae.eval()
    tfm = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    scaling = float(getattr(vae.config, "scaling_factor", 0.13025))
    report = {"scaling_factor": scaling, "image_size": args.image_size, "dtype": "float16", "encode_dtype": "float32", "splits": {}}

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        npy = out / f"{split}_vae_latents_f16.npy"
        paths = list_split_images(Path(args.images_root), split)
        if npy.is_file() and npy.stat().st_size > 1000 and not args.force:
            arr = np.load(npy, mmap_mode="r")
            sample = np.asarray(arr[: min(32, len(arr))], dtype=np.float32)
            if np.isfinite(sample).all():
                print(f"[SKIP] {split} {arr.shape} finite OK")
                report["splits"][split] = {"n": int(arr.shape[0]), "skipped": True, "path": str(npy)}
                continue
            print(f"[WARN] {split} cache has NaNs — rebuilding")
            npy.unlink()

        h = w = args.image_size // 8
        mm = np.lib.format.open_memmap(str(npy), mode="w+", dtype=np.float16, shape=(len(paths), 4, h, w))
        bs = max(1, int(args.batch_size))
        n_bad = 0
        for start in tqdm(range(0, len(paths), bs), desc=f"vae-{split}"):
            batch_paths = paths[start : start + bs]
            imgs = torch.stack([tfm(Image.open(p).convert("RGB")) for p in batch_paths]).to(
                device=device, dtype=torch.float32
            )
            with torch.no_grad():
                dist = vae.encode(imgs).latent_dist
                lat = dist.mean * scaling  # mean more stable than mode in fp issues
            arr = lat.detach().cpu().numpy()
            if not np.isfinite(arr).all():
                n_bad += 1
                # retry with sample()
                with torch.no_grad():
                    lat = dist.sample() * scaling
                arr = lat.detach().cpu().numpy()
            if not np.isfinite(arr).all():
                raise RuntimeError(f"VAE NaN at batch start={start}")
            mm[start : start + len(batch_paths)] = arr.astype(np.float16)
            del imgs, lat, arr
        # validate
        chk = np.asarray(mm[: min(64, len(paths))], dtype=np.float32)
        report["splits"][split] = {
            "n": len(paths),
            "path": str(npy),
            "shape": [len(paths), 4, h, w],
            "finite_frac": float(np.isfinite(chk).mean()),
            "mean": float(np.mean(chk)),
            "std": float(np.std(chk)),
            "n_bad_batches_retried": n_bad,
        }
        del mm
        print(f"[OK] {split} finite={report['splits'][split]['finite_frac']} mean={report['splits'][split]['mean']:.4f}")

    (out / "vae_latent_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
