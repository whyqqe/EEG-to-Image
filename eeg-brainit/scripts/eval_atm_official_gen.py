#!/usr/bin/env python3
"""Download official ATM generated images (sub-08) and compute image metrics vs GT."""

from __future__ import annotations

import argparse
import json
import os
import tarfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from eeg_brainit.utils.metrics import pixel_correlation, ssim_simple


def list_test_images(images_root: Path) -> list[Path]:
    root = images_root / "test_images"
    paths = []
    for d in sorted([p for p in root.iterdir() if p.is_dir()]):
        imgs = sorted(list(d.glob("*.jpg")) + list(d.glob("*.JPEG")) + list(d.glob("*.png")))
        paths.append(imgs[0])
    return paths


def extract_subject(tar_path: Path, subject: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / subject
    if marker.is_dir() and any(marker.rglob("*.png")):
        print(f"[INFO] already extracted {marker}")
        return marker
    with tarfile.open(tar_path, "r:gz") as tar:
        members = [m for m in tar.getmembers() if subject in m.name and (m.name.endswith(".png") or m.name.endswith(".jpg"))]
        print(f"[INFO] extracting {len(members)} files for {subject}")
        for m in tqdm(members, desc="extract"):
            tar.extract(m, path=out_dir)
    # Find the subject root folder inside extract
    cands = list(out_dir.rglob(subject))
    dirs = [c for c in cands if c.is_dir()]
    if not dirs:
        raise FileNotFoundError(f"No {subject} folder after extract under {out_dir}")
    # Prefer deepest / generated_imgs/sub-08 style
    dirs = sorted(dirs, key=lambda p: len(p.parts), reverse=True)
    return dirs[0]


def collect_gen_paths(subject_dir: Path) -> list[Path]:
    """Prefer one image per concept folder, sorted."""
    # Common layouts: subject_dir/<concept>/*.png OR subject_dir/*.png
    subdirs = sorted([p for p in subject_dir.iterdir() if p.is_dir()])
    paths: list[Path] = []
    if subdirs:
        for d in subdirs:
            imgs = sorted(list(d.glob("*.png")) + list(d.glob("*.jpg")))
            if imgs:
                paths.append(imgs[0])
    else:
        paths = sorted(list(subject_dir.glob("*.png")) + list(subject_dir.glob("*.jpg")))
    return paths


@torch.no_grad()
def metrics(gen_paths: list[Path], gt_paths: list[Path], device: torch.device) -> dict:
    import open_clip

    n = min(len(gen_paths), len(gt_paths))
    to_tensor = transforms.Compose(
        [transforms.Resize((256, 256), antialias=True), transforms.ToTensor()]
    )
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14", pretrained="laion2b_s32b_b79k", device=device
    )
    model.eval()
    pixs, ssims, clips = [], [], []
    for i in tqdm(range(n), desc="metrics"):
        g = Image.open(gen_paths[i]).convert("RGB")
        t = Image.open(gt_paths[i]).convert("RGB")
        gp = to_tensor(g).unsqueeze(0).to(device)
        gt = to_tensor(t).unsqueeze(0).to(device)
        pixs.append(pixel_correlation(gp, gt))
        ssims.append(ssim_simple(gp, gt))
        ge = F.normalize(model.encode_image(preprocess(g).unsqueeze(0).to(device)).float(), dim=-1)
        te = F.normalize(model.encode_image(preprocess(t).unsqueeze(0).to(device)).float(), dim=-1)
        clips.append(float((ge * te).sum()))
    return {
        "n": n,
        "pixcorr": float(np.mean(pixs)),
        "ssim": float(np.mean(ssims)),
        "clip_cosine": float(np.mean(clips)),
        "gen_root_sample": str(gen_paths[0].parent) if gen_paths else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="sub-08")
    parser.add_argument("--images-root", default="/project/peilab/why/data/images_set")
    parser.add_argument("--output-dir", default="outputs/eval/atm_official_gen_sub08")
    parser.add_argument("--cache-dir", default="/project/peilab/why/cache/eeg-brainit/hf")
    args = parser.parse_args()

    project = ROOT
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = project / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", args.cache_dir)
    os.environ.setdefault("HF_HUB_CACHE", str(Path(args.cache_dir) / "hub"))
    os.environ.setdefault("OPENCLIP_CACHE_DIR", "/project/peilab/why/cache/eeg-brainit/open_clip")

    print("[INFO] downloading generated_imgs.tar.gz (may take a while)...")
    tar_path = hf_hub_download(
        repo_id="LidongYang/EEG_Image_decode",
        repo_type="dataset",
        filename="generated_imgs.tar.gz",
        cache_dir=str(Path(args.cache_dir) / "hub"),
        local_dir=str(Path(args.cache_dir) / "atm_generated"),
        local_dir_use_symlinks=False,
    )
    tar_path = Path(tar_path)
    print(f"[INFO] tar={tar_path} size={tar_path.stat().st_size/1e9:.2f}GB")

    extract_root = out_dir / "extracted"
    subject_dir = extract_subject(tar_path, args.subject, extract_root)
    print(f"[INFO] subject_dir={subject_dir}")
    gen_paths = collect_gen_paths(subject_dir)
    gt_paths = list_test_images(Path(args.images_root))
    print(f"[INFO] gen={len(gen_paths)} gt={len(gt_paths)}")
    if len(gen_paths) < 10:
        # Debug listing
        print("[WARN] few gens; tree sample:")
        for p in list(subject_dir.rglob("*"))[:40]:
            print(" ", p)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = metrics(gen_paths, gt_paths, device)
    report = {
        "subject": args.subject,
        "source": "LidongYang/EEG_Image_decode generated_imgs.tar.gz",
        "subject_dir": str(subject_dir),
        "metrics": m,
        "note": "Official ATM reconstructed images vs THINGS-EEG2 test GT",
    }
    out_json = out_dir / "metrics.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(
        f"[OK] n={m['n']} PixCorr={m['pixcorr']:.4f} SSIM={m['ssim']:.4f} "
        f"CLIP={m['clip_cosine']:.4f}"
    )
    print(f"[OK] wrote {out_json}")


if __name__ == "__main__":
    main()
