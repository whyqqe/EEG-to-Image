#!/usr/bin/env python3
"""Build Fusion-space GT targets (HVF) for train/test image indices."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
BRAIN_HIVE = Path(os.environ.get("BRAIN_HIVE", "/project/peilab/why/Brain-HIVE"))
sys.path.insert(0, str(BRAIN_HIVE))

from nmb_paths import FUSION_PRIOR, PROJ_META  # noqa: E402
from main.models_adapter import FusionEncoderModel  # noqa: E402


def paths_from_indices(images_root: Path, obj_idx: np.ndarray, img_idx: np.ndarray, split: str) -> list[Path]:
    root = images_root / ("training_images" if split == "train" else "test_images")
    concept_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    paths: list[Path] = []
    for o, im in zip(obj_idx, img_idx):
        d = concept_dirs[int(o)]
        if split == "train":
            imgs = sorted(list(d.glob("*.jpg")) + list(d.glob("*.JPEG")) + list(d.glob("*.png")))
            paths.append(imgs[int(im)])
        else:
            imgs = sorted(list(d.glob("*.jpg")) + list(d.glob("*.JPEG")) + list(d.glob("*.png")))
            paths.append(imgs[0])
    return paths


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)


@torch.no_grad()
def encode_vae_batch(
    vae: torch.nn.Module,
    paths: list[Path],
    device: torch.device,
    batch_size: int,
    size: int = 128,
) -> np.ndarray:
    tf = transforms.Compose(
        [
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    out: list[np.ndarray] = []
    for i in tqdm(range(0, len(paths), batch_size), desc="vae"):
        batch_paths = paths[i : i + batch_size]
        xs = torch.stack([tf(Image.open(p).convert("RGB")) for p in batch_paths]).to(device)
        lat = vae.encode(xs).latent_dist.sample() * vae.config.scaling_factor
        out.append(lat.reshape(lat.shape[0], -1).float().cpu().numpy())
    return np.concatenate(out, axis=0)


@torch.no_grad()
def encode_clip_b_batch(
    model,
    paths: list[Path],
    device: torch.device,
    batch_size: int,
    size: int = 224,
) -> np.ndarray:
    tf = transforms.Compose(
        [
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    out: list[np.ndarray] = []
    for i in tqdm(range(0, len(paths), batch_size), desc="clip-b"):
        batch_paths = paths[i : i + batch_size]
        xs = torch.stack([tf(Image.open(p).convert("RGB")) for p in batch_paths]).to(device)
        emb = model.encode_image(xs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        out.append(emb.float().cpu().numpy())
    return np.concatenate(out, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-dir", type=str, default="", help="NB embed dir with index_obj/img_*.npy")
    ap.add_argument("--images-root", type=str, default="/project/peilab/why/data/images_set")
    ap.add_argument("--clip-h-train", type=str, required=True)
    ap.add_argument("--clip-h-test", type=str, required=True)
    ap.add_argument("--prior-path", type=str, default=str(FUSION_PRIOR))
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--vae-id", type=str, default="stabilityai/sdxl-vae")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    images_root = Path(args.images_root)
    embed_dir = Path(args.embed_dir) if args.embed_dir else None

    clip_h_tr = np.load(args.clip_h_train).astype(np.float32)
    clip_h_te = np.load(args.clip_h_test).astype(np.float32)

    if embed_dir and embed_dir.is_dir():
        obj_tr = np.load(embed_dir / "index_obj_train.npy")
        img_tr = np.load(embed_dir / "index_img_train.npy")
        obj_te = np.load(embed_dir / "index_obj_test.npy")
        img_te = np.load(embed_dir / "index_img_test.npy")
        train_paths = paths_from_indices(images_root, obj_tr, img_tr, "train")
        test_paths = paths_from_indices(images_root, obj_te, img_te, "test")
        n_tr, n_te = len(train_paths), len(test_paths)
        clip_h_tr, clip_h_te = clip_h_tr[:n_tr], clip_h_te[:n_te]
    else:
        raise ValueError("embed-dir required for EEG-aligned fusion targets")

    from diffusers import AutoencoderKL
    import open_clip

    vae = AutoencoderKL.from_pretrained(args.vae_id).to(device).eval()
    clip_b, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device
    )
    clip_b.eval()

    vae_tr = encode_vae_batch(vae, train_paths, device, args.batch_size)
    vae_te = encode_vae_batch(vae, test_paths, device, args.batch_size)
    clip_b_tr = encode_clip_b_batch(clip_b, train_paths, device, args.batch_size)
    clip_b_te = encode_clip_b_batch(clip_b, test_paths, device, args.batch_size)

    fusion_model = FusionEncoderModel.from_pretrained(args.prior_path, subfolder="fusion_encoder").to(device).eval()

    @torch.no_grad()
    def fuse(clip_h, clip_b, vae_emb) -> np.ndarray:
        embs = {
            "CLIP-ViT-H-14-laion2B-s32B-b79K": torch.from_numpy(clip_h).to(device),
            "CLIP-ViT-B-32-laion2B-s34B-b79K": torch.from_numpy(clip_b).to(device),
            "vae": torch.from_numpy(vae_emb).to(device),
        }
        outs = []
        bs = args.batch_size
        n = clip_h.shape[0]
        for i in range(0, n, bs):
            batch = {k: v[i : i + bs] for k, v in embs.items()}
            z = fusion_model(batch).float().cpu().detach().numpy()
            outs.append(z)
        return l2(np.concatenate(outs, axis=0))

    fusion_tr = fuse(l2(clip_h_tr), clip_b_tr, vae_tr)
    fusion_te = fuse(l2(clip_h_te), clip_b_te, vae_te)

    np.save(out_dir / "fusion_train.npy", fusion_tr.astype(np.float32))
    np.save(out_dir / "fusion_test.npy", fusion_te.astype(np.float32))
    meta = {
        "n_train": int(fusion_tr.shape[0]),
        "n_test": int(fusion_te.shape[0]),
        "dim": int(fusion_tr.shape[1]),
        "proj_meta": PROJ_META,
        "prior_path": args.prior_path,
        "vae_size": 128,
    }
    (out_dir / "fusion_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(f"[OK] {out_dir}")


if __name__ == "__main__":
    main()
