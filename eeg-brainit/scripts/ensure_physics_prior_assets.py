#!/usr/bin/env python3
"""Ensure assets for Physics-Prior LoRA adaptation experiments.

Uses existing ATM bridge CLIP-H/14 train features + extracts 200-way test CLIP.
Does not download SDXL/Brain-IT. Writes only under outputs/atm_bridge and cache/.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path("/project/peilab/why/cache/eeg-brainit")


def _setup_env() -> None:
    os.environ.setdefault("HOME", str(CACHE / "xdg-home"))
    os.environ.setdefault("HF_HOME", str(CACHE / "hf"))
    os.environ.setdefault("HF_HUB_CACHE", str(CACHE / "hf/hub"))
    os.environ.setdefault("OPENCLIP_CACHE_DIR", str(CACHE / "open_clip"))
    os.environ.setdefault("XDG_CACHE_HOME", str(CACHE / "xdg"))
    os.environ.setdefault("TMPDIR", str(CACHE / "tmp"))


def extract_test_clip(images_dir: Path, meta_path: Path, out_path: Path, device: torch.device) -> None:
    import open_clip

    meta = np.load(meta_path, allow_pickle=True).item()
    files = list(meta["test_img_files"])
    concepts = list(meta["test_img_concepts"])
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14", pretrained="laion2b_s32b_b79k"
    )
    model = model.to(device).eval()
    embs = []
    with torch.no_grad():
        for fname, concept in zip(files, concepts):
            # test_images/<concept>/<fname>
            path = images_dir / "test_images" / concept / fname
            if not path.is_file():
                # fallback: search one level
                cands = list((images_dir / "test_images").glob(f"*/{fname}"))
                if not cands:
                    raise FileNotFoundError(path)
                path = cands[0]
            img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            emb = model.encode_image(img)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            embs.append(emb.squeeze(0).float().cpu().numpy())
    arr = np.stack(embs, 0).astype(np.float32)
    np.save(out_path, arr)
    side = {
        "n": int(arr.shape[0]),
        "dim": int(arr.shape[1]),
        "model": "ViT-H-14/laion2b_s32b_b79k",
        "files": files,
        "concepts": concepts,
        "path": str(out_path),
    }
    out_path.with_suffix(".json").write_text(json.dumps(side, indent=2), encoding="utf-8")
    print(f"[OK] test CLIP {arr.shape} -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atm-bridge-dir", default="outputs/atm_bridge")
    parser.add_argument("--images-dir", default="/project/peilab/why/data/images_set")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    _setup_env()

    bridge = ROOT / args.atm_bridge_dir
    train_clip = bridge / "clip_img_train_1024.npy"
    if not train_clip.is_file():
        raise FileNotFoundError(f"missing ATM bridge train CLIP: {train_clip}")
    print(f"[OK] train CLIP {np.load(train_clip, mmap_mode='r').shape}")

    missing = []
    for sub in [f"sub-{i:02d}" for i in range(1, 11)]:
        for name in (f"{sub}_train_eeg_avg_1024.npy", f"{sub}_test_eeg_1024.npy"):
            if not (bridge / name).is_file():
                missing.append(name)
        eeg = ROOT / "data/processed/things-eeg2" / sub / "train_eeg.npy"
        if not eeg.is_file():
            missing.append(str(eeg))
    if missing:
        raise FileNotFoundError(f"missing assets: {missing[:8]}...")
    print("[OK] THINGS-EEG2 raw + ATM embeddings for sub-01..10")

    test_clip = bridge / "clip_img_test_1024.npy"
    if test_clip.is_file() and not args.force:
        print(f"[SKIP] exists {test_clip} shape={np.load(test_clip, mmap_mode='r').shape}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        extract_test_clip(
            Path(args.images_dir),
            Path(args.images_dir) / "image_metadata.npy",
            test_clip,
            device,
        )

    # quick ATM ceiling on sub-08 test
    atm = np.load(bridge / "sub-08_test_eeg_1024.npy")
    img = np.load(test_clip)
    atm = atm / (np.linalg.norm(atm, axis=1, keepdims=True) + 1e-8)
    img = img / (np.linalg.norm(img, axis=1, keepdims=True) + 1e-8)
    sim = atm @ img.T
    top1 = float((sim.argmax(1) == np.arange(len(atm))).mean())
    print(f"[INFO] ATM teacher ceiling sub-08 200-way top1={top1*100:.2f}%")
    (bridge / "physics_prior_assets.json").write_text(
        json.dumps(
            {
                "train_clip": str(train_clip),
                "test_clip": str(test_clip),
                "atm_ceiling_sub08_top1": top1,
                "n_subjects": 10,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("[OK] assets ready")


if __name__ == "__main__":
    main()
