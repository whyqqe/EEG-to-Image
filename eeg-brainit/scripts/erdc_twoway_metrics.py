#!/usr/bin/env python3
"""Two-way comparison (2WC) metrics in the ENIGMA/ATM reporting style.

For each sample i and each feature space, score =
  mean_j≠i 1[ sim(gt_i, recon_i) > sim(gt_i, recon_j) ]
Chance ≈ 50%. Writes JSON under --output-json.
"""

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


@torch.no_grad()
def twoway(sim_mat: np.ndarray) -> float:
    """sim_mat[i,j] = similarity between gt_i features and recon_j features."""
    n = sim_mat.shape[0]
    correct = 0
    total = 0
    for i in range(n):
        s_ii = sim_mat[i, i]
        for j in range(n):
            if i == j:
                continue
            correct += float(s_ii > sim_mat[i, j])
            total += 1
    return float(correct / max(total, 1))


@torch.no_grad()
def encode_bundle(paths: list[Path], device: torch.device) -> dict[str, np.ndarray]:
    import open_clip
    from torchvision.models import (
        AlexNet_Weights,
        EfficientNet_B1_Weights,
        Inception_V3_Weights,
        alexnet,
        efficientnet_b1,
        inception_v3,
    )

    n = len(paths)
    # CLIP ViT-H (same as our pipeline)
    clip_m, _, clip_pre = open_clip.create_model_and_transforms(
        "ViT-H-14", pretrained="laion2b_s32b_b79k", device=device
    )
    clip_m.eval()
    clips = []
    for p in tqdm(paths, desc="clip"):
        x = clip_pre(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
        clips.append(F.normalize(clip_m.encode_image(x).float(), dim=-1).cpu())
    del clip_m
    clip = torch.cat(clips, dim=0).numpy()

    alex = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1).features.to(device).eval()
    alex_tf = AlexNet_Weights.IMAGENET1K_V1.transforms()
    feats: dict[str, torch.Tensor] = {}

    def hook(name):
        def fn(_m, _i, o):
            feats[name] = o

        return fn

    h2 = alex[5].register_forward_hook(hook("l2"))
    h5 = alex[12].register_forward_hook(hook("l5"))
    a2, a5 = [], []
    for p in tqdm(paths, desc="alex"):
        feats.clear()
        alex(alex_tf(Image.open(p).convert("RGB")).unsqueeze(0).to(device))
        a2.append(feats["l2"].flatten(1).cpu())
        a5.append(feats["l5"].flatten(1).cpu())
    h2.remove()
    h5.remove()
    del alex
    a2 = F.normalize(torch.cat(a2), dim=-1).numpy()
    a5 = F.normalize(torch.cat(a5), dim=-1).numpy()

    inc = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False).to(device).eval()
    inc.fc = torch.nn.Identity()
    inc_tf = Inception_V3_Weights.IMAGENET1K_V1.transforms()
    ig = []
    for p in tqdm(paths, desc="inc"):
        ig.append(F.normalize(inc(inc_tf(Image.open(p).convert("RGB")).unsqueeze(0).to(device)).float(), dim=-1).cpu())
    del inc
    ig = torch.cat(ig).numpy()

    return {"clip": clip, "alex2": a2, "alex5": a5, "inception": ig}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen-dir", type=str, required=True)
    parser.add_argument("--images-root", type=str, default="/project/peilab/why/data/images_set")
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    cache = Path("/project/peilab/why/cache/eeg-brainit")
    os.environ.setdefault("TORCH_HOME", str(cache / "torch"))
    os.environ.setdefault("OPENCLIP_CACHE_DIR", str(cache / "open_clip"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen_dir = Path(args.gen_dir)
    if not gen_dir.is_absolute():
        gen_dir = ROOT / gen_dir
    gt = list_test_images(Path(args.images_root))
    n = len(gt) if args.max_images <= 0 else min(len(gt), args.max_images)
    gens = list_gen(gen_dir, n)
    gt = gt[:n]

    cache_dir = gen_dir.parent / "_twoway_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    g_cache = cache_dir / f"{gen_dir.name}_feats.npz"
    t_cache = cache_dir / "gt_feats.npz"

    if g_cache.is_file():
        g = dict(np.load(g_cache))
    else:
        g = encode_bundle(gens, device)
        np.savez(g_cache, **g)
    if t_cache.is_file():
        t = dict(np.load(t_cache))
    else:
        t = encode_bundle(gt, device)
        np.savez(t_cache, **t)

    report = {"n": n, "tag": args.tag or gen_dir.name, "gen_dir": str(gen_dir), "twoway": {}}
    for key in ["clip", "alex2", "alex5", "inception"]:
        sim = t[key] @ g[key].T  # (n,n) gt_i vs recon_j
        report["twoway"][key] = twoway(sim)
        print(f"[INFO] 2WC {key}={report['twoway'][key]*100:.2f}%")

    out = Path(args.output_json)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[OK] wrote {out}")


if __name__ == "__main__":
    main()
