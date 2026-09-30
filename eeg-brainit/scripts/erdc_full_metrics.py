#!/usr/bin/env python3
"""ATM/MindEye-style full image metrics for ERDC generations.

Reports: PixCorr, SSIM, CLIP cosine, AlexNet(2/5), Inception, EfficientNet-B1.
All correlations are mean Pearson over paired (gen, GT) feature maps / vectors.
Writes only under --output-json (and optional --report-dir).
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
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from eeg_brainit.utils.metrics import pixel_correlation, ssim_simple
from eval_atm_pipeline import list_test_images


def _pearson_flat(a: torch.Tensor, b: torch.Tensor) -> float:
    """a,b: (N, D) -> mean pearson over rows."""
    a = a - a.mean(dim=1, keepdim=True)
    b = b - b.mean(dim=1, keepdim=True)
    num = (a * b).sum(dim=1)
    den = a.norm(dim=1) * b.norm(dim=1).clamp_min(1e-8)
    return float((num / den).mean().item())


def list_gen_images(gen_dir: Path, n: int) -> list[Path]:
    paths = []
    for i in range(n):
        p = gen_dir / f"{i:03d}.png"
        if not p.is_file():
            # allow jpg
            p = gen_dir / f"{i:03d}.jpg"
        if not p.is_file():
            raise FileNotFoundError(p)
        paths.append(p)
    return paths


@torch.no_grad()
def eval_pair_folder(
    gen_dir: Path,
    gt_paths: list[Path],
    device: torch.device,
    max_images: int = 0,
) -> dict:
    import open_clip
    from torchvision import transforms
    from torchvision.models import (
        AlexNet_Weights,
        EfficientNet_B1_Weights,
        Inception_V3_Weights,
        alexnet,
        efficientnet_b1,
        inception_v3,
    )

    n = len(gt_paths) if max_images <= 0 else min(len(gt_paths), max_images)
    gen_paths = list_gen_images(gen_dir, n)
    gt_paths = gt_paths[:n]

    to_tensor = transforms.Compose(
        [
            transforms.Resize((256, 256), antialias=True),
            transforms.ToTensor(),
        ]
    )

    # --- pixel / ssim / CLIP ---
    clip_model, _, clip_pre = open_clip.create_model_and_transforms(
        "ViT-H-14", pretrained="laion2b_s32b_b79k", device=device
    )
    clip_model.eval()

    pixs, ssims, clips = [], [], []
    for i in tqdm(range(n), desc=f"pix-clip[{gen_dir.name}]"):
        g = Image.open(gen_paths[i]).convert("RGB")
        t = Image.open(gt_paths[i]).convert("RGB")
        gp = to_tensor(g).unsqueeze(0).to(device)
        gt = to_tensor(t).unsqueeze(0).to(device)
        pixs.append(pixel_correlation(gp, gt))
        ssims.append(ssim_simple(gp, gt))
        ge = F.normalize(clip_model.encode_image(clip_pre(g).unsqueeze(0).to(device)).float(), dim=-1)
        te = F.normalize(clip_model.encode_image(clip_pre(t).unsqueeze(0).to(device)).float(), dim=-1)
        clips.append(float((ge * te).sum()))
    del clip_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # --- AlexNet / Inception / EffNet ---
    alex = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1).features.to(device).eval()
    alex_tf = AlexNet_Weights.IMAGENET1K_V1.transforms()

    inc = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False).to(device).eval()
    inc.fc = torch.nn.Identity()
    inc_tf = Inception_V3_Weights.IMAGENET1K_V1.transforms()

    eff = efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V1).to(device).eval()
    eff.classifier = torch.nn.Identity()
    eff_tf = EfficientNet_B1_Weights.IMAGENET1K_V1.transforms()

    # hooks for alex layers
    alex_feats: dict[str, torch.Tensor] = {}

    def _hook(name):
        def fn(_m, _i, o):
            alex_feats[name] = o

        return fn

    # features: 0..12; layer2 ~ index 5 (ReLU after conv2), layer5 ~ index 12
    h2 = alex[5].register_forward_hook(_hook("l2"))
    h5 = alex[12].register_forward_hook(_hook("l5"))

    a2g, a2t, a5g, a5t = [], [], [], []
    ig, it, eg, et = [], [], [], []

    for i in tqdm(range(n), desc=f"cnn[{gen_dir.name}]"):
        g = Image.open(gen_paths[i]).convert("RGB")
        t = Image.open(gt_paths[i]).convert("RGB")

        alex_feats.clear()
        alex(alex_tf(g).unsqueeze(0).to(device))
        g2, g5 = alex_feats["l2"].flatten(1), alex_feats["l5"].flatten(1)
        alex_feats.clear()
        alex(alex_tf(t).unsqueeze(0).to(device))
        t2, t5 = alex_feats["l2"].flatten(1), alex_feats["l5"].flatten(1)
        a2g.append(g2.cpu())
        a2t.append(t2.cpu())
        a5g.append(g5.cpu())
        a5t.append(t5.cpu())

        ig.append(inc(inc_tf(g).unsqueeze(0).to(device)).float().cpu())
        it.append(inc(inc_tf(t).unsqueeze(0).to(device)).float().cpu())
        eg.append(eff(eff_tf(g).unsqueeze(0).to(device)).float().cpu())
        et.append(eff(eff_tf(t).unsqueeze(0).to(device)).float().cpu())

    h2.remove()
    h5.remove()

    a2g, a2t = torch.cat(a2g), torch.cat(a2t)
    a5g, a5t = torch.cat(a5g), torch.cat(a5t)
    ig, it = torch.cat(ig), torch.cat(it)
    eg, et = torch.cat(eg), torch.cat(et)

    out = {
        "n": n,
        "gen_dir": str(gen_dir),
        "pixcorr": float(np.mean(pixs)),
        "ssim": float(np.mean(ssims)),
        "clip_cosine": float(np.mean(clips)),
        "alexnet2": _pearson_flat(a2g, a2t),
        "alexnet5": _pearson_flat(a5g, a5t),
        "inception": _pearson_flat(ig, it),
        "effnet_b1": _pearson_flat(eg, et),
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen-dir", type=str, required=True)
    parser.add_argument("--images-root", type=str, default="/project/peilab/why/data/images_set")
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    cache_root = Path("/project/peilab/why/cache/eeg-brainit")
    os.environ.setdefault("TORCH_HOME", str(cache_root / "torch"))
    os.environ.setdefault("OPENCLIP_CACHE_DIR", str(cache_root / "open_clip"))
    os.environ.setdefault("HF_HOME", str(cache_root / "hf"))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_root / "hf" / "hub"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen_dir = Path(args.gen_dir)
    if not gen_dir.is_absolute():
        gen_dir = ROOT / gen_dir
    gt = list_test_images(Path(args.images_root))
    metrics = eval_pair_folder(gen_dir, gt, device, max_images=args.max_images)
    if args.tag:
        metrics["tag"] = args.tag

    out = Path(args.output_json)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(
        f"[OK] {args.tag or gen_dir.name}: "
        f"Pix={metrics['pixcorr']:.4f} SSIM={metrics['ssim']:.4f} CLIP={metrics['clip_cosine']:.4f} "
        f"A2={metrics['alexnet2']:.4f} A5={metrics['alexnet5']:.4f} "
        f"Inc={metrics['inception']:.4f} Eff={metrics['effnet_b1']:.4f}"
    )
    print(f"[OK] wrote {out}")


if __name__ == "__main__":
    main()
