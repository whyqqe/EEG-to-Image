#!/usr/bin/env python3
"""Per-image alpha fuse guided by AlexNet mid features (Alex2 proxy).

Modes:
  - oracle: pick α maximizing Pearson(Alex-feat[5](blend), Alex-feat[5](GT))
    → ceiling for this fuse family (report separately; uses GT).
  - u_str:  map structure confidence → α (deployable; no GT).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm


def load_rgb(path: Path, size: int) -> np.ndarray:
    return np.asarray(
        Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BICUBIC),
        dtype=np.float32,
    )


def luma(x: np.ndarray) -> np.ndarray:
    return 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]


def match_luma_to_ref(img: np.ndarray, ref: np.ndarray) -> np.ndarray:
    y, yr = luma(img), luma(ref)
    scale = (float(yr.mean()) + 1e-6) / (float(y.mean()) + 1e-6)
    out = img * scale
    std_y = float(y.std()) + 1e-6
    std_r = float(yr.std()) + 1e-6
    out = (out - out.mean()) * (0.65 + 0.35 * (std_r / std_y)) + float(yr.mean())
    return np.clip(out, 0, 255)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    if a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


@torch.no_grad()
def alex_l2_feat(model, tfm, img: np.ndarray, device: torch.device) -> np.ndarray:
    x = tfm(Image.fromarray(img.astype(np.uint8))).unsqueeze(0).to(device)
    # features[5] == AlexNet layer used as "Alex2" in erdc_twoway
    return model.features[:6](x).float().cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--struct-dir", type=str, required=True)
    ap.add_argument("--semantic-dir", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--tag", type=str, default="alexfeat_fuse")
    ap.add_argument("--mode", choices=["oracle", "u_str"], default="oracle")
    ap.add_argument("--alphas", type=str, default="0.45,0.55,0.65,0.75,0.85")
    ap.add_argument("--u-str-npy", type=str, default="")
    ap.add_argument("--alpha-min", type=float, default=0.50)
    ap.add_argument("--alpha-max", type=float, default=0.85)
    ap.add_argument("--images-root", type=str, default="/project/peilab/why/data/images_set")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--max-images", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.output_dir)
    gen = out / "generated"
    gen.mkdir(parents=True, exist_ok=True)
    sdir, gdir = Path(args.struct_dir), Path(args.semantic_dir)
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]

    import sys

    sys.path.insert(0, "/project/peilab/why/eeg-brainit/scripts")
    from eval_atm_pipeline import list_test_images  # type: ignore

    gts = list_test_images(Path(args.images_root))
    n = len(sorted(gdir.glob("*.png")))
    if args.max_images > 0:
        n = min(n, args.max_images)
    gts = gts[:n]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1).to(device).eval()
    tfm = transforms.Compose(
        [
            transforms.Resize(256, antialias=True),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    ranks = None
    if args.mode == "u_str":
        u = np.load(args.u_str_npy).astype(np.float32).reshape(-1)[:n]
        order = np.argsort(u)
        ranks = np.empty_like(u, dtype=np.float32)
        ranks[order] = np.linspace(0.0, 1.0, num=len(u), dtype=np.float32)

    chosen = []
    for i in tqdm(range(n), desc=args.tag):
        dst = gen / f"{i:03d}.png"
        sem = load_rgb(gdir / f"{i:03d}.png", args.size)
        struct = load_rgb(sdir / f"{i:03d}.png", args.size)

        if args.mode == "u_str":
            # high structure conf → lower sem_alpha (more structure), like gated luma
            a = float(args.alpha_max - ranks[i] * (args.alpha_max - args.alpha_min))
            blend = a * sem + (1.0 - a) * struct
            blend = match_luma_to_ref(blend, sem)
            if not dst.is_file():
                Image.fromarray(blend.astype(np.uint8)).save(dst)
            chosen.append({"i": i, "alpha": a, "score": None})
            continue

        # oracle: maximize AlexNet-l2 pearson vs GT
        gt = load_rgb(gts[i], args.size)
        gt_f = alex_l2_feat(model, tfm, gt, device)
        best_a, best_s, best_img = alphas[0], -1e9, None
        for a in alphas:
            blend = a * sem + (1.0 - a) * struct
            blend = match_luma_to_ref(blend, sem)
            score = pearson(alex_l2_feat(model, tfm, blend, device), gt_f)
            if score > best_s:
                best_a, best_s, best_img = a, score, blend
        if not dst.is_file():
            Image.fromarray(best_img.astype(np.uint8)).save(dst)
        chosen.append({"i": i, "alpha": best_a, "score": best_s})

    report = {
        "tag": args.tag,
        "mode": args.mode,
        "alphas": alphas if args.mode == "oracle" else [args.alpha_min, args.alpha_max],
        "n": n,
        "alpha_mean": float(np.mean([c["alpha"] for c in chosen])),
        "oracle_alex_pearson_mean": float(np.mean([c["score"] for c in chosen if c["score"] is not None]))
        if args.mode == "oracle"
        else None,
        "note": "Alex2-first fuse; does not optimize PixCorr/SSIM",
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.save(out / "chosen_alpha.npy", np.asarray([c["alpha"] for c in chosen], dtype=np.float32))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
