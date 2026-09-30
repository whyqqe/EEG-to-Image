#!/usr/bin/env python3
"""Lightweight scorer for the IP-mass diagnostic: PixCorr, SSIM, CLIP only.

Why a separate script: the full seven-metric eval also computes FID + SwAV,
which dominate runtime. For a scale sweep we only need the two collapsing
metrics (PixCorr/SSIM) plus one semantic metric (CLIP 2-way) as a control, so
the sweep stays interactive.

Also reports three structural diagnostics that separate "layout lost" from
"mere blur":
  layout_corr   Pearson corr of 8x8-downsampled grayscale (very low frequency)
  hf_energy     high-frequency energy (Laplacian variance) on grayscale
  inter_div     mean pairwise L1 between generated images (diversity within row)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def l2n(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8, None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", type=str, required=True)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--output-json", type=str, default="")
    ap.add_argument("--images-root", type=str, default="/project/peilab/why/data/images_set")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--clip-model", type=str, default="ViT-H-14")
    ap.add_argument("--clip-pretrained", type=str, default="laion2b_s32b_b79k")
    args = ap.parse_args()

    sys.path.insert(0, "/project/peilab/why/eeg-brainit/scripts")
    sys.path.insert(0, "/project/peilab/why/NeuroBridge/scripts/nda")
    from eval_atm_pipeline import list_test_images  # type: ignore
    from eval_standard7 import lowlevel  # type: ignore

    gen_dir = Path(args.gen_dir)
    gt = list_test_images(Path(args.images_root))
    gens = sorted(gen_dir.glob("*.png"))
    if len(gens) != len(gt):
        raise SystemExit(f"[FATAL] {len(gens)} generated vs {len(gt)} GT")

    ll = lowlevel(gens, gt)

    # ---- structural diagnostics on a subset (CPU, fast) ----
    from numpy.lib.stride_tricks import sliding_window_view
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    lay, hf, means = [], [], []
    n = len(gens)
    for p, t in zip(gens, gt):
        g = np.asarray(Image.open(p).convert("RGB").resize((425, 425),
                       Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        tt = np.asarray(Image.open(t).convert("RGB").resize((425, 425),
                        Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        gg = g.mean(-1)
        tgg = tt.mean(-1)
        # 8x8 block-mean = very low frequency (layout)
        blk = gg.reshape(8, 53, 8, 53).mean(axis=(1, 3))
        tblk = tgg.reshape(8, 53, 8, 53).mean(axis=(1, 3))
        lay.append(float(np.corrcoef(blk.reshape(-1), tblk.reshape(-1))[0, 1]))
        w = sliding_window_view(gg, (3, 3))
        hf.append(float(np.einsum("...ij,ij->...", w, k).var()))
        means.append(g.reshape(-1, 3).mean(0))

    idx = np.linspace(0, n - 1, 40).astype(int)
    sub = np.stack([np.asarray(Image.open(gens[i]).convert("RGB").resize((64, 64)),
                               dtype=np.float32) / 255.0 for i in idx])
    div = float(np.mean([np.abs(sub[a] - sub[b]).mean()
                         for a in range(0, 40, 4) for b in range(a + 1, 40, 4)]))

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.clip_model, pretrained=args.clip_pretrained, device=dev)
    model = model.to(dev).eval()

    def enc(paths):
        out = []
        with torch.no_grad():
            for i in range(0, len(paths), 32):
                b = torch.stack([preprocess(Image.open(p).convert("RGB"))
                                 for p in paths[i:i + 32]]).to(dev)
                out.append(model.encode_image(b).float().cpu().numpy())
        return l2n(np.concatenate(out, 0))

    fg, ft = enc(gens), enc(gt)
    sim = ft @ fg.T
    own = np.diag(sim)
    rng = np.random.default_rng(0)
    other = (np.arange(len(gt)) + rng.integers(1, len(gt), size=len(gt))) % len(gt)
    clip_2way = float((own > sim[np.arange(len(gt)), other]).mean())

    rep = {
        "tag": args.tag or gen_dir.parent.name,
        "gen_dir": str(gen_dir),
        "pixcorr": float(ll["pixcorr"]),
        "ssim": float(ll["ssim"]),
        "clip_2way": clip_2way,
        "clip_cosine": float((fg * ft).sum(1).mean()),
        "layout_corr": float(np.mean(lay)),
        "hf_energy": float(np.mean(hf)),
        "inter_div": div,
        "mean_color_std_across_rows": float(np.std(np.stack(means), axis=0).mean()),
    }
    print(f"[{rep['tag']}] pix={rep['pixcorr']:.4f} ssim={rep['ssim']:.4f} "
          f"clip2w={rep['clip_2way']:.3f} layout={rep['layout_corr']:.4f} "
          f"hf={rep['hf_energy']:.5f} div={rep['inter_div']:.4f}")
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(rep, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
