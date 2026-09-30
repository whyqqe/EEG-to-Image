#!/usr/bin/env python3
"""NeuroWeave v3 M8: Spatial cycle consistency.

Encode generated images with SDXL VAE and compare to EEG-predicted latents
(pred_vae_test_scaled.npy). Equivalence class is small → aligns with fidelity.

Also reports depth cycle if pred_depth available.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

NB_ROOT = Path("/project/peilab/why/NeuroBridge")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", type=str, required=True)
    ap.add_argument("--pred-vae", type=str, required=True)
    ap.add_argument("--pred-depth", type=str, default="")
    ap.add_argument("--output-json", type=str, required=True)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--scaling-factor", type=float, default=0.13025)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--max-images", type=int, default=200)
    args = ap.parse_args()

    gen_dir = Path(args.gen_dir)
    pred_vae = np.load(args.pred_vae).astype(np.float32)
    n = min(args.max_images, len(pred_vae), len(list(gen_dir.glob("*.png"))))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    sys.path.insert(0, str(NB_ROOT / "scripts" / "nda"))
    from train_eeg_vae_head import resolve_vae  # type: ignore

    hub = Path(os.environ.get("HF_HUB_CACHE", "/project/peilab/why/cache/eeg-brainit/hf/hub"))
    vae = resolve_vae(hub, device)
    vae.eval()

    gens = []
    for i in range(n):
        p = gen_dir / f"{i:03d}.png"
        if not p.is_file():
            raise FileNotFoundError(p)
        gens.append(p)

    # encode generated images → latents
    enc = []
    with torch.no_grad():
        for s in tqdm(range(0, n, 8), desc="vae-encode"):
            imgs = []
            for i in range(s, min(s + 8, n)):
                im = Image.open(gens[i]).convert("RGB").resize((512, 512), Image.Resampling.BICUBIC)
                t = torch.from_numpy(np.asarray(im).astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1)
                imgs.append(t)
            batch = torch.stack(imgs).to(device)
            # diffusers AutoencoderKL
            posterior = vae.encode(batch).latent_dist
            z = posterior.mean * args.scaling_factor
            enc.append(z.float().cpu().numpy())
    enc = np.concatenate(enc, 0).astype(np.float32)
    assert enc.shape == pred_vae[:n].shape, (enc.shape, pred_vae[:n].shape)

    # per-image pearson / cosine on flattened latents
    pears, cosines = [], []
    for i in range(n):
        a = enc[i].ravel()
        b = pred_vae[i].ravel()
        a0 = a - a.mean()
        b0 = b - b.mean()
        pears.append(float((a0 * b0).sum() / (np.linalg.norm(a0) * np.linalg.norm(b0) + 1e-8)))
        cosines.append(float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)))

    report = {
        "tag": args.tag or gen_dir.parent.name,
        "n": n,
        "vae_latent_pearson": float(np.mean(pears)),
        "vae_latent_cosine": float(np.mean(cosines)),
        "note": "spatial cycle: VAE(gen) vs EEG-predicted VAE latent (small equivalence class)",
    }

    if args.pred_depth and Path(args.pred_depth).is_file():
        # optional: rough depth cycle via luminance of gen vs pred_depth
        pd = np.load(args.pred_depth).astype(np.float32)
        dpears = []
        for i in range(n):
            im = np.asarray(Image.open(gens[i]).convert("L").resize((64, 64), Image.Resampling.BILINEAR)).astype(np.float32) / 255.0
            a = im.ravel()
            b = pd[i].ravel()
            a0, b0 = a - a.mean(), b - b.mean()
            dpears.append(float((a0 * b0).sum() / (np.linalg.norm(a0) * np.linalg.norm(b0) + 1e-8)))
        report["depth_luma_pearson"] = float(np.mean(dpears))

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in report if k != "note"}, indent=2))


if __name__ == "__main__":
    main()
