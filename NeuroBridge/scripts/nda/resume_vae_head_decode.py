#!/usr/bin/env python3
"""Resume-decode: rebuild pred_lowlevel_rgb_512 from a trained EEG->VAE head checkpoint.

Used when train_eeg_vae_head.py finished training (checkpoint + pred_vae_test.npy
saved) but crashed at the optional --decode-rgb stage (e.g. bad diffusers env).
Reuses the exact VAEHead/VAE/decode logic from train_eeg_vae_head.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

NB_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NB_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_eeg_vae_head import VAEHead, decode_latents, resolve_vae  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    out = Path(args.output_dir)
    ck_path = Path(args.checkpoint)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ck = torch.load(ck_path, map_location=device, weights_only=False)
    head = VAEHead(
        in_dim=int(ck["in_dim"]), spatial=int(ck["spatial"]), ch=int(ck["ch"])
    ).to(device)
    head.load_state_dict(ck["state_dict"])
    head.eval()

    mean = torch.as_tensor(ck["target_mean"], device=device, dtype=torch.float32).view(1, -1, 1, 1)
    std = torch.as_tensor(ck["target_std"], device=device, dtype=torch.float32).view(1, -1, 1, 1)

    # predicted test latents were saved by train; recompute identically for robustness
    pred_te_np = np.load(out / "pred_vae_test.npy").astype(np.float32)
    pred_te = torch.from_numpy(pred_te_np).to(device)

    import os
    hub = Path(os.environ.get("HF_HUB_CACHE", "/project/peilab/why/cache/eeg-brainit/hf/hub"))
    vae = resolve_vae(hub, device)

    rgb_dir = out / "pred_lowlevel_rgb_512"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    scaling = float(ck.get("scaling_factor", 0.13025))
    bs = 8
    from tqdm import tqdm
    for start in tqdm(range(0, len(pred_te), bs), desc="decode-rgb-resume"):
        chunk = pred_te[start : start + bs]
        imgs = decode_latents(vae, chunk, scaling)
        for j, im in enumerate(imgs):
            im.save(rgb_dir / f"{start + j:03d}.png")

    n_imgs = len(list(rgb_dir.glob("*.png")))
    report = {
        "pipeline": "eeg_vae_lowlevel_head_resume_decode",
        "resumed_from": str(ck_path),
        "best_epoch": int(ck.get("epoch", -1)),
        "best_metrics": ck.get("metrics", {}),
        "pred_rgb_dir": str(rgb_dir),
        "n_rgb": n_imgs,
        "scaling_factor": scaling,
    }
    # do NOT overwrite an existing full train report; write a decode-only marker
    (out / "vae_head_decode_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    # mark pipeline-complete with the same file the run script guards on
    if not (out / "vae_head_report.json").is_file():
        full = dict(report)
        full["pipeline"] = "eeg_vae_lowlevel_head (resume-decoded after training)"
        (out / "vae_head_report.json").write_text(json.dumps(full, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if n_imgs != 200:
        print(f"[WARN] expected 200 RGB but got {n_imgs}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
