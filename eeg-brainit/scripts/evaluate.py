#!/usr/bin/env python3
"""Evaluate a checkpoint: report token shapes and basic virtual-fMRI metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from eeg_brainit.data import SyntheticSmokeDataset, ThingsEEG2Dataset
from eeg_brainit.models import EEGBrainITPipeline
from eeg_brainit.utils.config import load_config
from eeg_brainit.utils.metrics import pixel_correlation, ssim_simple


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--max-batches", type=int, default=20)
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg.get("project_root", Path.cwd()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EEGBrainITPipeline.from_config(cfg, project_root=root).to(device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    manifest = root / cfg.get("data", {}).get("manifest", "data/processed/manifest.jsonl")
    if manifest.is_file() and manifest.stat().st_size > 0:
        ds = ThingsEEG2Dataset(manifest=manifest, root=root, split="val")
    else:
        ds = SyntheticSmokeDataset(n=16)
    loader = DataLoader(ds, batch_size=4, shuffle=False)

    ssim_sum = pix_sum = 0.0
    n = 0
    for i, batch in enumerate(loader):
        if i >= args.max_batches:
            break
        out = model(batch["spectrogram"].to(device))
        vf = out["virtual_fmri"]
        if vf.shape[1] >= 2:
            ssim_sum += ssim_simple(vf[:, 0:1], vf[:, 1:2])
            pix_sum += pixel_correlation(vf[:, 0:1], vf[:, 1:2])
        n += 1
    print(f"[OK] batches={n} vf_ssim={ssim_sum / max(n,1):.4f} vf_pixcorr={pix_sum / max(n,1):.4f}")


if __name__ == "__main__":
    main()
