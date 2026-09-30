#!/usr/bin/env python3
"""Encode train/test ViT-H 1024-d embeds from NB checkpoint (+ optional warm-start head)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

NB_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NB_ROOT))

from module.dataset import EEGPreImageDataset  # noqa: E402
from module.eeg_encoder.model import EEGProject  # noqa: E402
from module.projector import ProjectorLinear  # noqa: E402

DEFAULT_CHANNELS = [
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nb-root", type=str, default=str(NB_ROOT))
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--warm-start", type=str, default="")
    ap.add_argument("--subject", type=int, default=8)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    root = Path(args.nb_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    eeg_dir = str(root / "data/things_eeg/preprocessed_eeg")
    rn50_dir = str(root / "data/things_eeg/image_feature/RN50")
    train_ds = EEGPreImageDataset(
        [args.subject], eeg_dir, DEFAULT_CHANNELS, [0, 250],
        rn50_dir, "", False, [], True, False, None, True, False, False, False,
    )
    test_ds = EEGPreImageDataset(
        [args.subject], eeg_dir, DEFAULT_CHANNELS, [0, 250],
        rn50_dir, "", False, [], True, False, None, False, False, False, False,
    )

    latent_dim = int(train_ds.image_features.shape[-1])
    model = EEGProject(
        feature_dim=latent_dim,
        eeg_sample_points=int(train_ds.num_sample_points),
        channels_num=int(train_ds.channels_num),
    ).to(device)
    eeg_projector = ProjectorLinear(latent_dim, 512).to(device)
    head = torch.nn.Linear(latent_dim, 1024).to(device)

    ckpt = torch.load(Path(args.checkpoint) if Path(args.checkpoint).is_absolute() else root / args.checkpoint,
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    eeg_projector.load_state_dict(ckpt["eeg_projector_state_dict"])
    if args.warm_start:
        ws = Path(args.warm_start)
        if not ws.is_absolute():
            ws = root / ws
        wsd = torch.load(ws, map_location=device, weights_only=False)
        if "eeg_head_vith1024_state_dict" in wsd:
            head.load_state_dict(wsd["eeg_head_vith1024_state_dict"])

    @torch.no_grad()
    def run(ds, tag: str):
        model.eval()
        head.eval()
        eeg_projector.eval()
        vith, proj = [], []
        for batch in tqdm(DataLoader(ds, batch_size=512, shuffle=False), desc=tag):
            eeg = batch[0].to(device)
            raw = model(eeg)
            vith.append(head(raw).float().cpu().numpy())
            proj.append(eeg_projector(raw).float().cpu().numpy())
        v = np.concatenate(vith, axis=0)
        p = np.concatenate(proj, axis=0)
        v = v / np.linalg.norm(v, axis=1, keepdims=True).clip(1e-8)
        np.save(out / f"decode_vith1024_{tag}_clip_1024.npy", v.astype(np.float32))
        np.save(out / f"z_eeg_proj_{tag}.npy", p.astype(np.float32))

    run(train_ds, "train")
    run(test_ds, "test")
    print(f"[OK] {out}")


if __name__ == "__main__":
    main()
