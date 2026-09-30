#!/usr/bin/env python3
"""Encode EEG with SharedSpecific checkpoint → z_eeg_proj_{train,test}.npy for NVOL."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

NB_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NB_ROOT))
sys.path.insert(0, str(NB_ROOT / "scripts" / "nda"))

from module.dataset import EEGPreImageDataset  # noqa: E402
from module.projector import ProjectorLinear  # noqa: E402
from ss_modules import SharedSpecificEncoder  # noqa: E402

DEFAULT_CHANNELS = [
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nb-root", type=str, default=str(NB_ROOT))
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--subject", type=int, default=8)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    root = Path(args.nb_root)
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(Path(args.checkpoint) if Path(args.checkpoint).is_absolute() else root / args.checkpoint,
                      map_location=device, weights_only=False)
    subjects = [int(s) for s in ckpt.get("subjects", [args.subject])]
    img_dim = int(ckpt["img_dim"])
    feature_dim = int(ckpt["feature_dim"])
    eeg_len = int(ckpt["eeg_sample_points"])
    channels_num = int(ckpt["channels_num"])

    model = SharedSpecificEncoder(
        subject_ids=subjects,
        feature_dim=img_dim,
        eeg_sample_points=eeg_len,
        channels_num=channels_num,
        n_extra_blocks=int(ckpt.get("n_extra_blocks", 1)),
        use_adapter=True,
    ).to(device)
    eeg_projector = ProjectorLinear(img_dim, feature_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    eeg_projector.load_state_dict(ckpt["eeg_projector_state_dict"])
    model.eval()
    eeg_projector.eval()

    eeg_dir = str(root / "data/things_eeg/preprocessed_eeg")
    rn50_dir = str(root / "data/things_eeg/image_feature/RN50")

    for train_flag, tag in ((True, "train"), (False, "test")):
        ds = EEGPreImageDataset(
            [args.subject], eeg_dir, DEFAULT_CHANNELS, [0, 250],
            rn50_dir, "", False, [], True, False, None, train_flag, False, False, False,
        )
        xs = []
        with torch.no_grad():
            for batch in DataLoader(ds, batch_size=512, shuffle=False):
                eeg, _img, _t, sid, *_ = batch
                eeg, sid = eeg.to(device), sid.to(device)
                xs.append(eeg_projector(model(eeg, sid)).cpu().numpy())
        arr = np.concatenate(xs).astype(np.float32)
        np.save(out / f"z_eeg_proj_{tag}.npy", arr)
        print(f"[OK] {tag} {arr.shape}")


if __name__ == "__main__":
    main()
