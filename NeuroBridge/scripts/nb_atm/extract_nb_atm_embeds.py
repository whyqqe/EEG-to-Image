#!/usr/bin/env python3
"""Extract z_eeg raw/proj from NB train checkpoint (EEGProject or ATMS)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

NB_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NB_ROOT))
sys.path.insert(0, str(NB_ROOT / "scripts" / "nb_atm"))

from module.dataset import EEGPreImageDataset  # noqa: E402
from nb_encoder_factory import encode_eeg, load_train_checkpoint  # noqa: E402

DEFAULT_CHANNELS = [
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2",
]


@torch.no_grad()
def encode_split(model, eeg_projector, loader, device, encoder_type):
    raw_list, proj_list, sid_list = [], [], []
    for batch in tqdm(loader, desc="encode"):
        eeg = batch[0].to(device)
        sid = batch[3].to(device)
        if encoder_type == "atm":
            out = model(eeg, sid)
        else:
            out = model(eeg)
        proj = eeg_projector(out)
        raw_list.append(out.float().cpu().numpy())
        proj_list.append(proj.float().cpu().numpy())
        sid_list.append(sid.cpu().numpy())
    return (
        np.concatenate(raw_list, axis=0).astype(np.float32),
        np.concatenate(proj_list, axis=0).astype(np.float32),
        np.concatenate(sid_list, axis=0),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nb-root", type=str, default=str(NB_ROOT))
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--encoder-type", type=str, default="atm", choices=["atm", "eegproject"])
    ap.add_argument("--subject", type=int, default=8)
    ap.add_argument("--image-feature-dir", type=str, default="data/things_eeg/image_feature/ViT-H-14")
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--proj-out-dim", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    root = Path(args.nb_root)
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = root / ckpt_path
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eeg_dir = str(root / "data/things_eeg/preprocessed_eeg")
    img_dir = args.image_feature_dir if Path(args.image_feature_dir).is_absolute() else str(root / args.image_feature_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_ds = EEGPreImageDataset(
        [args.subject], eeg_dir, DEFAULT_CHANNELS, [0, 250], img_dir, "",
        False, [], True, False, None, True, False, False, False,
    )
    test_ds = EEGPreImageDataset(
        [args.subject], eeg_dir, DEFAULT_CHANNELS, [0, 250], img_dir, "",
        False, [], True, False, None, False, False, False, False,
    )
    feat_dim = int(train_ds.image_features.shape[-1])
    model, eeg_projector, meta = load_train_checkpoint(
        ckpt_path, args.encoder_type, feat_dim,
        int(train_ds.num_sample_points), int(train_ds.channels_num),
        args.proj_out_dim, device,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    tr_raw, tr_proj, _ = encode_split(model, eeg_projector, train_loader, device, meta["encoder_type"])
    te_raw, te_proj, _ = encode_split(model, eeg_projector, test_loader, device, meta["encoder_type"])

    np.save(out_dir / "z_eeg_raw_train.npy", tr_raw)
    np.save(out_dir / "z_eeg_proj_train.npy", tr_proj)
    np.save(out_dir / "z_eeg_raw_test.npy", te_raw)
    np.save(out_dir / "z_eeg_proj_test.npy", te_proj)

    report = {
        "checkpoint": str(ckpt_path),
        "encoder_type": meta["encoder_type"],
        "feat_dim": feat_dim,
        "proj_dim": int(tr_proj.shape[1]),
        "train_n": int(tr_raw.shape[0]),
        "test_n": int(te_raw.shape[0]),
    }
    (out_dir / "meta.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
