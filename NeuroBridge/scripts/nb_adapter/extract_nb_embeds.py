#!/usr/bin/env python3
"""Extract frozen NeuroBridge EEG embeddings for adapter training.

Saves:
  z_eeg_proj_{train,test}.npy  — SSP eeg_projector output (default adapter input, 512-d)
  z_eeg_raw_{train,test}.npy   — EEGProject backbone output (1024-d)
"""
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

from module.dataset import EEGPreImageDataset  # noqa: E402
from module.eeg_encoder.model import EEGProject  # noqa: E402
from module.projector import ProjectorLinear  # noqa: E402

DEFAULT_CHANNELS = [
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2",
]


@torch.no_grad()
def encode_split(
    model: torch.nn.Module,
    eeg_projector: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw_list, proj_list, obj_list, img_list = [], [], [], []
    for batch in tqdm(loader, desc="encode"):
        eeg, _img, _txt, _sid, obj_idx, img_idx, _rep = batch
        eeg = eeg.to(device)
        raw = model(eeg)
        proj = eeg_projector(raw)
        raw_list.append(raw.float().cpu().numpy())
        proj_list.append(proj.float().cpu().numpy())
        obj_list.append(obj_idx.numpy())
        img_list.append(img_idx.numpy())
    return (
        np.concatenate(raw_list, axis=0).astype(np.float32),
        np.concatenate(proj_list, axis=0).astype(np.float32),
        np.concatenate(obj_list, axis=0),
        np.concatenate(img_list, axis=0),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--nb-root", type=str, default=str(NB_ROOT))
    p.add_argument(
        "--checkpoint",
        type=str,
        default="results/things_eeg/intra-subjects/20260901-074319-sub-08/checkpoint_test_best.pth",
    )
    p.add_argument("--subject", type=int, default=8)
    p.add_argument("--eeg-data-dir", type=str, default="data/things_eeg/preprocessed_eeg")
    p.add_argument("--image-feature-dir", type=str, default="data/things_eeg/image_feature/RN50")
    p.add_argument("--output-dir", type=str, default="outputs/nb_adapter/sub-08/embeds")
    p.add_argument("--feature-dim", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--device", type=str, default="cuda:0")
    args = p.parse_args()

    root = Path(args.nb_root)
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = root / ckpt_path
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    eeg_dir = args.eeg_data_dir if Path(args.eeg_data_dir).is_absolute() else str(root / args.eeg_data_dir)
    img_dir = (
        args.image_feature_dir
        if Path(args.image_feature_dir).is_absolute()
        else str(root / args.image_feature_dir)
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device} ckpt={ckpt_path}")

    # Dummy dirs for unused aug/text; image_aug=False so only image_*.npy needed.
    train_ds = EEGPreImageDataset(
        [args.subject],
        eeg_dir,
        DEFAULT_CHANNELS,
        [0, 250],
        img_dir,
        "",
        False,
        [],
        True,
        False,
        None,
        True,
        False,
        False,
        False,
    )
    test_ds = EEGPreImageDataset(
        [args.subject],
        eeg_dir,
        DEFAULT_CHANNELS,
        [0, 250],
        img_dir,
        "",
        False,
        [],
        True,
        False,
        None,
        False,
        False,
        False,
        False,
    )

    latent_dim = int(train_ds.image_features.shape[-1])
    channels_num = int(train_ds.channels_num)
    eeg_len = int(train_ds.num_sample_points)
    print(
        f"[INFO] train={len(train_ds)} test={len(test_ds)} "
        f"latent={latent_dim} ch={channels_num} T={eeg_len}"
    )

    model = EEGProject(
        feature_dim=latent_dim, eeg_sample_points=eeg_len, channels_num=channels_num
    ).to(device)
    eeg_projector = ProjectorLinear(latent_dim, args.feature_dim).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    eeg_projector.load_state_dict(ckpt["eeg_projector_state_dict"])
    model.eval()
    eeg_projector.eval()

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    tr_raw, tr_proj, tr_obj, tr_img = encode_split(model, eeg_projector, train_loader, device)
    te_raw, te_proj, te_obj, te_img = encode_split(model, eeg_projector, test_loader, device)

    np.save(out_dir / "z_eeg_raw_train.npy", tr_raw)
    np.save(out_dir / "z_eeg_proj_train.npy", tr_proj)
    np.save(out_dir / "z_eeg_raw_test.npy", te_raw)
    np.save(out_dir / "z_eeg_proj_test.npy", te_proj)
    np.save(out_dir / "index_obj_train.npy", tr_obj)
    np.save(out_dir / "index_img_train.npy", tr_img)
    np.save(out_dir / "index_obj_test.npy", te_obj)
    np.save(out_dir / "index_img_test.npy", te_img)

    meta = {
        "subject": args.subject,
        "checkpoint": str(ckpt_path),
        "latent_dim": latent_dim,
        "feature_dim": args.feature_dim,
        "channels": DEFAULT_CHANNELS,
        "train_n": int(tr_proj.shape[0]),
        "test_n": int(te_proj.shape[0]),
        "raw_dim": int(tr_raw.shape[1]),
        "proj_dim": int(tr_proj.shape[1]),
        "note": "Default adapter input is z_eeg_proj_*; paired with ATM ViT-H clip_img_*_1024 by flat index.",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[OK] wrote embeds to {out_dir}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
