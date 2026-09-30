#!/usr/bin/env python3
"""Calibrate a missing subject adapter on frozen SharedSpecific encoder (MindBridge-style)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

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


def l2(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--subject", type=int, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--nb-root", type=str, default=str(NB_ROOT))
    args = ap.parse_args()

    root = Path(args.nb_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    subjects = [int(s) for s in ckpt.get("subjects", [])]
    img_dim = int(ckpt["img_dim"])
    feature_dim = int(ckpt["feature_dim"])
    model = SharedSpecificEncoder(
        subject_ids=subjects,
        feature_dim=img_dim,
        eeg_sample_points=int(ckpt["eeg_sample_points"]),
        channels_num=int(ckpt["channels_num"]),
        n_extra_blocks=int(ckpt.get("n_extra_blocks", 1)),
        use_adapter=True,
    ).to(device)
    eeg_projector = ProjectorLinear(img_dim, feature_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    eeg_projector.load_state_dict(ckpt["eeg_projector_state_dict"])
    img_projector = ProjectorLinear(img_dim, feature_dim).to(device)
    if "img_projector_state_dict" in ckpt:
        img_projector.load_state_dict(ckpt["img_projector_state_dict"])
    img_projector.eval()
    for p in img_projector.parameters():
        p.requires_grad_(False)

    sid = int(args.subject)
    model.add_subject(sid)
    # move newly added modules to device
    model = model.to(device)
    model.train_only_subject(sid)
    for p in eeg_projector.parameters():
        p.requires_grad_(False)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    eeg_dir = str(root / "data/things_eeg/preprocessed_eeg")
    rn50_dir = str(root / "data/things_eeg/image_feature/RN50")
    ds = EEGPreImageDataset(
        [sid], eeg_dir, DEFAULT_CHANNELS, [0, 250],
        rn50_dir, "", False, [], True, False, None, True, False, False, False,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=2)

    for ep in range(1, args.epochs + 1):
        losses = []
        for batch in tqdm(loader, desc=f"calib-sub{sid:02d}-{ep}"):
            eeg, img, _t, sid_t, *_ = batch
            eeg, img, sid_t = eeg.to(device), img.to(device), sid_t.to(device)
            z = eeg_projector(model(eeg, sid_t))
            y = img_projector(img)
            logits = l2(z) @ l2(y).T / 0.07
            labels = torch.arange(z.shape[0], device=device)
            loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        print(f"[ep {ep}] loss={np.mean(losses):.4f}")

    # save calibrated ckpt + encode
    ckpt_out = {
        **{k: ckpt[k] for k in ckpt if k not in ("model_state_dict",)},
        "model_state_dict": model.state_dict(),
        "eeg_projector_state_dict": eeg_projector.state_dict(),
        "img_projector_state_dict": img_projector.state_dict(),
        "subjects": model.subject_ids,
        "calibrated_subject": sid,
    }
    torch.save(ckpt_out, out / f"checkpoint_ss_calib_sub{sid:02d}.pth")

    model.eval()
    eeg_projector.eval()
    for train_flag, tag in ((True, "train"), (False, "test")):
        ds2 = EEGPreImageDataset(
            [sid], eeg_dir, DEFAULT_CHANNELS, [0, 250],
            rn50_dir, "", False, [], True, False, None, train_flag, False, False, False,
        )
        xs = []
        with torch.no_grad():
            for batch in DataLoader(ds2, batch_size=512, shuffle=False):
                eeg, _img, _t, sid_t, *_ = batch
                eeg, sid_t = eeg.to(device), sid_t.to(device)
                xs.append(eeg_projector(model(eeg, sid_t)).cpu().numpy())
        arr = np.concatenate(xs).astype(np.float32)
        np.save(out / f"z_ret_sub{sid:02d}_{tag}.npy", arr)
        np.save(out / f"z_eeg_proj_{tag}.npy", arr)
        print(f"[OK] {tag} {arr.shape}")

if __name__ == "__main__":
    main()
