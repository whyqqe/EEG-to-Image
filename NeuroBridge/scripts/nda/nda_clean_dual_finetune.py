#!/usr/bin/env python3
"""Track T: dual-objective clean+CPA fine-tune of official NeuroBridge EEGProject.

Primary selection metric: clean 200-way Top-1 (main-table protocol).
Also logs CPA Top-1 every epoch. Optional mid-layer (HCF) InfoNCE for T1b.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

NB_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NB_ROOT))

from module.dataset import EEGPreImageDataset  # noqa: E402
from module.eeg_encoder.model import EEGProject  # noqa: E402
from module.projector import ProjectorLinear  # noqa: E402
from module.util import retrieve_all  # noqa: E402

DEFAULT_CHANNELS = [
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2",
]


def l2n(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1)


def flatten_gallery(x: np.ndarray) -> np.ndarray:
    """(N,D) / (N,1,D) / (1,N,1,D) / (R,N,K,D) → (N,D)."""
    a = np.asarray(x)
    if a.ndim == 4:
        a = a.mean(axis=(0, 2))
    elif a.ndim == 3:
        if a.shape[0] in (1654, 200):
            a = a.mean(axis=1)
        elif a.shape[1] in (1654, 200):
            a = a.mean(axis=(0, 2)) if a.ndim == 4 else a.mean(axis=0)
            if a.ndim == 3:
                a = a.mean(axis=1)
        else:
            a = a.reshape(a.shape[0], -1, a.shape[-1]).mean(axis=1)
    if a.ndim != 2:
        raise ValueError(f"cannot flatten gallery {x.shape} → {a.shape}")
    return a.astype(np.float32)


def flatten_train_imgs(x: np.ndarray) -> np.ndarray:
    """RN50 train (1654,10,D) or CPA (1,1654,10,D) → (16540,D)."""
    a = np.asarray(x)
    if a.ndim == 4:
        a = a[0]  # (1654,10,D)
    if a.ndim == 3 and a.shape[0] == 1654 and a.shape[1] == 10:
        return a.reshape(-1, a.shape[-1]).astype(np.float32)
    if a.ndim == 2 and a.shape[0] == 16540:
        return a.astype(np.float32)
    raise ValueError(f"unexpected train image feat shape {x.shape}")


class DualGalleryDataset(Dataset):
    """EEG + paired clean/CPA image features (same object/image index)."""

    def __init__(
        self,
        subject: int,
        eeg_dir: str,
        rn50_dir: str,
        clean_feats: np.ndarray,
        cpa_feats: np.ndarray,
        hcf: np.ndarray | None,
        train: bool,
    ):
        self.base = EEGPreImageDataset(
            [subject], eeg_dir, DEFAULT_CHANNELS, [0, 250],
            rn50_dir, "", False, [], True, False, None, train, False, False, False,
        )
        assert len(self.base) == len(clean_feats) == len(cpa_feats), (
            f"len mismatch eeg={len(self.base)} clean={len(clean_feats)} cpa={len(cpa_feats)}"
        )
        if hcf is not None:
            assert len(hcf) == len(self.base), f"hcf {len(hcf)} != eeg {len(self.base)}"
        self.clean = clean_feats
        self.cpa = cpa_feats
        self.hcf = hcf

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        eeg, _img, _txt, sid, obj, img_i, rep = self.base[index]
        out = {
            "eeg": eeg,
            "sid": sid,
            "clean": torch.tensor(self.clean[index], dtype=torch.float32),
            "cpa": torch.tensor(self.cpa[index], dtype=torch.float32),
            "obj": obj,
            "img_i": img_i,
        }
        if self.hcf is not None:
            out["hcf"] = torch.tensor(self.hcf[index], dtype=torch.float32)
        return out


def info_nce(a: torch.Tensor, b: torch.Tensor, temp: float = 0.07) -> torch.Tensor:
    a, b = l2n(a), l2n(b)
    logits = (a @ b.T) / temp
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


@torch.no_grad()
def eval_dual(
    model: nn.Module,
    eeg_proj: nn.Module,
    img_proj: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    eeg_proj.eval()
    img_proj.eval()
    zs, cleans, cpas = [], [], []
    for batch in loader:
        eeg = batch["eeg"].to(device)
        z = l2n(eeg_proj(model(eeg)))
        zs.append(z.cpu().numpy())
        cleans.append(l2n(img_proj(batch["clean"].to(device))).cpu().numpy())
        cpas.append(l2n(img_proj(batch["cpa"].to(device))).cpu().numpy())
    z = np.concatenate(zs)
    clean = np.concatenate(cleans)
    cpa = np.concatenate(cpas)
    # concept-level: test already 200; train uses image-level (not primary)
    t5c, t1c, n = retrieve_all(z, clean, True)
    t5a, t1a, _ = retrieve_all(z, cpa, True)
    return {
        "n": int(n),
        "top1_clean": 100.0 * t1c / n,
        "top5_clean": 100.0 * t5c / n,
        "top1_cpa": 100.0 * t1a / n,
        "top5_cpa": 100.0 * t5a / n,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nb-root", type=str, default=str(NB_ROOT))
    ap.add_argument("--init-checkpoint", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--subject", type=int, default=8)
    ap.add_argument("--num-epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--lambda-clean", type=float, default=1.0)
    ap.add_argument("--lambda-cpa", type=float, default=0.5)
    ap.add_argument("--lambda-mid", type=float, default=0.25, help="HCF mid-layer InfoNCE weight (0=off)")
    ap.add_argument("--hcf-train", type=str, default="")
    ap.add_argument("--hcf-test", type=str, default="")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temp", type=float, default=0.07)
    args = ap.parse_args()

    root = Path(args.nb_root)
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    eeg_dir = str(root / "data/things_eeg/preprocessed_eeg")
    rn50_dir = str(root / "data/things_eeg/image_feature/RN50")
    aug_dir = root / "data/things_eeg/image_feature/RN50/GaussianBlur-GaussianNoise-LowResolution-Mosaic"

    clean_train = flatten_train_imgs(np.load(Path(rn50_dir) / "image_train.npy"))
    clean_test = flatten_gallery(np.load(Path(rn50_dir) / "image_test.npy"))
    cpa_train = flatten_train_imgs(np.load(aug_dir / "train.npy"))
    cpa_test = flatten_gallery(np.load(aug_dir / "test.npy"))

    hcf_train = hcf_test = None
    if args.lambda_mid > 0 and args.hcf_train and args.hcf_test:
        hcf_train = np.load(args.hcf_train).astype(np.float32)
        hcf_test = np.load(args.hcf_test).astype(np.float32)
        print(f"[INFO] HCF mid-layer on: train={hcf_train.shape} test={hcf_test.shape}")

    train_ds = DualGalleryDataset(
        args.subject, eeg_dir, rn50_dir, clean_train, cpa_train, hcf_train, True
    )
    test_ds = DualGalleryDataset(
        args.subject, eeg_dir, rn50_dir, clean_test, cpa_test, hcf_test, False
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=200, shuffle=False)

    img_dim = int(clean_train.shape[-1])
    eeg_len = int(train_ds.base.num_sample_points)
    channels_num = int(train_ds.base.channels_num)
    feature_dim = 512

    ckpt_path = Path(args.init_checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = root / ckpt_path
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = EEGProject(feature_dim=img_dim, eeg_sample_points=eeg_len, channels_num=channels_num).to(device)
    eeg_proj = ProjectorLinear(img_dim, feature_dim).to(device)
    img_proj = ProjectorLinear(img_dim, feature_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    eeg_proj.load_state_dict(ckpt["eeg_projector_state_dict"])
    img_proj.load_state_dict(ckpt["img_projector_state_dict"])

    mid_proj = None
    if hcf_train is not None:
        mid_proj = ProjectorLinear(img_dim, int(hcf_train.shape[-1])).to(device)

    # freeze image projector (gallery geometry fixed); tune EEG path
    for p in img_proj.parameters():
        p.requires_grad = False
    params = list(model.parameters()) + list(eeg_proj.parameters())
    if mid_proj is not None:
        params += list(mid_proj.parameters())
    opt = optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    # baseline before finetune
    base_metrics = eval_dual(model, eeg_proj, img_proj, test_loader, device)
    print(f"[BASE] clean_top1={base_metrics['top1_clean']:.1f}% cpa_top1={base_metrics['top1_cpa']:.1f}%")

    history = [{"phase": "init", **base_metrics}]
    best_clean, best_epoch = -1.0, 0

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        eeg_proj.train()
        if mid_proj is not None:
            mid_proj.train()
        ep_loss = 0.0
        for batch in tqdm(train_loader, desc=f"dual-ft-{epoch}"):
            eeg = batch["eeg"].to(device)
            clean = batch["clean"].to(device)
            cpa = batch["cpa"].to(device)
            opt.zero_grad()
            raw = model(eeg)
            z = eeg_proj(raw)
            zc = img_proj(clean)
            za = img_proj(cpa)
            loss = args.lambda_clean * info_nce(z, zc, args.temp)
            loss = loss + args.lambda_cpa * info_nce(z, za, args.temp)
            if mid_proj is not None:
                loss = loss + args.lambda_mid * info_nce(mid_proj(raw), batch["hcf"].to(device), args.temp)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            ep_loss += float(loss.item())

        metrics = eval_dual(model, eeg_proj, img_proj, test_loader, device)
        row = {
            "phase": "finetune",
            "epoch": epoch,
            "loss": ep_loss / max(len(train_loader), 1),
            **metrics,
        }
        history.append(row)
        print(
            f"[ep {epoch}] loss={row['loss']:.4f} "
            f"clean_top1={metrics['top1_clean']:.1f}% cpa_top1={metrics['top1_cpa']:.1f}%"
        )
        if metrics["top1_clean"] > best_clean:
            best_clean, best_epoch = metrics["top1_clean"], epoch
            payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "eeg_projector_state_dict": eeg_proj.state_dict(),
                "img_projector_state_dict": img_proj.state_dict(),
                "top1_clean": metrics["top1_clean"],
                "top5_clean": metrics["top5_clean"],
                "top1_cpa": metrics["top1_cpa"],
                "top5_cpa": metrics["top5_cpa"],
                "lambda_clean": args.lambda_clean,
                "lambda_cpa": args.lambda_cpa,
                "lambda_mid": args.lambda_mid,
                "init_checkpoint": str(ckpt_path),
                "design": "T1a dual clean+CPA (+optional T1b HCF)",
            }
            if mid_proj is not None:
                payload["mid_projector_state_dict"] = mid_proj.state_dict()
            torch.save(payload, out / "checkpoint_clean_dual_best.pth")

    # encode test embeds with best ckpt
    best = torch.load(out / "checkpoint_clean_dual_best.pth", map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    eeg_proj.load_state_dict(best["eeg_projector_state_dict"])
    model.eval()
    eeg_proj.eval()
    emb_dir = out / "embeds"
    emb_dir.mkdir(exist_ok=True)
    for tag, ds in (("train", train_ds), ("test", test_ds)):
        xs = []
        with torch.no_grad():
            for batch in DataLoader(ds, batch_size=512, shuffle=False):
                z = l2n(eeg_proj(model(batch["eeg"].to(device))))
                xs.append(z.cpu().numpy())
        arr = np.concatenate(xs).astype(np.float32)
        np.save(emb_dir / f"z_eeg_proj_{tag}.npy", arr)
        print(f"[OK] {tag} embeds {arr.shape}")

    report = {
        "pipeline": "nda_clean_dual_finetune",
        "track": "T",
        "init_checkpoint": str(ckpt_path),
        "baseline": base_metrics,
        "best_epoch": best_epoch,
        "best_top1_clean": best_clean,
        "best_top1_cpa": best.get("top1_cpa"),
        "best_top5_clean": best.get("top5_clean"),
        "lambda_clean": args.lambda_clean,
        "lambda_cpa": args.lambda_cpa,
        "lambda_mid": args.lambda_mid,
        "checkpoint": str(out / "checkpoint_clean_dual_best.pth"),
        "selection": "max top1_clean (main-table protocol)",
    }
    (out / "clean_dual_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(history).to_csv(out / "clean_dual_history.csv", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
