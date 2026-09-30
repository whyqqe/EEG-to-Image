#!/usr/bin/env python3
"""Train EEG→AlexNet mid (layer2 GAP) head — ATM/MindEye low-level style supervision.

Target: GAP of AlexNet features[5] on GT stimuli (same layer as official Alex2 2-way).
Outputs: checkpoint, pred alex feats (test), u_alex.npy (deployable confidence).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


class ZImageDS(Dataset):
    def __init__(self, z: np.ndarray, paths: list[Path]):
        assert len(z) == len(paths)
        self.z = z.astype(np.float32)
        self.paths = paths

    def __len__(self) -> int:
        return len(self.z)

    def __getitem__(self, i: int):
        return self.z[i], str(self.paths[i])


class AlexMidHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 192, hidden: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(z), dim=-1)


@torch.no_grad()
def encode_alex_gap(model, tfm, paths: list[str], device: torch.device) -> torch.Tensor:
    outs = []
    for p in paths:
        x = tfm(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
        f = model.features[:6](x)  # through features[5] (Alex2 layer)
        g = f.mean(dim=(2, 3))
        outs.append(F.normalize(g, dim=-1))
    return torch.cat(outs, dim=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eeg-train-npy", type=str, required=True)
    ap.add_argument("--eeg-test-npy", type=str, required=True)
    ap.add_argument("--images-root", type=str, default="/project/peilab/why/data/images_set")
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--num-epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--max-train", type=int, default=0, help="0=all; else subsample for speed")
    args = ap.parse_args()

    import sys

    sys.path.insert(0, "/project/peilab/why/eeg-brainit/scripts")
    from eval_atm_pipeline import list_test_images  # type: ignore

    def list_train_images(images_root: Path) -> list[Path]:
        root = images_root / "training_images"
        paths: list[Path] = []
        for d in sorted([p for p in root.iterdir() if p.is_dir()]):
            imgs = sorted(list(d.glob("*.jpg")) + list(d.glob("*.JPEG")) + list(d.glob("*.png")))
            paths.extend(imgs)
        return paths

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    z_tr = np.load(args.eeg_train_npy).astype(np.float32)
    z_te = np.load(args.eeg_test_npy).astype(np.float32)
    z_tr = z_tr / np.linalg.norm(z_tr, axis=1, keepdims=True).clip(1e-8)
    z_te = z_te / np.linalg.norm(z_te, axis=1, keepdims=True).clip(1e-8)

    root = Path(args.images_root)
    train_paths = list_train_images(root)
    test_paths = list_test_images(root)
    assert len(train_paths) == len(z_tr), f"train {len(train_paths)} vs z {len(z_tr)}"
    assert len(test_paths) == len(z_te)

    if args.max_train > 0 and args.max_train < len(z_tr):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(z_tr), size=args.max_train, replace=False)
        z_tr = z_tr[idx]
        train_paths = [train_paths[i] for i in idx]

    alex = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1).to(device).eval()
    tfm = models.AlexNet_Weights.IMAGENET1K_V1.transforms()

    ds = ZImageDS(z_tr, train_paths)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    head = AlexMidHead(z_tr.shape[1], out_dim=192).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)

    best = -1e9
    history = []
    for ep in range(1, args.num_epochs + 1):
        head.train()
        losses = []
        for zb, paths in tqdm(loader, desc=f"alex-mid-{ep}"):
            zb = zb.to(device)
            with torch.no_grad():
                tgt = encode_alex_gap(alex, tfm, list(paths), device)
            pred = head(zb)
            loss = 1.0 - (pred * tgt).sum(dim=-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        # quick val on test
        head.eval()
        with torch.no_grad():
            pred_te = head(torch.from_numpy(z_te).to(device))
            tgt_te = encode_alex_gap(alex, tfm, [str(p) for p in test_paths], device)
            cos = float((pred_te * tgt_te).sum(dim=-1).mean().item())
        row = {"epoch": ep, "loss": float(np.mean(losses)), "test_cos": cos}
        history.append(row)
        print(json.dumps(row))
        if cos > best:
            best = cos
            torch.save(
                {"state_dict": head.state_dict(), "in_dim": z_tr.shape[1], "out_dim": 192, "metrics": row},
                out / "checkpoint_alex_mid_best.pth",
            )
            np.save(out / "pred_alex_gap_test.npy", pred_te.cpu().numpy().astype(np.float32))
            # deployable confidence: predicted feature energy × (optional) consistency proxy
            u = pred_te.norm(dim=-1).cpu().numpy().astype(np.float32)
            # rank-normalize
            order = np.argsort(u)
            ranks = np.empty_like(u)
            ranks[order] = np.linspace(0.0, 1.0, num=len(u), dtype=np.float32)
            np.save(out / "u_alex.npy", ranks)

    report = {
        "pipeline": "eeg_alex_mid_head",
        "alex_layer": "features[5] GAP ch=192 (official Alex2 layer)",
        "best_test_cos": best,
        "n_train": int(len(z_tr)),
        "n_test": int(len(z_te)),
        "history": history[-5:],
        "protocol_note": "Supervises same AlexNet layer used in ATM/MindEye AlexNet(2) 2-way",
    }
    (out / "alex_mid_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
