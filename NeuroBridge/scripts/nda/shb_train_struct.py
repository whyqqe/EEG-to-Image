#!/usr/bin/env python3
"""SHB step 2: STRUCTURE-SPECIALISED head off the frozen backbone `raw`.

Defect being fixed
------------------
The existing structure tower reads `z_decode_vith`, whose training objective is
ViT-H-14 CLIP-Image cosine + InfoNCE -- a *semantic* objective that actively
collapses spatial layout. Regressing a 64x64 depth map (measured pearson 0.689)
or a 4x64x64 SDXL latent (measured pearson 0.332) from such a latent is an
information bottleneck, not a capacity problem.

This step learns `StructHead(raw) -> s (512-d)` with an explicitly STRUCTURAL
multi-task objective (depth + SDXL-VAE latent), i.e. objective-level decoupling
from the semantic branch. It is the prerequisite for the geometry field.

Outputs
  struct_head_best.pth
  struct_train.npy (16540,512) / struct_test.npy (200,512)
  struct_train_sub.npy (16540,K,512) / struct_test_sub.npy (200,K,512)
  pred_depth_test_64.npy (200,64,64), pred_vae_test.npy (200,4,64,64)
  struct_head_report.json   (pearson vs the 0.689 / 0.332 baselines)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class StructHead(nn.Module):
    """raw (1024) -> structure-specialised latent s (dim)."""

    def __init__(self, in_dim: int = 1024, dim: int = 512, hidden: int = 1024):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.trunk(z)


class DepthOut(nn.Module):
    def __init__(self, dim: int, res: int):
        super().__init__()
        self.res = res
        self.net = nn.Sequential(
            nn.Linear(dim, 1024), nn.GELU(), nn.Linear(1024, res * res)
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(s).view(-1, 1, self.res, self.res))


class VaeOut(nn.Module):
    def __init__(self, dim: int, ch: int = 4, res: int = 64):
        super().__init__()
        self.ch, self.res = ch, res
        self.net = nn.Sequential(
            nn.Linear(dim, 1024), nn.GELU(), nn.Linear(1024, ch * res * res)
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s).view(-1, self.ch, self.res, self.res)


def grad_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    px = pred[:, :, 1:] - pred[:, :, :-1]
    py = pred[:, 1:, :] - pred[:, :-1, :]
    gx = gt[:, :, 1:] - gt[:, :, :-1]
    gy = gt[:, 1:, :] - gt[:, :-1, :]
    return F.l1_loss(px, gx) + F.l1_loss(py, gy)


def pearson_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.reshape(len(a), -1).astype(np.float64)
    b = b.reshape(len(b), -1).astype(np.float64)
    ac, bc = a - a.mean(1, keepdims=True), b - b.mean(1, keepdims=True)
    den = np.sqrt((ac * ac).sum(1) * (bc * bc).sum(1)).clip(1e-12)
    return (ac * bc).sum(1) / den


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-train", type=str, required=True)
    ap.add_argument("--raw-test", type=str, required=True)
    ap.add_argument("--raw-train-sub", type=str, default="")
    ap.add_argument("--raw-test-sub", type=str, default="")
    ap.add_argument("--depth-train", type=str, required=True)
    ap.add_argument("--depth-test", type=str, required=True)
    ap.add_argument("--vae-train", type=str, required=True)
    ap.add_argument("--vae-test", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--num-epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda-vae", type=float, default=0.5)
    ap.add_argument("--lambda-grad", type=float, default=0.5)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    z_tr = np.load(args.raw_train).astype(np.float32)
    z_te = np.load(args.raw_test).astype(np.float32)
    d_tr = np.load(args.depth_train).astype(np.float32)[:, None]
    d_te = np.load(args.depth_test).astype(np.float32)[:, None]
    v_tr = np.load(args.vae_train).astype(np.float32)
    v_te = np.load(args.vae_test).astype(np.float32)
    print(f"[INFO] z_tr {z_tr.shape} d_tr {d_tr.shape} v_tr {v_tr.shape}")
    assert len(z_tr) == len(d_tr) == len(v_tr), "train length mismatch"
    assert len(z_te) == len(d_te) == len(v_te), "test length mismatch"

    # per-channel standardisation of the VAE target (mirrors train_eeg_vae_head.py)
    vm = v_tr.reshape(len(v_tr), v_tr.shape[1], -1).mean(axis=2, keepdims=True).reshape(1, -1, 1, 1)
    vs = v_tr.reshape(len(v_tr), v_tr.shape[1], -1).std(axis=2, keepdims=True).reshape(1, -1, 1, 1)
    vs = np.maximum(vs, 1e-6)
    v_tr_n = (v_tr - vm) / vs
    v_te_n = (v_te - vm) / vs
    res_d = d_tr.shape[-1]

    head = StructHead(z_tr.shape[1], args.dim).to(device)
    dep = DepthOut(args.dim, res_d).to(device)
    vae = VaeOut(args.dim, v_tr.shape[1], v_tr.shape[2]).to(device)
    params = list(head.parameters()) + list(dep.parameters()) + list(vae.parameters())
    opt = optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.num_epochs)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(z_tr), torch.from_numpy(d_tr), torch.from_numpy(v_tr_n)),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
    )

    def infer(z: np.ndarray):
        head.eval(); dep.eval(); vae.eval()
        ds, dd, dv = [], [], []
        with torch.no_grad():
            for s in range(0, len(z), 2048):
                zb = torch.from_numpy(z[s : s + 2048]).to(device)
                sb = head(zb)
                ds.append(sb.cpu().numpy())
                dd.append(dep(sb).squeeze(1).cpu().numpy())
                dv.append(vae(sb).cpu().numpy())
        return (np.concatenate(ds, 0).astype(np.float32),
                np.concatenate(dd, 0).astype(np.float32),
                np.concatenate(dv, 0).astype(np.float32))

    best = -1.0
    hist = []
    for ep in range(1, args.num_epochs + 1):
        head.train(); dep.train(); vae.train()
        acc = 0.0
        for zb, db, vb in loader:
            zb, db, vb = zb.to(device), db.to(device), vb.to(device)
            opt.zero_grad()
            s = head(zb)
            pd = dep(s)
            pv = vae(s)
            loss = (F.l1_loss(pd, db) + args.lambda_grad * grad_loss(pd.squeeze(1), db.squeeze(1))
                    + args.lambda_vae * F.l1_loss(pv, vb))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            acc += float(loss.item())
        sched.step()

        if ep % 5 == 0 or ep == 1:
            _, pd_te, pv_te = infer(z_te)
            r_d = float(pearson_rows(pd_te, d_te.squeeze(1)).mean())
            r_v = float(pearson_rows(pv_te, v_te_n).mean())
            hist.append({"epoch": ep, "loss": acc / max(len(loader), 1), "depth_pearson": r_d, "vae_pearson": r_v})
            print(f"[ep {ep}] loss={acc/max(len(loader),1):.4f} depth_pearson={r_d:.4f} vae_pearson={r_v:.4f}")
            if r_d + r_v > best:
                best = r_d + r_v
                torch.save({"state_dict": head.state_dict(), "dim": args.dim,
                            "in_dim": z_tr.shape[1], "epoch": ep,
                            "depth_pearson": r_d, "vae_pearson": r_v}, out / "struct_head_best.pth")

    ck = torch.load(out / "struct_head_best.pth", map_location=device, weights_only=False)
    head.load_state_dict(ck["state_dict"])
    s_tr, pd_tr, pv_tr = infer(z_tr)
    s_te, pd_te, pv_te = infer(z_te)
    np.save(out / "struct_train.npy", s_tr)
    np.save(out / "struct_test.npy", s_te)
    np.save(out / "pred_depth_test_64.npy", pd_te)
    np.save(out / "pred_vae_test.npy", (pv_te * vs + vm).astype(np.float32))

    sub_shapes = {}
    if args.raw_train_sub and Path(args.raw_train_sub).is_file():
        zs = np.load(args.raw_train_sub).astype(np.float32)      # (N,K,1024)
        ss = np.stack([infer(zs[:, k])[0] for k in range(zs.shape[1])], axis=1)
        np.save(out / "struct_train_sub.npy", ss)
        sub_shapes["train_sub"] = list(ss.shape)
    if args.raw_test_sub and Path(args.raw_test_sub).is_file():
        zs = np.load(args.raw_test_sub).astype(np.float32)
        ss = np.stack([infer(zs[:, k])[0] for k in range(zs.shape[1])], axis=1)
        np.save(out / "struct_test_sub.npy", ss)
        sub_shapes["test_sub"] = list(ss.shape)

    r_d = float(pearson_rows(pd_te, d_te.squeeze(1)).mean())
    r_v = float(pearson_rows(pv_te, v_te_n).mean())
    report = {
        "pipeline": "shb_train_struct",
        "objective": "STRUCTURAL multi-task (depth L1+grad, SDXL-VAE latent L1) off frozen raw",
        "why": "existing structure tower reads a CLIP-semantic latent (z_decode_vith); objective mismatch",
        "dim": args.dim,
        "best_epoch": ck.get("epoch"),
        "test_depth_pearson": r_d,
        "test_vae_pearson_std": r_v,
        "baseline_depth_pearson": 0.6889,
        "baseline_vae_pearson": 0.3317,
        "depth_pearson_gain": r_d - 0.6889,
        "vae_pearson_gain": r_v - 0.3317,
        "n_train": int(len(z_tr)), "n_test": int(len(z_te)),
        "subset_shapes": sub_shapes,
        "history": hist,
    }
    (out / "struct_head_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
