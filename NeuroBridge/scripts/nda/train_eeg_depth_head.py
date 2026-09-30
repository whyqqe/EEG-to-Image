#!/usr/bin/env python3
"""Track S: train EEG→Depth head on GT DepthAnything labels (not neighbors).

Input: frozen NDA-SS / dual-finetune z_eeg embeds (N, D)
Target: train_depth_64.npy aligned 1:1 with train embeds
Output: checkpoint + predicted test depth RGB maps + COCA cn scales
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def depth_to_rgb(depth: np.ndarray) -> Image.Image:
    d = depth.astype(np.float32)
    d = (d - d.min()) / (d.max() - d.min() + 1e-8)
    u8 = (d * 255.0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(np.stack([u8, u8, u8], axis=-1))


class DepthHead(nn.Module):
    """z (D,) → 64×64 depth via MLP + light conv refine."""

    def __init__(self, in_dim: int, out_res: int = 64, hidden: int = 1024):
        super().__init__()
        self.out_res = out_res
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_res * out_res),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 1, 3, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z).view(-1, 1, self.out_res, self.out_res)
        x = torch.sigmoid(x + self.refine(x))
        return x.squeeze(1)  # (B,H,W)


def grad_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    def grads(t):
        dx = t[:, :, 1:] - t[:, :, :-1]
        dy = t[:, 1:, :] - t[:, :-1, :]
        return dx, dy

    px, py = grads(pred)
    gx, gy = grads(gt)
    return F.l1_loss(px, gx) + F.l1_loss(py, gy)


@torch.no_grad()
def eval_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)
    mae = float(np.mean(np.abs(pred - gt)))
    # per-sample pearson
    rs = []
    for i in range(len(pred)):
        a, b = pred[i].ravel(), gt[i].ravel()
        if a.std() < 1e-8 or b.std() < 1e-8:
            rs.append(0.0)
        else:
            rs.append(float(np.corrcoef(a, b)[0, 1]))
    return {"mae": mae, "pearson_mean": float(np.mean(rs)), "pearson_median": float(np.median(rs))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eeg-train-npy", type=str, required=True)
    ap.add_argument("--eeg-test-npy", type=str, required=True)
    ap.add_argument("--depth-train-npy", type=str, required=True)
    ap.add_argument("--depth-test-npy", type=str, required=True)
    ap.add_argument(
        "--val-split-json",
        default="",
        help="leakfree.py split; when given, checkpoint selection uses held-in train concepts "
        "instead of the TEST set (audit finding P1/A5)",
    )
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--num-epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda-grad", type=float, default=0.5)
    ap.add_argument("--rgb-size", type=int, default=512)
    ap.add_argument("--cn-min", type=float, default=0.35)
    ap.add_argument("--cn-max", type=float, default=0.60)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pred_dir = out / "pred_depth_rgb_512"
    pred_dir.mkdir(exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    z_tr = np.load(args.eeg_train_npy).astype(np.float32)
    z_te = np.load(args.eeg_test_npy).astype(np.float32)
    d_tr = np.load(args.depth_train_npy).astype(np.float32)
    d_te = np.load(args.depth_test_npy).astype(np.float32)
    assert len(z_tr) == len(d_tr), f"train mismatch {len(z_tr)} vs {len(d_tr)}"
    assert len(z_te) == len(d_te), f"test mismatch {len(z_te)} vs {len(d_te)}"
    # l2 eeg
    z_tr = z_tr / np.linalg.norm(z_tr, axis=1, keepdims=True).clip(1e-8)
    z_te = z_te / np.linalg.norm(z_te, axis=1, keepdims=True).clip(1e-8)

    # ---- LEAK-FREE: select on held-in train concepts, not on the TEST depth maps ----
    # Audit finding P1/A5: this script used to pick best_pearson on the TEST set.
    z_va, d_va = None, None
    if args.val_split_json:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import leakfree as LF

        sp = LF.load(args.val_split_json)
        fi = LF.rows_for(sp, "fit", len(z_tr))
        vi = LF.rows_for(sp, "val_b", len(z_tr))
        assert len(set(fi.tolist()) & set(vi.tolist())) == 0
        z_va, d_va = z_tr[vi], d_tr[vi]
        z_tr, d_tr = z_tr[fi], d_tr[fi]
        print(f"[leakfree] fit {z_tr.shape[0]} rows | val {z_va.shape[0]} rows (held-in train concepts)")

    ds = TensorDataset(torch.from_numpy(z_tr), torch.from_numpy(d_tr))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    head = DepthHead(in_dim=z_tr.shape[1], out_res=d_tr.shape[-1]).to(device)
    opt = optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    history = []
    best_pearson, best_epoch = -1.0, 0

    for epoch in range(1, args.num_epochs + 1):
        head.train()
        ep = 0.0
        for zb, db in tqdm(loader, desc=f"depth-{epoch}"):
            zb, db = zb.to(device), db.to(device)
            opt.zero_grad()
            pred = head(zb)
            loss = F.l1_loss(pred, db) + args.lambda_grad * grad_loss(pred, db)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            ep += float(loss.item())

        head.eval()
        with torch.no_grad():
            pred_te = head(torch.from_numpy(z_te).to(device)).cpu().numpy()
            sel_pred = None if z_va is None else head(torch.from_numpy(z_va).to(device)).cpu().numpy()
        metrics = eval_metrics(pred_te, d_te)
        if sel_pred is None:
            sel = metrics
        else:
            sel = eval_metrics(sel_pred, d_va)
        row = {"epoch": epoch, "loss": ep / max(len(loader), 1), **metrics,
               "val_pearson": sel["pearson_mean"], "val_mae": sel["mae"],
               "selected_on": "val" if sel_pred is not None else "test(contaminated)"}
        history.append(row)
        print(f"[ep {epoch}] loss={row['loss']:.4f} val_pearson={sel['pearson_mean']:.3f} "
              f"test_pearson={metrics['pearson_mean']:.3f} val_mae={sel['mae']:.4f}")
        if sel["pearson_mean"] > best_pearson:
            best_pearson, best_epoch = sel["pearson_mean"], epoch
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": head.state_dict(),
                    "in_dim": z_tr.shape[1],
                    "out_res": d_tr.shape[-1],
                    "metrics": metrics,
                    "val_metrics": sel,
                },
                out / "checkpoint_depth_head_best.pth",
            )
            np.save(out / "pred_depth_test_64.npy", pred_te.astype(np.float32))

    # reload best + export RGB + COCA scales
    ckpt = torch.load(out / "checkpoint_depth_head_best.pth", map_location=device, weights_only=False)
    head.load_state_dict(ckpt["state_dict"])
    head.eval()
    with torch.no_grad():
        pred_te = head(torch.from_numpy(z_te).to(device)).cpu().numpy()
    np.save(out / "pred_depth_test_64.npy", pred_te.astype(np.float32))

    # u_str = per-sample pearson vs GT (oracle proxy for routing quality at train time);
    # at test we don't have GT for routing in deployment — use prediction sharpness + self-consistency.
    # Deployable u_str: normalized spatial std of prediction (structure present) × confidence.
    u_str = []
    for i in range(len(pred_te)):
        p = pred_te[i]
        sharp = float(p.std())
        # upsample export
        t = torch.from_numpy(p)[None, None]
        hi = F.interpolate(t, size=(args.rgb_size, args.rgb_size), mode="bicubic", align_corners=False)
        hi = hi.squeeze().numpy()
        depth_to_rgb(hi).save(pred_dir / f"{i:03d}.png")
        u_str.append(sharp)
    u = np.asarray(u_str, dtype=np.float32)
    # map sharpness → [0,1] via rank percentile
    ranks = u.argsort().argsort().astype(np.float32) / max(len(u) - 1, 1)
    cn = args.cn_min + ranks * (args.cn_max - args.cn_min)
    np.save(out / "u_str.npy", ranks)
    np.save(out / "cn_scale_coca.npy", cn.astype(np.float32))
    # fixed mild scales for ablation
    np.save(out / "cn_scale_fixed05.npy", np.full(len(u), 0.5, dtype=np.float32))

    final = eval_metrics(pred_te, d_te)
    report = {
        "pipeline": "eeg_depth_head",
        "track": "S",
        "best_epoch": best_epoch,
        "best_pearson": best_pearson,
        "final_test": final,
        "cn_min": args.cn_min,
        "cn_max": args.cn_max,
        "pred_rgb_dir": str(pred_dir),
        "eeg_train": args.eeg_train_npy,
        "eeg_test": args.eeg_test_npy,
        "n_train": int(len(z_tr)),
        "n_test": int(len(z_te)),
    }
    (out / "depth_head_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    import pandas as pd

    pd.DataFrame(history).to_csv(out / "depth_head_history.csv", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
