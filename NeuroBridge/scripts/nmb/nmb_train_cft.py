#!/usr/bin/env python3
"""Train Memory-Conditional Flow Transport: (NB proj + RAG mem) -> Fusion space."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "nb_adapter"))
from train_nb_adapter import concept_split, eval_cos, l2_np, retrieval_metrics  # noqa: E402


class MLPTransport(nn.Module):
    def __init__(self, din: int, dout: int = 1024, hidden: int = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(din, hidden),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(hidden, dout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@torch.no_grad()
def predict_batch(model: nn.Module, x: np.ndarray, device: torch.device, bs: int = 512) -> np.ndarray:
    model.eval()
    outs = []
    xt = torch.from_numpy(x.astype(np.float32))
    for i in range(0, len(xt), bs):
        outs.append(model(xt[i : i + bs].to(device)).cpu().numpy())
    return np.concatenate(outs, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-dir", type=str, required=True)
    ap.add_argument("--mem-train", type=str, required=True)
    ap.add_argument("--mem-test", type=str, required=True)
    ap.add_argument("--fusion-mem-train", type=str, default="")
    ap.add_argument("--fusion-mem-test", type=str, default="")
    ap.add_argument("--fusion-train", type=str, required=True)
    ap.add_argument("--fusion-test", type=str, required=True)
    ap.add_argument("--gallery", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--use-raw", action="store_true")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    embed_dir = Path(args.embed_dir)
    z_proj_tr = np.load(embed_dir / "z_eeg_proj_train.npy")
    z_proj_te = np.load(embed_dir / "z_eeg_proj_test.npy")
    mem_tr = np.load(args.mem_train).astype(np.float32)
    mem_te = np.load(args.mem_test).astype(np.float32)
    y_tr = np.load(args.fusion_train).astype(np.float32)
    y_te = np.load(args.fusion_test).astype(np.float32)
    gallery = np.load(args.gallery).astype(np.float32)

    parts_tr = [z_proj_tr, mem_tr]
    parts_te = [z_proj_te, mem_te]
    if args.fusion_mem_train and args.fusion_mem_test:
        parts_tr.append(np.load(args.fusion_mem_train).astype(np.float32))
        parts_te.append(np.load(args.fusion_mem_test).astype(np.float32))
    if args.use_raw:
        parts_tr.append(np.load(embed_dir / "z_eeg_raw_train.npy"))
        parts_te.append(np.load(embed_dir / "z_eeg_raw_test.npy"))
    x_tr = np.concatenate(parts_tr, axis=1)
    x_te = np.concatenate(parts_te, axis=1)

    train_idx, val_idx, _ = concept_split(1654, 10, 0.1, args.seed)
    x_tr_t = torch.from_numpy(x_tr[train_idx])
    y_tr_t = torch.from_numpy(y_tr[train_idx])
    x_va_t = torch.from_numpy(x_tr[val_idx])
    y_va_t = torch.from_numpy(y_tr[val_idx])

    model = MLPTransport(int(x_tr.shape[1]), int(y_tr.shape[1])).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(x_tr_t, y_tr_t), batch_size=args.batch_size, shuffle=True)

    best_val, best_state, bad = -1.0, None, 0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        loss_sum, n = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = F.normalize(model(xb), dim=-1)
            tgt = F.normalize(yb, dim=-1)
            loss = F.mse_loss(pred, tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item()) * xb.shape[0]
            n += xb.shape[0]
        val_cos = eval_cos(model, x_va_t, y_va_t, device)
        print(f"[cft] ep={ep:03d} loss={loss_sum/max(n,1):.4f} val_cos={val_cos:.4f}")
        if val_cos > best_val + 1e-5:
            best_val, best_state, bad = val_cos, {k: v.cpu() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    if best_state:
        model.load_state_dict(best_state)

    pred_test = l2_np(predict_batch(model, x_te, device))
    np.save(out_dir / "cft_mlp_test_fusion.npy", pred_test.astype(np.float32))
    torch.save({"state_dict": model.state_dict(), "din": x_tr.shape[1]}, out_dir / "cft_mlp.pt")

    report = {
        "best_val_cos": best_val,
        "test_gt_cos": float(np.mean(np.sum(pred_test * l2_np(y_te), axis=1))),
        "test_retrieval": retrieval_metrics(pred_test, gallery),
        "seconds": time.time() - t0,
        "din": int(x_tr.shape[1]),
    }
    (out_dir / "cft_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
