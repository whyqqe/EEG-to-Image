#!/usr/bin/env python3
"""GACL-lite: Fusion->ViT-H bridge with decode-gap sample weighting.

Weights train samples where ViT-H memory RAG is weak vs GT (semantic gap),
so the bridge learns to inject Fusion semantics where low-level memory fails.
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
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
NB_ADAPTER = SCRIPT_DIR.parent / "nb_adapter"
sys.path.insert(0, str(NB_ADAPTER))

from train_nb_adapter import (  # noqa: E402
    MLPAdapter,
    concept_split,
    eval_cos,
    l2_np,
    predict,
    retrieval_metrics,
)


def cosine_mse_weighted(pred: torch.Tensor, target: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    per = ((pred - target) ** 2).sum(dim=-1)
    return (per * w).sum() / w.sum().clamp(min=1e-8)


def train_gacl(
    model: nn.Module,
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
    w_tr: torch.Tensor,
    x_va: torch.Tensor,
    y_va: torch.Tensor,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    patience: int,
) -> dict:
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(
        TensorDataset(x_tr, y_tr, w_tr),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    best_val = -1.0
    best_state = None
    bad = 0
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb, wb in loader:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            pred = model(xb)
            loss = cosine_mse_weighted(pred, yb, wb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        val_cos = eval_cos(model, x_va, y_va, device)
        print(f"[gacl] ep={ep:03d} val_cos={val_cos:.4f}")
        if val_cos > best_val + 1e-5:
            best_val = val_cos
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_val_cos": best_val, "model": model}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fusion-train", type=str, required=True)
    ap.add_argument("--clip-train", type=str, required=True)
    ap.add_argument("--clip-test", type=str, required=True)
    ap.add_argument("--mem-train", type=str, required=True, help="ViT-H RAG train embeds for gap weights")
    ap.add_argument("--gallery", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--fusion-test-src", type=str, nargs="*", default=[])
    ap.add_argument("--gap-beta", type=float, default=2.0, help="weight = 1 + beta*(1-cos(mem,clip_gt))")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    x_train = np.load(args.fusion_train).astype(np.float32)
    y_train = np.load(args.clip_train).astype(np.float32)
    y_test = np.load(args.clip_test).astype(np.float32)
    mem_train = l2_np(np.load(args.mem_train).astype(np.float32))
    gallery = np.load(args.gallery).astype(np.float32)

    cos_mem_gt = np.sum(mem_train * l2_np(y_train), axis=1)
    weights = (1.0 + args.gap_beta * np.clip(1.0 - cos_mem_gt, 0.0, 1.0)).astype(np.float32)
    print(f"[gacl] weight mean={weights.mean():.3f} min={weights.min():.3f} max={weights.max():.3f}")

    tr_idx, va_idx, _ = concept_split(1654, 10, 0.1, args.seed)
    x_tr = torch.from_numpy(x_train[tr_idx])
    y_tr = torch.from_numpy(y_train[tr_idx])
    w_tr = torch.from_numpy(weights[tr_idx])
    x_va = torch.from_numpy(x_train[va_idx])
    y_va = torch.from_numpy(y_train[va_idx])

    pack = train_gacl(
        MLPAdapter(1024, 1024),
        x_tr,
        y_tr,
        w_tr,
        x_va,
        y_va,
        device,
        args.epochs,
        args.lr,
        args.weight_decay,
        args.batch_size,
        args.patience,
    )
    model = pack.pop("model")
    torch.save({"state_dict": model.state_dict(), "gap_beta": args.gap_beta}, out_dir / "gacl_mlp_adapter.pt")

    report: dict = {"gap_beta": args.gap_beta, **pack, "predictions": {}}
    for entry in args.fusion_test_src:
        tag, path = entry.split(":", 1)
        pred = predict(model, np.load(path).astype(np.float32), device)
        out_npy = out_dir / f"gacl_mlp_{tag}_test_clip_1024.npy"
        np.save(out_npy, pred)
        gt_cos = float(np.mean(np.sum(l2_np(pred) * l2_np(y_test), axis=1)))
        retr = retrieval_metrics(pred, gallery)
        report["predictions"][tag] = {"path": str(out_npy), "gt_cos": gt_cos, "retrieval": retr}
        print(f"[gacl/{tag}] gt_cos={gt_cos:.4f}")

    (out_dir / "gacl_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[OK] {out_dir / 'gacl_report.json'}")


if __name__ == "__main__":
    main()
