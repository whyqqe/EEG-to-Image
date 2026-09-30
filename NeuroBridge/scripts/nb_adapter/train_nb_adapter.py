#!/usr/bin/env python3
"""Train Linear / MLP adapters: NeuroBridge z → ViT-H CLIP-1024.

Default input: z_eeg_proj (512). Target: atm_bridge clip_img_{train,test}_1024.npy
Val split: hold out 10% of train *concepts* (not random samples) for early stopping.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def l2_np(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)


def retrieval_metrics(query: np.ndarray, gallery: np.ndarray) -> dict:
    q = l2_np(query.astype(np.float32))
    g = l2_np(gallery.astype(np.float32))
    sim = q @ g.T
    n = sim.shape[0]
    ranks = []
    hits1 = hits5 = 0
    for i in range(n):
        order = np.argsort(-sim[i])
        rank = int(np.where(order == i)[0][0]) + 1
        ranks.append(rank)
        hits1 += int(rank == 1)
        hits5 += int(rank <= 5)
    paired = float(np.mean([sim[i, i] for i in range(n)]))
    # shuffled baseline
    perm = np.random.default_rng(0).permutation(n)
    shuffled = float(np.mean([sim[i, perm[i]] for i in range(n)]))
    return {
        "n": n,
        "top1": hits1 / n,
        "top5": hits5 / n,
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(np.mean(ranks)),
        "paired_cos": paired,
        "shuffled_cos": shuffled,
        "cos_gap": paired - shuffled,
    }


class LinearAdapter(nn.Module):
    def __init__(self, din: int, dout: int):
        super().__init__()
        self.proj = nn.Linear(din, dout, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class MLPAdapter(nn.Module):
    def __init__(self, din: int, dout: int, hidden: int | None = None, dropout: float = 0.0):
        super().__init__()
        h = hidden or (2 * max(din, dout))
        self.net = nn.Sequential(
            nn.Linear(din, h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, dout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def cosine_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    return F.mse_loss(pred, target)


@torch.no_grad()
def eval_cos(model: nn.Module, x: torch.Tensor, y: torch.Tensor, device: torch.device, bs: int = 2048) -> float:
    model.eval()
    cos_sum = 0.0
    n = 0
    for i in range(0, len(x), bs):
        xb = x[i : i + bs].to(device)
        yb = y[i : i + bs].to(device)
        pred = F.normalize(model(xb), dim=-1)
        tgt = F.normalize(yb, dim=-1)
        cos_sum += float((pred * tgt).sum(dim=-1).sum().item())
        n += xb.shape[0]
    return cos_sum / max(n, 1)


def concept_split(n_concepts: int = 1654, n_img: int = 10, val_frac: float = 0.1, seed: int = 0):
    rng = np.random.default_rng(seed)
    n_val = max(1, int(round(n_concepts * val_frac)))
    val_concepts = set(rng.choice(n_concepts, size=n_val, replace=False).tolist())
    train_idx, val_idx = [], []
    for c in range(n_concepts):
        base = c * n_img
        ids = list(range(base, base + n_img))
        if c in val_concepts:
            val_idx.extend(ids)
        else:
            train_idx.extend(ids)
    return np.asarray(train_idx), np.asarray(val_idx), sorted(val_concepts)


def train_one(
    name: str,
    model: nn.Module,
    x_tr: torch.Tensor,
    y_tr: torch.Tensor,
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
    loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=batch_size, shuffle=True, drop_last=False)

    best_val = -1.0
    best_state = None
    best_epoch = -1
    bad = 0
    history = []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        n = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = cosine_mse_loss(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item()) * xb.shape[0]
            n += xb.shape[0]
        train_cos = eval_cos(model, x_tr, y_tr, device)
        val_cos = eval_cos(model, x_va, y_va, device)
        row = {"epoch": ep, "loss": loss_sum / max(n, 1), "train_cos": train_cos, "val_cos": val_cos}
        history.append(row)
        print(f"[{name}] ep={ep:03d} loss={row['loss']:.4f} train_cos={train_cos:.4f} val_cos={val_cos:.4f}")
        if val_cos > best_val + 1e-5:
            best_val = val_cos
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"[{name}] early stop at ep={ep} best_ep={best_epoch} best_val={best_val:.4f}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "name": name,
        "best_epoch": best_epoch,
        "best_val_cos": best_val,
        "seconds": time.time() - t0,
        "history": history,
        "model": model,
    }


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, device: torch.device, bs: int = 2048) -> np.ndarray:
    model.eval()
    outs = []
    xt = torch.from_numpy(x.astype(np.float32))
    for i in range(0, len(xt), bs):
        pred = model(xt[i : i + bs].to(device))
        pred = F.normalize(pred, dim=-1)
        outs.append(pred.float().cpu().numpy())
    return np.concatenate(outs, axis=0).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-dir", type=str, required=True)
    ap.add_argument("--clip-train", type=str, required=True)
    ap.add_argument("--clip-test", type=str, required=True)
    ap.add_argument("--gallery", type=str, required=True, help="test ViT-H gallery for retrieval")
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--input-key", type=str, default="proj", choices=["proj", "raw"])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr-linear", type=float, default=1e-3)
    ap.add_argument("--lr-mlp", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--mlp-hidden", type=int, default=0, help="0 => 2*max(din,dout)")
    ap.add_argument("--mlp-dropout", type=float, default=0.0)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--skip-mlp", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    embed_dir = Path(args.embed_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    key = "proj" if args.input_key == "proj" else "raw"
    x_train = np.load(embed_dir / f"z_eeg_{key}_train.npy")
    x_test = np.load(embed_dir / f"z_eeg_{key}_test.npy")
    y_train = np.load(args.clip_train).astype(np.float32)
    y_test = np.load(args.clip_test).astype(np.float32)
    gallery = np.load(args.gallery).astype(np.float32)

    if x_train.shape[0] != y_train.shape[0]:
        raise RuntimeError(f"train n mismatch NB={x_train.shape[0]} CLIP={y_train.shape[0]}")
    if x_test.shape[0] != y_test.shape[0]:
        raise RuntimeError(f"test n mismatch NB={x_test.shape[0]} CLIP={y_test.shape[0]}")

    n_concepts = 1654
    n_img = 10
    if x_train.shape[0] != n_concepts * n_img:
        # fallback: treat each sample independently for val
        print(f"[WARN] unexpected train n={x_train.shape[0]}; using random sample val split")
        rng = np.random.default_rng(args.seed)
        idx = rng.permutation(len(x_train))
        n_val = max(1, int(round(len(x_train) * args.val_frac)))
        val_idx, train_idx = idx[:n_val], idx[n_val:]
        val_concepts = []
    else:
        train_idx, val_idx, val_concepts = concept_split(n_concepts, n_img, args.val_frac, args.seed)

    x_tr_t = torch.from_numpy(x_train[train_idx])
    y_tr_t = torch.from_numpy(y_train[train_idx])
    x_va_t = torch.from_numpy(x_train[val_idx])
    y_va_t = torch.from_numpy(y_train[val_idx])

    din, dout = int(x_train.shape[1]), int(y_train.shape[1])
    print(f"[INFO] input={key} din={din} dout={dout} train={len(train_idx)} val={len(val_idx)} device={device}")

    results = {
        "input_key": key,
        "din": din,
        "dout": dout,
        "n_train_fit": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_val_concepts": len(val_concepts),
        "oracle_test": retrieval_metrics(y_test, gallery),
        "adapters": {},
    }

    specs = [("linear", LinearAdapter(din, dout), args.lr_linear)]
    if not args.skip_mlp:
        hidden = args.mlp_hidden or (2 * max(din, dout))
        specs.append(
            ("mlp", MLPAdapter(din, dout, hidden=hidden, dropout=args.mlp_dropout), args.lr_mlp)
        )

    for name, model, lr in specs:
        pack = train_one(
            name,
            model,
            x_tr_t,
            y_tr_t,
            x_va_t,
            y_va_t,
            device,
            epochs=args.epochs,
            lr=lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            patience=args.patience,
        )
        m = pack.pop("model")
        pred_test = predict(m, x_test, device)
        pred_train = predict(m, x_train, device)
        np.save(out_dir / f"{name}_test_clip_1024.npy", pred_test)
        np.save(out_dir / f"{name}_train_clip_1024.npy", pred_train)
        ckpt = {
            "name": name,
            "input_key": key,
            "din": din,
            "dout": dout,
            "state_dict": m.state_dict(),
            "best_epoch": pack["best_epoch"],
            "best_val_cos": pack["best_val_cos"],
        }
        if name == "mlp":
            ckpt["hidden"] = args.mlp_hidden or (2 * max(din, dout))
            ckpt["dropout"] = args.mlp_dropout
        torch.save(ckpt, out_dir / f"{name}_adapter.pt")

        test_ret = retrieval_metrics(pred_test, gallery)
        # also cosine to GT ViT-H (paired)
        gt_cos = float(np.mean(np.sum(l2_np(pred_test) * l2_np(y_test), axis=1)))
        pack_out = {
            **{k: pack[k] for k in ("best_epoch", "best_val_cos", "seconds")},
            "test_gt_cos": gt_cos,
            "test_retrieval": test_ret,
            "history_tail": pack["history"][-5:],
        }
        results["adapters"][name] = pack_out
        print(f"[OK] {name}: test_gt_cos={gt_cos:.4f} top1={test_ret['top1']:.3f} top5={test_ret['top5']:.3f}")

    # raw NB projected retrieval in RN50 space is not comparable to ViT-H gallery;
    # still report identity baseline: zero adapter using mean-centered ridge closed form optional skip.

    (out_dir / "adapter_eval.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[OK] wrote {out_dir / 'adapter_eval.json'}")


if __name__ == "__main__":
    main()
