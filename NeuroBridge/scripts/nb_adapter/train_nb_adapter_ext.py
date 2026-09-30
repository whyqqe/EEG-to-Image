#!/usr/bin/env python3
"""Extended NeuroBridge→ViT-H adapters beyond Linear/MLP.

Architectures:
  ridge          — closed-form ridge regression (L2-normalized targets)
  linear_nobias  — bias-free linear
  mlp_deep       — 3-layer MLP
  mlp_res        — residual MLP bottleneck
  cfm            — conditional flow matching (OT path, velocity MLP)
  diffprior      — ATM-style DiffusionPriorUNet (cond=z_NB, target=z_H)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Reuse helpers from the base trainer
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from train_nb_adapter import (  # noqa: E402
    LinearAdapter,
    MLPAdapter,
    concept_split,
    cosine_mse_loss,
    eval_cos,
    l2_np,
    predict,
    retrieval_metrics,
)

BRAINIT_SRC = Path("/project/peilab/why/eeg-brainit/src")
sys.path.insert(0, str(BRAINIT_SRC))
from eeg_brainit.models.atm_diffusion_prior import (  # noqa: E402
    AtmDiffusionPriorPipe,
    DiffusionPriorUNet,
)


class LinearNoBias(nn.Module):
    def __init__(self, din: int, dout: int):
        super().__init__()
        self.proj = nn.Linear(din, dout, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class MLPDeep(nn.Module):
    def __init__(self, din: int, dout: int, hidden: int | None = None, dropout: float = 0.05):
        super().__init__()
        h = hidden or (2 * max(din, dout))
        self.net = nn.Sequential(
            nn.Linear(din, h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, dout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPResidual(nn.Module):
    def __init__(self, din: int, dout: int, hidden: int | None = None, dropout: float = 0.05):
        super().__init__()
        h = hidden or (2 * max(din, dout))
        self.in_proj = nn.Linear(din, dout)
        self.block = nn.Sequential(
            nn.Linear(dout, h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, dout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.in_proj(x)
        return y + self.block(y)


class CFMVelocity(nn.Module):
    """v_theta(x_t, t, cond) predicting velocity on OT path x_t=(1-t)x0+t*x1."""

    def __init__(self, embed_dim: int, cond_dim: int, hidden: int = 2048):
        super().__init__()
        self.t_embed = nn.Sequential(nn.Linear(1, 128), nn.SiLU(), nn.Linear(128, 128))
        self.net = nn.Sequential(
            nn.Linear(embed_dim + cond_dim + 128, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # t: (B,) in [0,1]
        te = self.t_embed(t.view(-1, 1))
        return self.net(torch.cat([x_t, cond, te], dim=-1))


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float = 1e-2) -> np.ndarray:
    """Solve min ||XW - Y||^2 + alpha||W||^2 ; return W (din, dout)."""
    x = x.astype(np.float64)
    y = l2_np(y.astype(np.float32)).astype(np.float64)
    xtx = x.T @ x
    xty = x.T @ y
    din = xtx.shape[0]
    reg = alpha * np.eye(din, dtype=np.float64)
    # scale alpha relative to mean diagonal for stability
    reg *= max(float(np.trace(xtx) / din), 1.0)
    w = np.linalg.solve(xtx + reg, xty)
    return w.astype(np.float32)


def train_regression(
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
    loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=batch_size, shuffle=True)
    best_val, best_epoch, best_state, bad = -1.0, -1, None, 0
    history = []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        loss_sum = n = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = cosine_mse_loss(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item()) * xb.shape[0]
            n += xb.shape[0]
        train_cos = eval_cos(model, x_tr, y_tr, device)
        val_cos = eval_cos(model, x_va, y_va, device)
        history.append({"epoch": ep, "loss": loss_sum / max(n, 1), "train_cos": train_cos, "val_cos": val_cos})
        print(f"[{name}] ep={ep:03d} loss={history[-1]['loss']:.4f} train_cos={train_cos:.4f} val_cos={val_cos:.4f}")
        if val_cos > best_val + 1e-5:
            best_val, best_epoch = val_cos, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"[{name}] early stop ep={ep} best={best_epoch} val={best_val:.4f}")
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


def train_cfm(
    model: CFMVelocity,
    cond_tr: torch.Tensor,
    y_tr: torch.Tensor,
    cond_va: torch.Tensor,
    y_va: torch.Tensor,
    device: torch.device,
    epochs: int,
    lr: float,
    batch_size: int,
    patience: int,
    sample_steps: int,
) -> dict:
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(cond_tr, y_tr), batch_size=batch_size, shuffle=True)
    best_val, best_epoch, best_state, bad = -1.0, -1, None, 0
    history = []
    t0 = time.time()

    def sample_cos(cond: torch.Tensor, y: torch.Tensor) -> float:
        pred = sample_cfm(model, cond, device, steps=sample_steps)
        pred = F.normalize(pred, dim=-1)
        tgt = F.normalize(y.to(device), dim=-1)
        return float((pred * tgt).sum(dim=-1).mean().item())

    for ep in range(1, epochs + 1):
        model.train()
        loss_sum = n = 0.0
        for cond, y1 in loader:
            cond, y1 = cond.to(device), y1.to(device)
            y1 = F.normalize(y1, dim=-1)
            x0 = torch.randn_like(y1)
            t = torch.rand(y1.shape[0], device=device)
            # OT path
            t_ = t.view(-1, 1)
            x_t = (1.0 - t_) * x0 + t_ * y1
            v_target = y1 - x0
            v_pred = model(x_t, t, cond)
            loss = F.mse_loss(v_pred, v_target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item()) * y1.shape[0]
            n += y1.shape[0]
        # cheap val: subsample
        with torch.no_grad():
            idx = torch.randperm(len(cond_va))[: min(2048, len(cond_va))]
            val_cos = sample_cos(cond_va[idx], y_va[idx])
            train_cos = sample_cos(cond_tr[idx[: min(2048, len(cond_tr))]], y_tr[idx[: min(2048, len(cond_tr))]])
        history.append({"epoch": ep, "loss": loss_sum / max(n, 1), "train_cos": train_cos, "val_cos": val_cos})
        print(f"[cfm] ep={ep:03d} loss={history[-1]['loss']:.4f} train_cos={train_cos:.4f} val_cos={val_cos:.4f}")
        if val_cos > best_val + 1e-5:
            best_val, best_epoch = val_cos, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"[cfm] early stop ep={ep} best={best_epoch} val={best_val:.4f}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "name": "cfm",
        "best_epoch": best_epoch,
        "best_val_cos": best_val,
        "seconds": time.time() - t0,
        "history": history,
        "model": model,
    }


@torch.no_grad()
def sample_cfm(model: CFMVelocity, cond: torch.Tensor, device: torch.device, steps: int = 20) -> torch.Tensor:
    model.eval()
    cond = cond.to(device)
    x = torch.randn(cond.shape[0], 1024, device=device)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((cond.shape[0],), i / steps, device=device)
        v = model(x, t, cond)
        x = x + dt * v
    return x


def train_diffprior(
    cond_tr: torch.Tensor,
    y_tr: torch.Tensor,
    cond_va: torch.Tensor,
    y_va: torch.Tensor,
    device: torch.device,
    epochs: int,
    lr: float,
    batch_size: int,
    patience: int,
    sample_steps: int,
) -> dict:
    prior = DiffusionPriorUNet(embed_dim=1024, cond_dim=int(cond_tr.shape[1]), dropout=0.1).to(device)
    pipe = AtmDiffusionPriorPipe(prior, device)
    opt = torch.optim.AdamW(prior.parameters(), lr=lr, weight_decay=1e-2)
    loader = DataLoader(TensorDataset(cond_tr, y_tr), batch_size=batch_size, shuffle=True)
    best_val, best_epoch, best_state, bad = -1.0, -1, None, 0
    history = []
    t0 = time.time()

    def val_cos() -> float:
        idx = torch.randperm(len(cond_va))[: min(512, len(cond_va))]
        with torch.no_grad():
            pred = pipe.generate(
                cond_va[idx].to(device),
                num_inference_steps=sample_steps,
                guidance_scale=5.0,
            )
            pred = F.normalize(pred, dim=-1)
            tgt = F.normalize(y_va[idx].to(device), dim=-1)
            return float((pred * tgt).sum(dim=-1).mean().item())

    for ep in range(1, epochs + 1):
        prior.train()
        loss_sum = n = 0.0
        for cond, y1 in loader:
            cond, y1 = cond.to(device), y1.to(device)
            y1 = F.normalize(y1, dim=-1)
            noise = torch.randn_like(y1)
            timesteps = torch.randint(
                0, pipe.scheduler.config.num_train_timesteps, (y1.shape[0],), device=device
            ).long()
            noisy = pipe.scheduler.add_noise(y1, noise, timesteps)
            pred = prior(noisy, timesteps.float(), cond)
            loss = F.mse_loss(pred, noise)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item()) * y1.shape[0]
            n += y1.shape[0]
        # validate every epoch but sampling is slow — every 2 epochs after ep 1
        if ep == 1 or ep % 2 == 0 or ep == epochs:
            vc = val_cos()
        else:
            vc = best_val if best_val > 0 else 0.0
        history.append({"epoch": ep, "loss": loss_sum / max(n, 1), "val_cos": vc})
        print(f"[diffprior] ep={ep:03d} loss={history[-1]['loss']:.4f} val_cos={vc:.4f}")
        if vc > best_val + 1e-5:
            best_val, best_epoch = vc, ep
            best_state = {k: v.detach().cpu().clone() for k, v in prior.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"[diffprior] early stop ep={ep} best={best_epoch} val={best_val:.4f}")
                break
    if best_state is not None:
        prior.load_state_dict(best_state)
    return {
        "name": "diffprior",
        "best_epoch": best_epoch,
        "best_val_cos": best_val,
        "seconds": time.time() - t0,
        "history": history,
        "pipe": pipe,
        "prior": prior,
    }


def pack_eval(name: str, pred_test: np.ndarray, y_test: np.ndarray, gallery: np.ndarray, train_meta: dict) -> dict:
    gt_cos = float(np.mean(np.sum(l2_np(pred_test) * l2_np(y_test), axis=1)))
    ret = retrieval_metrics(pred_test, gallery)
    return {
        "best_epoch": train_meta.get("best_epoch"),
        "best_val_cos": train_meta.get("best_val_cos"),
        "seconds": train_meta.get("seconds"),
        "test_gt_cos": gt_cos,
        "test_retrieval": ret,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-dir", type=str, required=True)
    ap.add_argument("--clip-train", type=str, required=True)
    ap.add_argument("--clip-test", type=str, required=True)
    ap.add_argument("--gallery", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--input-key", type=str, default="proj", choices=["proj", "raw"])
    ap.add_argument("--archs", type=str, default="ridge,linear_nobias,mlp_deep,mlp_res,cfm,diffprior")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--cfm-epochs", type=int, default=40)
    ap.add_argument("--diff-epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--cfm-steps", type=int, default=20)
    ap.add_argument("--diff-steps", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    embed_dir = Path(args.embed_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    key = args.input_key
    x_train = np.load(embed_dir / f"z_eeg_{key}_train.npy")
    x_test = np.load(embed_dir / f"z_eeg_{key}_test.npy")
    y_train = np.load(args.clip_train).astype(np.float32)
    y_test = np.load(args.clip_test).astype(np.float32)
    gallery = np.load(args.gallery).astype(np.float32)

    train_idx, val_idx, val_concepts = concept_split(1654, 10, 0.1, args.seed)
    x_tr_t = torch.from_numpy(x_train[train_idx])
    y_tr_t = torch.from_numpy(y_train[train_idx])
    x_va_t = torch.from_numpy(x_train[val_idx])
    y_va_t = torch.from_numpy(y_train[val_idx])
    din, dout = int(x_train.shape[1]), int(y_train.shape[1])
    archs = [a.strip() for a in args.archs.split(",") if a.strip()]
    print(f"[INFO] din={din} dout={dout} archs={archs} device={device}")

    results = {
        "input_key": key,
        "din": din,
        "dout": dout,
        "n_val_concepts": len(val_concepts),
        "oracle_test": retrieval_metrics(y_test, gallery),
        "adapters": {},
    }

    for name in archs:
        t_arch = time.time()
        if name == "ridge":
            # fit on train split only
            w = fit_ridge(x_train[train_idx], y_train[train_idx], alpha=1e-2)
            pred_test = l2_np(x_test @ w)
            pred_train = l2_np(x_train @ w)
            # val cos
            pred_va = l2_np(x_train[val_idx] @ w)
            val_cos = float(np.mean(np.sum(pred_va * l2_np(y_train[val_idx]), axis=1)))
            meta = {"best_epoch": 0, "best_val_cos": val_cos, "seconds": time.time() - t_arch}
            np.save(out_dir / f"{name}_test_clip_1024.npy", pred_test.astype(np.float32))
            np.save(out_dir / f"{name}_train_clip_1024.npy", pred_train.astype(np.float32))
            np.save(out_dir / f"{name}_W.npy", w)
            results["adapters"][name] = pack_eval(name, pred_test, y_test, gallery, meta)
            print(f"[OK] {name}: gt_cos={results['adapters'][name]['test_gt_cos']:.4f} top1={results['adapters'][name]['test_retrieval']['top1']:.3f}")
            continue

        if name == "linear_nobias":
            model = LinearNoBias(din, dout)
            pack = train_regression(name, model, x_tr_t, y_tr_t, x_va_t, y_va_t, device, args.epochs, 1e-3, 1e-4, args.batch_size, args.patience)
        elif name == "mlp_deep":
            model = MLPDeep(din, dout)
            pack = train_regression(name, model, x_tr_t, y_tr_t, x_va_t, y_va_t, device, args.epochs, 3e-4, 1e-4, args.batch_size, args.patience)
        elif name == "mlp_res":
            model = MLPResidual(din, dout)
            pack = train_regression(name, model, x_tr_t, y_tr_t, x_va_t, y_va_t, device, args.epochs, 3e-4, 1e-4, args.batch_size, args.patience)
        elif name == "cfm":
            model = CFMVelocity(embed_dim=dout, cond_dim=din, hidden=2048)
            pack = train_cfm(model, x_tr_t, y_tr_t, x_va_t, y_va_t, device, args.cfm_epochs, 3e-4, args.batch_size, args.patience, args.cfm_steps)
            m = pack.pop("model")
            pred_test = l2_np(sample_cfm(m, torch.from_numpy(x_test), device, steps=args.cfm_steps).cpu().numpy())
            pred_train = l2_np(sample_cfm(m, torch.from_numpy(x_train), device, steps=args.cfm_steps).cpu().numpy())
            torch.save({"name": name, "state_dict": m.state_dict(), "din": din, "dout": dout, "steps": args.cfm_steps}, out_dir / f"{name}_adapter.pt")
            np.save(out_dir / f"{name}_test_clip_1024.npy", pred_test.astype(np.float32))
            np.save(out_dir / f"{name}_train_clip_1024.npy", pred_train.astype(np.float32))
            results["adapters"][name] = pack_eval(name, pred_test, y_test, gallery, pack)
            print(f"[OK] {name}: gt_cos={results['adapters'][name]['test_gt_cos']:.4f} top1={results['adapters'][name]['test_retrieval']['top1']:.3f}")
            continue
        elif name == "diffprior":
            pack = train_diffprior(x_tr_t, y_tr_t, x_va_t, y_va_t, device, args.diff_epochs, 1e-3, min(args.batch_size, 512), max(args.patience, 6), args.diff_steps)
            pipe = pack.pop("pipe")
            prior = pack.pop("prior")
            with torch.no_grad():
                outs = []
                xt = torch.from_numpy(x_test.astype(np.float32))
                for i in range(0, len(xt), 64):
                    outs.append(
                        pipe.generate(xt[i : i + 64].to(device), num_inference_steps=args.diff_steps, guidance_scale=5.0)
                        .float()
                        .cpu()
                        .numpy()
                    )
                pred_test = l2_np(np.concatenate(outs, axis=0))
            torch.save({"name": name, "state_dict": prior.state_dict(), "din": din, "dout": dout, "steps": args.diff_steps}, out_dir / f"{name}_adapter.pt")
            np.save(out_dir / f"{name}_test_clip_1024.npy", pred_test.astype(np.float32))
            results["adapters"][name] = pack_eval(name, pred_test, y_test, gallery, pack)
            print(f"[OK] {name}: gt_cos={results['adapters'][name]['test_gt_cos']:.4f} top1={results['adapters'][name]['test_retrieval']['top1']:.3f}")
            continue
        else:
            raise ValueError(f"unknown arch {name}")

        m = pack.pop("model")
        pred_test = predict(m, x_test, device)
        pred_train = predict(m, x_train, device)
        torch.save({"name": name, "state_dict": m.state_dict(), "din": din, "dout": dout}, out_dir / f"{name}_adapter.pt")
        np.save(out_dir / f"{name}_test_clip_1024.npy", pred_test)
        np.save(out_dir / f"{name}_train_clip_1024.npy", pred_train)
        results["adapters"][name] = pack_eval(name, pred_test, y_test, gallery, pack)
        print(f"[OK] {name}: gt_cos={results['adapters'][name]['test_gt_cos']:.4f} top1={results['adapters'][name]['test_retrieval']['top1']:.3f}")

    (out_dir / "adapter_eval_ext.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[OK] wrote {out_dir / 'adapter_eval_ext.json'}")


if __name__ == "__main__":
    main()
