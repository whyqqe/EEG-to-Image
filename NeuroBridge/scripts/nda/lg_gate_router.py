#!/usr/bin/env python3
"""LG-Gate router: EEG->structure-confidence u via OUT-OF-FOLD depth quality.

Why OOF labels: the deployed pure-intra DepthHead was trained on ALL train
samples, so its train predictions would be saturated (overfit) and give a
degenerate router label. We instead do K-fold cross-validation: each fold we
train a fresh DepthHead on the other folds (sub-08 only) and predict the held
out fold. The OOF pearson(pred_depth, GT_depth) per sample is an unbiased
"how decodable is this sample's structure from EEG" label.

  u_label_i = 0.5 + 0.5 * pearson( DepthHead_fold(z_i), GT_depth_i )

Router: MLP(z_decode_vith_l2) -> sigmoid u, trained on ALL OOF labels.
Test: u_hat_test; diagnostic = spearman(u_hat_test, u_true_test) where
u_true_test uses a final DepthHead trained on all train (deployed-quality head).

GT depth caches are image-side (Depth-Anything) — allowed for intra.
All EEG weights trained on sub-08 ONLY.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

NB_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NB_ROOT / "scripts" / "nda"))

from train_eeg_depth_head import DepthHead, grad_loss  # noqa: E402


def l2(x: np.ndarray) -> np.ndarray:
    return (x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-8)).astype(np.float32)


def pearson_flat(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.reshape(len(a), -1)
    b = b.reshape(len(b), -1)
    ac = a - a.mean(axis=1, keepdims=True)
    bc = b - b.mean(axis=1, keepdims=True)
    num = (ac * bc).sum(axis=1)
    den = np.sqrt((ac * ac).sum(axis=1) * (bc * bc).sum(axis=1)).clip(1e-8)
    return (num / den).astype(np.float32)


class GateRouter(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(z)).squeeze(-1)


@torch.no_grad()
def head_predict(head: nn.Module, z: np.ndarray, device: torch.device, bs: int = 2048) -> np.ndarray:
    head.eval()
    out = []
    for s in range(0, len(z), bs):
        zb = torch.from_numpy(z[s : s + bs]).to(device)
        out.append(head(zb).cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def train_depth_head(
    z_tr: np.ndarray, d_tr: np.ndarray, z_va: np.ndarray, d_va: np.ndarray,
    device: torch.device, epochs: int, bs: int, lr: float, seed: int,
) -> nn.Module:
    torch.manual_seed(seed)
    head = DepthHead(in_dim=z_tr.shape[1], out_res=d_tr.shape[-1]).to(device)
    opt = optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    ds = TensorDataset(torch.from_numpy(z_tr), torch.from_numpy(d_tr))
    loader = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True)
    best_p, best_sd = -1.0, None
    for ep in range(1, epochs + 1):
        head.train()
        for zb, db in loader:
            zb, db = zb.to(device), db.to(device)
            opt.zero_grad()
            pred = head(zb)
            loss = nn.functional.l1_loss(pred, db) + 0.5 * grad_loss(pred, db)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
        pv = head_predict(head, z_va, device)
        p = float(np.mean(pearson_flat(pv, d_va)))
        if p > best_p:
            best_p = p
            best_sd = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_sd)
    return head


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--z-train-npy", type=str, required=True)
    ap.add_argument("--z-test-npy", type=str, required=True)
    ap.add_argument("--depth-gt-train-npy", type=str, required=True)
    ap.add_argument("--depth-gt-test-npy", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--num-folds", type=int, default=3)
    ap.add_argument("--fold-epochs", type=int, default=30)
    ap.add_argument("--final-epochs", type=int, default=25)
    ap.add_argument("--router-epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    z_tr = l2(np.load(args.z_train_npy).astype(np.float32))
    z_te = l2(np.load(args.z_test_npy).astype(np.float32))
    d_tr = np.load(args.depth_gt_train_npy).astype(np.float32)
    d_te = np.load(args.depth_gt_test_npy).astype(np.float32)
    assert len(z_tr) == len(d_tr), f"train {len(z_tr)} vs {len(d_tr)}"
    n = len(z_tr)

    # ---- K-fold OOF structure-quality labels (no leakage: head never sees its fold) ----
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, args.num_folds)
    u_oof = np.zeros(n, dtype=np.float32)
    for fi, val_idx in enumerate(folds):
        tr_idx = np.concatenate([folds[k] for k in range(args.num_folds) if k != fi])
        print(f"[fold {fi+1}/{args.num_folds}] train={len(tr_idx)} val={len(val_idx)}")
        h = train_depth_head(
            z_tr[tr_idx], d_tr[tr_idx], z_tr[val_idx], d_tr[val_idx],
            device, args.fold_epochs, args.batch_size, args.lr, args.seed + fi,
        )
        pv = head_predict(h, z_tr[val_idx], device)
        u_oof[val_idx] = 0.5 + 0.5 * pearson_flat(pv, d_tr[val_idx])
    u_oof = np.clip(u_oof, 0.0, 1.0).astype(np.float32)
    np.save(out / "u_label_train_oof.npy", u_oof)
    print(f"[INFO] OOF u: mean={u_oof.mean():.3f} std={u_oof.std():.3f} min={u_oof.min():.3f} max={u_oof.max():.3f}")

    # ---- final deployed-quality head on all train -> test u_true (diagnostic only) ----
    h_final = train_depth_head(
        z_tr, d_tr, z_te, d_te, device, args.final_epochs, args.batch_size, args.lr, args.seed + 99,
    )
    pred_te = head_predict(h_final, z_te, device)
    u_te = np.clip(0.5 + 0.5 * pearson_flat(pred_te, d_te), 0.0, 1.0).astype(np.float32)
    np.save(out / "u_true_test.npy", u_te)
    np.save(out / "pred_depth_test_64.npy", pred_te.astype(np.float32))
    print(f"[INFO] u_true_test mean={u_te.mean():.3f} std={u_te.std():.3f}")

    # ---- Router: predict u from EEG ----
    zz = torch.from_numpy(z_tr)
    yy = torch.from_numpy(u_oof)
    # early-stop split inside train
    rng2 = np.random.RandomState(args.seed + 1)
    idx2 = rng2.permutation(n)
    nval = max(1, int(n * 0.05))
    v_idx, t_idx = idx2[:nval], idx2[nval:]
    loader = DataLoader(
        TensorDataset(zz[t_idx], yy[t_idx]), batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    xv = zz[v_idx].to(device)
    yv = yy[v_idx].to(device)
    router = GateRouter(in_dim=z_tr.shape[1]).to(device)
    opt = optim.AdamW(router.parameters(), lr=args.lr, weight_decay=1e-4)
    best_vc, best_sd = -1.0, None
    hist = []
    for ep in range(1, args.router_epochs + 1):
        router.train()
        acc = 0.0
        for zb, ub in loader:
            zb, ub = zb.to(device), ub.to(device)
            opt.zero_grad()
            loss = nn.functional.mse_loss(router(zb), ub)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)
            opt.step()
            acc += float(loss.item())
        router.eval()
        with torch.no_grad():
            vp = router(xv).cpu().numpy()
        vc = float(np.corrcoef(vp, yv.cpu().numpy())[0, 1]) if vp.std() > 1e-8 and yv.std() > 1e-8 else 0.0
        hist.append({"epoch": ep, "loss": acc / max(len(loader), 1), "val_corr": vc})
        if vc > best_vc:
            best_vc = vc
            best_sd = {k: v.detach().cpu().clone() for k, v in router.state_dict().items()}
        if ep % 10 == 0 or ep == 1:
            print(f"[router ep {ep}] loss={acc/max(len(loader),1):.5f} val_corr={vc:.3f}")

    router.load_state_dict(best_sd)
    router.eval()
    with torch.no_grad():
        u_hat_te = router(torch.from_numpy(z_te).to(device)).cpu().numpy().astype(np.float32)
    np.save(out / "u_hat_test.npy", u_hat_te)

    from scipy.stats import spearmanr

    sp = float(spearmanr(u_hat_te, u_te).statistic) if u_te.std() > 1e-8 and u_hat_te.std() > 1e-8 else 0.0
    report = {
        "pipeline": "lg_gate_router",
        "supervision": "K-fold OOF 0.5+0.5*pearson(DepthHead(z),GT_depth); heads trained sub-08 only",
        "router": "MLP z_decode_vith_l2 -> sigmoid u",
        "n_train": n,
        "n_test": len(z_te),
        "num_folds": args.num_folds,
        "fold_epochs": args.fold_epochs,
        "final_epochs": args.final_epochs,
        "router_val_corr_best": float(best_vc),
        "test_spearman_uhat_vs_utrue": sp,
        "u_oof_mean": float(u_oof.mean()),
        "u_true_test_mean": float(u_te.mean()),
        "u_hat_test_mean": float(u_hat_te.mean()),
    }
    torch.save({"state_dict": best_sd, "in_dim": z_tr.shape[1], **report}, out / "router_best.pth")
    (out / "router_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
