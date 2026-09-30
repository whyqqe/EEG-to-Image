#!/usr/bin/env python3
"""方案 A: structure-specialised EEG encoding branch.

WHY (measured, sub-08 intra)
----------------------------
The shipped structural path is a single 1024-d frozen feature (z_decode_vith)
fed to two INDEPENDENT L1 heads. Measured on the shipped VAE head:

    per-sample pearson         +0.3317
    per-sample std_ratio        0.390      (<1 => collapse)
    cross-sample spread ratio   0.428      <-- KEY: different EEGs differ by only 43%

A structurally identical MLP trained from scratch reproduced 0.418 spread --
so the bottleneck is NOT head capacity. Information-source decomposition:

    raw EEG (63ch x 50t)   -> VAE : pearson +0.2914  spread(std_ratio) 0.293
    z_decode_vith (1024d)  -> VAE : pearson +0.2690  spread(std_ratio) 0.381
    GT CLIP (semantic ORACLE) -> VAE : pearson +0.3167  spread 0.541  <-- ceiling
    z_decode_vith -> GT CLIP (semantics) : pearson +0.7996

=> fusion-side tricks (gating / CFM / transport / ControlNet tuning) are bounded
   by the information content of the conditioning signal. The only escapes are
   (a) an encoder that carries MORE structure, and (b) retrieving real images.

This script tests (a): a branch that learns structural features end-to-end,
sharing a scene code between the VAE decoder and the depth decoder, with an
explicit cross-decoder consistency term.

DECISION GATE
-------------
    spread > 0.55  -> the architectural story is viable, go for SSIM
    spread ~= 0.43 -> hard information ceiling -> pivot to retrieval (方案 B)

Leak-free: checkpoints are chosen on held-in val_b concepts (never the test set).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import leakfree as LF  # noqa: E402
from train_eeg_vae_head import VAEHead  # noqa: E402
from train_eeg_depth_head import DepthHead, grad_loss  # noqa: E402


# ---------------------------------------------------------------- metrics
def structure_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Everything we need to judge 'is the structure path collapsed?'."""
    P = pred - pred.mean(axis=1, keepdims=True)
    G = gt - gt.mean(axis=1, keepdims=True)
    den = np.linalg.norm(P, axis=1) * np.linalg.norm(G, axis=1)
    ok = den > 1e-8
    pearson = float(np.mean((P * G).sum(1)[ok] / den[ok]))
    std_ratio = float(np.mean(pred.std(axis=1) / gt.std(axis=1).clip(1e-8)))
    spread = float(pred.std(axis=0).mean() / gt.std(axis=0).mean())
    rng = np.random.default_rng(0)
    idx = rng.choice(len(pred), min(60, len(pred)), replace=False)
    dp = np.linalg.norm(pred[idx][:, None] - pred[idx][None], axis=2).mean()
    dg = np.linalg.norm(gt[idx][:, None] - gt[idx][None], axis=2).mean()
    return {
        "pearson": round(pearson, 4),
        "std_ratio": round(std_ratio, 4),
        "spread": round(spread, 4),
        "pair_dist_ratio": round(float(dp / max(dg, 1e-8)), 4),
    }


# ---------------------------------------------------------------- model
class StructEncoder(nn.Module):
    """EEG (+ optional frozen feature) -> scene code -> {VAE latent, depth}.

    The SHARED scene code h is the architectural consistency mechanism: both
    decoders must explain the same scene. An explicit cross-decoder term
    (VAE latent -> depth probe fitted on GT) is added on top.
    """

    def __init__(self, in_dim: int, code: int = 512, hidden: int = 2048,
                 spatial: int = 64, ch: int = 4, depth_hidden: int = 1024):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, code), nn.GELU(),
        )
        self.scene = nn.Sequential(nn.Linear(code, code), nn.GELU())
        self.vae_head = VAEHead(in_dim=code, hidden=hidden // 2, spatial=spatial, ch=ch)
        self.depth_head = DepthHead(in_dim=code, out_res=spatial, hidden=depth_hidden)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.scene(self.encoder(x))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encode(x)
        return self.vae_head(h), self.depth_head(h)


# ---------------------------------------------------------------- data
def pooled_eeg(path: Path, k: int) -> np.ndarray:
    """(concepts, reps, trials, ch, time) -> (N, ch*(time//k)) trial-averaged.

    Mirrors the standard THINGS-EEG2 preprocessing: average repeats, then
    temporal pooling on the remaining time axis.
    """
    arr = np.load(path).astype(np.float32)
    if arr.ndim == 5:
        arr = arr.mean(axis=2)                       # average trials
        arr = arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2], arr.shape[3])
    elif arr.ndim == 4:
        arr = arr.reshape(arr.shape[0] * arr.shape[1], arr.shape[2], arr.shape[3])
    N, C, T = arr.shape
    k = max(1, min(k, T))
    n = T // k
    arr = arr[:, :, : n * k].reshape(N, C, n, k).mean(axis=3)
    return arr.reshape(N, C * n)


def norm_rows(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-8)


def fit_consistency_probe(vae_fit: np.ndarray, depth_fit: np.ndarray,
                          ch: int = 4, res: int = 64, lam: float = 1e-2):
    """Ridge  pooled(VAE latent) -> depth, fitted on GT pairs (fit split only).

    Gives a cheap 'does my VAE latent describe the same scene as my depth map'
    test, without running DepthAnything inside the training loop.
    """
    b = res // 8
    X = vae_fit.reshape(len(vae_fit), ch, res, res).reshape(len(vae_fit), ch, b, 8, b, 8)
    X = X.mean(axis=(3, 5)).reshape(len(vae_fit), ch * b * b).astype(np.float32)
    Y = depth_fit.reshape(len(depth_fit), -1).astype(np.float32)
    mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True).clip(1e-6)
    Xn = (X - mu) / sd
    A = Xn.T @ Xn + lam * np.eye(Xn.shape[1], dtype=np.float32)
    W = np.linalg.solve(A, Xn.T @ Y)
    return {"mu": mu, "sd": sd, "W": W, "ch": ch, "b": b, "res": res}


def apply_probe(probe: dict, vae_pred: torch.Tensor, depth_like: torch.Tensor) -> torch.Tensor:
    """(B,ch,res,res) VAE latent -> predicted depth (B,res,res) via the GT probe."""
    B, ch, res, _ = vae_pred.shape
    b = probe["b"]
    x = vae_pred.reshape(B, ch, b, 8, b, 8).mean(dim=(3, 5)).reshape(B, ch * b * b)
    mu = torch.as_tensor(probe["mu"], device=x.device, dtype=x.dtype)
    sd = torch.as_tensor(probe["sd"], device=x.device, dtype=x.dtype)
    W = torch.as_tensor(probe["W"], device=x.device, dtype=x.dtype)
    return (((x - mu) / sd) @ W).reshape(-1, res, res)


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eeg-train-npy", required=True, help="raw preprocessed EEG train (.npy)")
    ap.add_argument("--eeg-test-npy", required=True)
    ap.add_argument("--feat-train-npy", default="", help="frozen z_decode_vith train (.npy)")
    ap.add_argument("--feat-test-npy", default="")
    ap.add_argument("--vae-train-npy", required=True)
    ap.add_argument("--vae-test-npy", required=True)
    ap.add_argument("--depth-train-npy", required=True)
    ap.add_argument("--depth-test-npy", required=True)
    ap.add_argument("--val-split-json", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--variant", default="multi_full",
                    choices=["multi_full", "vae_only", "frozen_only"])
    ap.add_argument("--time-pool", type=int, default=5)
    ap.add_argument("--code-dim", type=int, default=512)
    ap.add_argument("--num-epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lambda-depth", type=float, default=1.0)
    ap.add_argument("--lambda-grad", type=float, default=0.5)
    ap.add_argument("--lambda-cons", type=float, default=0.5)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ---------------- inputs
    use_eeg = args.variant != "frozen_only"
    parts_tr, parts_te, desc = [], [], []
    if use_eeg:
        e_tr = pooled_eeg(Path(args.eeg_train_npy), args.time_pool)
        e_te = pooled_eeg(Path(args.eeg_test_npy), args.time_pool)
        parts_tr.append(e_tr); parts_te.append(e_te)
        desc.append(f"raw EEG {e_tr.shape[1]}")
    if args.feat_train_npy:
        f_tr = np.load(args.feat_train_npy).astype(np.float32)
        f_te = np.load(args.feat_test_npy).astype(np.float32)
        parts_tr.append(f_tr); parts_te.append(f_te)
        desc.append(f"frozen feat {f_tr.shape[1]}")
    X_tr = np.hstack(parts_tr).astype(np.float32)
    X_te = np.hstack(parts_te).astype(np.float32)
    mu, sd = X_tr.mean(0, keepdims=True), X_tr.std(0, keepdims=True).clip(1e-6)
    X_tr = (X_tr - mu) / sd
    X_te = (X_te - mu) / sd
    print(f"[DATA] variant={args.variant}  input = {' + '.join(desc)} -> {X_tr.shape[1]}d")

    V_tr = np.load(args.vae_train_npy).astype(np.float32)
    V_te = np.load(args.vae_test_npy).astype(np.float32)
    D_tr = np.load(args.depth_train_npy).astype(np.float32)
    D_te = np.load(args.depth_test_npy).astype(np.float32)
    if D_tr.ndim == 2:
        r = int(round(D_tr.shape[1] ** 0.5))
        D_tr = D_tr.reshape(len(D_tr), r, r)
        D_te = D_te.reshape(len(D_te), r, r)
    assert len(X_tr) == len(V_tr) == len(D_tr), (len(X_tr), len(V_tr), len(D_tr))
    assert len(X_te) == len(V_te) == len(D_te), (len(X_te), len(V_te), len(D_te))
    res = V_tr.shape[-1]
    ch = V_tr.shape[1]
    print(f"[DATA] VAE target {V_tr.shape}  depth target {D_tr.shape}")

    # ---------------- leak-free split
    sp = LF.load(args.val_split_json)
    fi = LF.rows_for(sp, "fit", len(X_tr))
    vi = LF.rows_for(sp, "val_b", len(X_tr))
    assert len(set(fi.tolist()) & set(vi.tolist())) == 0
    print(f"[SPLIT] fit {len(fi)} rows | val_b {len(vi)} rows (held-in concepts; test never used for selection)")

    # target scaler from FIT only
    t_mu = V_tr[fi].mean(axis=(0, 2, 3), keepdims=True).astype(np.float32)
    t_sd = V_tr[fi].std(axis=(0, 2, 3), keepdims=True).astype(np.float32).clip(1e-3)
    Vn = (V_tr - t_mu) / t_sd
    Vn_te = (V_te - t_mu) / t_sd

    Xf = torch.from_numpy(X_tr[fi]); Vf = torch.from_numpy(Vn[fi]).view(len(fi), ch, res, res)
    Df = torch.from_numpy(D_tr[fi])
    Xv = torch.from_numpy(X_tr[vi]); Vv = torch.from_numpy(Vn[vi]).view(len(vi), ch, res, res)
    Dv = torch.from_numpy(D_tr[vi])

    loader = DataLoader(TensorDataset(Xf, Vf, Df), batch_size=args.batch_size,
                        shuffle=True, drop_last=True, num_workers=0)

    # ---------------- model
    model = StructEncoder(in_dim=X_tr.shape[1], code=args.code_dim,
                          spatial=res, ch=ch).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] StructEncoder params={n_par/1e6:.2f}M  code={args.code_dim}")
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.num_epochs)

    multi = args.variant != "vae_only"
    probe = None
    if multi:
        probe = fit_consistency_probe(V_tr[fi], D_tr[fi], ch=ch, res=res)
        print("[CONS] GT probe  pooled(VAE latent) -> depth  fitted on fit-split pairs")

    t_mu_t = torch.from_numpy(t_mu).to(device)
    t_sd_t = torch.from_numpy(t_sd).to(device)

    history, best = [], {"val_mae": 1e9, "epoch": 0}
    bad = 0
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        acc = {"vae": 0.0, "depth": 0.0, "cons": 0.0}
        nb = 0
        for xb, vb, db in tqdm(loader, desc=f"ab-{args.variant}-{epoch}", leave=False):
            xb, vb, db = xb.to(device), vb.to(device), db.to(device)
            opt.zero_grad()
            pv, pd = model(xb)
            loss_v = F.l1_loss(pv, vb)
            loss = loss_v
            lc = torch.zeros((), device=device)
            if multi:
                loss_d = F.l1_loss(pd, db) + args.lambda_grad * grad_loss(pd, db)
                lc = F.l1_loss(apply_probe(probe, pv, pd), pd)
                loss = loss_v + args.lambda_depth * loss_d + args.lambda_cons * lc
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            acc["vae"] += float(loss_v); acc["depth"] += float(loss_d) if multi else 0.0
            acc["cons"] += float(lc); nb += 1
        sched.step()
        if nb == 0:
            raise RuntimeError("all batches non-finite")
        for k in acc:
            acc[k] /= nb

        # ------- selection on held-in val_b (leak-free)
        model.eval()
        with torch.no_grad():
            pv, _ = model(Xv.to(device))
            val_mae = float(F.l1_loss(pv, Vv.to(device)))
        row = {"epoch": epoch, "loss_vae": round(acc["vae"], 4),
               "loss_depth": round(acc["depth"], 4), "loss_cons": round(acc["cons"], 4),
               "val_mae": round(val_mae, 4), "selected_on": "val_b"}
        history.append(row)
        if val_mae < best["val_mae"] - 1e-5:
            best = {"val_mae": val_mae, "epoch": epoch}
            bad = 0
            torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                        "in_dim": X_tr.shape[1], "code": args.code_dim,
                        "spatial": res, "ch": ch, "variant": args.variant,
                        "target_mean": t_mu.squeeze(), "target_std": t_sd.squeeze(),
                        "input_mean": mu.squeeze(), "input_std": sd.squeeze(),
                        "val_mae": val_mae}, out / "checkpoint_struct_best.pth")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[EARLY] no val improvement for {args.patience} epochs, stop at {epoch}")
                break
        if epoch % 10 == 0 or epoch == 1:
            print(f"[ep {epoch:3d}] vae={acc['vae']:.4f} depth={acc['depth']:.4f} "
                  f"cons={acc['cons']:.4f} val_mae={val_mae:.4f} best_ep={best['epoch']}")

    if not (out / "checkpoint_struct_best.pth").is_file():
        raise RuntimeError("no checkpoint saved")

    # ---------------- report on TEST (report only, never used to select)
    ck = torch.load(out / "checkpoint_struct_best.pth", map_location=device, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    with torch.no_grad():
        pv, pd = model(torch.from_numpy(X_te).to(device))
        pv = (pv * t_sd_t + t_mu_t).cpu().numpy()
        pd = pd.cpu().numpy()
    flat = lambda a: a.reshape(len(a), -1)
    m_new = structure_metrics(flat(pv), flat(V_te))

    # regression baseline from the SAME feature, single L1 head (shipped recipe)
    print("\n[REF] ridge z_decode_vith -> VAE latent (linear reference on test)")
    if args.feat_train_npy:
        Ftr = np.load(args.feat_train_npy).astype(np.float32)
        Fte = np.load(args.feat_test_npy).astype(np.float32)
        a = Ftr[fi]; b = Fte
        am, asd = a.mean(0, keepdims=True), a.std(0, keepdims=True).clip(1e-6)
        an = (a - am) / asd; bn = (b - am) / asd
        Y = flat(V_tr[fi])
        W = np.linalg.solve(an.T @ an + 1e-2 * np.eye(an.shape[1], dtype=np.float32), an.T @ Y)
        m_ridge = structure_metrics(bn @ W, flat(V_te))
    else:
        m_ridge = None

    with torch.no_grad():
        z_struct_te = model.encode(torch.from_numpy(X_te).to(device)).cpu().numpy()

    np.save(out / "pred_vae_test.npy", pv.astype(np.float16))
    np.save(out / "pred_depth_test.npy", pd.astype(np.float32))
    np.save(out / "z_struct_test.npy", z_struct_te.astype(np.float32))
    pandas.DataFrame(history).to_csv(out / "struct_encoder_history.csv", index=False)

    report = {
        "variant": args.variant,
        "input": desc,
        "input_dim": int(X_tr.shape[1]),
        "params_m": round(n_par / 1e6, 3),
        "best_epoch": best["epoch"],
        "best_val_mae": round(best["val_mae"], 4),
        "selected_on": "val_b (held-in concepts)",
        "test_metrics_pred_vae": m_new,
        "reference_ridge_frozen_feat": m_ridge,
        "shipped_vae_head_reference": {"pearson": 0.3317, "std_ratio": 0.390,
                                       "spread": 0.428, "pair_dist_ratio": 0.409},
        "gate": {"spread_target": 0.55,
                 "verdict": ("PASS - architecture viable" if m_new["spread"] > 0.55
                             else "FAIL - information ceiling confirmed, pivot to retrieval")},
        "n_fit": int(len(fi)), "n_val": int(len(vi)), "n_test": int(len(X_te)),
    }
    (out / "struct_encoder_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n" + json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
