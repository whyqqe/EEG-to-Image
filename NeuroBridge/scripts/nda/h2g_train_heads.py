#!/usr/bin/env python3
"""HCMA-2G: dual tower with a GRANULARITY axis orthogonal to the modality axis.

THEORY (why this shape)
-----------------------
Current HCMA has two DOWNSTREAM USES (semantics->IP-Adapter, structure->latent
init/ControlNet) but only ONE GRANULARITY: every alignment target is a global
pooled vector. The chain rule says the second term is unsupervised, not zero:

    I(e; y) = I(e; y_glob) + I(e; y_loc | y_glob)

Three consequences this module is built to fix, each with a nameable failure in
the shipped model:

(1) COLLAPSE IS THE L2/L1 OPTIMUM, not a training bug.
    A full-resolution regression target has argmin = E[y|e], whose variance is
    strictly below Var[y]. The shipped VAE head reaches std_ratio 0.390 while the
    achievable level is ~0.514 (full band) / ~0.773 (low band). Fix: heteroscedastic
    head (predict mean AND log-variance) so the irreducible variance is modelled
    rather than averaged away, plus MI-based band weighting so capacity is not
    spent on a band the signal cannot carry.

(2) TWO INDEPENDENT STRUCTURE HEADS ARE UNCONSTRAINED.
    Shipped: VAE head 12 epochs vs depth head 5 epochs, mutually unconstrained, and
    measured cn_scale effects are small. Fix: coarse-to-fine RESIDUAL factorisation
    with an explicit consistency term, instead of two parallel heads.

(3) THE ALIGNMENT OBJECTIVE IS BLIND TO HUBNESS.
    cosine/InfoNCE is invariant to orthogonal transforms, and hubness is a property
    of the covariance geometry (measured: raw margin 0.1329 > whiten margin 0.0970
    yet raw Top-1 is 7.5pp worse; hub_skew 3.00 -> 0.63 under whitening). So the
    objective supplies NO gradient pressure on the geometry that governs failure.
    Fix: VICReg variance + covariance terms (explicitly non-invariant), applied PER
    GRANULARITY.

Band allocation is MEASURED, not tuned: the per-band weight is the fit-split R^2 of
z -> band, so a band the signal cannot carry automatically receives ~zero weight
and the residual head models it as uncertainty (ABSTAIN) instead of averaging to it.
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
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import leakfree as LF  # noqa: E402


# ------------------------------------------------------------------ helpers
def radial_full(H: int, W: int) -> np.ndarray:
    """Full-spectrum radius grid, shape (H, W) -- for numpy fft2-based analysis."""
    fy = np.fft.fftfreq(H)[:, None]
    fx = np.fft.fftfreq(W)[None, :]
    return np.sqrt(fy**2 + fx**2) / 0.5


def radial_half(H: int, W: int) -> np.ndarray:
    """Half-spectrum radius grid, shape (H, W//2+1) -- MUST match torch.fft.rfft2.

    rfft2 returns the non-redundant half spectrum, so a full (H, W) mask cannot be
    broadcast against it. This grid is the one the denoising-time anchoring uses.
    """
    fy = np.fft.fftfreq(H)[:, None]
    fx = np.fft.rfftfreq(W)[None, :]
    return np.sqrt(fy**2 + fx**2) / 0.5


def band_split(x: torch.Tensor, r: torch.Tensor, cut: float):
    """(B,C,H,W) -> (low, high) with r < cut as the low band.

    `r` must be a HALF-spectrum radius grid (H, W//2+1) on x's device.
    """
    Fx = torch.fft.rfft2(x.float())
    m = (r < cut).float()[None, None]
    lo = torch.fft.irfft2(Fx * m, s=x.shape[-2:])
    hi = torch.fft.irfft2(Fx * (1.0 - m), s=x.shape[-2:])
    return lo, hi


def band_energy_ratio(resid: np.ndarray, tot: np.ndarray, r: np.ndarray,
                      lo: float, hi: float) -> float:
    """Fraction of BAND energy explained, i.e. a proper band-limited R^2.

    Computed in the Fourier domain so that 'how much of this band is predictable'
    is measured per band rather than as a global variance proxy.
    """
    m = (r >= lo) & (r < hi)
    if m.sum() == 0:
        return 0.0
    def e(x):
        F = np.fft.fft2(x, axes=(-2, -1))
        return float((np.abs(F[..., m]) ** 2).mean())
    den = e(tot)
    if den < 1e-12:
        return 0.0
    return float(max(0.0, 1.0 - e(resid) / den))


def vicreg(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Variance + covariance regulariser (non-invariant -> reaches hubness geometry)."""
    if x.shape[0] < 2:
        return torch.zeros((), device=x.device)
    x = x - x.mean(0, keepdim=True)
    std = torch.sqrt(x.var(0) + eps)
    var_loss = F.relu(1.0 - std).mean()
    d = x.shape[1]
    cov = (x.T @ x) / (x.shape[0] - 1)
    off = cov - torch.diag(torch.diag(cov))
    return var_loss + (off**2).sum() / d


def info_nce(a: torch.Tensor, b: torch.Tensor, temp: float = 0.07) -> torch.Tensor:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    logits = a @ b.T / temp
    tgt = torch.arange(a.shape[0], device=a.device)
    return 0.5 * (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.T, tgt))


def structure_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    P = pred - pred.mean(1, keepdims=True)
    G = gt - gt.mean(1, keepdims=True)
    den = np.linalg.norm(P, axis=1) * np.linalg.norm(G, axis=1)
    ok = den > 1e-8
    return {
        "pearson": round(float(np.mean((P * G).sum(1)[ok] / den[ok])), 4),
        "std_ratio": round(float(np.mean(pred.std(1) / gt.std(1).clip(1e-8))), 4),
        "spread": round(float(pred.std(0).mean() / gt.std(0).mean()), 4),
    }


# ------------------------------------------------------------------ model
class H2G(nn.Module):
    """Shared trunk + one head per (modality, granularity) cell."""

    def __init__(self, in_dim: int, code: int = 768, res: int = 64, ch: int = 4,
                 n_cells: int = 36, loc_dim: int = 1024, sem_dim: int = 1024):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, code), nn.GELU(),
            nn.Linear(code, code), nn.GELU(),
        )
        b = res // 8
        self._b, self._res, self._ch = b, res, ch
        # --- semantic global (kept as the IP-Adapter source; protects high-level metrics)
        self.sem_glob = nn.Sequential(nn.Linear(code, 768), nn.GELU(), nn.Linear(768, sem_dim))
        # --- semantic local (the previously UNSUPERVISED term). NOTE: `n_cells` is
        #     already G*G (the local target is (N, G*G, D)), so it must NOT be squared.
        self.sem_loc = nn.Sequential(nn.Linear(code, 1024), nn.GELU(),
                                     nn.Linear(1024, n_cells * loc_dim))
        self._cells, self._loc_dim = n_cells, loc_dim
        # --- structural global: low band (high-MI regime)
        self.struct_glob = nn.Sequential(
            nn.Linear(code, 1024), nn.GELU(),
            nn.Linear(1024, 128 * b * b), nn.GELU())
        self.struct_glob_up = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, ch, 3, padding=1))
        nn.init.zeros_(self.struct_glob_up[-1].weight)
        nn.init.zeros_(self.struct_glob_up[-1].bias)
        # --- structural residual: high band, HETEROSCEDASTIC (mean + logvar) => can abstain
        self.struct_res = nn.Sequential(nn.Linear(code, 512), nn.GELU(),
                                        nn.Linear(512, ch * res * res))
        self.struct_logvar = nn.Parameter(torch.zeros(1, ch, res, res))

    def encode(self, x):
        return self.trunk(x)

    def forward(self, x):
        h = self.encode(x)
        sg = self.struct_glob_up(self.struct_glob(h).view(-1, 128, self._b, self._b))
        sr = self.struct_res(h).view(-1, self._ch, self._res, self._res)
        lv = self.struct_logvar.expand_as(sr)
        return {
            "h": h,
            "sem_glob": self.sem_glob(h),
            "sem_loc": self.sem_loc(h).view(-1, self._cells, self._loc_dim),
            "struct_glob": sg,
            "struct_res": sr,
            "logvar": lv,
        }


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sem-train-npy", required=True, help="frozen EEG feature z (train)")
    ap.add_argument("--sem-test-npy", required=True)
    ap.add_argument("--vae-train-npy", required=True)
    ap.add_argument("--vae-test-npy", required=True)
    ap.add_argument("--local-train-npy", required=True, help="DINOv2 patch-grid targets (train)")
    ap.add_argument("--local-test-npy", required=True)
    ap.add_argument("--global-train-npy", default="", help="optional global semantic target override")
    ap.add_argument("--global-test-npy", default="")
    ap.add_argument("--concept-train-npy", default="", help="(N,) int concept ids for deployable prompts")
    ap.add_argument("--val-split-json", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--variant", default="full", choices=["full", "nogeom", "noloc", "noresid"])
    ap.add_argument("--cut", type=float, default=0.125, help="low/high band split radius")
    ap.add_argument("--code", type=int, default=768)
    ap.add_argument("--num-epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lambda-loc", type=float, default=1.0)
    ap.add_argument("--lambda-glob-sem", type=float, default=0.3)
    ap.add_argument("--lambda-struct", type=float, default=1.0)
    ap.add_argument("--lambda-cons", type=float, default=0.5)
    ap.add_argument("--lambda-geo", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    Z = np.load(args.sem_train_npy).astype(np.float32)
    Zte = np.load(args.sem_test_npy).astype(np.float32)
    V = np.load(args.vae_train_npy).astype(np.float32)
    Vte = np.load(args.vae_test_npy).astype(np.float32)
    L = np.load(args.local_train_npy).astype(np.float32)
    Lte = np.load(args.local_test_npy).astype(np.float32)
    assert len(Z) == len(V) == len(L), (len(Z), len(V), len(L))
    assert len(Zte) == len(Vte) == len(Lte), (len(Zte), len(Vte), len(Lte))
    res, ch = V.shape[-1], V.shape[1]
    n_cells = L.shape[1]          # local target is (N, G*G, D) -> n_cells is already G*G
    loc_dim = L.shape[2]
    # z normalization fitted on TRAIN only
    zmu, zsd = Z.mean(0, keepdims=True), Z.std(0, keepdims=True).clip(1e-6)
    Z = (Z - zmu) / zsd
    Zte = (Zte - zmu) / zsd
    print(f"[DATA] z {Z.shape} | vae {V.shape} | local {L.shape} (cells={n_cells} dim={loc_dim})")

    sp = LF.load(args.val_split_json)
    fi = LF.rows_for(sp, "fit", len(Z))
    vi = LF.rows_for(sp, "val_b", len(Z))
    assert len(set(fi.tolist()) & set(vi.tolist())) == 0
    print(f"[SPLIT] fit={len(fi)} val_b={len(vi)} (held-in concepts; test never used for selection)")

    # ---------------- MI-based band allocation (measured, leak-free)
    # R^2 is estimated OUT-OF-SAMPLE on val_a so the band weights measure genuine
    # predictability rather than the capacity of the linear fit (a fit-row R^2 would
    # read ~1.0 on high-dimensional noise).
    r_full = radial_full(res, res)
    r_half = radial_half(res, res)
    r_t = torch.from_numpy(r_half).float()
    band_edges = [0.0, 0.0625, 0.125, 0.25, 0.5, 2.0]
    weights_lo, weights_hi, diag = [], [], []
    ai = LF.rows_for(sp, "val_a", len(Z))
    lam = 1e-1
    Zf = Z[fi]
    W = np.linalg.solve(Zf.T @ Zf + lam * np.eye(Zf.shape[1], dtype=np.float32),
                        Zf.T @ V[fi].reshape(len(fi), -1))
    Za = Z[ai]
    resid = (Za @ W - V[ai].reshape(len(ai), -1)).reshape(len(ai), ch, res, res)
    tot_sig = V[ai].reshape(len(ai), ch, res, res)
    for lo, hi in zip(band_edges[:-1], band_edges[1:]):
        r2 = band_energy_ratio(resid, tot_sig, r_full, lo, hi)
        diag.append({"band": f"{lo}-{hi}", "band_r2_oos": round(r2, 5)})
        (weights_lo if hi <= args.cut else weights_hi).append(r2)
    w_lo = float(np.mean(weights_lo)) if weights_lo else 0.0
    w_hi = float(np.mean(weights_hi)) if weights_hi else 0.0
    # Floor both bands before normalising. The floor is stated rather than hidden:
    # a zero weight would starve the structural heads of any direct gradient, and the
    # measured linear R^2 is a LOWER bound on MI (a nonlinear relation would not show
    # up in the ridge probe). The floor preserves the ordering the measurement gives.
    FLOOR = 0.05
    w_lo_f, w_hi_f = max(w_lo, FLOOR), max(w_hi, FLOOR)
    tot = w_lo_f + w_hi_f
    a_lo, a_hi = w_lo_f / tot, w_hi_f / tot
    print(f"[MI] band R^2 profile (out-of-sample on val_a): {diag}")
    print(f"[MI] raw={w_lo:.4f}/{w_hi:.4f} -> weight low={a_lo:.4f} high={a_hi:.4f} (floor {FLOOR})")
    if w_hi < 0.15:
        print("[MI] high band R^2 is near noise: the residual head ABSTAINS "
              "(models uncertainty; its mean is not used for conditioning)")

    # ---------------- tensors
    # The global semantic target MUST be the DINOv2 global vector (N, 1024). Falling
    # back to the local tensor here would silently make sem_dim = grid and broadcast
    # wrongly against the (B, D) head output.
    if not args.global_train_npy or not args.global_test_npy:
        raise SystemExit("--global-train-npy and --global-test-npy are required "
                         "(DINOv2 global targets, shape (N, 1024))")
    Lg = np.load(args.global_train_npy).astype(np.float32)
    Lg_te = np.load(args.global_test_npy).astype(np.float32)
    if Lg.ndim != 2 or Lg_te.ndim != 2:
        raise SystemExit(f"global targets must be 2D, got {Lg.shape} / {Lg_te.shape}")
    if len(Lg) != len(Z) or len(Lg_te) != len(Zte):
        raise SystemExit(f"global target rows {len(Lg)}/{len(Lg_te)} != z rows {len(Z)}/{len(Zte)}")
    cid = (np.load(args.concept_train_npy).astype(np.int64)
           if args.concept_train_npy else np.zeros(len(Z), dtype=np.int64))

    def T(a, idx=None):
        a = a if idx is None else a[idx]
        return torch.from_numpy(np.ascontiguousarray(a))

    Vt = T(V).view(len(V), ch, res, res)
    lo_t, hi_t = band_split(Vt, r_t, args.cut)
    ds = TensorDataset(T(Z, fi), T(V, fi).view(len(fi), ch, res, res),
                       T(L, fi), T(Lg, fi), T(cid, fi))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)

    model = H2G(in_dim=Z.shape[1], code=args.code, res=res, ch=ch, n_cells=n_cells,
                loc_dim=loc_dim, sem_dim=Lg.shape[1]).to(device)
    print(f"[MODEL] variant={args.variant} params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.num_epochs)

    use_loc = args.variant != "noloc"
    use_geo = args.variant != "nogeom"
    use_res = args.variant != "noresid"
    r_dev = r_t.to(device)

    Vv = T(V, vi).view(len(vi), ch, res, res)
    Zv = T(Z, vi)
    Lv = T(L, vi)
    history, best, bad = [], {"val": 1e9, "epoch": 0}, 0
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        acc = {}
        nb = 0
        for zb, vb, lb, gb, cb in tqdm(loader, desc=f"h2g-{args.variant}-{epoch}", leave=False):
            zb, vb, lb, gb = zb.to(device), vb.to(device), lb.to(device), gb.to(device)
            opt.zero_grad()
            o = model(zb)
            l_lo, l_hi = band_split(vb, r_dev, args.cut)
            # (1) semantic global
            loss_sg = (1 - F.cosine_similarity(o["sem_glob"], gb, dim=-1)).mean() \
                + info_nce(o["sem_glob"], gb)
            loss = args.lambda_glob_sem * loss_sg
            # (2) semantic local -- the previously unsupervised term
            loss_sl = torch.zeros((), device=device)
            if use_loc:
                loss_sl = (1 - F.cosine_similarity(o["sem_loc"], lb, dim=-1)).mean() \
                    + info_nce(o["sem_loc"].mean(1), lb.mean(1))
                loss = loss + args.lambda_loc * loss_sl
            # (3) structural global: low band, weighted by its MEASURED R^2
            loss_stg = F.l1_loss(o["struct_glob"], l_lo)
            loss = loss + args.lambda_struct * a_lo * loss_stg
            # (4) structural residual: heteroscedastic NLL on the high band
            loss_str, loss_cons = torch.zeros((), device=device), torch.zeros((), device=device)
            if use_res:
                mu, lv = o["struct_res"], o["logvar"].clamp(-8, 8)
                loss_str = (0.5 * (torch.exp(-lv) * (mu - l_hi)**2 + lv)).mean()
                loss = loss + args.lambda_struct * a_hi * loss_str
                # coarse-to-fine consistency: the two heads must explain ONE scene
                loss_cons = F.l1_loss(o["struct_glob"] + mu, vb)
                loss = loss + args.lambda_cons * loss_cons
            # (5) geometry: non-invariant, reaches hubness
            if use_geo:
                loss = loss + args.lambda_geo * (
                    vicreg(o["h"]) + vicreg(o["struct_glob"].flatten(1)))
                if use_loc:
                    loss = loss + args.lambda_geo * vicreg(o["sem_loc"].mean(1))
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for k, v in (("sg", loss_sg), ("sl", loss_sl), ("stg", loss_stg),
                         ("str", loss_str), ("cons", loss_cons)):
                acc[k] = acc.get(k, 0.0) + float(v)
            nb += 1
        sched.step()
        if nb == 0:
            raise RuntimeError("all batches non-finite")
        for k in acc:
            acc[k] /= nb

        model.eval()
        with torch.no_grad():
            ov = model(Zv.to(device))
            val = float(F.l1_loss(ov["struct_glob"], band_split(Vv.to(device), r_dev, args.cut)[0]))
            if use_loc:
                val += float((1 - F.cosine_similarity(ov["sem_loc"], Lv.to(device), dim=-1)).mean())
        row = {"epoch": epoch, "val": round(val, 5), "selected_on": "val_b", **
               {k: round(v, 5) for k, v in acc.items()}}
        history.append(row)
        if val < best["val"] - 1e-6:
            best = {"val": val, "epoch": epoch}
            bad = 0
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "variant": args.variant,
                        "in_dim": Z.shape[1], "code": args.code, "res": res, "ch": ch,
                        "cells": n_cells, "loc_dim": loc_dim, "cut": args.cut,
                        "a_lo": a_lo, "a_hi": a_hi, "zmu": zmu.squeeze(), "zsd": zsd.squeeze(),
                        "val": val}, out / "checkpoint_h2g_best.pth")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[EARLY] stop at {epoch}")
                break
        if epoch % 10 == 0 or epoch == 1:
            print(f"[ep {epoch:3d}] sg={acc['sg']:.4f} sl={acc['sl']:.4f} stg={acc['stg']:.4f} "
                  f"str={acc['str']:.4f} cons={acc['cons']:.4f} val={val:.4f}")

    if not (out / "checkpoint_h2g_best.pth").is_file():
        raise RuntimeError("no checkpoint saved")

    # ---------------- export test-side conditions (report only; never selected on)
    ck = torch.load(out / "checkpoint_h2g_best.pth", map_location=device, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    Zte_t = torch.from_numpy(Zte).to(device)
    outs = []
    with torch.no_grad():
        for i in range(0, len(Zte_t), 256):
            outs.append({k: v.cpu().numpy() for k, v in model(Zte_t[i:i + 256]).items()})
    agg = {k: np.concatenate([o[k] for o in outs], 0) for k in outs[0]}

    low_te, high_te = band_split(torch.from_numpy(Vte).view(len(Vte), ch, res, res), r_t, args.cut)
    low_te, high_te = low_te.numpy(), high_te.numpy()
    anchors = {
        "anchor_lowband": agg["struct_glob"],                          # ABSTAIN: low only
        "anchor_low_plus_res": agg["struct_glob"] + agg["struct_res"],  # ablation: include residual mean
    }
    for nm, a in anchors.items():
        np.save(out / f"pred_{nm}_test.npy", a.astype(np.float32))
    np.save(out / "sem_glob_test.npy", agg["sem_glob"].astype(np.float32))
    np.save(out / "sem_loc_test.npy", agg["sem_loc"].astype(np.float32))
    np.save(out / "h_test.npy", agg["h"].astype(np.float32))

    flat = lambda a: a.reshape(len(a), -1)
    rep = {
        "variant": args.variant, "best_epoch": best["epoch"], "best_val": round(best["val"], 5),
        "cut": args.cut, "mi_allocation": {"low": round(a_lo, 4), "high": round(a_hi, 4)},
        "band_r2_profile": diag,
        "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 3),
        "test_metrics": {
            "anchor_lowband": structure_metrics(flat(anchors["anchor_lowband"]), flat(Vte)),
            "anchor_low_plus_res": structure_metrics(flat(anchors["anchor_low_plus_res"]), flat(Vte)),
            "lowband_target_itself": structure_metrics(flat(low_te), flat(Vte)),
        },
        "shipped_vae_head_reference": {"pearson": 0.3317, "std_ratio": 0.390, "spread": 0.428},
        "note": ("semantic side frozen upstream; structure heads replaced by 2x2 granularity. "
                 "high band ABSTAINS: its mean is not used for conditioning, only its uncertainty is modelled."),
    }
    (out / "h2g_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    pd.DataFrame(history).to_csv(out / "h2g_history.csv", index=False)
    print("\n" + json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
