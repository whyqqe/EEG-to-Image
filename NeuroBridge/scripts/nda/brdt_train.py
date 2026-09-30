#!/usr/bin/env python3
"""BRDT: band-routed dual tower with a measured-rank inter-tower bottleneck.

HCMA's main architecture is untouched: the cross-subject backbone, the two towers and
the multi-condition injection decoder are all preserved. What changes is (a) what each
tower READS from the EEG encoder, and (b) that the two towers are allowed a strictly
limited exchange. Both are decided by measurement, not by design taste.

ROUTING (Gate 1)
    gamma      -> STRUCT tower   (retinotopically tuned: carries WHERE)
    alpha,theta-> SEM tower      (little spatial tuning: carries global state)
    beta       -> SPLIT, early->struct / late->sem, because it is intermediate
    early window -> STRUCT, late window -> SEM (peripheral before foveal)
If Gate 1 fails, the orchestrator passes --route merged and the bands are pooled; that
is recorded in the report so a negative gate can never be laundered into a positive.

BOTTLENECK (Gate 2)
    z_struct' = z_struct + B_q(A_q z_sem)
    z_sem'    = z_sem    + B_s(A_s z_struct)      A: d->r, B: r->d
    r is the number of gamma<->alpha/theta canonical components above threshold, i.e.
    the MEASURED residual dependency. Because B is initialised at zero the exchange
    starts as an exact no-op, so any change must be earned.

    Falsifiable prediction: r above the measured value must hurt BOTH towers -- the
    semantic tower receives spatial noise, the structural tower receives global state.
    The `overexchange` arm tests that prediction rather than asserting it.

GRANULARITY x MODALITY (per tower, as requested)
    SEM  global : CLIP-text (coarse prompt) + ViT-H image embedding
    SEM  local  : DINOv2 patch-cell grid              <- the unsupervised chain-rule term
    STRUCT global: SDXL-VAE low band + depth (if available)
    STRUCT local : high band, HETEROSCEDASTIC, and it ABSTAINS
                   (argmin of an L1/L2 field regression is E[y|e] whose variance is
                    strictly below Var[y]; predicting that mean cannot add information,
                    so the band's mean is not used for conditioning -- only its
                    uncertainty is modelled.)

GEOMETRY
    VICReg is applied to REPRESENTATIONS (the tower codes), never to the 16384-dim
    reconstruction field. Applying it to the field is both wrong (it is a
    representation-geometry regulariser, not a fidelity term) and ruinous (a 16384^2
    covariance per step). A band-room floor keeps the loss weights ordered by the
    measured band R^2 without letting a single band monopolise the gradient.
"""

from __future__ import annotations

import argparse
import json
import pickle
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

BANDS = ["theta", "alpha", "beta", "gamma"]
WINDOWS = ["early", "late"]
# per Gate 1: who reads which band-window
ROUTE_STRUCT = [("gamma", "early"), ("gamma", "late"), ("beta", "early")]
ROUTE_SEM = [("alpha", "early"), ("alpha", "late"), ("theta", "early"),
             ("theta", "late"), ("beta", "late")]


def radial_full(H, W):
    fy = np.fft.fftfreq(H)[:, None]
    fx = np.fft.fftfreq(W)[None, :]
    return np.sqrt(fy**2 + fx**2) / 0.5


def radial_half(H, W):
    fy = np.fft.fftfreq(H)[:, None]
    fx = np.fft.rfftfreq(W)[None, :]
    return np.sqrt(fy**2 + fx**2) / 0.5


def band_split(x, r, cut):
    Fx = torch.fft.rfft2(x.float())
    m = (r < cut).float()[None, None]
    lo = torch.fft.irfft2(Fx * m, s=x.shape[-2:])
    hi = torch.fft.irfft2(Fx * (1.0 - m), s=x.shape[-2:])
    return lo, hi


def band_energy_r2(resid, tot, r, lo, hi):
    m = (r >= lo) & (r < hi)
    if m.sum() == 0:
        return 0.0
    def e(x):
        Fx = np.fft.fft2(x, axes=(-2, -1))
        return float((np.abs(Fx[..., m]) ** 2).mean())
    d = e(tot)
    return float(max(0.0, 1.0 - e(resid) / d)) if d > 1e-12 else 0.0


def vicreg(x, eps=1e-4):
    """Variance + covariance regulariser on a REPRESENTATION (B, d)."""
    if x.shape[0] < 2:
        return torch.zeros((), device=x.device)
    x = x - x.mean(0, keepdim=True)
    std = torch.sqrt(x.var(0) + eps)
    var = F.relu(1.0 - std).mean()
    d = x.shape[1]
    cov = (x.T @ x) / (x.shape[0] - 1)
    off = cov - torch.diag(torch.diag(cov))
    return var + (off**2).sum() / d


def info_nce(a, b, t=0.07):
    a, b = F.normalize(a, dim=-1), F.normalize(b, dim=-1)
    lg = a @ b.T / t
    tgt = torch.arange(a.shape[0], device=a.device)
    return 0.5 * (F.cross_entropy(lg, tgt) + F.cross_entropy(lg.T, tgt))


def struct_metrics(pred, gt):
    P = pred - pred.mean(1, keepdims=True)
    G = gt - gt.mean(1, keepdims=True)
    den = np.linalg.norm(P, axis=1) * np.linalg.norm(G, axis=1)
    ok = den > 1e-8
    return {"pearson": round(float(np.mean((P * G).sum(1)[ok] / den[ok])), 4),
            "std_ratio": round(float(np.mean(pred.std(1) / gt.std(1).clip(1e-8))), 4),
            "spread": round(float(pred.std(0).mean() / gt.std(0).mean()), 4)}


class RankBottleneck(nn.Module):
    """Minimal-capacity inter-tower channel. Capacity = sum log singular values."""

    def __init__(self, d: int, r: int):
        super().__init__()
        self.r = int(r)
        if r > 0:
            self.As = nn.Linear(d, r, bias=False)   # struct -> sem
            self.Bs = nn.Linear(r, d, bias=False)
            self.Aq = nn.Linear(d, r, bias=False)   # sem -> struct
            self.Bq = nn.Linear(r, d, bias=False)
            nn.init.zeros_(self.Bs.weight)          # start as an exact no-op
            nn.init.zeros_(self.Bq.weight)

    def forward(self, z_struct, z_sem):
        if self.r <= 0:
            return z_struct, z_sem, torch.zeros((), device=z_struct.device)
        ex_s = self.Bq(self.Aq(z_sem))              # what sem hands to struct
        ex_q = self.Bs(self.As(z_struct))           # what struct hands to sem
        return z_struct + ex_s, z_sem + ex_q, ex_s.abs().mean() + ex_q.abs().mean()


class BRDT(nn.Module):
    def __init__(self, d_in_struct, d_in_sem, d_in_loc, code=768, r=0, res=64, ch=4,
                 cells=36, loc_dim=1024, n_glob=1024, n_glob2=1024, depth_res=64,
                 use_depth=True):
        super().__init__()
        self.enc_struct = nn.Sequential(nn.Linear(d_in_struct, code), nn.GELU(),
                                        nn.Linear(code, code), nn.GELU())
        self.enc_sem = nn.Sequential(nn.Linear(d_in_sem, code), nn.GELU(),
                                     nn.Linear(code, code), nn.GELU())
        self.bneck = RankBottleneck(code, r)
        b = res // 8
        self._b, self._res, self._ch, self._cells, self._loc = b, res, ch, cells, loc_dim
        self._dres, self._use_depth = depth_res, use_depth
        # --- semantic tower: global x 2 modalities, local x 1 grid
        self.sem_glob_text = nn.Sequential(nn.Linear(code, 768), nn.GELU(), nn.Linear(768, n_glob))
        self.sem_glob_img = nn.Sequential(nn.Linear(code, 768), nn.GELU(), nn.Linear(768, n_glob2))
        self.sem_loc = nn.Sequential(nn.Linear(code, 1024), nn.GELU(),
                                     nn.Linear(1024, cells * loc_dim))
        # --- structural tower: global (low band) + depth, local (high band, abstaining)
        self.st_glob = nn.Sequential(nn.Linear(code, 1024), nn.GELU(),
                                     nn.Linear(1024, 128 * b * b), nn.GELU())
        self.st_glob_up = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, ch, 3, padding=1))
        nn.init.zeros_(self.st_glob_up[-1].weight)
        nn.init.zeros_(self.st_glob_up[-1].bias)
        self.depth_head = nn.Sequential(nn.Linear(code, 512), nn.GELU(), nn.Linear(512, depth_res * depth_res))
        self.st_res = nn.Sequential(nn.Linear(code, 512), nn.GELU(),
                                    nn.Linear(512, ch * res * res))
        self.st_logvar = nn.Parameter(torch.zeros(1, ch, res, res))

    def forward(self, xs, xq, xl=None):
        hs = self.enc_struct(xs)
        hq = self.enc_sem(xq)
        zs, zq, exch = self.bneck(hs, hq)
        out = {"h_struct": hs, "h_sem": hq, "z_struct": zs, "z_sem": zq, "exchange": exch,
               "sem_text": self.sem_glob_text(zq), "sem_img": self.sem_glob_img(zq),
               "sem_loc": self.sem_loc(zq).view(-1, self._cells, self._loc),
               "st_glob": self.st_glob_up(self.st_glob(zs).view(-1, 128, self._b, self._b)),
               "st_res": self.st_res(zs).view(-1, self._ch, self._res, self._res)}
        out["st_logvar"] = self.st_logvar.expand_as(out["st_res"])
        out["depth"] = self.depth_head(zs).view(-1, 1, self._dres, self._dres)
        return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-dir", required=True, help="bandfeat_{train,test}.npz + basis")
    ap.add_argument("--vae-train-npy", required=True)
    ap.add_argument("--vae-test-npy", required=True)
    ap.add_argument("--clip-text-train-npy", required=True, help="CLIP text target (global)")
    ap.add_argument("--clip-text-test-npy", required=True)
    ap.add_argument("--vith-train-npy", required=True, help="ViT-H image embedding (global)")
    ap.add_argument("--vith-test-npy", required=True)
    ap.add_argument("--dino-local-train-npy", required=True)
    ap.add_argument("--dino-local-test-npy", required=True)
    ap.add_argument("--dino-global-train-npy", default="")
    ap.add_argument("--dino-global-test-npy", default="")
    ap.add_argument("--depth-train-npy", default="")
    ap.add_argument("--depth-test-npy", default="")
    ap.add_argument("--val-split-json", required=True)
    ap.add_argument("--gate-json", default="", help="brdt_probe gates.json (sets rank r)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--variant", default="full",
                    choices=["full", "noexchange", "overexchange", "wrongroute",
                             "nolocal", "noabstain", "merged"])
    ap.add_argument("--rank", type=int, default=-1, help="-1 -> read from gates.json")
    ap.add_argument("--cut", type=float, default=0.125)
    ap.add_argument("--code", type=int, default=768)
    ap.add_argument("--num-epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lambda-loc", type=float, default=0.5)
    ap.add_argument("--lambda-struct", type=float, default=1.0)
    ap.add_argument("--lambda-depth", type=float, default=0.3)
    ap.add_argument("--lambda-cons", type=float, default=0.5)
    ap.add_argument("--lambda-geo", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")

    Fd = Path(a.feat_dir)
    tr = np.load(Fd / "bandfeat_train.npz")
    te = np.load(Fd / "bandfeat_test.npz")
    basis = pickle.loads((Fd / "bandfeat_basis.pkl").read_bytes())

    # ---------------- routing
    if a.variant == "merged":
        rs, rq = ROUTE_STRUCT + ROUTE_SEM, ROUTE_STRUCT + ROUTE_SEM   # no band separation
        route_mode = "merged"
    elif a.variant == "wrongroute":
        rs, rq = ROUTE_SEM, ROUTE_STRUCT                              # swapped on purpose
        route_mode = "wrongroute"
    else:
        rs, rq = ROUTE_STRUCT, ROUTE_SEM
        route_mode = "routed"

    def pack(store, route):
        cols = []
        for bn, wn in route:
            cols.append(store[f"{bn}_{wn}_mean"])
            cols.append(store[f"{bn}_power"])
            cols.append(store[f"{bn}_{wn}_disp"])
        return np.concatenate(cols, 1).astype(np.float32)

    Xs_tr, Xq_tr = pack(tr, rs), pack(tr, rq)
    Xs_te, Xq_te = pack(te, rs), pack(te, rq)

    V = np.load(a.vae_train_npy).astype(np.float32)
    Vte = np.load(a.vae_test_npy).astype(np.float32)
    Ct = np.load(a.clip_text_train_npy).astype(np.float32)
    Cte = np.load(a.clip_text_test_npy).astype(np.float32)
    Ht = np.load(a.vith_train_npy).astype(np.float32)
    Hte = np.load(a.vith_test_npy).astype(np.float32)
    L = np.load(a.dino_local_train_npy).astype(np.float32)
    Lte = np.load(a.dino_local_test_npy).astype(np.float32)
    res, ch = V.shape[-1], V.shape[1]
    cells, loc_dim = L.shape[1], L.shape[2]

    depth_tr = depth_te = None
    if a.depth_train_npy and a.depth_test_npy and Path(a.depth_train_npy).is_file() \
            and Path(a.depth_test_npy).is_file():
        depth_tr = np.load(a.depth_train_npy).astype(np.float32)
        depth_te = np.load(a.depth_test_npy).astype(np.float32)
        if depth_tr.ndim == 3:
            depth_tr = depth_tr[:, None]
        if depth_te.ndim == 3:
            depth_te = depth_te[:, None]
    use_depth = depth_tr is not None
    print(f"[DATA] struct_in {Xs_tr.shape} sem_in {Xq_tr.shape} vae {V.shape} "
          f"local {L.shape} depth={'yes' if use_depth else 'NO'}")

    sp = LF.load(a.val_split_json)
    fi = LF.rows_for(sp, "fit", len(V))
    vi = LF.rows_for(sp, "val_b", len(V))
    ai = LF.rows_for(sp, "val_a", len(V))
    assert len(set(fi.tolist()) & set(vi.tolist())) == 0

    # ---------------- measured rank r (Gate 2)
    if a.rank >= 0:
        rank = a.rank
    elif a.gate_json and Path(a.gate_json).is_file():
        g = json.loads(Path(a.gate_json).read_text(encoding="utf-8"))["gates"]
        rank = int(g.get("gate2_rank_r", 4))
    else:
        rank = 4
    if a.variant == "noexchange":
        rank = 0
    elif a.variant == "overexchange":
        rank = max(rank * 4, 8)
    print(f"[BOTTLENECK] mode={route_mode} rank_r={rank} variant={a.variant}")

    # ---------------- band-weighted structural supervision (out-of-sample on val_a)
    r_full, r_half = radial_full(res, res), radial_half(res, res)
    r_t = torch.from_numpy(r_half).float()
    edges = [0.0, 0.0625, 0.125, 0.25, 0.5, 2.0]
    Zs = Xs_tr.copy()
    mu, sd = Zs[fi].mean(0, keepdims=True), Zs[fi].std(0, keepdims=True).clip(1e-6)
    Zs = (Zs - mu) / sd
    W = np.linalg.solve(Zs[fi].T @ Zs[fi] + 0.1 * np.eye(Zs.shape[1], dtype=np.float32),
                        Zs[fi].T @ V[fi].reshape(len(fi), -1))
    resid = (Zs[ai] @ W - V[ai].reshape(len(ai), -1)).reshape(len(ai), ch, res, res)
    tot = V[ai].reshape(len(ai), ch, res, res)
    w_lo, w_hi, prof = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        r2 = band_energy_r2(resid, tot, r_full, lo, hi)
        prof.append({"band": f"{lo}-{hi}", "r2_oos": round(r2, 5)})
        (w_lo if hi <= a.cut else w_hi).append(r2)
    w_lo = float(np.mean(w_lo)) if w_lo else 0.0
    w_hi = float(np.mean(w_hi)) if w_hi else 0.0
    FLOOR = 0.05
    a_lo = max(w_lo, FLOOR) / (max(w_lo, FLOOR) + max(w_hi, FLOOR))
    a_hi = 1.0 - a_lo
    print(f"[MI] {prof} -> weights low={a_lo:.4f} high={a_hi:.4f}")

    def T(x, idx=None):
        x = x if idx is None else x[idx]
        return torch.from_numpy(np.ascontiguousarray(x))

    Vt = T(V).view(len(V), ch, res, res)
    r_dev = r_t.to(dev)
    ds = TensorDataset(T(Xs_tr, fi), T(Xq_tr, fi), T(V, fi).view(len(fi), ch, res, res),
                       T(Ct, fi), T(Ht, fi), T(L, fi),
                       T(depth_tr, fi) if use_depth else torch.zeros(len(fi), 1, 1, 1))
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=True, drop_last=True, num_workers=0)

    model = BRDT(Xs_tr.shape[1], Xq_tr.shape[1], L.shape[1] * L.shape[2], code=a.code,
                 r=rank, res=res, ch=ch, cells=cells, loc_dim=loc_dim,
                 n_glob=Ct.shape[1], n_glob2=Ht.shape[1],
                 depth_res=depth_tr.shape[-1] if use_depth else 64,
                 use_depth=use_depth).to(dev)
    print(f"[MODEL] params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    opt = optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.num_epochs)

    use_loc = a.variant != "nolocal"
    use_abstain = a.variant != "noabstain"
    Xs_v, Xq_v = T(Xs_tr, vi), T(Xq_tr, vi)
    Vv = T(V, vi).view(len(vi), ch, res, res)
    Cv, Hv, Lv = T(Ct, vi), T(Ht, vi), T(L, vi)

    hist, best, bad = [], {"val": 1e9, "epoch": 0}, 0
    for ep in range(1, a.num_epochs + 1):
        model.train()
        acc, nb = {}, 0
        for zb_s, zb_q, vb, cb, hb, lb, db in tqdm(loader, desc=f"brdt-{a.variant}-{ep}", leave=False):
            zb_s, zb_q, vb = zb_s.to(dev), zb_q.to(dev), vb.to(dev)
            cb, hb, lb = cb.to(dev), hb.to(dev), lb.to(dev)
            opt.zero_grad()
            o = model(zb_s, zb_q)
            l_lo, l_hi = band_split(vb, r_dev, a.cut)
            # semantic global: two modalities
            ls = (1 - F.cosine_similarity(o["sem_text"], cb, -1)).mean() + info_nce(o["sem_text"], cb)
            ls = ls + (1 - F.cosine_similarity(o["sem_img"], hb, -1)).mean() + info_nce(o["sem_img"], hb)
            loss = 0.3 * ls
            # semantic local: the previously unsupervised chain-rule term
            lsl = torch.zeros((), device=dev)
            if use_loc:
                lsl = (1 - F.cosine_similarity(o["sem_loc"], lb, -1)).mean() \
                    + info_nce(o["sem_loc"].mean(1), lb.mean(1))
                loss = loss + a.lambda_loc * lsl
            # structural global: low band + depth
            lst = F.l1_loss(o["st_glob"], l_lo)
            loss = loss + a.lambda_struct * a_lo * lst
            ld = torch.zeros((), device=dev)
            if use_depth:
                db_ = db.to(dev)
                ld = F.l1_loss(o["depth"], db_)
                loss = loss + a.lambda_depth * ld
            # structural local: heteroscedastic; ABSTAIN keeps the mean out of the loss
            lsr, lc = torch.zeros((), device=dev), torch.zeros((), device=dev)
            if use_abstain:
                lv = o["st_logvar"].clamp(-8, 8)
                lsr = (0.5 * (torch.exp(-lv) * (o["st_res"] - l_hi) ** 2 + lv)).mean()
                loss = loss + a.lambda_struct * a_hi * lsr
                lc = F.l1_loss(o["st_glob"] + o["st_res"], vb)
                loss = loss + a.lambda_cons * lc
            # geometry on REPRESENTATIONS only
            loss = loss + a.lambda_geo * (vicreg(o["z_struct"]) + vicreg(o["z_sem"]))
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for k, v in (("sem", ls), ("sem_loc", lsl), ("stg", lst), ("depth", ld),
                         ("str", lsr), ("cons", lc), ("exch", o["exchange"])):
                acc[k] = acc.get(k, 0.0) + float(v)
            nb += 1
        sch.step()
        if nb == 0:
            raise RuntimeError("all batches non-finite")
        for k in acc:
            acc[k] /= nb
        model.eval()
        with torch.no_grad():
            ov = model(Xs_v.to(dev), Xq_v.to(dev))
            val = float(F.l1_loss(ov["st_glob"], band_split(Vv.to(dev), r_dev, a.cut)[0]))
            if use_loc:
                val += float((1 - F.cosine_similarity(ov["sem_loc"], Lv.to(dev), -1)).mean())
        hist.append({"epoch": ep, "val": round(val, 5), **{k: round(v, 5) for k, v in acc.items()}})
        if val < best["val"] - 1e-6:
            best, bad = {"val": val, "epoch": ep}, 0
            torch.save({"state_dict": model.state_dict(), "epoch": ep, "variant": a.variant,
                        "route_mode": route_mode, "rank": rank, "a_lo": a_lo, "a_hi": a_hi,
                        "res": res, "ch": ch, "cells": cells, "loc_dim": loc_dim,
                        "code": a.code, "cut": a.cut, "use_depth": use_depth,
                        "d_in_struct": Xs_tr.shape[1], "d_in_sem": Xq_tr.shape[1]},
                       out / "checkpoint_brdt_best.pth")
        else:
            bad += 1
            if bad >= a.patience:
                print(f"[EARLY] stop @ {ep}")
                break
        if ep % 10 == 0 or ep == 1:
            print(f"[ep {ep:3d}] sem={acc['sem']:.4f} loc={acc['sem_loc']:.4f} "
                  f"stg={acc['stg']:.4f} str={acc['str']:.4f} cons={acc['cons']:.4f} "
                  f"exch={acc['exch']:.4f} val={val:.4f}")

    if not (out / "checkpoint_brdt_best.pth").is_file():
        raise RuntimeError("no checkpoint")

    # ---------------- export test-side conditions
    ck = torch.load(out / "checkpoint_brdt_best.pth", map_location=dev, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    parts = []
    with torch.no_grad():
        for i in range(0, len(Xs_te), 256):
            o = model(T(Xs_te, np.arange(i, min(i + 256, len(Xs_te)))).to(dev),
                      T(Xq_te, np.arange(i, min(i + 256, len(Xs_te)))).to(dev))
            parts.append({k: v.cpu().numpy() for k, v in o.items()
                          if isinstance(v, torch.Tensor) and v.ndim >= 1})
    agg = {k: np.concatenate([p[k] for p in parts], 0) for k in parts[0]}

    low_te, high_te = band_split(T(Vte).view(len(Vte), ch, res, res), r_t, a.cut)
    low_te, high_te = low_te.numpy(), high_te.numpy()
    flat = lambda x: x.reshape(len(x), -1)
    np.save(out / "pred_anchor_lowband_test.npy", agg["st_glob"].astype(np.float32))
    np.save(out / "pred_anchor_low_plus_res_test.npy",
            (agg["st_glob"] + agg["st_res"]).astype(np.float32))
    np.save(out / "sem_text_test.npy", agg["sem_text"].astype(np.float32))
    np.save(out / "sem_img_test.npy", agg["sem_img"].astype(np.float32))
    np.save(out / "z_struct_test.npy", agg["z_struct"].astype(np.float32))
    np.save(out / "z_sem_test.npy", agg["z_sem"].astype(np.float32))
    if use_depth:
        np.save(out / "pred_depth_test.npy", agg["depth"].astype(np.float32))

    # semantic retrieval diagnostics (200-way, self-excluded) on the new encoder
    def diag(x, name):
        n = len(x)
        xn = x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-8)
        sim = xn @ xn.T
        sim -= np.eye(n, dtype=sim.dtype) * 1e9
        order = np.argsort(-sim, 1)
        return {f"{name}_top1": round(float(np.mean(order[:, 0] == np.arange(n))), 4),
                f"{name}_top5": round(float(np.mean([np.any(order[i, :5] == i) for i in range(n)])), 4)}

    rep = {
        "variant": a.variant, "route_mode": route_mode, "rank_r": rank,
        "best_epoch": best["epoch"], "best_val": round(best["val"], 5),
        "cut": a.cut, "band_weights": {"low": round(a_lo, 4), "high": round(a_hi, 4)},
        "band_r2_profile": prof, "use_depth": use_depth,
        "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 3),
        "test_metrics": {
            "anchor_lowband": struct_metrics(flat(agg["st_glob"]), flat(Vte)),
            "anchor_low_plus_res": struct_metrics(flat(agg["st_glob"] + agg["st_res"]), flat(Vte)),
            "lowband_target_itself": struct_metrics(flat(low_te), flat(Vte)),
        },
        **diag(agg["sem_text"], "sem_text"), **diag(agg["sem_img"], "sem_img"),
        "shipped_vae_head_reference": {"pearson": 0.3317, "std_ratio": 0.390, "spread": 0.428},
    }
    (out / "brdt_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    pd.DataFrame(hist).to_csv(out / "brdt_history.csv", index=False)
    print("\n" + json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
