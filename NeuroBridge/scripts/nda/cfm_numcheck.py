#!/usr/bin/env python3
"""Is the stochastic-CFM failure conceptual or numerical?

CONTEXT
-------
cfm_rootcause_diag.py showed, on the 200 held-out samples:
    baseline z_decode_vith   top1=0.350  margin=0.2061
    deterministic cond-start top1=0.245  margin=0.1553   (current CondCFM design)
    stochastic CFM (32 steps) top1=0.010 margin=0.0074   <- collapsed
and oracle best-of-8 top1 was only 0.055, i.e. NOTHING rescues naive sampling.

WHY THIS MATTERS FOR THE DIRECTION
----------------------------------
If z0 ~ N(0,I) is INDEPENDENT of the condition c, then
    E[x1 | x0, c] = E[x1 | c]
so a correctly-trained velocity field should drive ANY starting noise to the
conditional mean, and sampling should be *no better and no worse* than the mean.
That predicts the 32-step failure is NUMERICAL: the trajectory has length
~|x1 - x0| ~ sqrt(1 + d) ~ 32, so 32 Euler steps of size 1.0 is far too coarse.

TEST
----
Train ONE stochastic CFM on (z_decode_vith -> CLIP ViT-H), then evaluate:
  * sampling with 32 / 128 / 512 steps
  * a plain conditional-mean MLP with the same trunk (the "no generative" ceiling)
If 512-step sampling matches the MLP mean, then generative sampling buys nothing
for this task and the alignment work must target the MEAN ESTIMATE.
If it stays collapsed even at 512 steps, the problem is conceptual instead.

Usage:
  python cfm_numcheck.py --intra-root outputs/intra_hcma_s/sub-08 \
    --clip-train .../clip_img_train_1024.npy --clip-test .../clip_img_test_1024.npy \
    --out outputs/cfm_diag/numcheck.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def l2(x): return F.normalize(x.float(), dim=-1)


def metrics(pred, gallery):
    P, G = l2(torch.from_numpy(pred)), l2(torch.from_numpy(gallery))
    S = (P @ G.T).numpy(); n = len(P)
    d = np.sum(S * np.eye(n), axis=1)
    off = (S.sum() - np.trace(S)) / (n * (n - 1))
    Pn = (P @ P.T).numpy()
    sp = (Pn.sum() - np.trace(Pn)) / (n * (n - 1))
    idx = np.argsort(-S, axis=1)
    return {
        "margin": round(float(d.mean() - off), 4),
        "top1": round(float((idx[:, :1] == np.arange(n)[:, None]).any(1).mean()), 4),
        "top5": round(float((idx[:, :5] == np.arange(n)[:, None]).any(1).mean()), 4),
        "spread": round(float(sp), 4),
        "norm": round(float(np.linalg.norm(pred, axis=1).mean()), 2),
    }


class Trunk(nn.Module):
    def __init__(self, cdim=1024, dim=1024, hidden=1024, tdim=128):
        super().__init__()
        self.tproj = nn.Sequential(nn.Linear(tdim, tdim * 2), nn.SiLU(), nn.Linear(tdim * 2, tdim))
        self.net = nn.Sequential(
            nn.Linear(dim + cdim + tdim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def temb(self, t):
        half = 64
        f = torch.exp(-math.log(1e4) * torch.arange(half, dtype=t.dtype) / (half - 1))
        a = t[:, None] * f[None, :] * 1000.0
        return self.tproj(torch.cat([torch.sin(a), torch.cos(a)], -1))


class Vel(Trunk):
    def forward(self, z, t, c): return self.net(torch.cat([z, c, self.temb(t)], -1))


class MeanMLP(nn.Module):
    """direct conditional-mean regressor (no generative component)"""
    def __init__(self, cdim=1024, dim=1024, hidden=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cdim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )
    def forward(self, c): return self.net(c)


@torch.no_grad()
def sample(m, cond, steps, gen):
    z = torch.randn(cond.shape[0], 1024, generator=gen)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((z.shape[0],), i * dt)
        z = z + dt * m(z, t, cond)
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--intra-root", required=True)
    ap.add_argument("--clip-train", required=True)
    ap.add_argument("--clip-test", required=True)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(0)
    root = Path(args.intra_root).resolve()
    z_tr = l2(torch.from_numpy(np.load(root / "train/z_decode_vith_train.npy").astype(np.float32)))
    z_te = l2(torch.from_numpy(np.load(root / "train/z_decode_vith_test.npy").astype(np.float32)))
    g_tr = l2(torch.from_numpy(np.load(args.clip_train).astype(np.float32)))
    g_te = l2(torch.from_numpy(np.load(args.clip_test).astype(np.float32)))

    res = {"baseline_z_decode_vith": metrics(z_te.numpy(), g_te.numpy())}

    # ---- train both models on the same budget ----
    vel, mean = Vel(), MeanMLP()
    o1 = torch.optim.AdamW(vel.parameters(), lr=2e-3, weight_decay=1e-4)
    o2 = torch.optim.AdamW(mean.parameters(), lr=2e-3, weight_decay=1e-4)
    bs = 512
    for s in range(args.steps):
        i = torch.randint(0, len(z_tr), (bs,))
        c, x1 = z_tr[i], g_tr[i]
        # FM
        x0 = torch.randn_like(x1); t = torch.rand(bs)
        zt = (1 - t[:, None]) * x0 + t[:, None] * x1
        l1 = F.mse_loss(vel(zt, t, c), x1 - x0)
        o1.zero_grad(set_to_none=True); l1.backward(); o1.step()
        # mean
        l2_ = F.mse_loss(mean(c), x1)
        o2.zero_grad(set_to_none=True); l2_.backward(); o2.step()
        if (s + 1) % 1000 == 0:
            print(f"  step {s+1}/{args.steps}  fm_loss={float(l1):.4f}  mean_loss={float(l2_):.4f}")
    vel.eval(); mean.eval()

    # ---- conditional mean ----
    with torch.no_grad():
        mu = l2(mean(z_te))
    res["cond_mean_mlp"] = metrics(mu.numpy(), g_te.numpy())

    # ---- FM sampling at increasing step counts ----
    for st in (32, 128, 512):
        g = torch.Generator().manual_seed(0)
        out = sample(vel, z_te, st, g)
        res[f"stochastic_cfm_{st}steps"] = metrics(out.numpy(), g_te.numpy())

    # ---- does sampling converge to the mean? ----
    g = torch.Generator().manual_seed(1)
    s1 = sample(vel, z_te, 512, g)
    m1 = torch.Generator().manual_seed(2)
    s2 = sample(vel, z_te, 512, m1)
    res["sample_consistency"] = {
        "cos_s1_vs_s2": round(float((l2(s1) * l2(s2)).sum(-1).mean()), 4),
        "cos_sample_vs_mean": round(float((l2(s1) * l2(mu)).sum(-1).mean()), 4),
        "note": "接近 1 => 采样退化为条件均值(噪声与条件独立,故无真实多样性)",
    }

    print("\n=== numerical vs conceptual check (200 held-out samples) ===")
    print(f"{'method':<28}{'margin':>8}{'top1':>7}{'top5':>7}{'spread':>8}{'|out|':>8}")
    for k in ["baseline_z_decode_vith", "cond_mean_mlp", "stochastic_cfm_32steps",
              "stochastic_cfm_128steps", "stochastic_cfm_512steps"]:
        r = res[k]
        print(f"{k:<28}{r['margin']:>8.4f}{r['top1']:>7.3f}{r['top5']:>7.3f}{r['spread']:>8.4f}{r['norm']:>8.2f}")
    c = res["sample_consistency"]
    print(f"\nsample-vs-sample cos (two seeds) = {c['cos_s1_vs_s2']}")
    print(f"sample-vs-conditional-mean cos   = {c['cos_sample_vs_mean']}")

    p = Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=2))
    print(f"[OK] -> {p}")


if __name__ == "__main__":
    main()
