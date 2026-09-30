#!/usr/bin/env python3
"""Diagnostic: why HCMA's CFM collapses, and does a proper stochastic CFM fix it?

CLAIM UNDER TEST
----------------
CondCFM.decode() starts at z = condition(c) with NO noise source:
    cond0 = self.condition(c); z = cond0.clone()   # <- no noise
so the ODE is a DETERMINISTIC map condition -> output. The only thing a
deterministic map can learn is E[target | condition] = the conditional mean.
That is exactly the mode-collapse that flow matching exists to cure -- so the
module cannot do what it is named after, and we measured the symptom:
    spread(z_cfm_f) = 0.9497 vs 0.3908 for real image embeddings (2.43x)
    margin 0.0215, 200-way Top-1 0.015 (worse than its own input, 0.160)

TEST
----
Train two velocity fields on the same intra data (EEG -> CLIP ViT-H target):
  A) "cond-start, no noise"  = current design, deterministic decode
  B) "noise-start, CFM"      = z0~N(0,I), condition via concat, integrate to t=1
Then measure, on the held-out 200 test samples:
  margin   = paired-cosine minus mean off-diagonal cosine (collapse-corrected)
  spread   = mean pairwise cosine inside the batch (1.0 = constant)
  200-way Top-1/Top-5
and for (B) the best-of-K ceiling under ORACLE selection -> is there headroom
for "sample K hypotheses + verify"? (oracle is a diagnostic, never a result)

Usage:
  python cfm_rootcause_diag.py --intra-root outputs/intra_hcma_s/sub-08 \
     --clip-train /project/peilab/why/eeg-brainit/outputs/atm_bridge/clip_img_train_1024.npy \
     --clip-test  /project/peilab/why/eeg-brainit/outputs/atm_bridge/clip_img_test_1024.npy \
     --steps 3000 --out outputs/cfm_diag/diag.json
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


def l2(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=-1)


# --------------------------------------------------------------------- metrics
def metrics(pred: np.ndarray, gallery: np.ndarray) -> dict:
    P, G = l2(torch.from_numpy(pred)), l2(torch.from_numpy(gallery))
    S = (P @ G.T).numpy()
    n = len(P)
    d = np.sum(S * np.eye(n), axis=1)
    off = (S.sum() - np.trace(S)) / (n * (n - 1))
    Pn = (P @ P.T).numpy()
    spread = (Pn.sum() - np.trace(Pn)) / (n * (n - 1))
    idx = np.argsort(-S, axis=1)
    t1 = float((idx[:, :1] == np.arange(n)[:, None]).any(1).mean())
    t5 = float((idx[:, :5] == np.arange(n)[:, None]).any(1).mean())
    return {
        "margin": round(float(d.mean() - off), 4),
        "paired": round(float(d.mean()), 4),
        "spread": round(float(spread), 4),
        "top1": round(t1, 4),
        "top5": round(t5, 4),
    }


# ----------------------------------------------------------------------- model
class Vel(nn.Module):
    """v(z_t, t, cond)"""

    def __init__(self, cond_dim=1024, dim=1024, hidden=1024, tdim=128):
        super().__init__()
        self.tproj = nn.Sequential(nn.Linear(tdim, tdim * 2), nn.SiLU(), nn.Linear(tdim * 2, tdim))
        self.net = nn.Sequential(
            nn.Linear(dim + cond_dim + tdim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def temb(self, t: torch.Tensor) -> torch.Tensor:
        half = 64
        f = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / (half - 1))
        a = t[:, None] * f[None, :] * 1000.0
        return self.tproj(torch.cat([torch.sin(a), torch.cos(a)], -1))

    def forward(self, z, t, c):
        return self.net(torch.cat([z, c, self.temb(t)], dim=-1))


@torch.no_grad()
def sample_cfm(m, cond, steps=32, n_sample=1, generator=None):
    """noise-start integration to t=1"""
    outs = []
    for _ in range(n_sample):
        z = torch.randn(cond.shape[0], m.dim_hint, generator=generator, device=cond.device)
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((z.shape[0],), i * dt, device=z.device)
            z = z + dt * m(z, t, cond)
        outs.append(z)
    return outs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--intra-root", required=True)
    ap.add_argument("--clip-train", required=True)
    ap.add_argument("--clip-test", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--n-sample", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device("cpu")

    root = Path(args.intra_root).resolve()
    # EEG condition: use the SAME feature the pipeline uses to drive structure
    z_tr = l2(torch.from_numpy(np.load(root / "train/z_decode_vith_train.npy").astype(np.float32)))
    z_te = l2(torch.from_numpy(np.load(root / "train/z_decode_vith_test.npy").astype(np.float32)))
    g_tr = l2(torch.from_numpy(np.load(args.clip_train).astype(np.float32)))
    g_te = l2(torch.from_numpy(np.load(args.clip_test).astype(np.float32)))
    print(f"[INFO] train {tuple(z_tr.shape)} -> {tuple(g_tr.shape)} ; test {tuple(z_te.shape)}")

    res: dict = {"n_train": int(len(z_tr)), "n_test": int(len(z_te))}

    # ---------------- baselines (already-existing conditions) ----------------
    res["baseline_z_decode_vith"] = metrics(z_te.numpy(), g_te.numpy())

    # ---------------- (B) proper stochastic CFM ----------------
    m = Vel().to(dev)
    m.dim_hint = 1024
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-4)
    bs = args.batch
    t0 = time.time()
    m.train()
    for step in range(args.steps):
        i = torch.randint(0, len(z_tr), (bs,))
        c, x1 = z_tr[i], g_tr[i]
        x0 = torch.randn_like(x1)
        t = torch.rand(bs)
        tb = t[:, None]
        zt = (1 - tb) * x0 + tb * x1
        v_target = x1 - x0
        loss = F.mse_loss(m(zt, t, c), v_target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step + 1) % max(args.steps // 6, 1) == 0:
            print(f"  [cfm] step {step+1}/{args.steps} loss={float(loss):.4f} ({time.time()-t0:.0f}s)")
    m.eval()

    g = torch.Generator(device=dev).manual_seed(args.seed)
    outs = sample_cfm(m, z_te, steps=32, n_sample=args.n_sample, generator=g)
    single = outs[0]
    res["stochastic_cfm_k1"] = metrics(single.numpy(), g_te.numpy())

    # ---------------- best-of-K selection ----------------
    # Two selectors over the SAME K candidates per test sample:
    #   oracle    : pick the candidate with the highest cosine to the TRUE image
    #               -> ceiling only, DIAGNOSTIC, never a reportable number
    #   consensus : pick the medoid (highest mean cosine to the other candidates)
    #               -> LABEL-FREE and deployable
    def topk_hits(cand: torch.Tensor, i: int) -> tuple[bool, bool]:
        sc = cand @ g_te.T
        order = torch.argsort(-sc)
        return bool(order[0] == i), bool((order[:5] == i).any())

    k_rows = []
    for k in [2, 4, 8]:
        if k > len(outs):
            continue
        hit_o = np.zeros(len(z_te), bool)
        hit_o5 = np.zeros(len(z_te), bool)
        hit_c = np.zeros(len(z_te), bool)
        hit_c5 = np.zeros(len(z_te), bool)
        for i in range(len(z_te)):
            C = l2(torch.stack([o[i] for o in outs[:k]]))       # (k, D)
            # oracle: pick the candidate closest to the TRUE image
            j = int((C @ g_te[i]).argmax())
            hit_o[i], hit_o5[i] = topk_hits(C[j], i)
            # consensus: medoid (label-free, deployable)
            sim = C @ C.T
            med = int(((sim.sum(1) - sim.diag()) / (k - 1)).argmax())
            hit_c[i], hit_c5[i] = topk_hits(C[med], i)
        k_rows.append({
            "k": k,
            "oracle_top1": round(float(hit_o.mean()), 4),
            "oracle_top5": round(float(hit_o5.mean()), 4),
            "consensus_top1": round(float(hit_c.mean()), 4),
            "consensus_top5": round(float(hit_c5.mean()), 4),
        })
    res["best_of_k"] = k_rows
    res["k1_top1"] = res["stochastic_cfm_k1"]["top1"]

    # ---------------- (A) current design: cond-start, no noise ----------------
    # trained identically but decode starts from cond and integrates backward,
    # mirroring CondCFM (target->cond interpolation, reverse ODE).
    m2 = Vel().to(dev)
    m2.dim_hint = 1024
    opt2 = torch.optim.AdamW(m2.parameters(), lr=args.lr, weight_decay=1e-4)
    for step in range(args.steps):
        i = torch.randint(0, len(z_tr), (bs,))
        c, x0 = z_tr[i], g_tr[i]          # x0 = target, x1 = condition (as in CondCFM)
        x1 = c
        t = torch.rand(bs)
        tb = t[:, None]
        zt = (1 - tb) * x0 + tb * x1
        loss = F.mse_loss(m2(zt, t, c), x1 - x0)
        opt2.zero_grad(set_to_none=True)
        loss.backward()
        opt2.step()
    m2.eval()
    with torch.no_grad():
        cond0 = z_te.clone()
        z = cond0.clone()
        dt = 1.0 / 32
        for i in range(32):
            t = torch.full((z.shape[0],), 1.0 - i * dt)
            z = z - dt * m2(z, t, z_te)
        det = z
    res["deterministic_cond_start_k1"] = metrics(det.numpy(), g_te.numpy())

    print("\n=== CFM root-cause diagnostic (all on the 200 held-out test samples) ===")
    for k in ["baseline_z_decode_vith", "deterministic_cond_start_k1", "stochastic_cfm_k1"]:
        r = res[k]
        print(f"{k:<30} margin={r['margin']:>7.4f} spread={r['spread']:>6.4f} "
              f"top1={r['top1']:>6.3f} top5={r['top5']:>6.3f}")
    print(f"\nbest-of-K selection (k1 top1 = {res['k1_top1']:.3f}):")
    print(f"  {'k':>3}{'oracle_T1':>11}{'oracle_T5':>11}{'consens_T1':>12}{'consens_T5':>12}")
    for r in k_rows:
        print(f"  {r['k']:>3}{r['oracle_top1']:>11.3f}{r['oracle_top5']:>11.3f}"
              f"{r['consensus_top1']:>12.3f}{r['consensus_top5']:>12.3f}")
    print("\n  oracle = ceiling (uses test gallery; DIAGNOSTIC ONLY)")
    print("  consensus = label-free medoid selector (deployable)")
    print(f"\n[REF] real image embeddings: spread=0.3908 ; 200-way chance=0.005")

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=2))
    print(f"[OK] -> {p}")


if __name__ == "__main__":
    main()
