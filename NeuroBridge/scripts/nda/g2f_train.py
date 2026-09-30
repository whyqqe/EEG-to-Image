#!/usr/bin/env python3
"""G2F: the user-designed dual tower, trained with leak-free objectives.

WHAT CHANGED VS g2_train.py (and why)
-------------------------------------
Measured on sub-08 (2026-09-11). All three models below read the SAME cached
EEG latent ``z_eeg_proj`` (512-d); the file is byte-identical across the three
pipelines (verified with np.array_equal), so every difference is the objective.

  condition                        pairwise-cos   top-1   top-5   2-way
  constant (train centroid)             1.000     0.005   0.025   0.520
  HCMA rag_soft5 memory  (used by 9/10 subjects)  0.670  0.060  0.200  0.760
  HCMA blend_nda_cfm_f_a40 (the gen condition)    0.778  0.060  0.185  0.755
  NDA-SS trainable decoder                        0.130  0.290  0.585  0.940

The gap is 5x in top-1 with an identical input. The cause is not the encoder and
not the target; it is the OBJECTIVE: NDA-SS carries a *differentiable* soft
memory inside the computation graph plus class-level supervision, while HCMA's
memory is a non-differentiable retrieval table (so the encoder was never trained
to be retrieval-friendly) and g2_train.py regressed a raw 512->1024 MLP.

Also note that cosine-to-target is NOT a usable metric here: a constant vector
scores 0.6147, higher than HCMA's own 0.5451, because every CLIP image embedding
shares a strong common direction (random pairs already sit at 0.378).

This script therefore keeps the user's architecture (cross-subject latent +
EEG encoder + parallel semantic/perceptual towers + multi-granularity heads +
CFM condition synthesiser) but replaces the objectives with:

  1. DIFFERENTIABLE SOFT MEMORY over the 16540 train image embeddings, with the
     true positive KEPT RETRIEVABLE (no leave-one-out mask). The mask is only
     meaningful when query and gallery are the same object in the same space, as
     in HCMA's ``rag_soft5`` (query == gallery == z_eeg_train), where row i
     matches itself at similarity 1.0 and returns its own target -- which is why
     ``rag_soft5_train`` reads 0.8891 against its own target. Here the query is a
     LEARNED projection while the gallery holds IMAGE embeddings, so no identity
     shortcut exists, and masking the true positive removes the only correct
     answer: the softmax average pins to the bank mean (measured mem_to_ip 0.639
     against a constant baseline of 0.6147, i.e. no better than a constant).
  2. CLASS-LEVEL CONTRASTIVE over the 1655 train concepts (derived from the
     unique rows of sem_concept_tmpl_train; the 200 test concepts are disjoint
     from the 1655 train concepts, verified -- intersection is empty), plus
     in-batch InfoNCE with MULTI-POSITIVE masking. A LOSO batch holds the same
     target image once per training subject, so a naive single-positive InfoNCE
     would treat another subject's row for the SAME image as a negative and push
     them apart. The InfoNCE term is what actually buys discriminability: the
     "be close to your own target" terms alone can be satisfied while the output
     drifts to the bank mean.
  3. Multi-granularity heads (overall / subject / background / detail) are
     FUSED into the IP embedding, so the user's design is functionally required
     rather than decorative:
         q_direct = h_ip(s)
         q_fused  = h_ip(s) + h_fuse([overall; subject; background; detail])
         q_mem    = softmax(l2(q)*l2(bank).T / tau) @ bank
  4. CFM conditioner that trans\ports the joint dual-tower code into a
     distribution over IP embeddings (sampled), per the user's design.

LEAK-FREE GUARANTEES
--------------------
  * test EEG is never touched during training; the memory bank and the class
    gallery are built ONLY from the 16540 train images / train concepts.
  * the 200 test concepts are disjoint from the 1655 train concepts, so the
    model cannot know a test class name.
  * checkpoint selection uses a held-out slice of TRAIN target indices, never
    the test set.
  * in-batch InfoNCE and the class gallery use train rows only; the memory bank
    is the 16540 train image embeddings only.
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

NB_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------- utils

def l2t(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(1e-8)


def l2n(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8, None)


def mlp(i: int, h: int, o: int, layers: int = 2, drop: float = 0.0) -> nn.Sequential:
    mods: list[nn.Module] = [nn.Linear(i, h), nn.GELU()]
    if drop > 0:
        mods.append(nn.Dropout(drop))
    for _ in range(layers - 1):
        mods += [nn.Linear(h, h), nn.GELU()]
        if drop > 0:
            mods.append(nn.Dropout(drop))
    mods.append(nn.Linear(h, o))
    return nn.Sequential(*mods)


class G2FNet(nn.Module):
    """Parallel dual tower + multi-granularity fusion + differentiable memory + CFM."""

    def __init__(self, in_dim: int = 512, code: int = 768, img_dim: int = 1280,
                 txt_dim: int = 1024, ip_dim: int = 1024, ch: int = 4,
                 spatial: int = 64, n_tex: int = 17, per_layers: int = 3,
                 drop: float = 0.15):
        super().__init__()
        self.ch, self.spatial, self.n_tex = ch, spatial, n_tex
        self.ip_dim = ip_dim

        # --- EEG trunk, split into the two towers
        self.sem_trunk = mlp(in_dim, code, code, 2, drop)
        self.per_trunk = mlp(in_dim, code, code, per_layers, drop)

        # --- semantic tower: image + 4 granularities + IP head + class head
        self.h_image = mlp(code, code, img_dim, 1, drop)
        self.h_overall = mlp(code, code, txt_dim, 1, drop)
        self.h_subject = mlp(code, code, txt_dim, 1, drop)
        self.h_background = mlp(code, code, txt_dim, 1, drop)
        self.h_detail = mlp(code, code, txt_dim, 1, drop)
        self.h_ip = mlp(code, code, ip_dim, 1, drop)
        # granularity -> IP fusion: makes the multi-granularity design functional
        self.h_fuse = mlp(4 * txt_dim, code, ip_dim, 1, drop)

        # --- perceptual tower: low-frequency latent + texture statistics
        self.h_struct = mlp(code, code, ch * spatial * spatial, 2, drop)
        self.h_texture = mlp(code, code, ch * n_tex, 1, drop)

        # --- CFM conditioner over the joint dual-tower code
        joint = img_dim + 4 * txt_dim + ip_dim + ch * n_tex + ch * 16
        self.cond_proj = nn.Sequential(nn.Linear(joint, code), nn.GELU(),
                                       nn.Linear(code, code))
        self.vel = nn.Sequential(nn.Linear(ip_dim + 1 + code, code), nn.GELU(),
                                 nn.Linear(code, code), nn.GELU(),
                                 nn.Linear(code, ip_dim))

    # ---- forward halves, so the same code serves training and export
    def sem_heads(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        s = self.sem_trunk(x)
        g = {
            "overall": self.h_overall(s),
            "subject": self.h_subject(s),
            "background": self.h_background(s),
            "detail": self.h_detail(s),
        }
        direct = self.h_ip(s)
        fused = direct + self.h_fuse(torch.cat([l2t(g["overall"]), l2t(g["subject"]),
                                                l2t(g["background"]), l2t(g["detail"])], -1))
        return {"image": self.h_image(s), "direct": direct, "fused": fused, **g, "_s": s}

    def per_heads(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        p = self.per_trunk(x)
        return {"struct": self.h_struct(p).view(-1, self.ch, self.spatial, self.spatial),
                "texture": self.h_texture(p).view(-1, self.ch, self.n_tex),
                "_p": p}

    def joint(self, sem: dict[str, torch.Tensor], per: dict[str, torch.Tensor]) -> torch.Tensor:
        parts = [l2t(sem["image"]), l2t(sem["overall"]), l2t(sem["subject"]),
                 l2t(sem["background"]), l2t(sem["detail"]), l2t(sem["fused"]),
                 l2t(per["texture"].flatten(1)),
                 l2t(F.adaptive_avg_pool2d(per["struct"], (4, 4)).flatten(1))]
        return self.cond_proj(torch.cat(parts, -1))

    # ---- differentiable top-k soft memory
    # NO leave-one-out mask by default. The mask is only meaningful when the
    # query and the gallery live in the SAME space and are the same object, as
    # in HCMA's rag_soft5 (query == gallery == z_eeg_train), where row i
    # retrieves itself at sim 1.0 and returns its own target -- that is the
    # degenerate shortcut, and it is why rag_soft5_train reads 0.8891 against its
    # own target. Here the query is a LEARNED projection h_ip(sem(z)) while the
    # gallery holds IMAGE embeddings, so no identity shortcut exists, and masking
    # the true positive removes the only correct answer.
    #
    # top-k rather than a full softmax over all 16540 entries: with a flat
    # similarity profile a full softmax averages the whole bank and returns its
    # mean. Measured (2026-09-11): full softmax gave mem_to_ip 0.6426 against a
    # CONSTANT baseline of 0.6147 -- i.e. the retrieved vector carried almost no
    # per-trial information -- and 2-way 0.65 against 0.52 for a constant.
    def memory(self, q: torch.Tensor, bank: torch.Tensor, tau: float, k: int = 16,
               loo_target_idx: torch.Tensor | None = None) -> torch.Tensor:
        qn, bn = l2t(q), l2t(bank)
        sim = qn @ bn.T
        if loo_target_idx is not None:
            sim = sim.clone()
            sim[torch.arange(sim.shape[0], device=sim.device), loo_target_idx] = -1e4
        topv, topi = sim.topk(min(k, sim.shape[1]), dim=-1)
        out = (torch.softmax(topv / tau, dim=-1).unsqueeze(-1) * bn[topi]).sum(1)
        return l2t(out)

    # ---- CFM velocity field; both ends scaled by sqrt(d) so the MSE is O(1)
    def vel_field(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.vel(torch.cat([x, t[:, None], cond], -1))

    @torch.no_grad()
    def sample_ip(self, cond: torch.Tensor, steps: int, seed: int) -> torch.Tensor:
        d = self.ip_dim
        g = torch.Generator(device=cond.device).manual_seed(seed)
        x = torch.randn(cond.shape[0], d, device=cond.device, generator=g)
        dt = 1.0 / steps
        for k in range(steps):
            t = torch.full((cond.shape[0],), k * dt, device=cond.device)
            x = x + dt * self.vel_field(x, t, cond)
        return l2t(x)


def cfm_loss(pred_v: torch.Tensor, x1: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred_v, x1 - x0)


def var_band(pred: torch.Tensor, target: torch.Tensor, lo: float = 0.4,
             hi: float = 2.0) -> torch.Tensor:
    """TWO-sided anti-degeneracy: keep predicted per-dim dispersion inside a band.

    g2_train.py used a one-sided floor with a fixed target of 0.5 for every head,
    which over-demanded the IP head and ignored the structure heads. Here the
    band is derived from each target's own dispersion. The upper bound matters:
    with only a floor the structural head drifted to 6.1x the target's
    dispersion, which cannot be corrected by the cosine objective because that
    is scale-invariant.
    """
    tn = target.std(0).norm()
    pn = pred.std(0).norm()
    return (F.relu(lo * tn - pn) + F.relu(pn - hi * tn)) / (tn + 1e-8)


def multi_pos_nce(pred: torch.Tensor, targ: torch.Tensor, tix: torch.Tensor,
                  tau: float) -> torch.Tensor:
    """In-batch InfoNCE with MULTI-positive masking.

    A LOSO batch contains the same target image once per training subject, so a
    naive single-positive InfoNCE would treat a different subject's row for the
    SAME image as a negative and actively push them apart. All rows sharing a
    target index are therefore treated as positives.

    This is the term that actually buys discriminability: a pure
    "be close to your own target" objective can be satisfied while the output
    collapses toward the bank mean (measured: 0.639 versus a constant baseline
    of 0.6147), because nothing forces separation from other images.
    """
    logits = (l2t(pred) @ l2t(targ).T) / tau
    pos = tix[:, None] == tix[None, :]
    neg = torch.full_like(logits, -1e9)
    lse_pos = torch.logsumexp(torch.where(pos, logits, neg), dim=-1)
    lse_all = torch.logsumexp(logits, dim=-1)
    return (lse_all - lse_pos).mean()


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-subjects", type=int, nargs="+", required=True)
    ap.add_argument("--test-subject", type=int, required=True)
    ap.add_argument("--z-root", type=str, default=str(NB_ROOT / "outputs/hcma_10subj"))
    ap.add_argument("--targets-dir", type=str, default=str(NB_ROOT / "outputs/g2/targets"))
    ap.add_argument("--ip-train-npy", type=str,
                    default="/project/peilab/why/eeg-brainit/outputs/atm_bridge/clip_img_train_1024.npy")
    ap.add_argument("--ip-test-npy", type=str,
                    default="/project/peilab/why/eeg-brainit/outputs/atm_bridge/clip_img_test_1024.npy")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=26)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--tau-nce", type=float, default=0.07)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--var-frac", type=float, default=0.40)
    # 0 = keep the true positive retrievable (correct here); 1 = HCMA-style mask
    ap.add_argument("--loo-memory", type=int, default=0)
    ap.add_argument("--w-mem", type=float, default=1.0)
    ap.add_argument("--w-nce", type=float, default=1.0)
    ap.add_argument("--w-cls", type=float, default=1.0)
    ap.add_argument("--w-cfm", type=float, default=0.5)
    ap.add_argument("--w-gran", type=float, default=0.4)
    ap.add_argument("--w-tex", type=float, default=0.4)
    ap.add_argument("--w-struct", type=float, default=1.0)
    ap.add_argument("--ode-steps", type=int, default=32)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", type=int, default=1)
    # smoke-test only: shrink the row sets; never used in production runs
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--limit-val", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    (out / "conds").mkdir(parents=True, exist_ok=True)
    T = Path(args.targets_dir)
    stag = f"sub-{args.test_subject:02d}"

    # -------------------------------------------------- load frozen targets
    ip_bank_np = np.load(args.ip_train_npy).astype(np.float32)      # (16540,1024)
    ip_test_np = np.load(args.ip_test_npy).astype(np.float32)       # (200,1024)
    bank = torch.from_numpy(l2n(ip_bank_np)).to(dev)
    n_bank = bank.shape[0]

    # class gallery + labels from the train concept-text targets only (leak-free)
    ctmp = np.load(T / "sem_concept_tmpl_train.npy").astype(np.float32)
    uniq, inv = np.unique(l2n(ctmp), axis=0, return_inverse=True)
    cls_gallery = torch.from_numpy(uniq).to(dev)
    labels_all = torch.from_numpy(inv.astype(np.int64))
    n_cls = cls_gallery.shape[0]

    sem_keys = ["image", "overall", "subject", "background", "detail"]
    tgt: dict[str, torch.Tensor] = {}
    for k in sem_keys:
        tgt[k] = torch.from_numpy(np.load(T / f"sem_{k}_train.npy").astype(np.float32))
    tgt["struct"] = torch.from_numpy(np.load(T / "perc_struct_train.npy").astype(np.float32))
    tgt["texture"] = torch.from_numpy(np.load(T / "perc_texture_train.npy").astype(np.float32))
    tgt["ip"] = torch.from_numpy(l2n(ip_bank_np))
    # move every target to the device ONCE. The training loop indexes these by
    # target row, so per-step host->device transfers would otherwise dominate.
    tgt = {k: v.to(dev) for k, v in tgt.items()}
    labels_all = labels_all.to(dev)
    cls_gallery = cls_gallery.to(dev)

    # -------------------------------------------------- gather train rows
    zs, idxs = [], []
    for s in args.train_subjects:
        z = np.load(f"{args.z_root}/sub-{s:02d}/zret/z_eeg_proj_train.npy").astype(np.float32)
        if z.shape[0] != n_bank:
            raise SystemExit(f"[FATAL] sub-{s:02d} z rows {z.shape[0]} != bank {n_bank}")
        zs.append(z)
        idxs.append(np.arange(n_bank))
    Ztr = np.concatenate(zs, 0)                     # (n_subj*16540, 512)
    tidx = np.concatenate(idxs, 0)                  # target/bank row of each training row
    n_tr = Ztr.shape[0]

    # held-out slice of TRAIN target indices -> leak-free checkpoint selection
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_bank)
    n_val = int(n_bank * args.val_frac)
    val_t = np.zeros(n_bank, dtype=bool)
    val_t[perm[:n_val]] = True
    is_val = val_t[tidx]

    tr_sel = np.where(~is_val)[0]
    va_sel = np.where(is_val)[0]
    # for val rows keep at most one row per subject so val is not 10x duplicated
    va_sel = va_sel[np.arange(len(va_sel)) % max(1, len(args.train_subjects)) == 0]
    if args.limit_train > 0:
        tr_sel = tr_sel[:args.limit_train]
    if args.limit_val > 0:
        va_sel = va_sel[:args.limit_val]

    Zte_np = np.load(f"{args.z_root}/{stag}/zret/z_eeg_proj_test.npy").astype(np.float32)
    ite_np = l2n(ip_test_np)

    Ztr_t = torch.from_numpy(Ztr)
    Zte_t = torch.from_numpy(Zte_np).to(dev)
    print(f"[g2f] train rows {n_tr} (valsel {len(va_sel)}, trainsel {len(tr_sel)}) "
          f"| bank {n_bank} | classes {n_cls} | test {Zte_t.shape}")

    model = G2FNet(drop=args.dropout).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[g2f] params {n_par/1e6:.2f}M  device {dev}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = max(1, len(tr_sel) // args.batch_size)
    total_steps = args.epochs * steps
    # OneCycleLR divides by the width of its warmup phase, which is degenerate for
    # very small schedules (smoke tests). Production runs (26 epochs x ~260 steps)
    # always take the cyclic path.
    if total_steps >= 20:
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.25)
    else:
        sched = torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0, total_iters=total_steps)

    # target dispersion floors for the anti-collapse hinge
    floors = {k: args.var_frac * float(tgt[k].std(0).norm()) for k in ("struct", "texture", "ip")}
    print("[g2f] var floors " + " ".join(f"{k}={v:.2f}" for k, v in floors.items()))

    best = {"score": -1e9, "epoch": -1}
    last_path, best_path = out / "last.pth", out / "best.pth"
    ck_path = out / "ckpt_meta.json"
    start_ep = 0
    if args.resume and last_path.is_file():
        try:
            ck = torch.load(last_path, map_location=dev, weights_only=False)
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
            start_ep = ck["epoch"] + 1
            best = ck.get("best", best)
            print(f"[g2f] resumed at epoch {start_ep} (best {best})")
        except Exception as e:                                     # noqa: BLE001
            print(f"[g2f] resume failed ({e}); starting fresh")

    def evaluate(sel: np.ndarray, tag: str) -> dict[str, float]:
        model.eval()
        acc: dict[str, list[float]] = {}
        with torch.no_grad():
            for i in range(0, len(sel), 1024):
                rows = sel[i:i + 1024]
                x = Ztr_t[rows].to(dev)
                ti = torch.from_numpy(tidx[rows]).to(dev)
                sem = model.sem_heads(x)
                per = model.per_heads(x)
                qf = l2t(sem["fused"])
                mem = model.memory(qf, bank, args.tau,
                                   loo_target_idx=(ti if args.loo_memory else None))
                tgtip = tgt["ip"][ti]
                lg = qf @ l2t(cls_gallery).T
                top1 = (lg.argmax(1) == labels_all[ti]).float().mean()
                v: dict[str, torch.Tensor] = {
                    "image": (l2t(sem["image"]) * l2t(tgt["image"][ti])).sum(-1).mean(),
                    "fused_to_ip": (qf * tgtip).sum(-1).mean(),
                    "mem_to_ip": (mem * tgtip).sum(-1).mean(),
                    "nce": multi_pos_nce(qf, tgtip, ti, args.tau_nce),
                    "cls_top1": top1,
                    "struct": (l2t(per["struct"].flatten(1)) *
                               l2t(tgt["struct"][ti].flatten(1))).sum(-1).mean(),
                    "struct_std": per["struct"].std(0).norm() / (tgt["struct"].std(0).norm() + 1e-8),
                    "texture_std": per["texture"].std(0).norm() / (tgt["texture"].std(0).norm() + 1e-8),
                }
                for k, t in v.items():
                    acc.setdefault(k, []).append(float(t.detach()))
        m = {k: float(np.mean(v)) for k, v in acc.items()}
        # The score keeps the leak-free semantic objective dominant and weights the
        # structural term down: g2_train.py let structure outbid semantics in
        # selection and the semantic output regressed. `nce` (lower better) enters
        # negatively because it is the term that forces discriminability.
        m["score"] = 0.30 * m["mem_to_ip"] + 0.15 * m["fused_to_ip"] + 0.20 * m["cls_top1"] \
            + 0.10 * m["image"] + 0.20 * m["struct"] - 0.15 * m["nce"]
        print(f"[{tag}] " + " ".join(f"{k}={m[k]:.4f}" for k in
                                     ("mem_to_ip", "fused_to_ip", "nce", "cls_top1", "image",
                                      "struct", "struct_std", "texture_std", "score")))
        model.train()
        return m

    history: list[dict] = []
    for ep in range(start_ep, args.epochs):
        model.train()
        perm2 = rng.permutation(len(tr_sel))
        run: dict[str, float] = {}
        nb = 0
        t0 = time.time()
        for b in range(steps):
            rows = tr_sel[perm2[b * args.batch_size:(b + 1) * args.batch_size]]
            if len(rows) < 2:
                continue
            x = Ztr_t[rows].to(dev)
            ti = torch.from_numpy(tidx[rows]).to(dev)
            sem = model.sem_heads(x)
            per = model.per_heads(x)

            # 1. multi-granularity + image regression
            l_sem = 0.0
            for k in ("image", "overall", "subject", "background", "detail"):
                wt = 1.0 if k == "image" else args.w_gran
                l_sem = l_sem + wt * (1 - (l2t(sem[k]) * l2t(tgt[k][ti])).sum(-1)).mean()
            l_sem = l_sem / (1.0 + 4 * args.w_gran)

            # 2. IP regression (direct vs fused) + differentiable memory.
            # The memory bank holds IMAGE embeddings and the query is a learned
            # projection, so the target IS the correct answer and must stay
            # retrievable (see the note on `memory`).
            ip_t = tgt["ip"][ti]
            loo = ti if args.loo_memory else None
            l_ip = (1 - (l2t(sem["direct"]) * ip_t).sum(-1)).mean() \
                + (1 - (l2t(sem["fused"]) * ip_t).sum(-1)).mean()
            mem = model.memory(sem["fused"], bank, args.tau, loo_target_idx=loo)
            l_mem = (1 - (mem * ip_t).sum(-1)).mean()

            # 3. discriminability: in-batch InfoNCE (multi-positive) + class-level
            # contrastive. Without these two the "match your own target" terms can
            # be satisfied while every output drifts toward the bank mean.
            l_nce = multi_pos_nce(sem["fused"], ip_t, ti, args.tau_nce) \
                + multi_pos_nce(mem, ip_t, ti, args.tau_nce)
            lg = l2t(sem["fused"]) @ l2t(cls_gallery).T
            l_cls = F.cross_entropy(lg / args.tau, labels_all[ti])

            # 4. perceptual tower
            l_struct = (1 - (l2t(per["struct"].flatten(1)) *
                             l2t(tgt["struct"][ti].flatten(1))).sum(-1)).mean()
            l_tex = (1 - (l2t(per["texture"].flatten(1)) *
                          l2t(tgt["texture"][ti].flatten(1))).sum(-1)).mean()

            # 5. CFM conditioner: both ends scaled by sqrt(d), see docstring
            cond = model.joint(sem, per)
            d = model.ip_dim
            x1 = (ip_t * math.sqrt(d)).detach()
            x0 = torch.randn_like(x1)
            t = torch.rand(x1.shape[0], device=dev)
            xt = (1 - t)[:, None] * x0 + t[:, None] * x1
            l_cfm = cfm_loss(model.vel_field(xt, t, cond), x1, x0)

            # 6. anti-degeneracy band (two-sided, target-derived)
            hinges = var_band(per["struct"], tgt["struct"][ti]) \
                + var_band(per["texture"], tgt["texture"][ti]) \
                + var_band(sem["fused"], ip_t)

            loss = l_sem + 0.8 * l_ip + args.w_mem * l_mem + args.w_nce * l_nce \
                + args.w_cls * l_cls + args.w_struct * l_struct + args.w_tex * l_tex \
                + args.w_cfm * l_cfm + 0.5 * hinges

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()

            for k, v in (("loss", loss), ("sem", l_sem), ("ip", l_ip), ("mem", l_mem),
                         ("nce", l_nce), ("cls", l_cls), ("struct", l_struct),
                         ("tex", l_tex), ("cfm", l_cfm)):
                run[k] = run.get(k, 0.0) + float(v.detach())
            nb += 1

        tr_m = {k: v / max(nb, 1) for k, v in run.items()}
        va_m = evaluate(va_sel, f"ep{ep} val")
        rec = {"epoch": ep, "lr": sched.get_last_lr()[0], "sec": round(time.time() - t0, 1),
               **{f"tr_{k}": round(v, 4) for k, v in tr_m.items()},
               **{f"va_{k}": round(v, 4) for k, v in va_m.items()}}
        history.append(rec)

        if va_m["score"] > best["score"]:
            best = {"score": va_m["score"], "epoch": ep, **{f"va_{k}": v for k, v in va_m.items()}}
            torch.save({"model": model.state_dict(), **best}, best_path)

        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": ep, "best": best}, last_path)

    # -------------------------------------------------- export (test rows only)
    if best_path.is_file():
        ck = torch.load(best_path, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        print(f"[g2f] loaded best epoch {ck.get('epoch')} score {ck.get('score'):.4f}")
    model.eval()
    ip_te_t = torch.from_numpy(ite_np).to(dev)
    outs: dict[str, list[np.ndarray]] = {}
    with torch.no_grad():
        for i in range(0, Zte_t.shape[0], 256):
            x = Zte_t[i:i + 256]
            sem = model.sem_heads(x)
            per = model.per_heads(x)
            qd, qf = l2t(sem["direct"]), l2t(sem["fused"])
            mem = model.memory(qf, bank, args.tau)             # NO LOO mask at test
            cond = model.joint(sem, per)
            cfm = model.sample_ip(cond, args.ode_steps, args.seed)
            vals = {"ip_direct": qd, "ip_fused": qf, "ip_mem": mem, "ip_cfm": cfm,
                    "lf_latent": per["struct"], "texture": per["texture"],
                    "image": l2t(sem["image"])}
            for k, v in vals.items():
                outs.setdefault(k, []).append(v.float().cpu().numpy())

    report: dict = {"protocol": "g2f", "test_subject": stag,
                    "train_subjects": args.train_subjects,
                    "params_m": round(n_par / 1e6, 3), "best": best, "history": history,
                    "n_classes": int(n_cls), "n_bank": int(n_bank)}
    for k, v in outs.items():
        a = np.concatenate(v, 0).astype(np.float32)
        if k.startswith("ip_"):
            a = l2n(a)
            report[f"{k}_cos_to_ip"] = float((a * ite_np).sum(-1).mean())
        np.save(out / "conds" / f"{k}_test.npy", a)

    # honest discrimination of the exported IP conditions (200-way, leak-free)
    def disc(a: np.ndarray) -> dict[str, float]:
        s = l2n(a) @ ite_np.T
        n = len(s)
        r = np.random.default_rng(0).permutation(n)
        ok = np.arange(n) != r
        return {"top1": float(np.mean([i in np.argsort(-s[i])[:1] for i in range(n)])),
                "top5": float(np.mean([i in np.argsort(-s[i])[:5] for i in range(n)])),
                "twoway": float(np.mean(s[np.arange(n), np.arange(n)][ok] >
                                        s[np.arange(n), r][ok]))}
    for k in ("ip_direct", "ip_fused", "ip_mem", "ip_cfm"):
        report[f"{k}_disc"] = disc(outs[k][0] if isinstance(outs[k], list) else outs[k])
    report["note"] = ("cos_to_ip is NOT usable as a quality metric here (a constant "
                      "vector scores 0.6147); use *_disc (top1/top5/twoway).")
    (out / "g2f_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("[g2f] " + json.dumps({k: v for k, v in report.items()
                                 if k.endswith("_disc") or k.endswith("_cos_to_ip")}, indent=2))
    print(f"[g2f] done -> {out}")


if __name__ == "__main__":
    main()
