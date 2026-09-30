#!/usr/bin/env python3
"""LOSO training for POP-Former + Cross-Brain Orthogonal Attention (CBOA).

Training strategy (scaled)
--------------------------
  • Model: d=512, 8 layers, 8 heads (~35M+ params)
  • Full THINGS-EEG2 train trials (max_per_sub=0)
  • 50 epochs, cosine LR with warmup, AdamW, grad clip
  • Epoch loop: encode → update memory bank + id basis → CBOA train
  • Losses: CLIP MSE+InfoNCE (+ATM distill) + identity constraints
      - CE(probe_ap, subject): encourage identity in aperiodic
      - CE(probe_per, subject) via GRL: discourage identity in periodic
  • Test: two-pass encode (null mem → CLIP neighbor memory → CBOA)

Outputs under outputs/pop_former/ (does not touch subject_align / st_gate).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("XFORMERS_DISABLED", "1")
os.environ.setdefault("HOME", "/project/peilab/why/cache/eeg-brainit/xdg-home")
os.environ.setdefault("HF_HOME", "/project/peilab/why/cache/eeg-brainit/hf")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eeg_brainit.data.things_eeg2_adapt import ThingsEEG2SubjectDataset, collate_batch
from eeg_brainit.models.pop_former import CrossBrainMemoryBank, POPFormer
from eeg_brainit.utils.config import ensure_dirs


ALL_SUBJECTS = [f"sub-{i:02d}" for i in range(1, 11)]
N_TRAIN_CLASSES = 1654


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lambd * g, None


def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sub_idx(name: str) -> int:
    return int(name.replace("sub-", "")) - 1


def l2_np(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


def retrieval_metrics(queries: np.ndarray, gallery: np.ndarray) -> dict[str, float]:
    q = l2_np(queries)
    g = l2_np(gallery)
    sim = q @ g.T
    n = sim.shape[0]
    gt = np.arange(n)
    order = np.argsort(-sim, axis=1)
    ranks = np.argmax(order == gt[:, None], axis=1)
    top1 = float((ranks < 1).mean())
    top5 = float((ranks < 5).mean())
    return {"top1": top1, "top5": top5}


def atm_ceiling(subject: str, atm_dir: Path, gallery: np.ndarray) -> dict[str, float]:
    p = atm_dir / f"{subject}_test_eeg_1024.npy"
    if not p.is_file():
        return {"top1": float("nan"), "top5": float("nan")}
    return retrieval_metrics(np.load(p).astype(np.float32), gallery)


def cosine_warmup_lr(step: int, warmup: int, total: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * float(step + 1) / float(max(warmup, 1))
    progress = (step - warmup) / float(max(total - warmup, 1))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def compute_loss(model, out, clip_img, subject_id, atm_emb, args, grl_lambda):
    tgt = clip_img.float()
    mse = F.mse_loss(out["clip_raw"], tgt)
    logits = out["clip_emb"] @ F.normalize(tgt, dim=-1).T / args.temp
    nce = F.cross_entropy(logits, torch.arange(logits.size(0), device=logits.device))
    loss = mse + args.lambda_nce * nce
    stats = {"mse": float(mse), "nce": float(nce)}

    if atm_emb is not None and args.lambda_atm > 0:
        la = F.mse_loss(out["clip_emb"], F.normalize(atm_emb.float(), dim=-1))
        loss = loss + args.lambda_atm * la
        stats["atm"] = float(la)

    ce_ap = F.cross_entropy(model.probe_ap(out["pooled_ap"]), subject_id.long())
    per_feat = grad_reverse(out["pooled_per"], grl_lambda)
    ce_per = F.cross_entropy(model.probe_per(per_feat), subject_id.long())
    loss = loss + args.lambda_id_ap * ce_ap + args.lambda_id_per * ce_per
    stats["id_ap"] = float(ce_ap)
    stats["id_per"] = float(ce_per)
    return loss, stats


@torch.no_grad()
def refresh_memory_and_basis(model, bank, loader, device, args):
    model.eval()
    feats, sids = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        t_per, t_ap = model.encode_streams(batch["eeg"])
        bank.update(t_per, batch["label"], momentum=args.mem_momentum)
        feats.append(model.pool(t_ap))
        sids.append(batch["subject_id"])
    if feats:
        model.update_id_basis(torch.cat(feats, 0), torch.cat(sids, 0), momentum=args.id_momentum)
    model.train()


def train_fold(model, bank, train_subs, args, device, tag: str):
    dss = [
        ThingsEEG2SubjectDataset(
            ROOT / args.eeg_root,
            ROOT / args.atm_bridge_dir,
            sub,
            split="train",
            max_samples=args.max_per_sub,
            seed=args.seed,
        )
        for sub in train_subs
    ]
    loader = DataLoader(
        ConcatDataset(dss),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_batch,
        num_workers=0,
    )
    # smaller loader for memory refresh (subset)
    refresh_loader = DataLoader(
        ConcatDataset(dss),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_batch,
        num_workers=0,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_ep = max(len(loader), 1)
    total_steps = steps_per_ep * args.epochs
    warmup = int(args.warmup_epochs * steps_per_ep)
    global_step = 0

    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for ep in range(1, args.epochs + 1):
        # refresh bank / id basis each epoch (skip ep1 until after first pass)
        if ep == 1 or ep % args.mem_refresh_every == 0:
            # limited refresh batches for speed
            limited = []
            for i, batch in enumerate(refresh_loader):
                limited.append(batch)
                if i + 1 >= args.mem_refresh_batches:
                    break
            model.eval()
            with torch.no_grad():
                feats, sids = [], []
                for batch in limited:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    t_per, t_ap = model.encode_streams(batch["eeg"])
                    bank.update(t_per, batch["label"], momentum=args.mem_momentum)
                    feats.append(model.pool(t_ap))
                    sids.append(batch["subject_id"])
                if feats:
                    model.update_id_basis(torch.cat(feats, 0), torch.cat(sids, 0), momentum=args.id_momentum)
            model.train()

        # GRL schedule
        p = ep / max(args.epochs, 1)
        grl = float(args.grl_max * (2.0 / (1.0 + math.exp(-10 * p)) - 1.0))
        mem_drop = args.mem_dropout

        loss_sum = n = 0
        for batch in loader:
            lr = cosine_warmup_lr(global_step, warmup, total_steps, args.lr)
            for g in opt.param_groups:
                g["lr"] = lr

            batch = {k: v.to(device) for k, v in batch.items()}
            # memory lookup by class label
            if random.random() < mem_drop or not bank.ready:
                mem = model.null_mem.expand(batch["eeg"].size(0), -1, -1)
                use_mem = False
            else:
                mem = bank.lookup(batch["label"], model.null_mem)
                use_mem = True

            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(batch["eeg"], mem=mem, use_memory=use_mem, use_private=True)
                loss, stats = compute_loss(
                    model, out, batch["clip_img"], batch["subject_id"], batch.get("atm_emb"), args, grl
                )

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()

            # online memory update
            with torch.no_grad():
                t_per, _ = model.encode_streams(batch["eeg"])
                bank.update(t_per, batch["label"], momentum=args.mem_momentum)

            loss_sum += float(loss) * batch["eeg"].size(0)
            n += batch["eeg"].size(0)
            global_step += 1

        print(
            f"[{tag} ep{ep:02d}/{args.epochs}] loss={loss_sum/max(n,1):.4f} "
            f"lr={lr:.2e} grl={grl:.3f} mem_ready={bank.ready} "
            f"id_basis={bool(model.id_basis_ready)} params={model.num_parameters()/1e6:.1f}M",
            flush=True,
        )
    return model


@torch.no_grad()
def encode_test_twopass(model, bank, subject, gallery, clip_train, args, device):
    ds = ThingsEEG2SubjectDataset(
        ROOT / args.eeg_root,
        ROOT / args.atm_bridge_dir,
        subject,
        split="test",
        max_samples=0,
        seed=args.seed,
    )
    model.eval()
    clip_train_t = torch.from_numpy(clip_train).float().to(device)
    embs = []
    bs = args.eval_batch_size
    for i0 in range(0, len(ds), bs):
        samples = [ds[i] for i in range(i0, min(i0 + bs, len(ds)))]
        batch = collate_batch(samples)
        x = batch["eeg"].to(device)
        # pass 1: null memory
        out1 = model(x, mem=None, use_memory=False, use_private=True)
        # pass 2: CLIP-neighbor memory from train classes
        mem = bank.lookup_clip_neighbors(
            out1["clip_emb"], clip_train_t, topk=args.mem_topk, null=model.null_mem
        )
        out2 = model(x, mem=mem, use_memory=True, use_private=True)
        embs.append(out2["clip_emb"].cpu().numpy())
    return np.concatenate(embs, 0)


@torch.no_grad()
def encode_test_nomem(model, subject, args, device):
    ds = ThingsEEG2SubjectDataset(
        ROOT / args.eeg_root,
        ROOT / args.atm_bridge_dir,
        subject,
        split="test",
        max_samples=0,
        seed=args.seed,
    )
    model.eval()
    embs = []
    bs = args.eval_batch_size
    for i0 in range(0, len(ds), bs):
        samples = [ds[i] for i in range(i0, min(i0 + bs, len(ds)))]
        batch = collate_batch(samples)
        out = model(batch["eeg"].to(device), mem=None, use_memory=False, use_private=True)
        embs.append(out["clip_emb"].cpu().numpy())
    return np.concatenate(embs, 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eeg-root", default="data/processed/things-eeg2")
    p.add_argument("--atm-bridge-dir", default="outputs/atm_bridge")
    p.add_argument("--output-dir", default="outputs/pop_former/loso_v1")
    p.add_argument("--subjects", default=",".join(ALL_SUBJECTS))
    # model scale
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--mem-tokens", type=int, default=64)
    p.add_argument("--id-rank", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.1)
    # train strategy
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--warmup-epochs", type=float, default=3.0)
    p.add_argument("--max-per-sub", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--temp", type=float, default=0.07)
    p.add_argument("--lambda-nce", type=float, default=0.5)
    p.add_argument("--lambda-atm", type=float, default=0.2)
    p.add_argument("--lambda-id-ap", type=float, default=0.2)
    p.add_argument("--lambda-id-per", type=float, default=0.1)
    p.add_argument("--grl-max", type=float, default=1.0)
    p.add_argument("--mem-dropout", type=float, default=0.15)
    p.add_argument("--mem-momentum", type=float, default=0.05)
    p.add_argument("--id-momentum", type=float, default=0.1)
    p.add_argument("--mem-refresh-every", type=int, default=1)
    p.add_argument("--mem-refresh-batches", type=int, default=40)
    p.add_argument("--mem-topk", type=int, default=8)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--clip-dim", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-folds", type=int, default=0)
    args = p.parse_args()
    if args.no_amp:
        args.amp = False

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / args.output_dir
    ensure_dirs(out_dir, out_dir / "checkpoints", out_dir / "folds")
    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    gallery = np.load(ROOT / args.atm_bridge_dir / "clip_img_test_1024.npy").astype(np.float32)
    clip_train = np.load(ROOT / args.atm_bridge_dir / "clip_img_train_1024.npy").astype(np.float32)
    # class-level train CLIP: average 10 reps
    clip_train_cls = clip_train.reshape(N_TRAIN_CLASSES, 10, -1).mean(1).astype(np.float32)

    print(
        f"[INFO] POP-Former/CBOA device={device} d={args.d_model} L={args.n_layers} "
        f"H={args.n_heads} epochs={args.epochs} bs={args.batch_size} lr={args.lr}",
        flush=True,
    )

    fold_rows = []
    test_subs = subjects[: args.max_folds] if args.max_folds > 0 else subjects
    for test_sub in test_subs:
        train_subs = [s for s in subjects if s != test_sub]
        print(f"\n[FOLD] test={test_sub} train={len(train_subs)}", flush=True)
        model = POPFormer(
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            clip_dim=args.clip_dim,
            dropout=args.dropout,
            n_subjects=10,
            mem_tokens=args.mem_tokens,
            id_rank=args.id_rank,
        ).to(device)
        print(f"[INFO] trainable params={model.num_parameters()/1e6:.2f}M", flush=True)
        bank = CrossBrainMemoryBank(
            N_TRAIN_CLASSES, args.d_model, mem_tokens=args.mem_tokens, device=device
        )
        model = train_fold(model, bank, train_subs, args, device, tag=f"pop-{test_sub}")
        ck = out_dir / "checkpoints" / f"pop_{test_sub}.pt"
        torch.save({"model": model.state_dict(), "test_sub": test_sub, "args": vars(args)}, ck)

        ceiling = atm_ceiling(test_sub, ROOT / args.atm_bridge_dir, gallery)
        q_nomem = encode_test_nomem(model, test_sub, args, device)
        q_cboa = encode_test_twopass(model, bank, test_sub, gallery, clip_train_cls, args, device)
        m_nomem = retrieval_metrics(q_nomem, gallery)
        m_cboa = retrieval_metrics(q_cboa, gallery)
        row = {
            "test_subject": test_sub,
            "atm_ceiling": ceiling,
            "settings": {
                "no_memory": m_nomem,
                "cboa_twopass": m_cboa,
            },
        }
        fold_rows.append(row)
        (out_dir / "folds" / f"{test_sub}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        print(
            f"[EVAL {test_sub}] nomem={m_nomem['top1']*100:.2f}% "
            f"cboa={m_cboa['top1']*100:.2f}% atm={ceiling['top1']*100:.2f}%",
            flush=True,
        )

    summary = {}
    for key in fold_rows[0]["settings"].keys():
        t1 = [r["settings"][key]["top1"] for r in fold_rows]
        t5 = [r["settings"][key]["top5"] for r in fold_rows]
        summary[key] = {
            "top1": {"mean": float(np.mean(t1)), "std": float(np.std(t1)), "per_subject": t1},
            "top5": {"mean": float(np.mean(t5)), "std": float(np.std(t5)), "per_subject": t5},
        }
    ceil1 = [r["atm_ceiling"]["top1"] for r in fold_rows]
    ceil5 = [r["atm_ceiling"]["top5"] for r in fold_rows]
    summary["atm_ceiling"] = {
        "top1": {"mean": float(np.nanmean(ceil1)), "std": float(np.nanstd(ceil1)), "per_subject": ceil1},
        "top5": {"mean": float(np.nanmean(ceil5)), "std": float(np.nanstd(ceil5)), "per_subject": ceil5},
    }
    (out_dir / "metrics.json").write_text(
        json.dumps({"folds": fold_rows, "summary": summary, "args": vars(args)}, indent=2),
        encoding="utf-8",
    )
    card = {
        k: {"top1_mean": v["top1"]["mean"], "top1_std": v["top1"]["std"], "top5_mean": v["top5"]["mean"]}
        for k, v in summary.items()
    }
    (out_dir / "JOB_COMPLETE.json").write_text(
        json.dumps({"status": "ok", "protocol": "POP-Former CBOA LOSO", "results": card}, indent=2),
        encoding="utf-8",
    )
    print("\n[SUMMARY]")
    for k, st in summary.items():
        print(
            f"  {k:16s} top1={st['top1']['mean']*100:.2f}±{st['top1']['std']*100:.2f}% "
            f"top5={st['top5']['mean']*100:.2f}%"
        )
    print("[OK]", out_dir / "JOB_COMPLETE.json")


if __name__ == "__main__":
    main()
