#!/usr/bin/env python3
"""LOSO training / evaluation for ST-GATE.

Protocol
--------
1) Multi-subject pretrain with episodic subject profiles:
   each batch item conditioned on M unlabeled same-subject SC-EEG (null dropout).
2) Held-out subject eval grid:
   M ∈ m_list (unlabeled train EEG → profile → transport+geometry)
   K ∈ k_list (optional labeled refine of hypernet only; projector frozen)
3) Report cosine retrieval Top-1/Top-5 (+ ATM ceiling). Geometry params from
   hypernet are used optionally via soft-CSLS logits at train time; at test we
   primarily report cosine on transported embeddings (stable), and optionally
   a geometry-aware score if --eval-geo.

Does NOT delete or overwrite subject_align / prior experiment outputs.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("XFORMERS_DISABLED", "1")
os.environ.setdefault("HOME", "/project/peilab/why/cache/eeg-brainit/xdg-home")
os.environ.setdefault("HF_HOME", "/project/peilab/why/cache/eeg-brainit/hf")
os.environ.setdefault("CC", "gcc")
os.environ.setdefault("CXX", "g++")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eeg_brainit.data.things_eeg2_adapt import ThingsEEG2SubjectDataset, collate_batch
from eeg_brainit.models.st_gate import (
    STGateModel,
    SubjectDiscriminator,
    soft_csls_logits,
    st_gate_losses,
)
from eeg_brainit.models.subject_context import SubjectContextBank
from eeg_brainit.utils.config import ensure_dirs


ALL_SUBJECTS = [f"sub-{i:02d}" for i in range(1, 11)]


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
    top1_idx = order[:, 0]
    counts = np.bincount(top1_idx, minlength=gallery.shape[0]).astype(np.float64)
    hub = 0.0 if counts.std() < 1e-8 else float(
        ((counts - counts.mean()) ** 3).mean() / (counts.std() ** 3 + 1e-8)
    )
    return {"top1": top1, "top5": top5, "hubness_skew": hub}


def atm_ceiling(subject: str, atm_bridge_dir: Path, gallery: np.ndarray) -> dict[str, float]:
    path = atm_bridge_dir / f"{subject}_test_eeg_1024.npy"
    if not path.is_file():
        return {"top1": float("nan"), "top5": float("nan"), "hubness_skew": float("nan")}
    return retrieval_metrics(np.load(path).astype(np.float32), gallery)


def adv_lambda(epoch: int, epochs: int, max_lambda: float) -> float:
    if max_lambda <= 0:
        return 0.0
    p = epoch / max(epochs, 1)
    return float(max_lambda * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0))


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]


def train_st_gate(model, disc, train_subs, bank, args, device, tag: str) -> STGateModel:
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
    params = list(model.parameters()) + list(disc.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    model.train()
    disc.train()
    for ep in range(1, args.epochs + 1):
        loss_sum = n = 0
        grl = adv_lambda(ep, args.epochs, args.grl_max)
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            use_null = random.random() < args.null_profile_p
            if use_null:
                out = model(batch["eeg"], use_null_profile=True)
            else:
                ctx = bank.sample_batch(batch["subject_id"], m=args.ctx_m, device=device)
                out = model(batch["eeg"], ctx_eeg=ctx)
            loss, _ = st_gate_losses(
                out,
                batch["clip_img"],
                subject_id=batch["subject_id"],
                disc=disc,
                atm_emb=batch.get("atm_emb"),
                gallery=None,
                temp=args.temp,
                lambda_nce=args.lambda_nce,
                lambda_atm=args.lambda_atm,
                lambda_adv=args.lambda_adv,
                lambda_geo=args.lambda_geo,
                grl_lambda=grl,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += float(loss) * batch["eeg"].size(0)
            n += batch["eeg"].size(0)
        sched.step()
        print(
            f"[{tag} ep{ep:02d}] loss={loss_sum / max(n, 1):.4f} "
            f"grl={grl:.3f} lr={sched.get_last_lr()[0]:.2e}",
            flush=True,
        )
    return model


def pick_mk_indices(n: int, m: int, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    n_cls = n // 10
    k_idx: list[int] = []
    if k > 0:
        cls = rng.choice(n_cls, size=min(k, n_cls), replace=False)
        for c in cls:
            k_idx.append(int(c) * 10 + int(rng.randint(0, 10)))
        while len(k_idx) < k:
            j = int(rng.randint(0, n))
            if j not in k_idx:
                k_idx.append(j)
        k_idx = k_idx[:k]
    ban = set(k_idx)
    cand = [i for i in range(n) if i not in ban]
    if m > 0:
        m_idx = rng.choice(cand, size=min(m, len(cand)), replace=False).astype(np.int64)
    else:
        m_idx = np.array([], dtype=np.int64)
    return np.array(k_idx, dtype=np.int64), m_idx


def refine_kshot(model: STGateModel, subject: str, k_idx: np.ndarray, profile: torch.Tensor, args, device):
    if len(k_idx) == 0:
        return model
    ds = ThingsEEG2SubjectDataset(
        ROOT / args.eeg_root,
        ROOT / args.atm_bridge_dir,
        subject,
        split="train",
        max_samples=0,
        seed=args.seed,
    )
    subset = Subset(ds, [int(i) for i in k_idx])
    loader = DataLoader(
        subset,
        batch_size=min(args.adapt_batch_size, max(1, len(k_idx))),
        shuffle=True,
        drop_last=False,
        collate_fn=collate_batch,
        num_workers=0,
    )
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.transport_parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(list(model.transport_parameters()), lr=args.adapt_lr, weight_decay=0.01)
    steps = 0
    max_steps = args.adapt_steps
    model.train()
    while steps < max_steps:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch["eeg"], profile=profile.to(device))
            loss, _ = st_gate_losses(
                out,
                batch["clip_img"],
                temp=args.temp,
                lambda_nce=args.lambda_nce,
                lambda_atm=0.0,
                lambda_adv=0.0,
                lambda_geo=0.0,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            steps += 1
            if steps >= max_steps:
                break
    for p in model.parameters():
        p.requires_grad_(True)
    model.eval()
    return model


@torch.no_grad()
def encode_test(
    model: STGateModel,
    subject: str,
    profile: torch.Tensor | None,
    gallery: np.ndarray,
    args,
    device,
) -> tuple[np.ndarray, dict[str, float] | None]:
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
    geo_rows = []
    g = torch.from_numpy(l2_np(gallery)).float().to(device)
    bs = 64
    for i0 in range(0, len(ds), bs):
        samples = [ds[i] for i in range(i0, min(i0 + bs, len(ds)))]
        batch = collate_batch(samples)
        x = batch["eeg"].to(device)
        if profile is None:
            out = model(x, use_null_profile=True)
        else:
            out = model(x, profile=profile.to(device))
        embs.append(out["clip_emb"].cpu().numpy())
        if args.eval_geo:
            logits = soft_csls_logits(
                out["clip_emb"],
                g,
                out["k_row"],
                out["k_col"],
                out["csls_w"],
                out["log_tau"].exp(),
            )
            geo_rows.append(logits.cpu().numpy())
    q = np.concatenate(embs, 0)
    geo_met = None
    if args.eval_geo and geo_rows:
        sim = np.concatenate(geo_rows, 0)
        n = sim.shape[0]
        gt = np.arange(n)
        order = np.argsort(-sim, axis=1)
        ranks = np.argmax(order == gt[:, None], axis=1)
        geo_met = {
            "top1": float((ranks < 1).mean()),
            "top5": float((ranks < 5).mean()),
            "hubness_skew": float("nan"),
        }
    return q, geo_met


def eval_setting(base_model, subject, bank, gallery, m, k, args, device, seed) -> dict:
    model = copy.deepcopy(base_model).to(device)
    n_train = bank.n[subject]
    k_idx, m_idx = pick_mk_indices(n_train, m=m, k=k, seed=seed)
    profile = None
    if m > 0:
        ctx = torch.from_numpy(np.asarray(bank.eeg[subject][m_idx], dtype=np.float32)).to(device)
        model.eval()
        with torch.no_grad():
            profile = model.build_profile(ctx)
    if k > 0:
        if profile is None:
            profile = model.null_profile.detach().clone()
        model = refine_kshot(model, subject, k_idx, profile, args, device)
    q, geo_met = encode_test(model, subject, profile, gallery, args, device)
    met = retrieval_metrics(q, gallery)
    met.update({"m": m, "k": k, "n_m": int(len(m_idx)), "n_k": int(len(k_idx))})
    if geo_met is not None:
        met["geo_top1"] = geo_met["top1"]
        met["geo_top5"] = geo_met["top5"]
    return met


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eeg-root", default="data/processed/things-eeg2")
    parser.add_argument("--atm-bridge-dir", default="outputs/atm_bridge")
    parser.add_argument("--output-dir", default="outputs/st_gate/loso_v1")
    parser.add_argument("--subjects", default=",".join(ALL_SUBJECTS))
    parser.add_argument("--backbone", default="atm_style")
    parser.add_argument("--nz", type=int, default=256)
    parser.add_argument("--profile-dim", type=int, default=128)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--max-per-sub", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--temp", type=float, default=0.07)
    parser.add_argument("--lambda-nce", type=float, default=0.5)
    parser.add_argument("--lambda-atm", type=float, default=0.2)
    parser.add_argument("--lambda-adv", type=float, default=0.1)
    parser.add_argument("--lambda-geo", type=float, default=0.25)
    parser.add_argument("--grl-max", type=float, default=1.0)
    parser.add_argument("--ctx-m", type=int, default=32)
    parser.add_argument("--null-profile-p", type=float, default=0.25)
    parser.add_argument("--m-list", default="0,50,100")
    parser.add_argument("--k-list", default="0,1,5,10,20")
    parser.add_argument("--adapt-lr", type=float, default=5e-4)
    parser.add_argument("--adapt-steps", type=int, default=100)
    parser.add_argument("--adapt-batch-size", type=int, default=16)
    parser.add_argument("--eval-geo", action="store_true")
    parser.add_argument("--clip-dim", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-folds", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / args.output_dir
    ensure_dirs(out_dir, out_dir / "checkpoints", out_dir / "folds")
    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    m_list = parse_int_list(args.m_list)
    k_list = parse_int_list(args.k_list)
    gallery = np.load(ROOT / args.atm_bridge_dir / "clip_img_test_1024.npy").astype(np.float32)
    print(
        f"[INFO] ST-GATE device={device} nz={args.nz} rank={args.rank} "
        f"M={m_list} K={k_list} epochs={args.epochs}",
        flush=True,
    )

    fold_rows = []
    test_subs = subjects[: args.max_folds] if args.max_folds > 0 else subjects
    for test_sub in test_subs:
        train_subs = [s for s in subjects if s != test_sub]
        print(f"\n[FOLD] test={test_sub} train={len(train_subs)}", flush=True)
        bank = SubjectContextBank(ROOT / args.eeg_root, train_subs + [test_sub], split="train")
        model = STGateModel(
            backbone=args.backbone,
            nz=args.nz,
            profile_dim=args.profile_dim,
            rank=args.rank,
            clip_dim=args.clip_dim,
        ).to(device)
        disc = SubjectDiscriminator(model.nz, n_subjects=10).to(device)
        model = train_st_gate(model, disc, train_subs, bank, args, device, tag=f"stgate-{test_sub}")
        ck = out_dir / "checkpoints" / f"stgate_{test_sub}.pt"
        torch.save({"model": model.state_dict(), "test_sub": test_sub, "args": vars(args)}, ck)

        ceiling = atm_ceiling(test_sub, ROOT / args.atm_bridge_dir, gallery)
        settings = {}
        for m in m_list:
            for k in k_list:
                key = f"m{m}_k{k}"
                settings[key] = eval_setting(
                    model,
                    test_sub,
                    bank,
                    gallery,
                    m=m,
                    k=k,
                    args=args,
                    device=device,
                    seed=args.seed + sub_idx(test_sub) * 10007 + m * 17 + k,
                )
                s = settings[key]
                extra = ""
                if "geo_top1" in s:
                    extra = f" geo={s['geo_top1']*100:.2f}%"
                print(
                    f"  [{key}] top1={s['top1']*100:.2f}% top5={s['top5']*100:.2f}%{extra}",
                    flush=True,
                )
        row = {"test_subject": test_sub, "atm_ceiling": ceiling, "settings": settings}
        fold_rows.append(row)
        (out_dir / "folds" / f"{test_sub}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        z = settings.get("m0_k0", {})
        best = max(settings.items(), key=lambda kv: kv[1]["top1"])
        print(
            f"[EVAL {test_sub}] zero={z.get('top1', 0)*100:.2f}% "
            f"best={best[0]}:{best[1]['top1']*100:.2f}% atm={ceiling['top1']*100:.2f}%",
            flush=True,
        )

    keys = list(fold_rows[0]["settings"].keys())
    summary = {}
    for key in keys:
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
    blob = {"folds": fold_rows, "summary": summary}
    (out_dir / "metrics.json").write_text(json.dumps(blob, indent=2), encoding="utf-8")
    card = {
        k: {"top1_mean": v["top1"]["mean"], "top1_std": v["top1"]["std"], "top5_mean": v["top5"]["mean"]}
        for k, v in summary.items()
    }
    (out_dir / "JOB_COMPLETE.json").write_text(
        json.dumps({"status": "ok", "protocol": "ST-GATE LOSO", "results": card}, indent=2),
        encoding="utf-8",
    )
    print("\n[SUMMARY]")
    for k, st in summary.items():
        print(
            f"  {k:12s} top1={st['top1']['mean']*100:.2f}±{st['top1']['std']*100:.2f}% "
            f"top5={st['top5']['mean']*100:.2f}%"
        )
    print("[OK]", out_dir / "JOB_COMPLETE.json")


if __name__ == "__main__":
    main()
