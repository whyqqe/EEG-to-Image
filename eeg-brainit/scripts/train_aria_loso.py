#!/usr/bin/env python3
"""LOSO training for ARIA (Anchor-Relative Inter-brain Alignment).

Train shared EEG encoder with:
  L = λ_rsa L_rsa + λ_rel L_rel + λ_abs L_abs_centered + λ_id L_id_grl [+ ATM]

Eval:
  - absolute cosine (baseline head)
  - relative anchor retrieval (main)
  - Procrustes N-shot on absolute space (N in {0,100,500,all})

Does not overwrite subject_align / pop_former outputs.
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
from torch.utils.data import ConcatDataset, DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eeg_brainit.data.things_eeg2_adapt import ThingsEEG2SubjectDataset, collate_batch
from eeg_brainit.models.aria import (
    ARIAEncoder,
    apply_procrustes,
    aria_loss,
    build_class_anchors,
    fit_orthogonal_procrustes,
    relative_from_anchors,
    subsample_anchors,
)
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
    order = np.argsort(-sim, axis=1)
    ranks = np.argmax(order == np.arange(n)[:, None], axis=1)
    return {"top1": float((ranks < 1).mean()), "top5": float((ranks < 5).mean())}


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


def train_fold(model, train_subs, args, device, tag: str):
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
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    if hasattr(model, "param_groups"):
        groups = model.param_groups(args.lr_backbone, args.lr, args.weight_decay)
        opt = torch.optim.AdamW(groups)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = max(len(loader), 1) * args.epochs
    warmup = int(args.warmup_epochs * max(len(loader), 1))
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    global_step = 0

    for ep in range(1, args.epochs + 1):
        # Delayed GRL: keep abs head stable first (abs_id mode)
        warm = int(getattr(args, "id_warmup_epochs", 0) or 0)
        if args.grl_max <= 0 or args.lambda_id <= 0:
            grl = 0.0
        elif ep <= warm:
            grl = 0.0
        else:
            p = (ep - warm) / float(max(args.epochs - warm, 1))
            grl = float(args.grl_max * (2.0 / (1.0 + math.exp(-10 * p)) - 1.0))
        loss_sum = n = 0
        acc = {"abs": 0.0, "abs_nce": 0.0, "atm": 0.0, "rsa": 0.0, "rel": 0.0, "id": 0.0}
        model.train()
        ep_steps = max(len(loader), 1)
        t_ep0 = __import__("time").time()
        for it, batch in enumerate(loader, start=1):
            lr = cosine_warmup_lr(global_step, warmup, steps, args.lr)
            lr_bb = cosine_warmup_lr(global_step, warmup, steps, args.lr_backbone)
            if hasattr(model, "param_groups") and len(opt.param_groups) > 1:
                opt.param_groups[0]["lr"] = lr_bb
                for g in opt.param_groups[1:]:
                    g["lr"] = lr
            else:
                for g in opt.param_groups:
                    g["lr"] = lr
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(batch["eeg"])
                loss, stats = aria_loss(
                    model,
                    out,
                    batch["clip_img"],
                    batch["subject_id"],
                    temp=args.temp,
                    lambda_rsa=args.lambda_rsa,
                    lambda_rel=args.lambda_rel,
                    lambda_abs=args.lambda_abs,
                    lambda_nce=args.lambda_nce,
                    lambda_id=args.lambda_id,
                    grl_lambda=grl,
                    atm_emb=batch.get("atm_emb"),
                    lambda_atm=args.lambda_atm,
                    abs_subject_center=args.abs_subject_center,
                )
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            bs = batch["eeg"].size(0)
            loss_sum += float(loss.detach()) * bs
            n += bs
            for k in acc:
                if k in stats:
                    acc[k] += float(stats[k]) * bs
            global_step += 1
            if args.log_every > 0 and (it % args.log_every == 0 or it == ep_steps):
                elapsed = __import__("time").time() - t_ep0
                it_s = it / max(elapsed, 1e-6)
                eta = (ep_steps - it) / max(it_s, 1e-6)
                print(
                    f"[{tag} ep{ep:02d} {it}/{ep_steps}] loss={float(loss.detach()):.4f} "
                    f"lr={lr:.2e} {it_s:.2f} it/s eta={eta/60:.1f}m",
                    flush=True,
                )
        parts = " ".join(f"{k}={acc[k]/max(n,1):.3f}" for k in acc if acc[k] != 0.0)
        print(
            f"[{tag} ep{ep:02d}/{args.epochs}] loss={loss_sum/max(n,1):.4f} "
            f"{parts} lr={lr:.2e} grl={grl:.3f} time={(__import__('time').time()-t_ep0)/60:.1f}m",
            flush=True,
        )
    return model


@torch.no_grad()
def encode_split(model, subject, split, args, device, max_samples=0):
    ds = ThingsEEG2SubjectDataset(
        ROOT / args.eeg_root,
        ROOT / args.atm_bridge_dir,
        subject,
        split=split,
        max_samples=max_samples,
        seed=args.seed,
    )
    model.eval()
    abs_list, clip_list = [], []
    bs = args.eval_batch_size
    for i0 in range(0, len(ds), bs):
        samples = [ds[i] for i in range(i0, min(i0 + bs, len(ds)))]
        batch = collate_batch(samples)
        out = model(batch["eeg"].to(device))
        abs_list.append(out["clip_emb"].cpu().numpy())
        if "clip_img" in batch:
            clip_list.append(batch["clip_img"].numpy())
    abs_emb = np.concatenate(abs_list, 0)
    clip_emb = np.concatenate(clip_list, 0) if clip_list else None
    return abs_emb, clip_emb, ds


def sample_indices(n: int, k: int, seed: int) -> np.ndarray:
    if k <= 0 or k >= n:
        return np.arange(n)
    rng = np.random.RandomState(seed)
    return np.sort(rng.choice(n, size=k, replace=False))


@torch.no_grad()
def identity_probe_accuracy(model, train_subs, args, device, max_per_sub: int = 200) -> dict[str, float]:
    """Fresh linear probe of subject ID from frozen clip_emb (not co-trained probe_id).

    Lower accuracy supports identity peeling / centering. Chance ≈ 1/len(train_subs).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model.eval()
    feats, labels = [], []
    for sub in train_subs:
        ds = ThingsEEG2SubjectDataset(
            ROOT / args.eeg_root,
            ROOT / args.atm_bridge_dir,
            sub,
            split="train",
            max_samples=max_per_sub,
            seed=args.seed,
        )
        sid = sub_idx(sub)
        bs = args.eval_batch_size
        for i0 in range(0, len(ds), bs):
            samples = [ds[i] for i in range(i0, min(i0 + bs, len(ds)))]
            batch = collate_batch(samples)
            out = model(batch["eeg"].to(device))
            feats.append(out["clip_emb"].float().cpu().numpy())
            labels.append(np.full((out["clip_emb"].size(0),), sid, dtype=np.int64))
    if not feats:
        return {"acc": float("nan"), "chance": float("nan"), "n": 0, "kind": "fresh_logreg"}
    x = np.concatenate(feats, 0)
    y = np.concatenate(labels, 0)
    uniq = sorted(set(y.tolist()))
    remap = {u: i for i, u in enumerate(uniq)}
    y_m = np.array([remap[v] for v in y], dtype=np.int64)
    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(len(y_m))
    n_te = max(int(0.25 * len(y_m)), 1)
    te, tr = idx[:n_te], idx[n_te:]
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, solver="lbfgs"),
    )
    clf.fit(x[tr], y_m[tr])
    acc = float(clf.score(x[te], y_m[te]))
    return {
        "acc": acc,
        "chance": 1.0 / max(len(uniq), 1),
        "n": int(len(y_m)),
        "kind": "fresh_logreg",
    }


def evaluate_fold(model, test_sub, train_subs, gallery, anchors_np, args, device):
    abs_test, _, _ = encode_split(model, test_sub, "test", args, device)
    anchors_t = torch.from_numpy(anchors_np).to(device)
    rel_test = relative_from_anchors(torch.from_numpy(abs_test).to(device), anchors_t).cpu().numpy()
    rel_gal = relative_from_anchors(torch.from_numpy(gallery).to(device), anchors_t).cpu().numpy()

    m_abs = retrieval_metrics(abs_test, gallery)
    m_rel = retrieval_metrics(rel_test, rel_gal)
    ceiling = atm_ceiling(test_sub, ROOT / args.atm_bridge_dir, gallery)

    # A2-1 inference: remove held-out subject's train mean (cheap identity offset)
    abs_tr, _, _ = encode_split(model, test_sub, "train", args, device, max_samples=0)
    mu = abs_tr.mean(0, keepdims=True)
    m_abs_mu = retrieval_metrics(abs_test - mu, gallery)

    proc = {}
    for n_name, n_val in [("n0", 0), ("n100", 100), ("n500", 500), ("n_all", 0)]:
        if n_name == "n0":
            proc[n_name] = m_abs
            continue
        abs_tr_p, clip_tr, ds_tr = encode_split(
            model, test_sub, "train", args, device, max_samples=0
        )
        if n_name != "n_all":
            idx = sample_indices(len(ds_tr), n_val, args.seed + sub_idx(test_sub))
            abs_tr_p, clip_tr = abs_tr_p[idx], clip_tr[idx]
        q = fit_orthogonal_procrustes(abs_tr_p, clip_tr)
        src_mean, tgt_mean = abs_tr_p.mean(0), clip_tr.mean(0)
        abs_cal = apply_procrustes(abs_test, q, src_mean, tgt_mean)
        proc[n_name] = retrieval_metrics(abs_cal, gallery)

    id_probe = identity_probe_accuracy(model, train_subs, args, device, max_per_sub=200)

    return {
        "test_subject": test_sub,
        "atm_ceiling": ceiling,
        "identity_probe": id_probe,
        "settings": {
            "absolute": m_abs,
            "absolute_mu": m_abs_mu,
            "relative": m_rel,
            **{f"procrustes_{k}": v for k, v in proc.items()},
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eeg-root", default="data/processed/things-eeg2")
    p.add_argument("--atm-bridge-dir", default="outputs/atm_bridge")
    p.add_argument("--output-dir", default="outputs/aria/loso_v1")
    p.add_argument("--subjects", default=",".join(ALL_SUBJECTS))
    p.add_argument("--nz", type=int, default=256)
    p.add_argument("--clip-dim", type=int, default=1024)
    p.add_argument("--anchor-k", type=int, default=512)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--warmup-epochs", type=float, default=2.0)
    p.add_argument("--max-per-sub", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--log-every", type=int, default=200, help="print every N steps (0=epoch only)")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--temp", type=float, default=0.07)
    p.add_argument("--backbone", default="atm", choices=["atm", "labram"],
                   help="atm=from-scratch ATM-style; labram=pretrained LaBraM fine-tune")
    p.add_argument("--lr-backbone", type=float, default=1e-5,
                   help="LaBraM backbone LR (head uses --lr)")
    p.add_argument("--unfreeze-last-n", type=int, default=2)
    p.add_argument("--train-patch-embed", action="store_true")
    p.add_argument("--mode", default="full", choices=["full", "abs_only", "abs_id", "abs_center"],
                   help="abs_only=gate; abs_id=delayed GRL; abs_center=A2-1 subject-centering")
    p.add_argument("--lambda-rsa", type=float, default=1.0)
    p.add_argument("--lambda-rel", type=float, default=0.5)
    p.add_argument("--lambda-abs", type=float, default=0.1)
    p.add_argument("--lambda-nce", type=float, default=0.5)
    p.add_argument("--lambda-id", type=float, default=0.1)
    p.add_argument("--lambda-atm", type=float, default=0.2)
    p.add_argument("--grl-max", type=float, default=1.0)
    p.add_argument("--id-warmup-epochs", type=int, default=5,
                   help="epochs with grl=0 before ramping identity adversarial")
    p.add_argument("--abs-subject-center", action="store_true",
                   help="InfoNCE after per-subject mean removal (on for abs_center)")
    p.add_argument("--baseline-abs-dir", default="outputs/aria/abs_only_v2",
                   help="optional abs_only JOB_COMPLETE for comparison in summary")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-folds", type=int, default=0, help="0=all 10 folds")
    args = p.parse_args()
    if args.no_amp:
        args.amp = False
    if args.mode == "abs_only":
        args.lambda_rsa = 0.0
        args.lambda_rel = 0.0
        args.lambda_id = 0.0
        args.grl_max = 0.0
        args.abs_subject_center = False
        if args.lambda_abs <= 0:
            args.lambda_abs = 1.0
    elif args.mode == "abs_id":
        # Stage-2: keep strong absolute; add delayed identity peel
        args.lambda_rsa = 0.0
        args.lambda_rel = 0.0
        args.abs_subject_center = False
        if args.lambda_abs <= 0:
            args.lambda_abs = 1.0
        if args.lambda_id <= 0:
            args.lambda_id = 0.15
        if args.grl_max <= 0:
            args.grl_max = 1.0
        if args.lambda_atm <= 0:
            args.lambda_atm = 0.3
    elif args.mode == "abs_center":
        # A2-1: strong abs + per-subject mean removal in InfoNCE (no GRL/RSA/rel)
        args.lambda_rsa = 0.0
        args.lambda_rel = 0.0
        args.lambda_id = 0.0
        args.grl_max = 0.0
        args.abs_subject_center = True
        if args.lambda_abs <= 0:
            args.lambda_abs = 1.0
        if args.lambda_atm <= 0:
            args.lambda_atm = 0.3

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / args.output_dir
    ensure_dirs(out_dir, out_dir / "checkpoints", out_dir / "folds")
    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    gallery = np.load(ROOT / args.atm_bridge_dir / "clip_img_test_1024.npy").astype(np.float32)
    clip_train = np.load(ROOT / args.atm_bridge_dir / "clip_img_train_1024.npy").astype(np.float32)
    anchors = subsample_anchors(build_class_anchors(clip_train), args.anchor_k, args.seed)

    print(
        f"[INFO] ARIA mode={args.mode} backbone={args.backbone} device={device} "
        f"epochs={args.epochs} bs={args.batch_size} "
        f"abs={args.lambda_abs} nce={args.lambda_nce} atm={args.lambda_atm} "
        f"rsa={args.lambda_rsa} rel={args.lambda_rel} id={args.lambda_id} "
        f"grl_max={args.grl_max} id_warm={args.id_warmup_epochs} "
        f"abs_center={args.abs_subject_center}",
        flush=True,
    )

    fold_rows = []
    test_subs = subjects[: args.max_folds] if args.max_folds > 0 else subjects
    for test_sub in test_subs:
        train_subs = [s for s in subjects if s != test_sub]
        print(f"\n[FOLD] test={test_sub} train={len(train_subs)}", flush=True)
        if args.backbone == "labram":
            from eeg_brainit.models.aria_labram import LabramARIAEncoder

            model = LabramARIAEncoder(
                n_subjects=10,
                clip_dim=args.clip_dim,
                unfreeze_last_n_blocks=args.unfreeze_last_n,
                train_patch_embed=args.train_patch_embed,
                device=device,
            ).to(device)
        else:
            model = ARIAEncoder(n_subjects=10, clip_dim=args.clip_dim, nz=args.nz).to(device)
        model.set_anchors(torch.from_numpy(anchors).to(device))
        print(f"[INFO] trainable params={model.num_parameters()/1e6:.2f}M", flush=True)
        model = train_fold(model, train_subs, args, device, tag=f"aria-{test_sub}")
        ck = out_dir / "checkpoints" / f"aria_{test_sub}.pt"
        torch.save({"model": model.state_dict(), "anchors": anchors, "args": vars(args)}, ck)
        row = evaluate_fold(model, test_sub, train_subs, gallery, anchors, args, device)
        fold_rows.append(row)
        (out_dir / "folds" / f"{test_sub}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        st = row["settings"]
        idp = row.get("identity_probe", {})
        print(
            f"[EVAL {test_sub}] abs={st['absolute']['top1']*100:.2f}% "
            f"abs_mu={st['absolute_mu']['top1']*100:.2f}% "
            f"rel={st['relative']['top1']*100:.2f}% "
            f"proc500={st['procrustes_n500']['top1']*100:.2f}% "
            f"id_probe={idp.get('acc', float('nan'))*100:.1f}% "
            f"(chance={idp.get('chance', float('nan'))*100:.1f}% kind={idp.get('kind','?')}) "
            f"atm={row['atm_ceiling']['top1']*100:.2f}%",
            flush=True,
        )

    summary = {}
    keys = fold_rows[0]["settings"].keys()
    for key in keys:
        t1 = [r["settings"][key]["top1"] for r in fold_rows]
        t5 = [r["settings"][key]["top5"] for r in fold_rows]
        summary[key] = {
            "top1": {"mean": float(np.mean(t1)), "std": float(np.std(t1)), "per_subject": t1},
            "top5": {"mean": float(np.mean(t5)), "std": float(np.std(t5)), "per_subject": t5},
        }
    ceil1 = [r["atm_ceiling"]["top1"] for r in fold_rows]
    summary["atm_ceiling"] = {
        "top1": {"mean": float(np.nanmean(ceil1)), "std": float(np.nanstd(ceil1)), "per_subject": ceil1},
    }
    id_accs = [r.get("identity_probe", {}).get("acc", float("nan")) for r in fold_rows]
    summary["identity_probe_acc"] = {
        "mean": float(np.nanmean(id_accs)),
        "std": float(np.nanstd(id_accs)),
        "per_subject": id_accs,
        "chance": float(np.nanmean([r.get("identity_probe", {}).get("chance", float("nan")) for r in fold_rows])),
    }
    # optional compare to abs_only baseline
    compare = None
    base_path = ROOT / args.baseline_abs_dir / "JOB_COMPLETE.json"
    if base_path.is_file():
        try:
            base = json.loads(base_path.read_text(encoding="utf-8"))
            b_abs = base.get("results", {}).get("absolute", {}).get("top1_mean")
            if b_abs is not None:
                compare = {
                    "baseline_dir": str(args.baseline_abs_dir),
                    "baseline_abs_top1": b_abs,
                    "this_abs_top1": summary["absolute"]["top1"]["mean"],
                    "delta_abs_top1": float(summary["absolute"]["top1"]["mean"] - b_abs),
                }
        except Exception:
            compare = None

    (out_dir / "metrics.json").write_text(
        json.dumps({"folds": fold_rows, "summary": summary, "args": vars(args), "compare_to_abs_only": compare}, indent=2),
        encoding="utf-8",
    )
    card = {
        k: {"top1_mean": v["top1"]["mean"], "top1_std": v["top1"].get("std", 0.0), "top5_mean": v["top5"]["mean"] if "top5" in v else None}
        for k, v in summary.items()
        if isinstance(v, dict) and "top1" in v
    }
    card["identity_probe_acc"] = {
        "mean": summary["identity_probe_acc"]["mean"],
        "std": summary["identity_probe_acc"]["std"],
        "chance": summary["identity_probe_acc"]["chance"],
    }
    pass_gate = float(summary.get("absolute", {}).get("top1", {}).get("mean", 0.0)) >= 0.08
    (out_dir / "JOB_COMPLETE.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "protocol": f"ARIA LOSO ({args.mode})",
                "abs_only_gate_pass": pass_gate if args.mode == "abs_only" else None,
                "gate_threshold_top1": 0.08,
                "compare_to_abs_only": compare,
                "results": card,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if args.mode == "abs_only":
        print(f"[GATE] abs top1 mean gate>=8%: {'PASS' if pass_gate else 'FAIL'}", flush=True)
    if compare:
        print(
            f"[COMPARE] abs_only={compare['baseline_abs_top1']*100:.2f}% "
            f"this={compare['this_abs_top1']*100:.2f}% "
            f"delta={compare['delta_abs_top1']*100:+.2f}pp",
            flush=True,
        )
    print(
        f"[id_probe] mean={summary['identity_probe_acc']['mean']*100:.1f}% "
        f"chance≈{summary['identity_probe_acc']['chance']*100:.1f}%",
        flush=True,
    )
    print("\n[SUMMARY]")
    for k, st in summary.items():
        if isinstance(st, dict) and "top1" in st:
            print(f"  {k:20s} top1={st['top1']['mean']*100:.2f}±{st['top1'].get('std',0)*100:.2f}%")
    print("[OK]", out_dir / "JOB_COMPLETE.json")


if __name__ == "__main__":
    main()
