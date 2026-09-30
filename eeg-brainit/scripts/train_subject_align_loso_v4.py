#!/usr/bin/env python3
"""LOSO v4: freeze projector + stable W_s adapt + mixup/adv ablations.

Fixes vs v3 (from 10-fold analysis):
  1) Never fine-tune projector (v3 n2000 dip from unfreeze_proj)
  2) No SATTC-on-W_s (SAW hurt cosine after subject align)
  3) Hybrid adapt: ridge closed-form for small N; conservative SGD for large N
  4) Ablation switch: both | mixup | adv | none
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

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eeg_brainit.data.things_eeg2_adapt import ThingsEEG2SubjectDataset, collate_batch
from eeg_brainit.models.subject_align import (
    SubjectAlignEEGEncoder,
    SubjectDiscriminator,
    enigma_loss,
    estimate_latent_targets,
    fit_align_ridge,
    grad_reverse,
    mixup_batch,
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


def init_align_from_mean(model: SubjectAlignEEGEncoder, sid: int, source_ids: list[int]) -> None:
    with torch.no_grad():
        w = torch.stack([model.align[i].weight.data for i in source_ids], 0).mean(0)
        b = torch.stack([model.align[i].bias.data for i in source_ids], 0).mean(0)
        model.align[sid].weight.copy_(w)
        model.align[sid].bias.copy_(b)


def reset_align_identity(model: SubjectAlignEEGEncoder, sid: int) -> None:
    with torch.no_grad():
        model.align[sid].weight.copy_(
            torch.eye(model.nz, device=model.align[sid].weight.device, dtype=model.align[sid].weight.dtype)
        )
        model.align[sid].bias.zero_()


def build_model(args) -> SubjectAlignEEGEncoder:
    return SubjectAlignEEGEncoder(
        n_subjects=10,
        clip_dim=args.clip_dim,
        backbone=args.backbone,
        nz=args.nz,
        n_filters=args.n_filters,
        emb=args.emb,
    )


def apply_ablation_flags(args) -> None:
    """Map --ablation to mixup/adv hyperparameters."""
    mode = args.ablation
    if mode == "both":
        pass  # keep CLI mixup/adv
    elif mode == "mixup":
        args.lambda_adv = 0.0
        args.lambda_adv_ce = 0.0
    elif mode == "adv":
        args.mixup_alpha = 0.0
        args.mixup_p = 0.0
    elif mode == "none":
        args.lambda_adv = 0.0
        args.lambda_adv_ce = 0.0
        args.mixup_alpha = 0.0
        args.mixup_p = 0.0
    else:
        raise ValueError(mode)


def adv_lambda(epoch: int, epochs: int, max_lambda: float) -> float:
    if max_lambda <= 0 or epochs <= 0:
        return 0.0
    p = epoch / max(epochs, 1)
    return float(max_lambda * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0))


def train_multsubj(model, disc, train_subs, args, device, tag: str) -> SubjectAlignEEGEncoder:
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
    train_ids = {sub_idx(s) for s in train_subs}
    for i, layer in enumerate(model.align):
        for p in layer.parameters():
            p.requires_grad_(i in train_ids)

    params = [p for p in model.parameters() if p.requires_grad]
    if args.lambda_adv > 0:
        params = params + list(disc.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    model.train()
    disc.train()
    for ep in range(1, args.epochs + 1):
        loss_sum = n = 0
        adv_sum = 0.0
        lam_adv = adv_lambda(ep, args.epochs, args.lambda_adv)
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            eeg, clip_img, sid = batch["eeg"], batch["clip_img"], batch["subject_id"]
            atm = batch.get("atm_emb")
            if args.mixup_alpha > 0 and random.random() < args.mixup_p:
                eeg, clip_img, sid, atm, _ = mixup_batch(
                    eeg, clip_img, sid, atm, alpha=args.mixup_alpha
                )
            out = model(eeg, sid, normalize=True)
            loss = enigma_loss(
                out["clip_raw"],
                out["clip_emb"],
                clip_img,
                temp=args.temp,
                lambda_nce=args.lambda_nce,
                atm_emb=atm,
                lambda_atm=args.lambda_atm,
            )
            if lam_adv > 0:
                h = model.encode_pre_align(eeg)
                logits_s = disc(grad_reverse(h, lam_adv))
                adv = F.cross_entropy(logits_s, sid.long())
                loss = loss + args.lambda_adv_ce * adv
                adv_sum += float(adv) * eeg.size(0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            loss_sum += float(loss) * eeg.size(0)
            n += eeg.size(0)
        sched.step()
        print(
            f"[{tag} ep{ep:02d}] loss={loss_sum / max(n, 1):.4f} "
            f"adv={adv_sum / max(n, 1):.4f} λgrl={lam_adv:.3f} lr={sched.get_last_lr()[0]:.2e}",
            flush=True,
        )
    model.unfreeze_all()
    return model


def pick_calib_indices(n: int, n_calib: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    if n_calib <= 0:
        return np.array([], dtype=np.int64)
    if n_calib >= n:
        return np.arange(n, dtype=np.int64)
    n_cls = n // 10
    per_cls = max(1, n_calib // max(n_cls, 1))
    idxs = []
    for c in rng.permutation(n_cls):
        reps = rng.choice(10, size=min(per_cls, 10), replace=False)
        for r in reps:
            idxs.append(int(c) * 10 + int(r))
            if len(idxs) >= n_calib:
                return np.array(idxs[:n_calib], dtype=np.int64)
    while len(idxs) < n_calib:
        j = int(rng.randint(0, n))
        if j not in idxs:
            idxs.append(j)
    return np.array(idxs[:n_calib], dtype=np.int64)


def adapt_ridge(
    model: SubjectAlignEEGEncoder,
    subject: str,
    calib_idx: np.ndarray,
    args,
    device,
) -> str:
    """Closed-form W_s via projector inversion + ridge (projector stays frozen)."""
    sid = sub_idx(subject)
    ds = ThingsEEG2SubjectDataset(
        ROOT / args.eeg_root,
        ROOT / args.atm_bridge_dir,
        subject,
        split="train",
        max_samples=0,
        seed=args.seed,
    )
    # accumulate h and z* in batches
    hs, zs = [], []
    model.eval()
    bs = 64
    idxs = [int(i) for i in calib_idx]
    for i0 in range(0, len(idxs), bs):
        chunk = idxs[i0 : i0 + bs]
        samples = [ds[j] for j in chunk]
        batch = collate_batch(samples)
        eeg = batch["eeg"].to(device)
        clip = batch["clip_img"].to(device)
        with torch.no_grad():
            h = model.encode_pre_align(eeg)
        z_tgt = estimate_latent_targets(model, eeg, clip, steps=args.ridge_inv_steps, lr=args.ridge_inv_lr)
        hs.append(h.detach())
        zs.append(z_tgt)
    h_all = torch.cat(hs, 0)
    z_all = torch.cat(zs, 0)
    w, b = fit_align_ridge(h_all, z_all, ridge=args.ridge_lambda)
    with torch.no_grad():
        model.align[sid].weight.copy_(w.to(model.align[sid].weight.dtype))
        model.align[sid].bias.copy_(b.to(model.align[sid].bias.dtype))
    return "ridge"


def adapt_sgd(
    model: SubjectAlignEEGEncoder,
    subject: str,
    calib_idx: np.ndarray,
    args,
    device,
) -> str:
    """SGD on W_s only (projector + backbone frozen)."""
    sid = sub_idx(subject)
    ds = ThingsEEG2SubjectDataset(
        ROOT / args.eeg_root,
        ROOT / args.atm_bridge_dir,
        subject,
        split="train",
        max_samples=0,
        seed=args.seed,
    )
    subset = Subset(ds, [int(i) for i in calib_idx])
    loader = DataLoader(
        subset,
        batch_size=min(args.adapt_batch_size, max(1, len(calib_idx))),
        shuffle=True,
        drop_last=False,
        collate_fn=collate_batch,
        num_workers=0,
    )
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.align[sid].parameters():
        p.requires_grad_(True)

    # schedule by N
    n = len(calib_idx)
    if n < 200:
        lr, max_steps = args.adapt_lr_tiny, args.adapt_steps_tiny
    elif n < 1000:
        lr, max_steps = args.adapt_lr_small, args.adapt_steps_small
    else:
        lr, max_steps = args.adapt_lr_large, args.adapt_steps_large

    opt = torch.optim.AdamW(list(model.align[sid].parameters()), lr=lr, weight_decay=0.01)
    model.train()
    steps = 0
    while steps < max_steps:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            batch["subject_id"] = torch.full_like(batch["subject_id"], sid)
            out = model(batch["eeg"], batch["subject_id"], normalize=True)
            loss = enigma_loss(
                out["clip_raw"],
                out["clip_emb"],
                batch["clip_img"],
                temp=args.temp,
                lambda_nce=args.lambda_nce,
                atm_emb=batch.get("atm_emb"),
                lambda_atm=args.lambda_atm * 0.5,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            steps += 1
            if steps >= max_steps:
                break
    model.eval()
    model.unfreeze_all()
    return f"sgd(lr={lr},steps={max_steps})"


def adapt_target(model, subject, calib_idx, args, device) -> str:
    if len(calib_idx) == 0:
        return "none"
    mode = args.adapt_mode
    if mode == "ridge":
        return adapt_ridge(model, subject, calib_idx, args, device)
    if mode == "sgd":
        return adapt_sgd(model, subject, calib_idx, args, device)
    # hybrid: ridge for small N, sgd for large
    if len(calib_idx) <= args.ridge_max_n:
        return adapt_ridge(model, subject, calib_idx, args, device)
    return adapt_sgd(model, subject, calib_idx, args, device)


@torch.no_grad()
def encode_test(model: SubjectAlignEEGEncoder, subject: str, args, device) -> np.ndarray:
    sid = sub_idx(subject)
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
    bs = 64
    for i0 in range(0, len(ds), bs):
        samples = [ds[i] for i in range(i0, min(i0 + bs, len(ds)))]
        batch = collate_batch(samples)
        x = batch["eeg"].to(device)
        sid_t = torch.full((x.size(0),), sid, device=device, dtype=torch.long)
        out = model(x, sid_t, normalize=True)
        embs.append(out["clip_emb"].cpu().numpy())
    return np.concatenate(embs, 0)


def eval_calib(base_state, subject, train_subs, gallery, n_calib, args, device) -> dict:
    model = build_model(args).to(device)
    model.load_state_dict(base_state)
    sid = sub_idx(subject)
    src = [sub_idx(s) for s in train_subs]
    if args.init_mode == "mean":
        init_align_from_mean(model, sid, src)
    else:
        reset_align_identity(model, sid)
    calib_idx = pick_calib_indices(16540, n_calib, seed=args.seed + sid * 1009 + n_calib)
    how = adapt_target(model, subject, calib_idx, args, device)
    q = encode_test(model, subject, args, device)
    met = retrieval_metrics(q, gallery)
    met.update({"n_calib": int(len(calib_idx)), "adapt": how, "also_proj": False})
    return met


def parse_int_list(s: str) -> list[int]:
    out = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(-1 if x.lower() == "all" else int(x))
    return out


def run_one(args) -> dict:
    apply_ablation_flags(args)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / args.output_dir
    ensure_dirs(out_dir, out_dir / "checkpoints", out_dir / "folds")
    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    calib_list = parse_int_list(args.calib_list)
    gallery = np.load(ROOT / args.atm_bridge_dir / "clip_img_test_1024.npy").astype(np.float32)
    print(
        f"[INFO] ablation={args.ablation} seed={args.seed} adapt={args.adapt_mode} "
        f"mixup={args.mixup_alpha}/{args.mixup_p} adv={args.lambda_adv}/{args.lambda_adv_ce} "
        f"calib={calib_list} device={device}",
        flush=True,
    )

    fold_rows = []
    test_subs = subjects[: args.max_folds] if args.max_folds > 0 else subjects
    for test_sub in test_subs:
        train_subs = [s for s in subjects if s != test_sub]
        print(f"\n[FOLD] test={test_sub} train={len(train_subs)}", flush=True)
        model = build_model(args).to(device)
        disc = SubjectDiscriminator(model.nz, n_subjects=10).to(device)
        model = train_multsubj(
            model, disc, train_subs, args, device, tag=f"v4-{args.ablation}-s{args.seed}-{test_sub}"
        )
        ck = out_dir / "checkpoints" / f"{args.ablation}_s{args.seed}_{test_sub}.pt"
        torch.save({"model": model.state_dict(), "args": vars(args), "test_sub": test_sub}, ck)
        base_state = copy.deepcopy(model.state_dict())
        ceiling = atm_ceiling(test_sub, ROOT / args.atm_bridge_dir, gallery)
        settings = {}
        for n_cal in calib_list:
            n_use = 16540 if n_cal < 0 else n_cal
            key = f"n{n_use}" if n_cal >= 0 else "n_all"
            settings[key] = eval_calib(base_state, test_sub, train_subs, gallery, n_use, args, device)
            s = settings[key]
            print(
                f"  [{key}] top1={s['top1']*100:.2f}% top5={s['top5']*100:.2f}% adapt={s['adapt']}",
                flush=True,
            )
        row = {
            "test_subject": test_sub,
            "ablation": args.ablation,
            "seed": args.seed,
            "atm_ceiling": ceiling,
            "settings": settings,
        }
        fold_rows.append(row)
        (out_dir / "folds" / f"{args.ablation}_s{args.seed}_{test_sub}.json").write_text(
            json.dumps(row, indent=2), encoding="utf-8"
        )
        z = settings.get("n0", {})
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
    blob = {
        "ablation": args.ablation,
        "seed": args.seed,
        "folds": fold_rows,
        "summary": summary,
    }
    (out_dir / f"metrics_{args.ablation}_s{args.seed}.json").write_text(
        json.dumps(blob, indent=2), encoding="utf-8"
    )
    card = {
        k: {"top1_mean": v["top1"]["mean"], "top1_std": v["top1"]["std"], "top5_mean": v["top5"]["mean"]}
        for k, v in summary.items()
    }
    print(f"\n[SUMMARY ablation={args.ablation} seed={args.seed}]")
    for k, st in summary.items():
        print(
            f"  {k:10s} top1={st['top1']['mean']*100:.2f}±{st['top1']['std']*100:.2f}% "
            f"top5={st['top5']['mean']*100:.2f}%"
        )
    return {"ablation": args.ablation, "seed": args.seed, "results": card}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eeg-root", default="data/processed/things-eeg2")
    parser.add_argument("--atm-bridge-dir", default="outputs/atm_bridge")
    parser.add_argument("--output-dir", default="outputs/subject_align_loso/loso_v4")
    parser.add_argument("--subjects", default=",".join(ALL_SUBJECTS))
    parser.add_argument("--backbone", default="atm_style", choices=["atm_style", "enigma"])
    parser.add_argument("--nz", type=int, default=256)
    parser.add_argument("--n-filters", type=int, default=80)
    parser.add_argument("--emb", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--max-per-sub", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--temp", type=float, default=0.07)
    parser.add_argument("--lambda-nce", type=float, default=0.5)
    parser.add_argument("--lambda-atm", type=float, default=0.2)
    parser.add_argument("--lambda-adv", type=float, default=1.0)
    parser.add_argument("--lambda-adv-ce", type=float, default=0.1)
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument("--mixup-p", type=float, default=0.5)
    parser.add_argument("--ablation", default="both", choices=["both", "mixup", "adv", "none"])
    parser.add_argument("--ablations", default="", help="comma list to run sequentially; overrides --ablation")
    parser.add_argument("--seeds", default="42", help="comma seeds")
    parser.add_argument("--clip-dim", type=int, default=1024)
    parser.add_argument("--calib-list", default="0,100,500,2000,5000,all")
    parser.add_argument("--init-mode", default="mean", choices=["identity", "mean"])
    parser.add_argument("--adapt-mode", default="hybrid", choices=["hybrid", "ridge", "sgd"])
    parser.add_argument("--ridge-max-n", type=int, default=1000)
    parser.add_argument("--ridge-lambda", type=float, default=1e-2)
    parser.add_argument("--ridge-inv-steps", type=int, default=40)
    parser.add_argument("--ridge-inv-lr", type=float, default=0.3)
    parser.add_argument("--adapt-lr-tiny", type=float, default=3e-4)
    parser.add_argument("--adapt-lr-small", type=float, default=5e-4)
    parser.add_argument("--adapt-lr-large", type=float, default=1e-3)
    parser.add_argument("--adapt-steps-tiny", type=int, default=400)
    parser.add_argument("--adapt-steps-small", type=int, default=800)
    parser.add_argument("--adapt-steps-large", type=int, default=6000)
    parser.add_argument("--adapt-batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-folds", type=int, default=0)
    args = parser.parse_args()

    ablations = [a.strip() for a in args.ablations.split(",") if a.strip()] or [args.ablation]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    all_cards = {}
    for abl in ablations:
        for seed in seeds:
            args.ablation = abl
            args.seed = seed
            # restore defaults before ablation mutates (copy from CLI once)
            # re-read mixup/adv defaults each loop
            args.mixup_alpha = 0.2
            args.mixup_p = 0.5
            args.lambda_adv = 1.0
            args.lambda_adv_ce = 0.1
            card = run_one(args)
            all_cards[f"{abl}_s{seed}"] = card["results"]

    out_dir = ROOT / args.output_dir
    ensure_dirs(out_dir)
    (out_dir / "JOB_COMPLETE.json").write_text(
        json.dumps({"status": "ok", "protocol": "subject-align LOSO v4", "results": all_cards}, indent=2),
        encoding="utf-8",
    )
    print("[OK]", out_dir / "JOB_COMPLETE.json")


if __name__ == "__main__":
    main()
