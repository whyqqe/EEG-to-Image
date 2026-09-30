#!/usr/bin/env python3
"""LOSO v3: subject-align + mixup/adversarial invariance + SATTC calibration.

Builds on v2 (ATM-style shared + W_s + ATM distill) with literature upgrades:
  1) Feature mixup (subject-invariant EEG embeddings, ACM'26-style)
  2) Subject-adversarial GRL on *pre-align* latents (DANN / Özdenizci)
  3) After W_s adaptation, attach SATTC-like label-free calibration
     (SAW+CW+Ada-CSLS+PoE) — viable once encoder is strong enough
     (v2 zero-shot ≈ SATTC standardized ATM ~9% Top-1)

Calibration N still only trains W_new (+ optional projector); test labels unused.
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
from eeg_brainit.models.cross_subject_tta import run_calibration_suite
from eeg_brainit.models.subject_align import (
    SubjectAlignEEGEncoder,
    SubjectDiscriminator,
    enigma_loss,
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


def reset_align_identity(model: SubjectAlignEEGEncoder, sid: int) -> None:
    layer = model.align[sid]
    with torch.no_grad():
        layer.weight.copy_(torch.eye(model.nz, device=layer.weight.device, dtype=layer.weight.dtype))
        layer.bias.zero_()


def init_align_from_mean(model: SubjectAlignEEGEncoder, sid: int, source_ids: list[int]) -> None:
    with torch.no_grad():
        w = torch.stack([model.align[i].weight.data for i in source_ids], 0).mean(0)
        b = torch.stack([model.align[i].bias.data for i in source_ids], 0).mean(0)
        model.align[sid].weight.copy_(w)
        model.align[sid].bias.copy_(b)


def build_model(args) -> SubjectAlignEEGEncoder:
    return SubjectAlignEEGEncoder(
        n_subjects=10,
        clip_dim=args.clip_dim,
        backbone=args.backbone,
        nz=args.nz,
        n_filters=args.n_filters,
        emb=args.emb,
    )


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

    params = [p for p in model.parameters() if p.requires_grad] + list(disc.parameters())
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


def adapt_target(
    model: SubjectAlignEEGEncoder,
    subject: str,
    calib_idx: np.ndarray,
    args,
    device,
    also_proj: bool,
) -> SubjectAlignEEGEncoder:
    sid = sub_idx(subject)
    if len(calib_idx) == 0:
        return model

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
    params = list(model.align[sid].parameters())
    if also_proj:
        for p in model.projector.parameters():
            p.requires_grad_(True)
        params += list(model.projector.parameters())

    opt = torch.optim.AdamW(params, lr=args.adapt_lr, weight_decay=0.01)
    steps_per_epoch = max(1, len(loader))
    target_steps = args.adapt_epochs * steps_per_epoch
    if len(calib_idx) < 500:
        max_steps = min(args.adapt_max_steps_small, max(args.adapt_min_steps, target_steps))
    else:
        max_steps = min(args.adapt_max_steps_large, max(args.adapt_min_steps, target_steps))

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
    return model


@torch.no_grad()
def encode_split(
    model: SubjectAlignEEGEncoder, subject: str, split: str, args, device, max_n: int = 0
) -> np.ndarray:
    sid = sub_idx(subject)
    ds = ThingsEEG2SubjectDataset(
        ROOT / args.eeg_root,
        ROOT / args.atm_bridge_dir,
        subject,
        split=split,
        max_samples=0,
        seed=args.seed,
    )
    model.eval()
    embs = []
    bs = 64
    n = len(ds) if max_n <= 0 else min(len(ds), max_n)
    for i0 in range(0, n, bs):
        samples = [ds[i] for i in range(i0, min(i0 + bs, n))]
        batch = collate_batch(samples)
        x = batch["eeg"].to(device)
        sid_t = torch.full((x.size(0),), sid, device=device, dtype=torch.long)
        out = model(x, sid_t, normalize=True)
        embs.append(out["clip_emb"].cpu().numpy())
    return np.concatenate(embs, 0)


def eval_calib(
    base_state: dict,
    subject: str,
    train_subs: list[str],
    gallery: np.ndarray,
    n_calib: int,
    args,
    device,
) -> dict:
    model = build_model(args).to(device)
    model.load_state_dict(base_state)
    sid = sub_idx(subject)
    src = [sub_idx(s) for s in train_subs]
    if args.init_mode == "mean":
        init_align_from_mean(model, sid, src)
    else:
        reset_align_identity(model, sid)

    ds_len = 16540
    calib_idx = pick_calib_indices(ds_len, n_calib, seed=args.seed + sid * 1009 + n_calib)
    also_proj = bool(args.unfreeze_proj_n > 0 and n_calib >= args.unfreeze_proj_n)
    if n_calib > 0:
        model = adapt_target(model, subject, calib_idx, args, device, also_proj=also_proj)

    q = encode_split(model, subject, "test", args, device)
    met = retrieval_metrics(q, gallery)
    met.update({"n_calib": int(len(calib_idx)), "init": args.init_mode, "also_proj": also_proj})

    if args.apply_sattc:
        q_calib = encode_split(model, subject, "train", args, device, max_n=2000)
        suite = run_calibration_suite(
            q, gallery, calib_queries=q_calib, csls_k=args.csls_k, beta=args.poe_beta
        )
        met["sattc"] = {
            k: {"top1": v["top1"], "top5": v["top5"], "hubness_skew": v["hubness_skew"]}
            for k, v in suite.items()
        }
        best_cal = max(suite.items(), key=lambda kv: kv[1]["top1"])
        met["sattc_best"] = best_cal[0]
        met["sattc_best_top1"] = best_cal[1]["top1"]
        met["sattc_best_top5"] = best_cal[1]["top5"]
    return met


def parse_int_list(s: str) -> list[int]:
    out = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        if x.lower() == "all":
            out.append(-1)
        else:
            out.append(int(x))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eeg-root", default="data/processed/things-eeg2")
    parser.add_argument("--atm-bridge-dir", default="outputs/atm_bridge")
    parser.add_argument("--output-dir", default="outputs/subject_align_loso/loso_v3")
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
    parser.add_argument("--clip-dim", type=int, default=1024)
    parser.add_argument("--calib-list", default="0,100,500,2000,5000,all")
    parser.add_argument("--init-mode", default="mean", choices=["identity", "mean"])
    parser.add_argument("--adapt-lr", type=float, default=1e-3)
    parser.add_argument("--adapt-epochs", type=int, default=12)
    parser.add_argument("--adapt-min-steps", type=int, default=100)
    parser.add_argument("--adapt-max-steps-small", type=int, default=1500)
    parser.add_argument("--adapt-max-steps-large", type=int, default=12000)
    parser.add_argument("--adapt-batch-size", type=int, default=64)
    parser.add_argument("--unfreeze-proj-n", type=int, default=2000)
    parser.add_argument("--apply-sattc", action="store_true", default=True)
    parser.add_argument("--no-sattc", action="store_true")
    parser.add_argument("--csls-k", type=int, default=10)
    parser.add_argument("--poe-beta", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-folds", type=int, default=0)
    args = parser.parse_args()
    if args.no_sattc:
        args.apply_sattc = False

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / args.output_dir
    ensure_dirs(out_dir, out_dir / "checkpoints", out_dir / "folds")
    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    calib_list = parse_int_list(args.calib_list)
    gallery = np.load(ROOT / args.atm_bridge_dir / "clip_img_test_1024.npy").astype(np.float32)
    print(
        f"[INFO] device={device} backbone={args.backbone} nz={args.nz} "
        f"epochs={args.epochs} mixup={args.mixup_alpha}/{args.mixup_p} "
        f"adv={args.lambda_adv}/{args.lambda_adv_ce} sattc={args.apply_sattc} "
        f"calib={calib_list}",
        flush=True,
    )

    fold_rows = []
    test_subs = subjects[: args.max_folds] if args.max_folds > 0 else subjects
    for test_sub in test_subs:
        train_subs = [s for s in subjects if s != test_sub]
        print(f"\n[FOLD] test={test_sub} train={len(train_subs)}", flush=True)
        model = build_model(args).to(device)
        disc = SubjectDiscriminator(model.nz, n_subjects=10).to(device)
        n_shared = sum(p.numel() for p in model.shared_parameters()) / 1e6
        print(f"[INFO] Nz={model.nz} shared≈{n_shared:.2f}M", flush=True)
        model = train_multsubj(model, disc, train_subs, args, device, tag=f"align3-{test_sub}")
        ck = out_dir / "checkpoints" / f"align_{test_sub}.pt"
        torch.save({"model": model.state_dict(), "test_sub": test_sub, "args": vars(args)}, ck)
        base_state = copy.deepcopy(model.state_dict())

        ceiling = atm_ceiling(test_sub, ROOT / args.atm_bridge_dir, gallery)
        settings = {}
        for n_cal in calib_list:
            n_use = 16540 if n_cal < 0 else n_cal
            key = f"n{n_use}" if n_cal >= 0 else "n_all"
            settings[key] = eval_calib(
                base_state, test_sub, train_subs, gallery, n_use, args, device
            )
            s = settings[key]
            extra = ""
            if "sattc_best_top1" in s:
                extra = f" sattc_best={s['sattc_best']}:{s['sattc_best_top1']*100:.2f}%"
            print(
                f"  [{key}] top1={s['top1']*100:.2f}% top5={s['top5']*100:.2f}% "
                f"proj={s['also_proj']}{extra}",
                flush=True,
            )

        row = {"test_subject": test_sub, "atm_ceiling": ceiling, "settings": settings}
        fold_rows.append(row)
        (out_dir / "folds" / f"{test_sub}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        z = settings.get("n0", {})
        best = max(settings.items(), key=lambda kv: kv[1]["top1"])
        best_sattc = max(
            ((k, v.get("sattc_best_top1", v["top1"])) for k, v in settings.items()),
            key=lambda kv: kv[1],
        )
        print(
            f"[EVAL {test_sub}] zero={z.get('top1', 0)*100:.2f}% "
            f"best={best[0]}:{best[1]['top1']*100:.2f}% "
            f"best_sattc={best_sattc[0]}:{best_sattc[1]*100:.2f}% "
            f"atm={ceiling['top1']*100:.2f}%",
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
        if args.apply_sattc:
            st1 = [r["settings"][key].get("sattc_best_top1", r["settings"][key]["top1"]) for r in fold_rows]
            st5 = [r["settings"][key].get("sattc_best_top5", r["settings"][key]["top5"]) for r in fold_rows]
            summary[f"{key}_sattc"] = {
                "top1": {"mean": float(np.mean(st1)), "std": float(np.std(st1)), "per_subject": st1},
                "top5": {"mean": float(np.mean(st5)), "std": float(np.std(st5)), "per_subject": st5},
            }
    ceil1 = [r["atm_ceiling"]["top1"] for r in fold_rows]
    ceil5 = [r["atm_ceiling"]["top5"] for r in fold_rows]
    summary["atm_ceiling"] = {
        "top1": {"mean": float(np.nanmean(ceil1)), "std": float(np.nanstd(ceil1)), "per_subject": ceil1},
        "top5": {"mean": float(np.nanmean(ceil5)), "std": float(np.nanstd(ceil5)), "per_subject": ceil5},
    }

    all_results = {"enigma_align_v3": {"folds": fold_rows, "summary": summary}}
    (out_dir / "metrics.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    card = {
        k: {"top1_mean": v["top1"]["mean"], "top1_std": v["top1"]["std"], "top5_mean": v["top5"]["mean"]}
        for k, v in summary.items()
    }
    (out_dir / "JOB_COMPLETE.json").write_text(
        json.dumps(
            {"status": "ok", "protocol": "subject-align LOSO v3 (mixup+adv+SATTC)", "results": card},
            indent=2,
        ),
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
