#!/usr/bin/env python3
"""NOD Dual-Teacher CLIP Bridge training.

Modes:
  dual         — InfoNCE(img) + MSE(img) + MSE(fMRI teacher)
  eeg_only     — InfoNCE(img) + MSE(img)
  fake_teacher — same as dual but teacher = shuffled image CLIP (ablation)

Frozen assets: OpenCLIP embeddings (precomputed), fMRI→CLIP teacher, SDXL/IP-Adapter (optional gen).
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
os.environ.setdefault("HF_HOME", "/project/peilab/why/cache/eeg-brainit/hf")
os.environ.setdefault("HF_HUB_CACHE", "/project/peilab/why/cache/eeg-brainit/hf/hub")

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from eeg_brainit.models.dual_teacher import (
    DualTeacherEEG2CLIP,
    FmriClipTeacher,
    dual_teacher_losses,
)
from eeg_brainit.models.eeg2fmri import retrieval_metrics
from eeg_brainit.utils.config import ensure_dirs


def _split_by_image_id(image_ids: list[str], val_frac: float, seed: int):
    rng = random.Random(seed)
    uniq = sorted(set(image_ids))
    rng.shuffle(uniq)
    n_val = max(1, int(len(uniq) * val_frac)) if len(uniq) > 1 else 0
    val_set = set(uniq[:n_val])
    train = [i for i, x in enumerate(image_ids) if x not in val_set]
    val = [i for i, x in enumerate(image_ids) if x in val_set]
    return train, val


def _cosine_lr(optimizer, epoch, epochs, base_lrs, warmup: int = 3):
    for pg, base_lr in zip(optimizer.param_groups, base_lrs):
        if epoch <= warmup:
            lr = base_lr * epoch / max(warmup, 1)
        else:
            t = (epoch - warmup) / max(epochs - warmup, 1)
            lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * t))
        pg["lr"] = lr


def load_subject(
    pairs_root: Path,
    clip_dir: Path,
    subject: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    index = json.loads((clip_dir / "index.json").read_text(encoding="utf-8"))
    emb = np.load(clip_dir / "embeddings.npy").astype(np.float32)
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    sub_dir = pairs_root / subject
    z = np.load(sub_dir / "pairs.npz")
    meta = json.loads((sub_dir / "pairs_meta.json").read_text(encoding="utf-8"))
    ch = list(meta["ch_names"])
    trials = meta["trials"]
    keep, clips, ids = [], [], []
    for i, t in enumerate(trials):
        iid = t["image_id"]
        if iid not in index:
            continue
        keep.append(i)
        clips.append(emb[index[iid]])
        ids.append(iid)
    eeg = z["eeg"][keep].astype(np.float32)
    fmri = z["fmri_roi"][keep].astype(np.float32)
    clip = np.stack(clips, 0).astype(np.float32)
    print(f"[INFO] {subject} n={len(keep)} eeg={eeg.shape} fmri={fmri.shape} clip={clip.shape}")
    return eeg, fmri, clip, ids, ch


@torch.no_grad()
def eval_retrieval(model, eeg, clip, device, bs: int = 128) -> dict:
    model.eval()
    preds = []
    for i in range(0, len(eeg), bs):
        out = model(torch.from_numpy(eeg[i : i + bs]).to(device))
        preds.append(out["clip_emb"].cpu())
    pred = torch.cat(preds, 0)
    return retrieval_metrics(pred, torch.from_numpy(clip))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-root", default="data/nod/processed/classmean")
    parser.add_argument("--clip-dir", default="data/nod/processed/clip_vit_h14_all")
    parser.add_argument("--subject", default="sub-01")
    parser.add_argument(
        "--fmri-teacher-ckpt",
        default="outputs/eval/nod_phase2_gt_fmri2image/sub-01_classmean/fmri2clip.pt",
    )
    parser.add_argument("--output-dir", default="outputs/nod_dual_teacher/sub-01_dual")
    parser.add_argument("--mode", choices=["dual", "eeg_only", "fake_teacher"], default="dual")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda-nce", type=float, default=1.0)
    parser.add_argument("--lambda-mse-img", type=float, default=1.0)
    parser.add_argument("--lambda-mse-fmri", type=float, default=0.25)
    parser.add_argument("--temp", type=float, default=0.07)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--train-frac", type=float, default=1.0, help="Subsample train images for low-data curves")
    parser.add_argument("--skip-gen", action="store_true")
    parser.add_argument("--max-images", type=int, default=8)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pairs_root = ROOT / args.pairs_root
    clip_dir = ROOT / args.clip_dir
    out_dir = ROOT / args.output_dir
    ensure_dirs(out_dir, out_dir / "checkpoints", out_dir / "generated")

    eeg, fmri, clip, ids, ch_names = load_subject(pairs_root, clip_dir, args.subject)
    train_idx, val_idx = _split_by_image_id(ids, args.val_frac, args.seed)
    if args.train_frac < 1.0:
        rng = random.Random(args.seed)
        k = max(1, int(len(train_idx) * args.train_frac))
        train_idx = rng.sample(train_idx, k)
    print(f"[INFO] mode={args.mode} train={len(train_idx)} val={len(val_idx)} device={device}")

    teacher = None
    use_fmri_teacher = args.mode in {"dual", "fake_teacher"}
    if use_fmri_teacher:
        if args.mode == "dual":
            tpath = ROOT / args.fmri_teacher_ckpt
            if not tpath.is_file():
                raise FileNotFoundError(tpath)
            teacher = FmriClipTeacher.from_checkpoint(str(tpath), device)
            print(f"[INFO] loaded fMRI teacher {tpath}")
        else:
            print("[INFO] fake teacher = shuffled image CLIP")

    model = DualTeacherEEG2CLIP(
        n_channels=eeg.shape[1],
        clip_dim=clip.shape[1],
        d_model=args.d_model,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    base_lrs = [args.lr]

    tr = TensorDataset(
        torch.from_numpy(eeg[train_idx]),
        torch.from_numpy(fmri[train_idx]),
        torch.from_numpy(clip[train_idx]),
    )
    loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True, drop_last=True)

    history = []
    best = -1.0
    best_path = out_dir / "checkpoints" / "best.pt"
    lam_fmri = 0.0 if args.mode == "eeg_only" else args.lambda_mse_fmri

    for epoch in range(1, args.epochs + 1):
        _cosine_lr(opt, epoch, args.epochs, base_lrs, warmup=3)
        model.train()
        loss_sum = n_seen = 0
        for eb, fb, cb in loader:
            eb, fb, cb = eb.to(device), fb.to(device), cb.to(device)
            pred = model(eb)["clip_emb"]
            z_f = None
            if use_fmri_teacher:
                if args.mode == "dual":
                    with torch.no_grad():
                        z_f = teacher(fb)
                else:
                    # fake teacher: batch-shuffled image CLIP
                    perm = torch.randperm(cb.size(0), device=device)
                    z_f = cb[perm]
            losses = dual_teacher_losses(
                pred,
                cb,
                z_f,
                temp=args.temp,
                lambda_nce=args.lambda_nce,
                lambda_mse_img=args.lambda_mse_img,
                lambda_mse_fmri=lam_fmri,
            )
            opt.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += float(losses["total"]) * eb.size(0)
            n_seen += eb.size(0)

        ret = eval_retrieval(model, eeg[val_idx], clip[val_idx], device)
        # GT teacher ceiling on same val
        gt_ret = None
        if teacher is not None:
            with torch.no_grad():
                z_gt = teacher(torch.from_numpy(fmri[val_idx]).to(device)).cpu()
            gt_ret = retrieval_metrics(z_gt, torch.from_numpy(clip[val_idx]))

        row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(n_seen, 1),
            "val_top1": ret["top1"],
            "val_top5": ret["top5"],
            "chance_top1": ret["chance_top1"],
            "gt_teacher_top1": None if gt_ret is None else gt_ret["top1"],
            "lr": opt.param_groups[0]["lr"],
            "mode": args.mode,
            "train_n": len(train_idx),
        }
        history.append(row)
        print(
            f"[{args.mode} {epoch:03d}] loss={row['train_loss']:.4f} "
            f"val_t1={ret['top1']*100:.2f}% t5={ret['top5']*100:.2f}% "
            f"chance={ret['chance_top1']*100:.2f}%"
            + (f" gt_t1={gt_ret['top1']*100:.2f}%" if gt_ret else "")
        )
        if ret["top1"] > best:
            best = ret["top1"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "ch_names": ch_names,
                    "clip_dim": int(clip.shape[1]),
                    "n_channels": int(eeg.shape[1]),
                    "best_val_top1": best,
                    "history": history,
                    "mode": args.mode,
                },
                best_path,
            )

    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    metrics = {
        "mode": args.mode,
        "subject": args.subject,
        "best_val_top1": best,
        "chance_top1": 1.0 / max(len(val_idx), 1),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "ckpt": str(best_path),
        "fmri_teacher": args.fmri_teacher_ckpt if args.mode == "dual" else None,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[OK] best_top1={best*100:.2f}% -> {best_path}")

    if not args.skip_gen and best_path.is_file():
        # optional generation with best ckpt
        ck = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        model.eval()
        take = val_idx[: args.max_images]
        with torch.no_grad():
            pred = model(torch.from_numpy(eeg[take]).to(device))["clip_emb"].cpu().numpy()
        sys.path.insert(0, str(ROOT / "scripts"))
        from eval_nod_phase2_gt_fmri2image import generate_images_sdxl, _make_grid, _resolve_stim

        images_root = ROOT / "data/nod/raw/ds005811/stimuli/ImageNet"
        gt_paths = []
        for i in take:
            p = _resolve_stim(images_root, ids[i])
            if p is None:
                raise FileNotFoundError(ids[i])
            gt_paths.append(p)
        gen_paths = generate_images_sdxl(
            pred, out_dir / "generated" / args.mode, device, max_images=args.max_images, tag=args.mode
        )
        _make_grid(gen_paths, gt_paths, out_dir / f"grid_{args.mode}.png")
        print(f"[OK] wrote {out_dir / f'grid_{args.mode}.png'}")


if __name__ == "__main__":
    main()
