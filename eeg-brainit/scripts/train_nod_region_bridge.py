#!/usr/bin/env python3
"""NOD Region-Bridge training: fMRI EVC/Ventral teachers + EEG spatiotemporal mapping.

Modes:
  region   — image CLIP + EVC teacher + Ventral teacher (main)
  eeg_only — image CLIP only
  global   — image CLIP + single global classmean teacher
  fake     — image CLIP + shuffled region teachers
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
os.environ.setdefault("HOME", "/project/peilab/why/cache/eeg-brainit/xdg-home")
os.environ.setdefault("NEUROMAPS_DATA", "/project/peilab/why/cache/eeg-brainit/neuromaps-data")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eeg_brainit.models.dual_teacher import FmriClipTeacher, dual_teacher_losses
from eeg_brainit.models.eeg2fmri import retrieval_metrics
from eeg_brainit.models.region_bridge import (
    OCC_CH,
    VENT_CH,
    RegionBridgeEEG,
    channel_index,
    time_slice,
)
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


def train_teacher(fmri: np.ndarray, clip: np.ndarray, train_idx, val_idx, device, epochs=25, lr=1e-3):
    """Quick GT ROI → CLIP teacher."""
    model = FmriClipTeacher(in_dim=fmri.shape[1], out_dim=clip.shape[1]).to(device)
    for p in model.parameters():
        p.requires_grad = True
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    tr = TensorDataset(torch.from_numpy(fmri[train_idx]), torch.from_numpy(clip[train_idx]))
    loader = DataLoader(tr, batch_size=128, shuffle=True, drop_last=True)
    best, best_state = -1.0, None
    for _ in range(1, epochs + 1):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = dual_teacher_losses(pred, y, None, lambda_mse_fmri=0.0)["total"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(torch.from_numpy(fmri[val_idx]).to(device)).cpu()
            ret = retrieval_metrics(pred, torch.from_numpy(clip[val_idx]))
        if ret["top1"] > best:
            best = ret["top1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.train()
    model.load_state_dict(best_state)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, best


@torch.no_grad()
def eval_img(model, eeg, clip, device, bs=128):
    model.eval()
    preds = []
    for i in range(0, len(eeg), bs):
        out = model(torch.from_numpy(eeg[i : i + bs]).to(device))
        preds.append(out["clip_img"].cpu())
    return retrieval_metrics(torch.cat(preds, 0), torch.from_numpy(clip))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-root", default="data/nod/processed/region_pairs")
    parser.add_argument("--subject", default="sub-01")
    parser.add_argument("--output-dir", default="outputs/nod_region_bridge/sub-01_region")
    parser.add_argument("--mode", choices=["region", "eeg_only", "global", "fake"], default="region")
    parser.add_argument(
        "--global-teacher-ckpt",
        default="outputs/eval/nod_phase2_gt_fmri2image/sub-01_classmean/fmri2clip.pt",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--teacher-epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda-nce", type=float, default=1.0)
    parser.add_argument("--lambda-mse-img", type=float, default=1.0)
    parser.add_argument("--lambda-evc", type=float, default=0.25)
    parser.add_argument("--lambda-vent", type=float, default=0.35)
    parser.add_argument("--lambda-global", type=float, default=0.25)
    parser.add_argument("--early-win", default="0.06,0.15")
    parser.add_argument("--late-win", default="0.15,0.35")
    parser.add_argument("--train-frac", type=float, default=1.0)
    parser.add_argument("--skip-gen", action="store_true")
    parser.add_argument("--max-images", type=int, default=8)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sub_dir = ROOT / args.pairs_root / args.subject
    z = np.load(sub_dir / "pairs.npz")
    meta = json.loads((sub_dir / "pairs_meta.json").read_text(encoding="utf-8"))
    eeg = z["eeg"].astype(np.float32)
    fmri_evc = z["fmri_evc"].astype(np.float32)
    fmri_vent = z["fmri_ventral"].astype(np.float32)
    fmri_global = z["fmri_global"].astype(np.float32) if "fmri_global" in z.files else None
    clip = z["clip"].astype(np.float32)
    ids = [t["image_id"] for t in meta["trials"]]
    ch_names = list(meta["ch_names"])
    sfreq = float(meta.get("sfreq", 250.0))
    tmin = float(meta.get("tmin", -0.1))

    train_idx, val_idx = _split_by_image_id(ids, args.val_frac, args.seed)
    if args.train_frac < 1.0:
        rng = random.Random(args.seed)
        train_idx = rng.sample(train_idx, max(1, int(len(train_idx) * args.train_frac)))

    out_dir = ROOT / args.output_dir
    ensure_dirs(out_dir, out_dir / "checkpoints", out_dir / "teachers", out_dir / "generated")
    print(f"[INFO] {args.subject} mode={args.mode} n={len(ids)} train={len(train_idx)} val={len(val_idx)}")

    teacher_evc = teacher_vent = teacher_global = None
    ceil: dict[str, float] = {}
    if args.mode in {"region", "fake"}:
        print("[INFO] training EVC/Ventral fMRI→CLIP teachers...")
        teacher_evc, ceil["evc"] = train_teacher(
            fmri_evc, clip, train_idx, val_idx, device, epochs=args.teacher_epochs
        )
        teacher_vent, ceil["ventral"] = train_teacher(
            fmri_vent, clip, train_idx, val_idx, device, epochs=args.teacher_epochs
        )
        torch.save(
            {"model": teacher_evc.state_dict(), "in_dim": fmri_evc.shape[1], "top1": ceil["evc"]},
            out_dir / "teachers" / "teacher_evc.pt",
        )
        torch.save(
            {"model": teacher_vent.state_dict(), "in_dim": fmri_vent.shape[1], "top1": ceil["ventral"]},
            out_dir / "teachers" / "teacher_ventral.pt",
        )
        print(f"[OK] teacher ceilings evc={ceil['evc']*100:.2f}% vent={ceil['ventral']*100:.2f}%")
    if args.mode == "global":
        if fmri_global is None:
            raise RuntimeError("global mode requires fmri_global in region pairs")
        teacher_global = FmriClipTeacher.from_checkpoint(str(ROOT / args.global_teacher_ckpt), device)
        with torch.no_grad():
            pred = teacher_global(torch.from_numpy(fmri_global[val_idx]).to(device)).cpu()
            ceil["global"] = retrieval_metrics(pred, torch.from_numpy(clip[val_idx]))["top1"]
        print(f"[OK] global teacher ceiling={ceil['global']*100:.2f}%")

    model = RegionBridgeEEG(n_channels=eeg.shape[1], clip_dim=clip.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    base_lrs = [args.lr]

    idx_occ = channel_index(ch_names, OCC_CH)
    idx_vent = channel_index(ch_names, VENT_CH)
    early = tuple(float(x) for x in args.early_win.split(","))
    late = tuple(float(x) for x in args.late_win.split(","))
    sl_early = time_slice(eeg.shape[-1], sfreq, tmin, early)
    sl_late = time_slice(eeg.shape[-1], sfreq, tmin, late)
    print(
        f"[INFO] early={early}->{sl_early.start}:{sl_early.stop} "
        f"late={late}->{sl_late.start}:{sl_late.stop} occ={len(idx_occ)} vent_ch={len(idx_vent)}"
    )

    if args.mode == "global":
        tensors = (
            torch.from_numpy(eeg[train_idx]),
            torch.from_numpy(fmri_global[train_idx]),
            torch.from_numpy(clip[train_idx]),
        )
    else:
        tensors = (
            torch.from_numpy(eeg[train_idx]),
            torch.from_numpy(fmri_evc[train_idx]),
            torch.from_numpy(fmri_vent[train_idx]),
            torch.from_numpy(clip[train_idx]),
        )
    loader = DataLoader(TensorDataset(*tensors), batch_size=args.batch_size, shuffle=True, drop_last=True)

    history, best = [], -1.0
    best_path = out_dir / "checkpoints" / "best.pt"

    for epoch in range(1, args.epochs + 1):
        _cosine_lr(opt, epoch, args.epochs, base_lrs, warmup=3)
        model.train()
        loss_sum = n_seen = 0
        for batch in loader:
            if args.mode == "global":
                eb, fg, cb = batch
                eb, fg, cb = eb.to(device), fg.to(device), cb.to(device)
                out = model(eb)
                losses = dual_teacher_losses(
                    out["clip_img"],
                    cb,
                    None,
                    lambda_nce=args.lambda_nce,
                    lambda_mse_img=args.lambda_mse_img,
                    lambda_mse_fmri=0.0,
                )
                with torch.no_grad():
                    tg = teacher_global(fg)
                total = losses["total"] + args.lambda_global * F.mse_loss(out["clip_img"], tg)
            else:
                eb, fe, fv, cb = batch
                eb, fe, fv, cb = eb.to(device), fe.to(device), fv.to(device), cb.to(device)
                out = model.forward_views(
                    eb, idx_occ=idx_occ, idx_vent=idx_vent, sl_early=sl_early, sl_late=sl_late
                )
                losses = dual_teacher_losses(
                    out["clip_img"],
                    cb,
                    None,
                    lambda_nce=args.lambda_nce,
                    lambda_mse_img=args.lambda_mse_img,
                    lambda_mse_fmri=0.0,
                )
                total = losses["total"]
                if args.mode == "region":
                    with torch.no_grad():
                        te = teacher_evc(fe)
                        tv = teacher_vent(fv)
                    total = total + args.lambda_evc * F.mse_loss(out["clip_evc"], te)
                    total = total + args.lambda_vent * F.mse_loss(out["clip_vent"], tv)
                elif args.mode == "fake":
                    perm_e = torch.randperm(cb.size(0), device=device)
                    perm_v = torch.randperm(cb.size(0), device=device)
                    with torch.no_grad():
                        te = teacher_evc(fe)[perm_e]
                        tv = teacher_vent(fv)[perm_v]
                    total = total + args.lambda_evc * F.mse_loss(out["clip_evc"], te)
                    total = total + args.lambda_vent * F.mse_loss(out["clip_vent"], tv)

            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += float(total) * eb.size(0)
            n_seen += eb.size(0)

        ret = eval_img(model, eeg[val_idx], clip[val_idx], device)
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(n_seen, 1),
            "val_top1": ret["top1"],
            "val_top5": ret["top5"],
            "chance_top1": ret["chance_top1"],
            "mode": args.mode,
            "teacher_ceilings": ceil,
        }
        history.append(row)
        print(
            f"[{args.mode} {epoch:03d}] loss={row['train_loss']:.4f} "
            f"val_t1={ret['top1']*100:.2f}% t5={ret['top5']*100:.2f}%"
        )
        if ret["top1"] > best:
            best = ret["top1"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "ch_names": ch_names,
                    "best_val_top1": best,
                    "history": history,
                    "ceilings": ceil,
                },
                best_path,
            )

    metrics = {
        "mode": args.mode,
        "subject": args.subject,
        "best_val_top1": best,
        "chance_top1": 1.0 / max(len(val_idx), 1),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "teacher_ceilings": ceil,
        "ckpt": str(best_path),
    }
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[OK] best_top1={best*100:.2f}% -> {best_path}")

    if not args.skip_gen and best_path.is_file():
        ck = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        model.eval()
        take = val_idx[: args.max_images]
        with torch.no_grad():
            pred = model(torch.from_numpy(eeg[take]).to(device))["clip_img"].cpu().numpy()
        sys.path.insert(0, str(ROOT / "scripts"))
        from eval_nod_phase2_gt_fmri2image import _make_grid, _resolve_stim, generate_images_sdxl

        images_root = ROOT / "data/nod/raw/ds005811/stimuli/ImageNet"
        gt_paths = [_resolve_stim(images_root, ids[i]) for i in take]
        if any(p is None for p in gt_paths):
            print("[WARN] skip gen: missing stimuli")
        else:
            gen = generate_images_sdxl(
                pred, out_dir / "generated" / args.mode, device, max_images=args.max_images, tag=args.mode
            )
            _make_grid(gen, gt_paths, out_dir / f"grid_{args.mode}.png")
            print(f"[OK] grid -> {out_dir / f'grid_{args.mode}.png'}")


if __name__ == "__main__":
    main()
