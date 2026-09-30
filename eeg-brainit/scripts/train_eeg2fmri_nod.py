#!/usr/bin/env python3
"""Phase-1 main: fine-tune NeuroBOLT (glb.pth) on NOD EEG→fMRI."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from eeg_brainit.models.eeg2fmri import EEG2fMRIModel, clip_losses, fmri_losses, retrieval_metrics
from eeg_brainit.models.neurobolt_eeg2fmri import NeuroBoltEEG2fMRI
from eeg_brainit.utils.config import ensure_dirs, load_config


def _split_indices(n: int, val_frac: float, seed: int) -> tuple[list[int], list[int]]:
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_val = max(1, int(n * val_frac)) if n > 1 else 0
    return idx[n_val:], idx[:n_val]


def _split_by_image_id(
    image_ids: list[str],
    val_frac: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Hold out whole images so train/val never share the same stimulus."""
    rng = random.Random(seed)
    uniq = sorted(set(image_ids))
    rng.shuffle(uniq)
    n_val = max(1, int(len(uniq) * val_frac)) if len(uniq) > 1 else 0
    val_set = set(uniq[:n_val])
    train_idx = [i for i, iid in enumerate(image_ids) if iid not in val_set]
    val_idx = [i for i, iid in enumerate(image_ids) if iid in val_set]
    if not train_idx or not val_idx:
        return _split_indices(len(image_ids), val_frac, seed)
    return train_idx, val_idx


def _cosine_lr(optimizer: torch.optim.Optimizer, epoch: int, epochs: int, base_lrs: list[float], warmup: int) -> list[float]:
    lrs = []
    for pg, base_lr in zip(optimizer.param_groups, base_lrs):
        if epoch <= warmup:
            lr = base_lr * epoch / max(warmup, 1)
        else:
            t = (epoch - warmup) / max(epochs - warmup, 1)
            lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * t))
        pg["lr"] = lr
        lrs.append(lr)
    return lrs


def _build_model(cfg: dict, ch_names: list[str], device: torch.device):
    mcfg = cfg.get("eeg2fmri", {})
    backbone = str(mcfg.get("backbone", "neurobolt"))
    heads_only = int(cfg.get("train", {}).get("heads_only_epochs", 0)) > 0
    if backbone == "neurobolt":
        model = NeuroBoltEEG2fMRI(
            ch_names=ch_names,
            num_rois=int(mcfg.get("num_rois", 200)),
            clip_dim=int(mcfg.get("clip_dim", 1024)),
            hidden=int(mcfg.get("hidden", 1024)),
            dropout=float(mcfg.get("dropout", 0.15)),
            use_clip_head=bool(mcfg.get("use_clip_head", False)),
            glb_ckpt=str(mcfg.get("glb_ckpt", "checkpoints/neurobolt/glb.pth")),
            patch_size=int(mcfg.get("patch_size", 200)),
            win_level=int(mcfg.get("win_level", 1)),
            unfreeze_last_n_blocks=int(mcfg.get("unfreeze_last_n_blocks", 2)),
            train_mss=bool(mcfg.get("train_mss", False)),
            use_mss=bool(mcfg.get("use_mss", False)),
            train_patch_embed=bool(mcfg.get("train_patch_embed", False)),
            heads_only=heads_only,
            head_depth=int(mcfg.get("head_depth", 1)),
        )
    else:
        model = EEG2fMRIModel.from_config(cfg, ch_names=ch_names)
    return model.to(device)


def _load_subjects(meta: dict, max_subjects: int = 0) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray | None, list[str], list[str]]:
    subjects = list(meta["subjects"].items())
    if max_subjects > 0:
        subjects = subjects[:max_subjects]
    eegs, fmris, clips, ch_ref, names, image_ids = [], [], [], None, [], []
    for sub_name, sub_meta in subjects:
        cache_path = Path(sub_meta["cache"])
        if not cache_path.is_file():
            raise FileNotFoundError(cache_path)
        pairs_meta_path = cache_path.parent / "pairs_meta.json"
        pairs_meta = json.loads(pairs_meta_path.read_text(encoding="utf-8")) if pairs_meta_path.is_file() else {}
        ch_names = pairs_meta.get("ch_names")
        if not ch_names:
            raise RuntimeError(f"ch_names missing in {pairs_meta_path}")
        if ch_ref is None:
            ch_ref = list(ch_names)
        elif list(ch_names) != ch_ref:
            raise RuntimeError(f"channel mismatch {sub_name}: {ch_names[:3]} vs {ch_ref[:3]}")
        z = np.load(str(cache_path))
        eegs.append(z["eeg"])
        fmris.append(z["fmri_roi"])
        if "clip" in z.files:
            clips.append(z["clip"])
        names.append(sub_name)
        trials = pairs_meta.get("trials") or []
        if len(trials) != len(z["eeg"]):
            image_ids.extend([f"{sub_name}:{i}" for i in range(len(z["eeg"]))])
        else:
            image_ids.extend([f"{sub_name}:{t.get('image_id', i)}" for i, t in enumerate(trials)])
        print(f"[INFO] loaded {sub_name} n={len(z['eeg'])} eeg={tuple(z['eeg'].shape)} rois={z['fmri_roi'].shape[1]}")
    eeg = np.concatenate(eegs, axis=0)
    fmri = np.concatenate(fmris, axis=0)
    clip = np.concatenate(clips, axis=0) if clips and len(clips) == len(eegs) else None
    assert ch_ref is not None
    return names, eeg, fmri, clip, ch_ref, image_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/nod_eeg2fmri.yaml")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = ROOT / cfg.get("output_dir", "outputs/nod_eeg2fmri")
    ensure_dirs(out_dir, out_dir / "checkpoints")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pairs_manifest = ROOT / cfg.get("data", {}).get("manifest", "data/nod/manifests/nod_pairs.json")
    backbone = str(cfg.get("eeg2fmri", {}).get("backbone", "neurobolt"))

    if args.smoke_only:
        meta_path = ROOT / "data/nod/processed/sub-01/pairs_meta.json"
        ch_names = json.loads(meta_path.read_text())["ch_names"] if meta_path.is_file() else [f"C{i}" for i in range(1, 63)]
        cfg.setdefault("eeg2fmri", {})
        cfg["eeg2fmri"]["num_rois"] = int(cfg["eeg2fmri"].get("num_rois", 200))
        model = _build_model(cfg, ch_names, device)
        x = torch.randn(2, len(ch_names), 226, device=device)
        out = model(x)
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in model.parameters())
        print("[INFO] smoke", {k: tuple(v.shape) for k, v in out.items() if hasattr(v, "shape")})
        print(f"[INFO] params trainable={n_train:,} / total={n_all:,} backbone={backbone}")
        return

    if not pairs_manifest.is_file():
        print(f"[WAIT] Missing {pairs_manifest}")
        return
    meta = json.loads(pairs_manifest.read_text(encoding="utf-8"))
    if meta.get("status") != "pairs" or not meta.get("subjects"):
        print(f"[WAIT] Manifest not ready: {pairs_manifest}")
        return

    max_subjects = int(cfg.get("data", {}).get("max_subjects", 0))
    names, eeg, fmri, clip, ch_names, image_ids = _load_subjects(meta, max_subjects=max_subjects)
    n, c, t = eeg.shape
    num_rois = int(fmri.shape[1])
    has_clip = clip is not None and bool(cfg.get("eeg2fmri", {}).get("use_clip_head", False))
    clip_dim = int(clip.shape[1]) if clip is not None else int(cfg.get("eeg2fmri", {}).get("clip_dim", 1024))
    print(
        f"[INFO] subjects={names} n={n} eeg=({c},{t}) rois={num_rois} "
        f"clip_head={has_clip} backbone={backbone}"
    )

    cfg.setdefault("eeg2fmri", {})
    cfg["eeg2fmri"]["in_dim"] = int(c)
    cfg["eeg2fmri"]["num_rois"] = int(num_rois)
    cfg["eeg2fmri"]["clip_dim"] = int(clip_dim)
    cfg["eeg2fmri"]["use_clip_head"] = bool(has_clip)

    model = _build_model(cfg, ch_names, device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"[INFO] device={device} trainable={n_train:,} / total={n_all:,}")

    seed = int(cfg.get("train", {}).get("seed", 42))
    split_mode = str(cfg.get("data", {}).get("split", "image_id"))
    if split_mode == "image_id":
        train_idx, val_idx = _split_by_image_id(image_ids, float(cfg.get("data", {}).get("val_frac", 0.1)), seed)
        print(f"[INFO] split=image_id train={len(train_idx)} val={len(val_idx)} unique_img={len(set(image_ids))}")
    else:
        train_idx, val_idx = _split_indices(n, float(cfg.get("data", {}).get("val_frac", 0.1)), seed)
        print(f"[INFO] split=trial train={len(train_idx)} val={len(val_idx)}")

    def _subset(idxs: list[int]):
        e = torch.from_numpy(eeg[idxs])
        f = torch.from_numpy(fmri[idxs])
        if has_clip and clip is not None:
            return TensorDataset(e, f, torch.from_numpy(clip[idxs]))
        return TensorDataset(e, f)

    bs = int(cfg.get("train", {}).get("batch_size", 16))
    nw = int(cfg.get("train", {}).get("num_workers", 0))
    train_loader = DataLoader(_subset(train_idx), batch_size=bs, shuffle=True, num_workers=nw, drop_last=True)
    val_loader = DataLoader(_subset(val_idx), batch_size=bs, shuffle=False, num_workers=nw)

    head_lr = float(cfg.get("train", {}).get("learning_rate", 1e-4))
    backbone_lr = float(cfg.get("train", {}).get("backbone_lr", head_lr * 0.1))
    wd = float(cfg.get("train", {}).get("weight_decay", 0.05))
    heads_only_epochs = int(cfg.get("train", {}).get("heads_only_epochs", 0))
    min_epochs = int(cfg.get("train", {}).get("min_epochs", 25))

    def _build_optim():
        groups = model.trainable_parameter_groups(backbone_lr, head_lr, wd)
        return torch.optim.AdamW(groups), [g["lr"] for g in groups]

    opt, base_lrs = _build_optim()
    epochs = int(args.max_epochs) if args.max_epochs > 0 else int(cfg.get("train", {}).get("epochs", 80))
    warmup = int(cfg.get("train", {}).get("warmup_epochs", 5))
    patience = int(cfg.get("train", {}).get("early_stop_patience", 20))
    grad_clip = float(cfg.get("train", {}).get("grad_clip", 1.0))

    lam_mse0 = float(cfg.get("loss", {}).get("lambda_mse", 1.0))
    lam_corr0 = float(cfg.get("loss", {}).get("lambda_corr", 1.5))
    lam_cos0 = float(cfg.get("loss", {}).get("lambda_cosine", 0.5))
    lam_fnce0 = float(cfg.get("loss", {}).get("lambda_nce", 0.15))
    lam_clip0 = float(cfg.get("loss", {}).get("lambda_clip", 0.0))
    nce_temp = float(cfg.get("loss", {}).get("nce_temp", 0.1))
    select_ema = float(cfg.get("train", {}).get("select_ema", 0.8))

    history = []
    best_score = -1e9
    best_path = out_dir / "checkpoints" / "best.pt"
    stale = 0
    ema_val_corr = None
    backbone_unfrozen = heads_only_epochs <= 0

    for epoch in range(1, epochs + 1):
        if (not backbone_unfrozen) and epoch > heads_only_epochs:
            print(
                f"[INFO] unfreezing NeuroBOLT last-{cfg['eeg2fmri'].get('unfreeze_last_n_blocks', 2)} "
                f"TS blocks (use_mss={cfg['eeg2fmri'].get('use_mss', False)})"
            )
            model.unfreeze_backbone()
            opt, base_lrs = _build_optim()
            backbone_unfrozen = True
            print(f"[INFO] trainable now={sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        lrs = _cosine_lr(opt, epoch, epochs, base_lrs, warmup)
        phase = "heads_only" if not backbone_unfrozen else "joint"
        lam_mse, lam_corr, lam_cos, lam_fnce = lam_mse0, lam_corr0, lam_cos0, lam_fnce0
        lam_clip = lam_clip0 if has_clip else 0.0

        model.train()
        tr_loss = tr_corr = 0.0
        n_tr = 0
        for batch in train_loader:
            if has_clip:
                eeg_b, y, clip_b = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            else:
                eeg_b, y = batch[0].to(device), batch[1].to(device)
                clip_b = None
            out = model(eeg_b)
            losses = fmri_losses(out["fmri_pred"], y, lam_mse, lam_corr, lam_cos, lam_fnce, nce_temp)
            total = losses["total"]
            if has_clip and clip_b is not None and "clip_pred" in out and lam_clip > 0:
                total = total + lam_clip * clip_losses(out["clip_pred"], clip_b, nce_temp)["total"]
            opt.zero_grad(set_to_none=True)
            total.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip)
            opt.step()
            bs_ = eeg_b.size(0)
            tr_loss += float(total) * bs_
            tr_corr += float(losses["corr"]) * bs_
            n_tr += bs_
        tr_loss /= max(n_tr, 1)
        tr_corr /= max(n_tr, 1)

        model.eval()
        s_loss = s_corr = 0.0
        n_va = 0
        preds_f, tgts_f = [], []
        with torch.no_grad():
            for batch in val_loader:
                if has_clip:
                    eeg_b, y = batch[0].to(device), batch[1].to(device)
                else:
                    eeg_b, y = batch[0].to(device), batch[1].to(device)
                out = model(eeg_b)
                losses = fmri_losses(out["fmri_pred"], y, lam_mse0, lam_corr0, lam_cos0, lam_fnce0, nce_temp)
                bs_ = eeg_b.size(0)
                s_loss += float(losses["total"]) * bs_
                s_corr += float(losses["corr"]) * bs_
                n_va += bs_
                preds_f.append(out["fmri_pred"].cpu())
                tgts_f.append(y.cpu())
        va_loss = s_loss / max(n_va, 1)
        va_corr = s_corr / max(n_va, 1)
        fmri_ret = retrieval_metrics(torch.cat(preds_f), torch.cat(tgts_f))

        if ema_val_corr is None:
            ema_val_corr = va_corr
        else:
            ema_val_corr = select_ema * ema_val_corr + (1.0 - select_ema) * va_corr

        # Primary gate: EMA val fMRI corr + retrieval.
        score = float(ema_val_corr) * 10.0 + float(fmri_ret["top5"]) * 5.0
        if score > best_score:
            best_score = score
            stale = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "cfg": cfg,
                    "epoch": epoch,
                    "val_corr": va_corr,
                    "ema_val_corr": ema_val_corr,
                    "fmri_top1": fmri_ret["top1"],
                    "fmri_top5": fmri_ret["top5"],
                    "phase": phase,
                    "backbone": backbone,
                    "subjects": names,
                },
                best_path,
            )
        elif epoch >= min_epochs:
            stale += 1

        row = {
            "epoch": epoch,
            "phase": phase,
            "lr_backbone": lrs[0] if len(lrs) > 1 else 0.0,
            "lr_head": lrs[-1] if lrs else None,
            "train_loss": tr_loss,
            "train_corr": tr_corr,
            "val_loss": va_loss,
            "val_corr": va_corr,
            "ema_val_corr": ema_val_corr,
            "fmri_top1": fmri_ret["top1"],
            "fmri_top5": fmri_ret["top5"],
            "chance_top1": fmri_ret["chance_top1"],
            "score": score,
        }
        history.append(row)
        print(
            f"[epoch {epoch:03d}] phase={phase} lr={lrs[-1]:.2e}/{lrs[0]:.2e} "
            f"loss={tr_loss:.4f} fmri_corr={tr_corr:.4f}/{va_corr:.4f}(ema={ema_val_corr:.4f}) "
            f"fmri_t5={fmri_ret['top5']:.4f} chance={fmri_ret['chance_top1']:.4f}"
        )
        if patience > 0 and epoch >= min_epochs and stale >= patience:
            print(f"[EARLY STOP] best_score={best_score:.4f}")
            break

    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    torch.save({"model": model.state_dict(), "cfg": cfg, "history": history}, out_dir / "checkpoints" / "last.pt")
    print(f"[OK] wrote {out_dir} best_score={best_score}")


if __name__ == "__main__":
    main()
