#!/usr/bin/env python3
"""NOD cascade toward NeuroBOLT → fMRI → BiT → Image.

Stage A: train NodRoiBitDecoder on GT classmean fMRI → CLIP
Stage B: continue NeuroBOLT FT with frozen decoder CLIP alignment
Stage C: retrieval + optional SDXL grids (GT vs pred)
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from eeg_brainit.models.eeg2fmri import clip_losses, fmri_losses, retrieval_metrics
from eeg_brainit.models.neurobolt_eeg2fmri import NeuroBoltEEG2fMRI
from eeg_brainit.models.nod_roi_bit import NodRoiBitDecoder
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


def _cosine_lr(optimizer, epoch, epochs, base_lrs, warmup):
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


def load_multiclip_subjects(pairs_root: Path, clip_dir: Path, max_subjects: int = 0):
    index = json.loads((clip_dir / "index.json").read_text(encoding="utf-8"))
    emb = np.load(clip_dir / "embeddings.npy").astype(np.float32)
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)

    subs = sorted([p.name for p in pairs_root.glob("sub-*") if p.is_dir()])
    if max_subjects > 0:
        subs = subs[:max_subjects]

    eegs, fmris, clips, ids, ch_ref = [], [], [], [], None
    for sub in subs:
        z = np.load(pairs_root / sub / "pairs.npz")
        meta = json.loads((pairs_root / sub / "pairs_meta.json").read_text(encoding="utf-8"))
        ch = meta.get("ch_names")
        if not ch:
            raise RuntimeError(f"missing ch_names in {sub}")
        if ch_ref is None:
            ch_ref = list(ch)
        elif list(ch) != ch_ref:
            raise RuntimeError(f"channel mismatch {sub}")
        trials = meta["trials"]
        keep = []
        clip_rows = []
        id_rows = []
        for i, t in enumerate(trials):
            iid = t["image_id"]
            if iid not in index:
                continue
            keep.append(i)
            clip_rows.append(emb[index[iid]])
            id_rows.append(f"{sub}:{iid}")
        if not keep:
            print(f"[WARN] {sub}: no CLIP overlap, skip")
            continue
        eegs.append(z["eeg"][keep])
        fmris.append(z["fmri_roi"][keep])
        clips.append(np.stack(clip_rows, 0))
        ids.extend(id_rows)
        print(f"[INFO] {sub} n={len(keep)}/{len(trials)} eeg={tuple(z['eeg'].shape[1:])} rois={z['fmri_roi'].shape[1]}")
    if not eegs:
        raise RuntimeError("no subjects with CLIP overlap")
    return (
        subs,
        np.concatenate(eegs, 0),
        np.concatenate(fmris, 0),
        np.concatenate(clips, 0),
        ids,
        ch_ref,
    )


def build_neurobolt(ch_names, num_rois, cfg_eeg2fmri, device, heads_only=True):
    mcfg = cfg_eeg2fmri
    model = NeuroBoltEEG2fMRI(
        ch_names=ch_names,
        num_rois=num_rois,
        clip_dim=int(mcfg.get("clip_dim", 1024)),
        hidden=int(mcfg.get("hidden", 512)),
        dropout=float(mcfg.get("dropout", 0.3)),
        use_clip_head=False,
        glb_ckpt=str(mcfg.get("glb_ckpt", "checkpoints/neurobolt/glb.pth")),
        patch_size=int(mcfg.get("patch_size", 200)),
        win_level=int(mcfg.get("win_level", 1)),
        unfreeze_last_n_blocks=int(mcfg.get("unfreeze_last_n_blocks", 2)),
        train_mss=False,
        use_mss=bool(mcfg.get("use_mss", False)),
        train_patch_embed=False,
        heads_only=heads_only,
        head_depth=int(mcfg.get("head_depth", 1)),
    ).to(device)
    return model


def stage_a_train_decoder(
    fmri,
    clip,
    train_idx,
    val_idx,
    device,
    out_dir: Path,
    epochs: int = 30,
    lr: float = 1e-4,
    batch_size: int = 64,
    brainit_ckpt: str | None = None,
):
    decoder = NodRoiBitDecoder(
        roi_dim=fmri.shape[1],
        brain_dim=512,
        num_brain_tokens=64,
        num_query_tokens=128,
        num_blocks=2,
        clip_dim=clip.shape[1],
        dropout=0.1,
        brainit_ckpt=brainit_ckpt,
    ).to(device)
    opt = torch.optim.AdamW(decoder.parameters(), lr=lr, weight_decay=0.05)
    base_lrs = [lr]
    best = -1.0
    best_state = None
    history = []
    tr = TensorDataset(torch.from_numpy(fmri[train_idx]), torch.from_numpy(clip[train_idx]))
    va_f = torch.from_numpy(fmri[val_idx]).to(device)
    va_c = torch.from_numpy(clip[val_idx]).to(device)
    loader = DataLoader(tr, batch_size=batch_size, shuffle=True, drop_last=True)

    for epoch in range(1, epochs + 1):
        _cosine_lr(opt, epoch, epochs, base_lrs, warmup=3)
        decoder.train()
        loss_sum = n_seen = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = decoder(xb)["clip_emb"]
            loss = clip_losses(pred, yb, temp=0.07)["total"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            opt.step()
            loss_sum += float(loss) * xb.size(0)
            n_seen += xb.size(0)
        decoder.eval()
        with torch.no_grad():
            pred = decoder(va_f)["clip_emb"]
            ret = retrieval_metrics(pred, va_c)
            cos = float((F.normalize(pred, dim=-1) * F.normalize(va_c, dim=-1)).sum(-1).mean())
        row = {"epoch": epoch, "train_loss": loss_sum / max(n_seen, 1), "val_cos": cos, **ret}
        history.append(row)
        print(f"[A {epoch:03d}] loss={row['train_loss']:.4f} cos={cos:.4f} top1={ret['top1']*100:.2f}% top5={ret['top5']*100:.2f}%")
        if ret["top1"] > best:
            best = ret["top1"]
            best_state = {k: v.detach().cpu().clone() for k, v in decoder.state_dict().items()}
            torch.save(
                {
                    "model": best_state,
                    "roi_dim": fmri.shape[1],
                    "clip_dim": clip.shape[1],
                    "brain_dim": 512,
                    "num_brain_tokens": 64,
                    "num_query_tokens": 128,
                    "num_blocks": 2,
                    "best_top1": best,
                    "history": history,
                },
                out_dir / "decoder_best.pt",
            )
    if best_state is not None:
        decoder.load_state_dict(best_state)
    return decoder, {"best_top1": best, "history": history}


def _match_moments(pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Calibrate pred to ref's per-dim mean/std (keeps decoder on-manifold)."""
    pred = pred.float()
    ref = ref.float()
    mu_p = pred.mean(dim=0, keepdim=True)
    sd_p = pred.std(dim=0, keepdim=True).clamp_min(1e-4)
    mu_r = ref.mean(dim=0, keepdim=True)
    sd_r = ref.std(dim=0, keepdim=True).clamp_min(1e-4)
    return (pred - mu_p) / sd_p * sd_r + mu_r


def stage_b0_adapt_decoder(
    eeg,
    fmri,
    clip,
    train_idx,
    val_idx,
    ch_names,
    phase1_ckpt: Path,
    decoder: NodRoiBitDecoder,
    device,
    out_dir: Path,
    epochs: int = 15,
    lr: float = 5e-5,
    batch_size: int = 64,
):
    """Freeze NeuroBOLT; adapt BiT decoder to predicted-fMRI distribution (+ keep GT)."""
    # Preserve Stage-A GT decoder before adaptation overwrites decoder_best.pt.
    stage_a_path = out_dir / "decoder_stage_a.pt"
    best_path = out_dir / "decoder_best.pt"
    if best_path.is_file() and not stage_a_path.is_file():
        import shutil

        shutil.copy2(best_path, stage_a_path)
        print(f"[B0] preserved Stage-A decoder -> {stage_a_path}")

    ck = torch.load(phase1_ckpt, map_location="cpu", weights_only=False)
    cfg = ck.get("cfg", {})
    mcfg = cfg.get("eeg2fmri", {})
    nb = build_neurobolt(ch_names, fmri.shape[1], mcfg, device, heads_only=True)
    nb.load_state_dict(ck["model"], strict=False)
    nb.eval()
    for p in nb.parameters():
        p.requires_grad = False

    for p in decoder.parameters():
        p.requires_grad = True
    decoder.train()
    opt = torch.optim.AdamW(decoder.parameters(), lr=lr, weight_decay=0.05)
    base_lrs = [lr]

    # Cache Phase-1 predictions once (fixed teacher features).
    print("[B0] caching Phase-1 fMRI predictions...")
    pred_all = []
    with torch.no_grad():
        x = torch.from_numpy(eeg)
        for i in range(0, len(x), 128):
            pred_all.append(nb(x[i : i + 128].to(device))["fmri_pred"].cpu())
    pred_all = torch.cat(pred_all, 0).numpy().astype(np.float32)

    tr = TensorDataset(
        torch.from_numpy(fmri[train_idx]),
        torch.from_numpy(pred_all[train_idx]),
        torch.from_numpy(clip[train_idx]),
    )
    loader = DataLoader(tr, batch_size=batch_size, shuffle=True, drop_last=True)
    va_f = torch.from_numpy(fmri[val_idx]).to(device)
    va_p = torch.from_numpy(pred_all[val_idx]).to(device)
    va_c = torch.from_numpy(clip[val_idx]).to(device)

    best = -1.0
    best_state = None
    history = []
    for epoch in range(1, epochs + 1):
        _cosine_lr(opt, epoch, epochs, base_lrs, warmup=2)
        decoder.train()
        loss_sum = n_seen = 0
        for gt, pred, cb in loader:
            gt, pred, cb = gt.to(device), pred.to(device), cb.to(device)
            pred_cal = _match_moments(pred, gt)
            # Mix GT + calibrated pred so ceiling does not collapse.
            loss_gt = clip_losses(decoder(gt)["clip_emb"], cb, temp=0.07)["total"]
            loss_pr = clip_losses(decoder(pred_cal)["clip_emb"], cb, temp=0.07)["total"]
            loss = 0.5 * loss_gt + 0.5 * loss_pr
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            opt.step()
            loss_sum += float(loss) * gt.size(0)
            n_seen += gt.size(0)

        decoder.eval()
        with torch.no_grad():
            ret_gt = retrieval_metrics(decoder(va_f)["clip_emb"], va_c)
            ret_pr = retrieval_metrics(decoder(_match_moments(va_p, va_f))["clip_emb"], va_c)
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(n_seen, 1),
            "gt_top1": ret_gt["top1"],
            "pred_top1": ret_pr["top1"],
            "gt_top5": ret_gt["top5"],
            "pred_top5": ret_pr["top5"],
        }
        history.append(row)
        print(
            f"[B0 {epoch:03d}] loss={row['train_loss']:.4f} "
            f"gt_t1={ret_gt['top1']*100:.2f}% pred_t1={ret_pr['top1']*100:.2f}% "
            f"pred_t5={ret_pr['top5']*100:.2f}%"
        )
        if ret_pr["top1"] > best:
            best = ret_pr["top1"]
            best_state = {k: v.detach().cpu().clone() for k, v in decoder.state_dict().items()}
            torch.save(
                {
                    "model": best_state,
                    "roi_dim": fmri.shape[1],
                    "clip_dim": clip.shape[1],
                    "brain_dim": 512,
                    "num_brain_tokens": 64,
                    "num_query_tokens": 128,
                    "num_blocks": 2,
                    "best_pred_top1": best,
                    "gt_top1": ret_gt["top1"],
                    "history": history,
                },
                out_dir / "decoder_best.pt",
            )
    if best_state is not None:
        decoder.load_state_dict(best_state)
    return decoder, history


def stage_b_align_neurobolt(
    eeg,
    fmri,
    clip,
    train_idx,
    val_idx,
    ch_names,
    phase1_ckpt: Path,
    decoder: NodRoiBitDecoder,
    device,
    out_dir: Path,
    epochs: int = 40,
    lr: float = 3e-5,
    backbone_lr: float = 3e-6,
    batch_size: int = 32,
    lambda_fmri: float = 1.0,
    lambda_bridge: float = 0.5,
    lambda_img: float = 0.05,
):
    """Align NeuroBOLT to decoder(GT) manifold + weak image CLIP (decoder frozen)."""
    ck = torch.load(phase1_ckpt, map_location="cpu", weights_only=False)
    cfg = ck.get("cfg", {})
    mcfg = cfg.get("eeg2fmri", {})
    model = build_neurobolt(ch_names, fmri.shape[1], mcfg, device, heads_only=True)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    print(f"[B] load Phase-1 missing={len(missing)} unexpected={len(unexpected)}")
    for p in decoder.parameters():
        p.requires_grad = False
    decoder.eval()

    opt = torch.optim.AdamW(model.trainable_parameter_groups(0.0, lr, 0.08))
    base_lrs = [g["lr"] for g in opt.param_groups]
    backbone_unfrozen = False
    best = -1e9
    history = []
    tr = TensorDataset(
        torch.from_numpy(eeg[train_idx]),
        torch.from_numpy(fmri[train_idx]),
        torch.from_numpy(clip[train_idx]),
    )
    loader = DataLoader(tr, batch_size=batch_size, shuffle=True, drop_last=True)
    va_e = torch.from_numpy(eeg[val_idx])
    va_f = torch.from_numpy(fmri[val_idx]).to(device)
    va_c = torch.from_numpy(clip[val_idx]).to(device)
    heads_only_epochs = max(8, epochs // 3)

    for epoch in range(1, epochs + 1):
        if (not backbone_unfrozen) and epoch > heads_only_epochs:
            print("[B] unfreezing last TS blocks")
            model.unfreeze_backbone()
            opt = torch.optim.AdamW(model.trainable_parameter_groups(backbone_lr, lr, 0.08))
            base_lrs = [g["lr"] for g in opt.param_groups]
            backbone_unfrozen = True

        # ramp image CLIP slowly
        lam_img = lambda_img * min(1.0, max(0.0, (epoch - heads_only_epochs) / max(epochs - heads_only_epochs, 1)))
        lam_bridge = lambda_bridge
        _cosine_lr(opt, epoch, epochs, base_lrs, warmup=3)

        model.train()
        loss_sum = corr_sum = n_seen = 0
        for eb, fb, cb in loader:
            eb, fb, cb = eb.to(device), fb.to(device), cb.to(device)
            out = model(eb)
            pred = out["fmri_pred"]
            fl = fmri_losses(pred, fb, 1.0, 1.5, 0.3, 0.1, 0.1)
            pred_cal = _match_moments(pred, fb)
            with torch.no_grad():
                teacher = decoder(fb)["clip_emb"]
            student = decoder(pred_cal)["clip_emb"]
            bridge = clip_losses(student, teacher, temp=0.1)["total"]
            img = clip_losses(student, cb, temp=0.07)["total"]
            total = lambda_fmri * fl["total"] + lam_bridge * bridge + lam_img * img
            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            loss_sum += float(total) * eb.size(0)
            corr_sum += float(fl["corr"]) * eb.size(0)
            n_seen += eb.size(0)

        model.eval()
        preds_f, preds_c = [], []
        with torch.no_grad():
            for i in range(0, len(va_e), batch_size):
                out = model(va_e[i : i + batch_size].to(device))
                pred = out["fmri_pred"]
                preds_f.append(pred)
                preds_c.append(decoder(_match_moments(pred, va_f[i : i + batch_size]))["clip_emb"])
            pred_f = torch.cat(preds_f, 0)
            pred_c = torch.cat(preds_c, 0)
            fl = fmri_losses(pred_f, va_f, 1.0, 1.5, 0.3, 0.1, 0.1)
            ret_f = retrieval_metrics(pred_f, va_f)
            ret_c = retrieval_metrics(pred_c, va_c)
            ret_gt = retrieval_metrics(decoder(va_f)["clip_emb"], va_c)
        score = float(fl["corr"]) * 10 + float(ret_c["top5"]) * 5 + float(ret_c["top1"]) * 20
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(n_seen, 1),
            "train_corr": corr_sum / max(n_seen, 1),
            "val_corr": float(fl["corr"]),
            "fmri_top1": ret_f["top1"],
            "fmri_top5": ret_f["top5"],
            "clip_top1": ret_c["top1"],
            "clip_top5": ret_c["top5"],
            "gt_clip_top1": ret_gt["top1"],
            "lam_img": lam_img,
            "score": score,
        }
        history.append(row)
        print(
            f"[B {epoch:03d}] loss={row['train_loss']:.4f} "
            f"fmri_corr={row['train_corr']:.4f}/{row['val_corr']:.4f} "
            f"clip_t1={ret_c['top1']*100:.2f}%/{ret_gt['top1']*100:.2f}%(gt) "
            f"clip_t5={ret_c['top5']*100:.2f}% lam_img={lam_img:.3f}"
        )
        if score > best:
            best = score
            torch.save(
                {
                    "model": model.state_dict(),
                    "cfg": cfg,
                    "epoch": epoch,
                    "val_corr": row["val_corr"],
                    "clip_top1": ret_c["top1"],
                    "clip_top5": ret_c["top5"],
                    "gt_clip_top1": ret_gt["top1"],
                    "history": history,
                    "decoder_path": str(out_dir / "decoder_best.pt"),
                    "use_moment_match": True,
                },
                out_dir / "neurobolt_aligned_best.pt",
            )
    return history


@torch.no_grad()
def _load_decoder(ck_path: Path, device: torch.device) -> NodRoiBitDecoder:
    dec_ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    decoder = NodRoiBitDecoder(
        roi_dim=int(dec_ck["roi_dim"]),
        brain_dim=int(dec_ck.get("brain_dim", 512)),
        num_brain_tokens=int(dec_ck.get("num_brain_tokens", 64)),
        num_query_tokens=int(dec_ck.get("num_query_tokens", 128)),
        num_blocks=int(dec_ck.get("num_blocks", 2)),
        clip_dim=int(dec_ck["clip_dim"]),
        dropout=0.1,
        brainit_ckpt=None,
    ).to(device)
    decoder.load_state_dict(dec_ck["model"])
    return decoder


def stage_c_eval_and_gen(
    eeg,
    fmri,
    clip,
    ids,
    val_idx,
    ch_names,
    out_dir: Path,
    device,
    images_root: Path,
    max_images: int = 8,
    gen: bool = True,
):
    from PIL import Image

    # local import of generation helpers
    sys.path.insert(0, str(ROOT / "scripts"))
    from eval_nod_phase2_gt_fmri2image import (
        _make_grid,
        _resolve_stim,
        generate_images_sdxl,
    )

    decoder = _load_decoder(out_dir / "decoder_best.pt", device)
    decoder.eval()

    nb_path = out_dir / "neurobolt_aligned_best.pt"
    nb_ck = torch.load(nb_path, map_location="cpu", weights_only=False)
    model = build_neurobolt(ch_names, fmri.shape[1], nb_ck.get("cfg", {}).get("eeg2fmri", {}), device, heads_only=True)
    model.load_state_dict(nb_ck["model"], strict=False)
    model.eval()

    use_mm = bool(nb_ck.get("use_moment_match", True))
    preds_f, preds_c, preds_c_raw = [], [], []
    bs = 64
    gt_f = torch.from_numpy(fmri[val_idx]).to(device)
    for i in range(0, len(val_idx), bs):
        idx = val_idx[i : i + bs]
        out = model(torch.from_numpy(eeg[idx]).to(device))
        pred = out["fmri_pred"]
        preds_f.append(pred.cpu())
        preds_c_raw.append(decoder(pred)["clip_emb"].cpu())
        pred_in = _match_moments(pred, gt_f[i : i + bs]) if use_mm else pred
        preds_c.append(decoder(pred_in)["clip_emb"].cpu())
    pred_f = torch.cat(preds_f)
    pred_c = torch.cat(preds_c)
    pred_c_raw = torch.cat(preds_c_raw)
    gt_c = torch.from_numpy(clip[val_idx])
    report = {
        "val_n": len(val_idx),
        "use_moment_match": use_mm,
        "fmri_retrieval": retrieval_metrics(pred_f, gt_f.cpu()),
        "cascade_clip_retrieval": retrieval_metrics(pred_c, gt_c),
        "cascade_clip_retrieval_raw": retrieval_metrics(pred_c_raw, gt_c),
        "gt_fmri_clip_retrieval": retrieval_metrics(decoder(gt_f)["clip_emb"].cpu(), gt_c),
        "neurobolt_ckpt": str(nb_path),
        "decoder_ckpt": str(out_dir / "decoder_best.pt"),
    }
    print(
        f"[C] cascade clip top1={report['cascade_clip_retrieval']['top1']*100:.2f}% "
        f"raw={report['cascade_clip_retrieval_raw']['top1']*100:.2f}% "
        f"gt_ceiling={report['gt_fmri_clip_retrieval']['top1']*100:.2f}%"
    )

    if gen:
        take = val_idx[:max_images]
        gt_paths = []
        for i in take:
            iid = ids[i].split(":", 1)[-1]
            p = _resolve_stim(images_root, iid)
            if p is None:
                raise FileNotFoundError(iid)
            gt_paths.append(p)
        with torch.no_grad():
            out = model(torch.from_numpy(eeg[take]).to(device))
            pred = out["fmri_pred"]
            gt_t = torch.from_numpy(fmri[take]).to(device)
            gt_clip = decoder(gt_t)["clip_emb"].cpu().numpy()
            pred_in = _match_moments(pred, gt_t) if use_mm else pred
            pred_clip = decoder(pred_in)["clip_emb"].cpu().numpy()
        p_gt = generate_images_sdxl(gt_clip, out_dir / "generated" / "gt_fmri", device, max_images=max_images, tag="gt_bit")
        p_pr = generate_images_sdxl(pred_clip, out_dir / "generated" / "cascade", device, max_images=max_images, tag="cascade")
        _make_grid(p_gt, gt_paths, out_dir / "grid_gt_fmri_bit.png")
        _make_grid(p_pr, gt_paths, out_dir / "grid_cascade.png")
        report["generation"] = {
            "gt_grid": str(out_dir / "grid_gt_fmri_bit.png"),
            "cascade_grid": str(out_dir / "grid_cascade.png"),
        }

    (out_dir / "cascade_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[OK] wrote {out_dir / 'cascade_metrics.json'}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-root", default="data/nod/processed/classmean")
    parser.add_argument("--clip-dir", default="data/nod/processed/clip_vit_h14_all")
    parser.add_argument("--fallback-clip-dir", default="data/nod/processed/clip_vit_h14")
    parser.add_argument("--phase1-ckpt", default="outputs/nod_eeg2fmri/neurobolt_classmean_v3/checkpoints/best.pt")
    parser.add_argument("--brainit-ckpt", default="checkpoints/brain_it/decoder_clipg_state_dict.pt")
    parser.add_argument("--images-root", default="data/nod/raw/ds005811/stimuli/ImageNet")
    parser.add_argument("--output-dir", default="outputs/nod_cascade/neurobolt_bit_v1")
    parser.add_argument("--max-subjects", type=int, default=0)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage-a-epochs", type=int, default=25)
    parser.add_argument("--stage-b0-epochs", type=int, default=15)
    parser.add_argument("--stage-b-epochs", type=int, default=40)
    parser.add_argument("--max-images", type=int, default=8)
    parser.add_argument("--skip-gen", action="store_true")
    parser.add_argument("--stage", default="all", choices=["all", "a", "b0", "b", "c"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pairs_root = ROOT / args.pairs_root
    clip_dir = ROOT / args.clip_dir
    if not (clip_dir / "index.json").is_file():
        clip_dir = ROOT / args.fallback_clip_dir
        print(f"[WARN] full CLIP missing, fallback {clip_dir}")
    out_dir = ROOT / args.output_dir
    ensure_dirs(out_dir, out_dir / "generated")

    names, eeg, fmri, clip, ids, ch_names = load_multiclip_subjects(
        pairs_root, clip_dir, max_subjects=args.max_subjects
    )
    train_idx, val_idx = _split_by_image_id(ids, args.val_frac, args.seed)
    print(f"[INFO] n={len(ids)} train={len(train_idx)} val={len(val_idx)} device={device} subjects~{names}")

    decoder = None
    if args.stage in {"all", "a"}:
        decoder, meta_a = stage_a_train_decoder(
            fmri,
            clip,
            train_idx,
            val_idx,
            device,
            out_dir,
            epochs=args.stage_a_epochs,
            brainit_ckpt=str(ROOT / args.brainit_ckpt),
        )
        (out_dir / "stage_a.json").write_text(json.dumps(meta_a, indent=2), encoding="utf-8")

    if args.stage in {"all", "b0", "b"}:
        if decoder is None:
            decoder = _load_decoder(out_dir / "decoder_best.pt", device)

    if args.stage in {"all", "b0"} or args.stage == "b":
        # Stage B always starts with B0 decoder adaptation unless only-c.
        hist_b0 = stage_b0_adapt_decoder(
            eeg,
            fmri,
            clip,
            train_idx,
            val_idx,
            ch_names,
            ROOT / args.phase1_ckpt,
            decoder,
            device,
            out_dir,
            epochs=args.stage_b0_epochs,
        )
        (out_dir / "stage_b0.json").write_text(json.dumps(hist_b0, indent=2), encoding="utf-8")
        decoder = _load_decoder(out_dir / "decoder_best.pt", device)

    if args.stage in {"all", "b"}:
        hist_b = stage_b_align_neurobolt(
            eeg,
            fmri,
            clip,
            train_idx,
            val_idx,
            ch_names,
            ROOT / args.phase1_ckpt,
            decoder,
            device,
            out_dir,
            epochs=args.stage_b_epochs,
        )
        (out_dir / "stage_b.json").write_text(json.dumps(hist_b, indent=2), encoding="utf-8")

    if args.stage in {"all", "c"}:
        stage_c_eval_and_gen(
            eeg,
            fmri,
            clip,
            ids,
            val_idx,
            ch_names,
            out_dir,
            device,
            ROOT / args.images_root,
            max_images=args.max_images,
            gen=not args.skip_gen,
        )


if __name__ == "__main__":
    main()
