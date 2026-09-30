#!/usr/bin/env python3
"""Evaluate Stage-4 EEG-Brain-IT checkpoint with CLIP retrieval.

Because the current pipeline predicts localized CLIP-space tokens (dim 1664) and
does not yet run diffusion image synthesis, this script reports:

1) Representation alignment via a ridge probe:
   mean-pooled EEG CLIP tokens -> OpenCLIP image embedding
2) Subject-wise 200-way identification on the THINGS-EEG2 test split
   (Top-1 / Top-5 / median rank)
3) Sanity metrics on virtual-fMRI outputs

Caches are written under outputs/eval/.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from eeg_brainit.data import ThingsEEG2Dataset
from eeg_brainit.models import EEGBrainITPipeline
from eeg_brainit.utils.config import ensure_dirs, load_config
from eeg_brainit.utils.metrics import pixel_correlation, ssim_simple


def deep_update(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


@torch.no_grad()
def extract_eeg_features(model, loader, device, max_batches: int = 0) -> dict:
    model.eval()
    feats, subjects, ids, image_paths = [], [], [], []
    vf_ssim = vf_pix = 0.0
    n_vf = 0
    used_clip_emb = False
    for i, batch in enumerate(tqdm(loader, desc="eeg-forward")):
        if max_batches > 0 and i >= max_batches:
            break
        out = model(batch["spectrogram"].to(device, non_blocking=True))
        if "clip_emb" in out:
            # Direct OpenCLIP-space embedding from ClipAlignHead.
            feats.append(F.normalize(out["clip_emb"].float(), dim=-1).cpu())
            used_clip_emb = True
        else:
            # Mean-pool localized CLIP tokens -> (B, D)
            tok = out["clip_tokens"].float().mean(dim=1)
            feats.append(tok.cpu())
        subjects.extend(batch["subject"])
        ids.extend(batch["id"])
        image_paths.extend(batch["image_path"])
        vf = out["virtual_fmri"]
        if vf.shape[1] >= 2:
            vf_ssim += ssim_simple(vf[:, 0:1], vf[:, 1:2])
            vf_pix += pixel_correlation(vf[:, 0:1], vf[:, 1:2])
            n_vf += 1
    return {
        "eeg_feat": torch.cat(feats, dim=0).numpy().astype(np.float32),
        "subjects": subjects,
        "ids": ids,
        "image_paths": image_paths,
        "vf_ssim": vf_ssim / max(n_vf, 1),
        "vf_pixcorr": vf_pix / max(n_vf, 1),
        "used_clip_emb": used_clip_emb,
    }


@torch.no_grad()
def encode_images(image_paths: list[str], model, preprocess, device, batch_size: int = 64) -> np.ndarray:
    embs = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="clip-images"):
        paths = image_paths[i : i + batch_size]
        imgs = []
        for p in paths:
            img = Image.open(p).convert("RGB")
            imgs.append(preprocess(img))
        x = torch.stack(imgs, dim=0).to(device)
        feat = model.encode_image(x)
        feat = F.normalize(feat.float(), dim=-1)
        embs.append(feat.cpu().numpy().astype(np.float32))
    return np.concatenate(embs, axis=0)


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float = 1e3) -> np.ndarray:
    """Return W for y ~= x @ W, with L2 regularization."""
    # x: (N, Dx), y: (N, Dy)
    xtx = x.T @ x
    d = xtx.shape[0]
    xtx = xtx + alpha * np.eye(d, dtype=np.float64)
    xty = x.T @ y.astype(np.float64)
    w = np.linalg.solve(xtx, xty)
    return w.astype(np.float32)


def retrieval_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    """pred/target are L2-normalized (N, D); evaluate N-way identification."""
    sim = pred @ target.T  # (N, N)
    ranks = []
    top1 = top5 = 0
    n = sim.shape[0]
    for i in range(n):
        order = np.argsort(-sim[i])
        rank = int(np.where(order == i)[0][0]) + 1
        ranks.append(rank)
        if rank == 1:
            top1 += 1
        if rank <= 5:
            top5 += 1
    ranks = np.asarray(ranks)
    return {
        "n": n,
        "top1": top1 / n,
        "top5": top5 / n,
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(ranks.mean()),
    }


def subjectwise_retrieval(
    eeg_feat: np.ndarray,
    img_feat: np.ndarray,
    subjects: list[str],
    image_paths: list[str],
    probe_w: np.ndarray,
) -> dict:
    """Map EEG -> CLIP space, then 200-way retrieval within each subject."""
    pred = eeg_feat.astype(np.float64) @ probe_w.astype(np.float64)
    pred = pred / np.linalg.norm(pred, axis=1, keepdims=True).clip(min=1e-8)
    img = img_feat / np.linalg.norm(img_feat, axis=1, keepdims=True).clip(min=1e-8)

    by_subj: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(subjects):
        by_subj[s].append(i)

    per = {}
    tops1, tops5, meds = [], [], []
    for s, idxs in sorted(by_subj.items()):
        # Deduplicate images within subject if needed: keep first occurrence mapping
        # Our test set is 200 unique images per subject with 1 EEG each.
        p = pred[idxs]
        t = img[idxs]
        # Ensure one-to-one by image path uniqueness
        paths = [image_paths[i] for i in idxs]
        if len(set(paths)) != len(paths):
            # Average EEG preds for duplicate images
            uniq = {}
            for local_i, path in enumerate(paths):
                uniq.setdefault(path, []).append(local_i)
            p2, t2 = [], []
            for path, locs in uniq.items():
                p2.append(p[locs].mean(axis=0))
                t2.append(t[locs[0]])
            p = np.stack(p2, axis=0)
            t = np.stack(t2, axis=0)
            p = p / np.linalg.norm(p, axis=1, keepdims=True).clip(min=1e-8)
        m = retrieval_metrics(p.astype(np.float32), t.astype(np.float32))
        per[s] = m
        tops1.append(m["top1"])
        tops5.append(m["top5"])
        meds.append(m["median_rank"])
    return {
        "per_subject": per,
        "macro_top1": float(np.mean(tops1)),
        "macro_top5": float(np.mean(tops5)),
        "macro_median_rank": float(np.mean(meds)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument(
        "--override",
        type=str,
        nargs="*",
        default=[],
        help="Optional yaml overlays (e.g. configs/clip_align.yaml)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/stage4_e2e/checkpoints/stage4_best_e30.pt",
    )
    parser.add_argument(
        "--direct-clip",
        action="store_true",
        help="If EEG feats are already 768-d CLIP emb, evaluate cosine retrieval without ridge",
    )
    parser.add_argument(
        "--subjects",
        type=str,
        nargs="*",
        default=None,
        help="Optional subject filter, e.g. sub-08",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--probe-train-split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--max-probe-batches", type=int, default=200, help="0 = all")
    parser.add_argument("--max-test-batches", type=int, default=0, help="0 = all test")
    parser.add_argument("--clip-model", type=str, default="ViT-L-14")
    parser.add_argument("--clip-pretrained", type=str, default="openai")
    parser.add_argument("--ridge-alpha", type=float, default=1e3)
    parser.add_argument("--output-dir", type=str, default="outputs/eval/stage4")
    parser.add_argument("--smoke", action="store_true", help="Tiny run for debugging")
    args = parser.parse_args()

    cfg = load_config(args.config)
    for ov in args.override:
        cfg = deep_update(cfg, load_config(ov))
    if args.subjects:
        cfg.setdefault("data", {})["subjects"] = list(args.subjects)
    root = Path(cfg.get("project_root", Path.cwd()))
    out_dir = root / args.output_dir
    ensure_dirs(out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")
    if device.type == "cuda":
        print(f"[INFO] GPU={torch.cuda.get_device_name(0)}")

    # Cache dirs for HF/open_clip
    cache_root = Path(cfg.get("cache_root", "/project/peilab/why/cache/eeg-brainit"))
    ensure_dirs(cache_root / "open_clip", cache_root / "hf")
    import os

    os.environ.setdefault("HF_HOME", str(cache_root / "hf"))
    os.environ.setdefault("OPENCLIP_CACHE_DIR", str(cache_root / "open_clip"))
    os.environ.setdefault("TORCH_HOME", str(cache_root / "torch"))

    model = EEGBrainITPipeline.from_config(cfg, project_root=root).to(device)
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        ckpt_path = root / args.checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    print(f"[INFO] loaded {ckpt_path} missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    manifest = root / cfg["data"]["manifest"]
    data_cfg = cfg["data"]
    subjects = data_cfg.get("subjects")
    if isinstance(subjects, str):
        subjects = [subjects]
    common = dict(
        manifest=manifest,
        root=root,
        image_size=int(data_cfg.get("image_size", 224)),
        n_fft=int(data_cfg.get("spectrogram", {}).get("n_fft", 64)),
        hop_length=int(data_cfg.get("spectrogram", {}).get("hop_length", 16)),
        target_time=int(data_cfg.get("spectrogram", {}).get("target_time", 64)),
        expected_channels=int(data_cfg.get("num_channels", 63)),
        subjects=subjects,
    )
    probe_ds = ThingsEEG2Dataset(**common, split=args.probe_train_split)
    test_ds = ThingsEEG2Dataset(**common, split="test")
    print(f"[INFO] subjects={subjects} probe={len(probe_ds)} test={len(test_ds)}")
    if args.smoke:
        args.max_probe_batches = 4
        args.max_test_batches = 8
        args.batch_size = min(args.batch_size, 4)
        print("[INFO] smoke mode: truncated batches")

    probe_loader = DataLoader(
        probe_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    print("[INFO] Extracting probe-set EEG features...")
    probe = extract_eeg_features(model, probe_loader, device, max_batches=args.max_probe_batches)
    print("[INFO] Extracting test-set EEG features...")
    test = extract_eeg_features(model, test_loader, device, max_batches=args.max_test_batches)

    import open_clip

    print(f"[INFO] Loading OpenCLIP {args.clip_model} / {args.clip_pretrained}")
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        args.clip_model, pretrained=args.clip_pretrained, device=device
    )
    clip_model.eval()

    print("[INFO] Encoding probe images...")
    probe_img = encode_images(probe["image_paths"], clip_model, preprocess, device)
    print("[INFO] Encoding test images...")
    test_img = encode_images(test["image_paths"], clip_model, preprocess, device)

    use_direct = bool(args.direct_clip or probe.get("used_clip_emb"))
    if use_direct and probe["eeg_feat"].shape[1] == probe_img.shape[1]:
        print("[INFO] Direct CLIP cosine retrieval (no ridge); identity probe")
        d = probe["eeg_feat"].shape[1]
        w = np.eye(d, dtype=np.float32)
        x_mean = np.zeros((1, d), dtype=np.float32)
        x_std = np.ones((1, d), dtype=np.float32)
        x_te = test["eeg_feat"]
        mode = "direct_clip"
    else:
        # Fit ridge probe on probe split
        x = probe["eeg_feat"]
        y = probe_img
        x_mean, x_std = x.mean(axis=0, keepdims=True), x.std(axis=0, keepdims=True).clip(min=1e-6)
        x_n = (x - x_mean) / x_std
        w = fit_ridge(x_n, y, alpha=args.ridge_alpha)
        print(f"[INFO] ridge probe fitted: X {x.shape} -> Y {y.shape}")
        x_te = (test["eeg_feat"] - x_mean) / x_std
        mode = "ridge"

    metrics = subjectwise_retrieval(x_te, test_img, test["subjects"], test["image_paths"], w)

    # Chance depends on gallery size per subject (typically 200 for THINGS-EEG2 test).
    gallery_ns = [m["n"] for m in metrics["per_subject"].values()]
    chance_top1 = float(np.mean([1.0 / max(n, 1) for n in gallery_ns])) if gallery_ns else 0.0
    summary = {
        "checkpoint": str(ckpt_path),
        "device": str(device),
        "clip_model": args.clip_model,
        "clip_pretrained": args.clip_pretrained,
        "probe_split": args.probe_train_split,
        "retrieval_mode": mode,
        "used_clip_emb": bool(probe.get("used_clip_emb")),
        "n_probe": int(probe["eeg_feat"].shape[0]),
        "n_test": int(test["eeg_feat"].shape[0]),
        "vf_ssim_test": test["vf_ssim"],
        "vf_pixcorr_test": test["vf_pixcorr"],
        "retrieval": {
            "macro_top1": metrics["macro_top1"],
            "macro_top5": metrics["macro_top5"],
            "macro_median_rank": metrics["macro_median_rank"],
            "chance_top1": chance_top1,
            "per_subject": metrics["per_subject"],
        },
    }

    out_json = out_dir / "metrics.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(
        out_dir / "features.npz",
        probe_eeg=probe["eeg_feat"],
        probe_img=probe_img,
        test_eeg=test["eeg_feat"],
        test_img=test_img,
        probe_w=w,
        x_mean=x_mean,
        x_std=x_std,
    )

    print("========== EVAL SUMMARY ==========")
    print(f"checkpoint: {ckpt_path}")
    print(f"mode={mode} used_clip_emb={summary['used_clip_emb']}")
    print(f"test N={summary['n_test']}  probe N={summary['n_probe']}")
    print(f"vf_ssim={summary['vf_ssim_test']:.4f}  vf_pixcorr={summary['vf_pixcorr_test']:.4f}")
    print(
        f"retrieval macro Top-1={metrics['macro_top1']*100:.2f}%  "
        f"Top-5={metrics['macro_top5']*100:.2f}%  "
        f"medRank={metrics['macro_median_rank']:.1f}  "
        f"(chance Top-1={chance_top1*100:.2f}%)"
    )
    for s, m in metrics["per_subject"].items():
        print(
            f"  {s}: top1={m['top1']*100:.1f}% top5={m['top5']*100:.1f}% "
            f"medRank={m['median_rank']:.1f} (n={m['n']})"
        )
    print(f"[OK] wrote {out_json}")


if __name__ == "__main__":
    main()
