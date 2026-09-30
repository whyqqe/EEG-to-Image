#!/usr/bin/env python3
"""Evaluate ATM-bridge ClipRefiner retrieval on THINGS-EEG2 test set.

Compares refined clip_emb vs raw ATM embeddings against OpenCLIP ViT-H/14
image features. Writes only under --output-dir.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from eeg_brainit.models.atm_bridge import AtmBrainITPipeline


def retrieval_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    sim = pred @ target.T
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
        "n": int(n),
        "top1": float(top1 / n),
        "top5": float(top5 / n),
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(ranks.mean()),
    }


def cos_stats(pred: np.ndarray, target: np.ndarray, seed: int = 0) -> dict:
    paired = float((pred * target).sum(1).mean())
    shuffled = float(
        (pred * target[np.random.RandomState(seed).permutation(len(pred))]).sum(1).mean()
    )
    return {"paired_cos": paired, "shuffled_cos": shuffled, "cos_gap": paired - shuffled}


def load_img_features(cache_path: Path, fallback_dir: Path) -> tuple[np.ndarray, str]:
    """Prefer local cache; else reuse ATM baseline cache if present."""
    candidates = [
        cache_path,
        fallback_dir / "test_ViT-H-14_laion2b_s32b_b79k_features.npy",
    ]
    for p in candidates:
        if p.is_file():
            feat = np.load(p).astype(np.float32)
            feat = feat / np.linalg.norm(feat, axis=1, keepdims=True).clip(min=1e-8)
            return feat, str(p)
    raise FileNotFoundError(
        f"No image feature cache at {cache_path} or {candidates[1]}; "
        "run eval_atm_baseline first or encode images."
    )


@torch.no_grad()
def refine_eeg(
    model: AtmBrainITPipeline,
    eeg: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    model.eval()
    outs = []
    x = torch.from_numpy(eeg.astype(np.float32))
    for i in range(0, len(x), batch_size):
        batch = x[i : i + batch_size].to(device)
        out = model(batch)
        outs.append(F.normalize(out["clip_emb"].float(), dim=-1).cpu().numpy())
    return np.concatenate(outs, axis=0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/atm_bridge_sub08/checkpoints/atm_stage1_best_e30.pt",
    )
    parser.add_argument("--subject", type=str, default="sub-08")
    parser.add_argument("--bridge-dir", type=str, default="outputs/atm_bridge")
    parser.add_argument("--config", type=str, default="configs/atm_bridge_sub08.yaml")
    parser.add_argument("--output-dir", type=str, default="outputs/eval/atm_bridge_sub08")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--img-feat-cache",
        type=str,
        default="",
        help="Optional path to test image CLIP features npy",
    )
    args = parser.parse_args()

    project = ROOT
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = project / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = project / ckpt_path
    bridge_dir = Path(args.bridge_dir)
    if not bridge_dir.is_absolute():
        bridge_dir = project / bridge_dir

    cache_root = Path("/project/peilab/why/cache/eeg-brainit")
    os.environ.setdefault("HF_HOME", str(cache_root / "hf"))
    os.environ.setdefault("OPENCLIP_CACHE_DIR", str(cache_root / "open_clip"))
    os.environ.setdefault("TORCH_HOME", str(cache_root / "torch"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")
    if device.type == "cuda":
        print(f"[INFO] GPU={torch.cuda.get_device_name(0)}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("cfg")
    if cfg is None:
        from eeg_brainit.utils.config import load_config

        cfg = load_config(str(project / args.config))
    subject = ckpt.get("subject", args.subject)
    print(f"[INFO] checkpoint={ckpt_path} epoch={ckpt.get('epoch')} stage={ckpt.get('stage')} subject={subject}")

    model = AtmBrainITPipeline.from_config(cfg, project_root=str(project)).to(device)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    print(f"[INFO] load_state_dict missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    eeg_path = bridge_dir / f"{subject}_test_eeg_1024.npy"
    if not eeg_path.is_file():
        raise FileNotFoundError(eeg_path)
    eeg_raw = np.load(eeg_path).astype(np.float32)
    eeg_raw = eeg_raw / np.linalg.norm(eeg_raw, axis=1, keepdims=True).clip(min=1e-8)
    print(f"[INFO] raw ATM test emb {eeg_raw.shape}")

    img_cache = Path(args.img_feat_cache) if args.img_feat_cache else out_dir / "test_ViT-H-14_features.npy"
    if not img_cache.is_absolute():
        img_cache = project / img_cache
    img_feat, img_src = load_img_features(
        img_cache, project / "outputs/eval/atm_baseline"
    )
    print(f"[INFO] image features from {img_src} shape={img_feat.shape}")
    if eeg_raw.shape[0] != img_feat.shape[0]:
        raise RuntimeError(f"eeg {eeg_raw.shape} vs img {img_feat.shape}")
    if eeg_raw.shape[1] != img_feat.shape[1]:
        raise RuntimeError(f"dim mismatch eeg {eeg_raw.shape[1]} vs img {img_feat.shape[1]}")

    eeg_ref = refine_eeg(model, eeg_raw, device, batch_size=args.batch_size)
    np.save(out_dir / f"{subject}_refined_clip_emb.npy", eeg_ref)

    raw_m = retrieval_metrics(eeg_raw, img_feat)
    raw_m.update(cos_stats(eeg_raw, img_feat))
    ref_m = retrieval_metrics(eeg_ref, img_feat)
    ref_m.update(cos_stats(eeg_ref, img_feat))

    summary = {
        "checkpoint": str(ckpt_path),
        "subject": subject,
        "epoch": ckpt.get("epoch"),
        "stage": ckpt.get("stage"),
        "n_test": int(eeg_raw.shape[0]),
        "feature_dim": int(eeg_raw.shape[1]),
        "img_feat_source": img_src,
        "chance_top1": 1.0 / float(eeg_raw.shape[0]),
        "raw_atm": raw_m,
        "refined": ref_m,
        "delta_top1": float(ref_m["top1"] - raw_m["top1"]),
        "delta_top5": float(ref_m["top5"] - raw_m["top5"]),
        "delta_cos_gap": float(ref_m["cos_gap"] - raw_m["cos_gap"]),
    }
    out_json = out_dir / "metrics.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("========== ATM BRIDGE EVAL ==========")
    print(
        f"RAW ATM   Top-1={raw_m['top1']*100:.2f}% Top-5={raw_m['top5']*100:.2f}% "
        f"med={raw_m['median_rank']:.1f} cos_gap={raw_m['cos_gap']:.4f}"
    )
    print(
        f"REFINED   Top-1={ref_m['top1']*100:.2f}% Top-5={ref_m['top5']*100:.2f}% "
        f"med={ref_m['median_rank']:.1f} cos_gap={ref_m['cos_gap']:.4f}"
    )
    print(
        f"DELTA     Top-1={summary['delta_top1']*100:+.2f}pp "
        f"Top-5={summary['delta_top5']*100:+.2f}pp "
        f"cos_gap={summary['delta_cos_gap']:+.4f}"
    )
    print(f"[OK] wrote {out_json}")


if __name__ == "__main__":
    main()
