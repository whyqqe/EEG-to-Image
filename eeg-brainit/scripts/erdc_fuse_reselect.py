#!/usr/bin/env python3
"""Reselect from an existing RAS/CN candidate bank with fused evidence score.

score = brain_cos(eeg, gen) + lambda * struct_cos(gen, neighbor_CLIP)
  - ip_only candidates: struct term = 0 (no structure evidence)
  - raises PixCorr when structure-supporting gens are under-selected by pure brain score

No regeneration. Writes selected_* under --output-dir and metrics.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from erdc_ras_closed_loop import l2, load_embed  # type: ignore
from eval_atm_pipeline import image_metrics, list_test_images  # type: ignore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cand-dir", type=str, required=True)
    parser.add_argument("--eeg-npy", type=str, required=True, help="test EEG/CLIP embeds (N,D)")
    parser.add_argument(
        "--gallery-npy",
        type=str,
        default="outputs/atm_bridge/clip_img_train_1024.npy",
    )
    parser.add_argument("--lambda-struct", type=float, default=0.35)
    parser.add_argument(
        "--struct-mode",
        type=str,
        default="retrieve",
        choices=["retrieve", "shuffle", "misalign", "zero"],
        help="How struct term uses neighbors: retrieve=aligned; shuffle=permute "
        "neighbor_idx; misalign=shift indices; zero=disable struct evidence.",
    )
    parser.add_argument("--top-k-brain", type=int, default=0, help="if >0, add topk_struct pick")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--images-root", type=str, default="/project/peilab/why/data/images_set")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int, default=0)
    args = parser.parse_args()

    project = ROOT
    cand_dir = Path(args.cand_dir)
    if not cand_dir.is_absolute():
        cand_dir = project / cand_dir
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = project / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = Path("/project/peilab/why/cache/eeg-brainit")
    os.environ.setdefault("OPENCLIP_CACHE_DIR", str(cache / "open_clip"))
    os.environ.setdefault("TORCH_HOME", str(cache / "torch"))

    specs = json.loads((cand_dir / "specs.json").read_text())
    nb_path = cand_dir / "neighbor_idx.npy"
    if not nb_path.is_file():
        nb_path = cand_dir.parent / "neighbor_idx.npy"
    if not nb_path.is_file():
        raise FileNotFoundError(f"neighbor_idx.npy not in {cand_dir} or parent")
    neighbor_idx = np.load(nb_path)
    if args.struct_mode == "shuffle":
        rng = np.random.RandomState(args.seed)
        neighbor_idx = neighbor_idx[rng.permutation(neighbor_idx.shape[0])]
    elif args.struct_mode == "misalign":
        neighbor_idx = np.roll(neighbor_idx, shift=1, axis=0)
    eeg_path = Path(args.eeg_npy)
    if not eeg_path.is_absolute():
        eeg_path = project / eeg_path
    gal_path = Path(args.gallery_npy)
    if not gal_path.is_absolute():
        gal_path = project / gal_path
    eeg = load_embed(eeg_path)
    gallery = load_embed(gal_path)

    feats = np.load(cand_dir / "clip_feats_flat.npy")
    n = neighbor_idx.shape[0] if args.max_images <= 0 else min(neighbor_idx.shape[0], args.max_images)
    k = len(specs)
    if feats.ndim == 3:
        feats = feats[:n]
    elif feats.ndim == 2:
        rows, d = feats.shape
        if rows == n * k:
            feats = feats.reshape(n, k, d)
        elif rows == n and d % k == 0:
            feats = feats[:n].reshape(n, k, d // k)
        else:
            raise ValueError(f"cannot reshape feats {feats.shape} for n={n} k={k}")
    else:
        raise ValueError(f"unexpected feats ndim={feats.ndim}")
    eeg = eeg[:n]

    brain = np.einsum("nd,nkd->nk", eeg, feats)
    struct = np.zeros_like(brain)
    for j, sp in enumerate(specs):
        rank = int(sp.get("neighbor_rank", -1))
        mode = sp.get("mode", "")
        if rank < 0 or mode in ("ip_only",):
            continue
        if args.struct_mode == "zero":
            continue
        # neighbor CLIP from train gallery
        nb = neighbor_idx[:n, rank]
        neigh_feat = gallery[nb]  # (n,d)
        struct[:, j] = np.einsum("nd,nd->n", feats[:, j, :], neigh_feat)

    fused = brain + float(args.lambda_struct) * struct
    np.save(out_dir / "brain_scores.npy", brain.astype(np.float32))
    np.save(out_dir / "struct_scores.npy", struct.astype(np.float32))
    np.save(out_dir / "fused_scores.npy", fused.astype(np.float32))

    rng = np.random.RandomState(args.seed)
    picks = {
        "brain": brain.argmax(1),
        "fused": fused.argmax(1),
        "struct": struct.argmax(1),
        "first": np.zeros(n, dtype=np.int64),
        "random": rng.randint(0, k, size=n),
    }
    if int(args.top_k_brain) > 0:
        tk = min(int(args.top_k_brain), k)
        topk_idx = np.zeros(n, dtype=np.int64)
        for i in range(n):
            top = np.argsort(-brain[i])[:tk]
            topk_idx[i] = top[struct[i, top].argmax()]
        picks["topk_struct"] = topk_idx

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gt = list_test_images(Path(args.images_root))
    report = {
        "n": n,
        "k": k,
        "lambda_struct": float(args.lambda_struct),
        "struct_mode": args.struct_mode,
        "cand_dir": str(cand_dir),
        "selection": {},
    }
    for name, idx in picks.items():
        odir = out_dir / f"selected_{name}"
        odir.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            src = cand_dir / f"{i:03d}_k{int(idx[i])}.png"
            dst = odir / f"{i:03d}.png"
            if not dst.is_file():
                Image.open(src).save(dst)
        metrics = image_metrics([odir / f"{i:03d}.png" for i in range(n)], gt[:n], device)
        report["selection"][name] = {
            "metrics": metrics,
            "pick_hist": {str(j): int((idx == j).sum()) for j in range(k)},
            "mean_brain": float(brain[np.arange(n), idx].mean()),
            "mean_struct": float(struct[np.arange(n), idx].mean()),
            "mean_fused": float(fused[np.arange(n), idx].mean()),
        }
        print(
            f"[INFO] {name:6s} CLIP={metrics['clip_cosine']:.4f} Pix={metrics['pixcorr']:.4f} "
            f"hist={report['selection'][name]['pick_hist']}"
        )

    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[OK] wrote {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
