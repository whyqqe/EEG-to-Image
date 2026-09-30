#!/usr/bin/env python
"""NW-v4 M6: multi-candidate render-verify selection.

For each of the 200 test trials we generate N candidates (N seeds of the same
condition set) and then pick ONE, using only signals we are allowed to have at test
time.  No GT images, no GT CLIP rows, no class names.

The two criteria are deliberately the two axes we are trying to buy:

  semantic  cos( CLIP(candidate), sem_cond_trial )   -- how close the render is to the
            image-CLIP row our own semantic head predicted for THIS trial.  This is the
            same quantity the IP-Adapter was asked to satisfy, so it is a consistency
            check rather than a new oracle.

  pixel     -L1( downsample(candidate), downsample(anchor_trial) )  -- how faithfully
            the render keeps the low-frequency layout our spatial path predicted for
            THIS trial.  The anchor is our own blurred low-level prediction, not GT.

score = alpha * semantic - beta * pixel_penalty, and we argmax over the N candidates.
Sweeping (alpha, beta) traces the semantic/pixel Pareto front from ONE set of N
generations, which is why this is cheap: N generations buy a whole curve.

Because the score matrices are saved to npz, a sweep costs no GPU time at all.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def l2n(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def encode_clip(paths: list[Path], device: str, batch: int = 32) -> np.ndarray:
    """OpenCLIP ViT-H/14, i.e. exactly the space the IP-Adapter condition lives in."""
    import os
    import torch
    import open_clip
    from PIL import Image

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14", pretrained="laion2b_s32b_b79k",
        cache_dir=os.environ["OPENCLIP_CACHE_DIR"], device=device)
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            xs = torch.stack([preprocess(Image.open(p).convert("RGB"))
                              for p in paths[i:i + batch]]).to(device)
            f = model.encode_image(xs).float()
            out.append(torch.nn.functional.normalize(f, dim=-1).cpu().numpy())
    del model
    torch.cuda.empty_cache()
    return np.concatenate(out, 0)


def low_freq(paths: list[Path], side: int = 64) -> np.ndarray:
    """Grayscale, downsampled low-frequency content.  This is the band the anchor owns."""
    from PIL import Image
    out = np.empty((len(paths), side * side), np.float32)
    for i, p in enumerate(paths):
        im = Image.open(p).convert("L").resize((side, side), Image.BILINEAR)
        out[i] = np.asarray(im, np.float32).ravel() / 255.0
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand-dirs", required=True,
                    help="comma-separated dirs, each containing generated/000.png..")
    ap.add_argument("--sem-cond", required=True, help="200x1024 image-CLIP condition bank")
    ap.add_argument("--anchor-dir", required=True,
                    help="200 low-level RGB images used as the img2img anchor")
    ap.add_argument("--scores-npz", required=True,
                    help="where to cache the semantic/pixel score matrices")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.0)
    ap.add_argument("--standardize", action="store_true",
                    help="divide each criterion by its global std so alpha/beta are "
                         "directly comparable tradeoff weights")
    ap.add_argument("--out-dir", default="",
                    help="if set, materialise the selection as <out-dir>/generated/")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--report", default="")
    args = ap.parse_args()

    dirs = [Path(d) for d in args.cand_dirs.split(",") if d]
    if not dirs:
        raise SystemExit("[FATAL] no candidate dirs")
    for d in dirs:
        missing = [i for i in range(200) if not (d / "generated" / f"{i:03d}.png").is_file()]
        if missing:
            raise SystemExit(f"[FATAL] {d} is missing {len(missing)} images "
                             f"(e.g. {missing[:5]})")
    n_cand = len(dirs)

    sp = Path(args.scores_npz)
    want = [f"{d.parent.name}" for d in dirs]
    z = None
    if sp.is_file():
        try:
            z = np.load(sp, allow_pickle=True)
        except Exception as e:                                  # noqa: BLE001
            print(f"[m6] cached scores unusable ({e}); recomputing")
            z = None
    if z is not None:
        cached = list(z["sem"].shape)
        # a stale cache from a different candidate pool would silently mislabel arms
        side = sp.with_suffix(".cands.json")
        have = json.loads(side.read_text()) if side.is_file() else None
        if cached[0] != n_cand or (have is not None and have != want):
            print(f"[m6] cache holds {cached[0]} candidates {have}; "
                  f"requested {n_cand} {want} -> recomputing")
            z = None
    if z is not None:
        sem, pix, cands = z["sem"], z["pix"], want
        print(f"[m6] loaded cached scores {sem.shape} from {sp}")
    else:
        sem_cond = l2n(np.load(args.sem_cond).astype(np.float32))
        anchor = low_freq([Path(args.anchor_dir) / f"{i:03d}.png" for i in range(200)])
        sem = np.empty((n_cand, 200), np.float32)
        pix = np.empty((n_cand, 200), np.float32)
        for ci, d in enumerate(dirs):
            paths = [d / "generated" / f"{i:03d}.png" for i in range(200)]
            f = encode_clip(paths, args.device)
            sem[ci] = (l2n(f) * sem_cond).sum(1)          # higher = closer to our condition
            pix[ci] = -np.abs(low_freq(paths) - anchor).mean(1)   # higher = closer to anchor
            print(f"[m6] cand {ci} {d.name}: sem {sem[ci].mean():+.4f}  pix {pix[ci].mean():+.4f}")
        cands = want
        sp.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(sp, sem=sem, pix=pix)
        sp.with_suffix(".cands.json").write_text(json.dumps(cands), encoding="utf-8")
        print(f"[m6] wrote score matrices -> {sp}  (+ .cands.json sidecar)")

    # per-candidate variance tells us whether the seeds even differ
    print(f"[m6] across-candidate std: sem {sem.std(0).mean():.4f}  pix {pix.std(0).mean():.4f}")
    if sem.std(0).mean() < 1e-5 and pix.std(0).mean() < 1e-5:
        print("[m6] WARN candidates are identical - selection is a no-op")

    if args.standardize:
        s_sem = float(sem.std()) or 1.0
        s_pix = float(pix.std()) or 1.0
        sem = sem / s_sem
        pix = pix / s_pix
        print(f"[m6] standardized: sem /= {s_sem:.5f}  pix /= {s_pix:.5f}")

    score = args.alpha * sem + args.beta * pix
    pick = score.argmax(0)
    chosen = np.bincount(pick, minlength=n_cand)
    print(f"[m6] alpha={args.alpha:g} beta={args.beta:g}  picks per candidate: "
          f"{dict(enumerate(chosen.tolist()))}")

    rep = {"n_candidates": n_cand, "candidates": cands,
           "alpha": args.alpha, "beta": args.beta, "standardized": bool(args.standardize),
           "picks": {str(i): int(c) for i, c in enumerate(chosen)},
           "score": {"sem_mean": float(sem.mean()), "pix_mean": float(pix.mean()),
                     "sem_std_across_cand": float(sem.std(0).mean()),
                     "pix_std_across_cand": float(pix.std(0).mean())}}

    if args.out_dir:
        gd = Path(args.out_dir) / "generated"
        gd.mkdir(parents=True, exist_ok=True)
        for i in range(200):
            src = dirs[pick[i]] / "generated" / f"{i:03d}.png"
            dst = gd / f"{i:03d}.png"
            if not dst.is_file():
                shutil.copyfile(src, dst)
        rep["out_dir"] = str(args.out_dir)
        print(f"[m6] materialised selection -> {gd}")

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(rep, indent=2), encoding="utf-8")
        print(f"[m6] wrote {args.report}")


if __name__ == "__main__":
    main()
