#!/usr/bin/env python3
"""Build hybrid depth maps: high u_str → EEG-pred; low → neighbor DepthAnything.

Uses symlinks when possible to avoid duplicating PNGs on disk.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--u-str-npy", type=str, required=True)
    ap.add_argument("--pred-depth-dir", type=str, required=True, help="{i:03d}.png")
    ap.add_argument("--neighbor-idx-npy", type=str, required=True)
    ap.add_argument("--neighbor-depth-dir", type=str, required=True, help="{nb:06d}.png")
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--thresh", type=float, default=0.45, help="rank-u_str >= thresh → use pred")
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    u = np.load(args.u_str_npy).astype(np.float32).reshape(-1)
    neigh = np.load(args.neighbor_idx_npy)
    pred_dir = Path(args.pred_depth_dir)
    nb_dir = Path(args.neighbor_depth_dir)

    n_pred = n_nb = 0
    choices = []
    for i in range(len(u)):
        dst = out / f"{i:03d}.png"
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        use_pred = bool(u[i] >= args.thresh)
        if use_pred:
            src = pred_dir / f"{i:03d}.png"
            if not src.is_file():
                raise FileNotFoundError(src)
            os.symlink(src.resolve(), dst)
            n_pred += 1
            choices.append("pred")
        else:
            nb = int(neigh[i, 0] if neigh.ndim > 1 else neigh[i])
            src = nb_dir / f"{nb:06d}.png"
            if not src.is_file():
                raise FileNotFoundError(src)
            # neighbor maps may be different size — materialize resized copy only when needed
            img = Image.open(src).convert("RGB")
            if img.size != (args.size, args.size):
                img = img.resize((args.size, args.size), Image.Resampling.BICUBIC)
                img.save(dst)
            else:
                os.symlink(src.resolve(), dst)
            n_nb += 1
            choices.append("neighbor")

    report = {
        "n": len(u),
        "thresh": args.thresh,
        "n_pred": n_pred,
        "n_neighbor": n_nb,
        "frac_pred": n_pred / max(len(u), 1),
        "output_dir": str(out),
    }
    (out / "hybrid_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "choices.json").write_text(json.dumps(choices), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
