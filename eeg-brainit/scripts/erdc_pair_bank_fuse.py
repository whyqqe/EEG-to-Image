#!/usr/bin/env python3
"""Build a 2-candidate bank from two selected folders and run fuse reselect."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from erdc_ras_closed_loop import encode_clip  # type: ignore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir-a", type=str, required=True, help="first candidate folder (k=0)")
    parser.add_argument("--dir-b", type=str, required=True, help="second candidate folder (k=1)")
    parser.add_argument("--label-a", type=str, default="a")
    parser.add_argument("--label-b", type=str, default="b")
    parser.add_argument("--eeg-npy", type=str, required=True)
    parser.add_argument("--lambda-struct", type=float, default=0.15)
    parser.add_argument("--gallery-npy", type=str, default="outputs/atm_bridge/clip_img_train_1024.npy")
    parser.add_argument("--neighbor-npy", type=str, default="", help="optional neighbor_idx for struct")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-images", type=int, default=0)
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    cand = out_root / "candidates"
    cand.mkdir(parents=True, exist_ok=True)

    dir_a = Path(args.dir_a)
    dir_b = Path(args.dir_b)
    if not dir_a.is_absolute():
        dir_a = ROOT / dir_a
    if not dir_b.is_absolute():
        dir_b = ROOT / dir_b

    n = min(
        sum(1 for i in range(10_000) if (dir_a / f"{i:03d}.png").is_file() or (dir_a / f"{i:03d}.jpg").is_file()),
        sum(1 for i in range(10_000) if (dir_b / f"{i:03d}.png").is_file() or (dir_b / f"{i:03d}.jpg").is_file()),
    )
    if args.max_images > 0:
        n = min(n, args.max_images)
    if n <= 0:
        raise RuntimeError(f"no paired images in {dir_a} and {dir_b}")

    specs = [
        {"k": 0, "mode": "pair_bank", "source": args.label_a, "neighbor_rank": 0},
        {"k": 1, "mode": "pair_bank", "source": args.label_b, "neighbor_rank": 0},
    ]
    (cand / "specs.json").write_text(json.dumps(specs, indent=2), encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feats = []
    for i in range(n):
        row = []
        for k, d in enumerate([dir_a, dir_b]):
            src = d / f"{i:03d}.png"
            if not src.is_file():
                src = d / f"{i:03d}.jpg"
            dst = cand / f"{i:03d}_k{k}.png"
            shutil.copy2(src, dst)
            row.append(encode_clip([dst], device)[0])
        feats.append(np.stack(row, axis=0))
    feats_flat = np.stack(feats, axis=0).astype(np.float32)
    np.save(cand / "clip_feats_flat.npy", feats_flat)

    nb_src = (
        Path(args.neighbor_npy)
        if args.neighbor_npy
        else ROOT / "outputs/erdc/w13_merged_turbo/neighbor_idx.npy"
    )
    if nb_src.is_file():
        shutil.copy2(nb_src, out_root / "neighbor_idx.npy")

    fuse_out = out_root / "fuse_run"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "erdc_fuse_reselect.py"),
        "--cand-dir",
        str(cand),
        "--eeg-npy",
        args.eeg_npy,
        "--lambda-struct",
        str(args.lambda_struct),
        "--gallery-npy",
        args.gallery_npy,
        "--output-dir",
        str(fuse_out),
    ]
    if args.max_images > 0:
        cmd.extend(["--max-images", str(args.max_images)])
    subprocess.run(cmd, check=True)
    print(f"[OK] pair bank fuse -> {fuse_out / 'selected_fused'}")


if __name__ == "__main__":
    main()
