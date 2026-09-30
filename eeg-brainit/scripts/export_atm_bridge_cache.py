#!/usr/bin/env python3
"""Export ATM EEG embeddings into a Brain-IT-friendly cache (safe).

Read-only inputs:
  eeg-to-image/checkpoints/atm/ATM_S_eeg_features_*.pt
  eeg-to-image/checkpoints/atm/ViT-H-14_features_train.pt

Writes only under eeg-brainit/outputs/atm_bridge/.
Does not launch generation or overwrite source checkpoints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--atm-dir",
        type=str,
        default="/project/peilab/why/eeg-to-image/checkpoints/atm",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/atm_bridge",
    )
    parser.add_argument("--subjects", type=str, nargs="*", default=None)
    args = parser.parse_args()

    root = Path("/project/peilab/why/eeg-brainit")
    atm_dir = Path(args.atm_dir)
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = args.subjects or [f"sub-{i:02d}" for i in range(1, 11)]
    meta = {"subjects": {}, "atm_dir": str(atm_dir), "clip_space": "OpenCLIP ViT-H-14 (1024-d)"}

    img_obj = torch.load(atm_dir / "ViT-H-14_features_train.pt", map_location="cpu", weights_only=False)
    img = img_obj["img_features"].float()
    img = F.normalize(img, dim=-1)
    np.save(out_dir / "clip_img_train_1024.npy", img.numpy().astype(np.float32))

    for sub in subjects:
        tr = torch.load(atm_dir / f"ATM_S_eeg_features_{sub}_train.pt", map_location="cpu", weights_only=False).float()
        te = torch.load(atm_dir / f"ATM_S_eeg_features_{sub}_test.pt", map_location="cpu", weights_only=False).float()
        # Average 4 repeats -> one emb per training image (16540).
        if tr.shape[0] % 16540 != 0:
            raise RuntimeError(f"{sub} train shape unexpected: {tuple(tr.shape)}")
        reps = tr.shape[0] // 16540
        tr_avg = tr.view(16540, reps, -1).mean(1)
        tr_avg = F.normalize(tr_avg, dim=-1)
        te_n = F.normalize(te, dim=-1)
        np.save(out_dir / f"{sub}_train_eeg_avg_1024.npy", tr_avg.numpy().astype(np.float32))
        np.save(out_dir / f"{sub}_test_eeg_1024.npy", te_n.numpy().astype(np.float32))
        # Quick train-matched sanity (not zero-shot): cosine gap on first 200
        idx = np.arange(200)
        p = tr_avg[idx].numpy()
        t = img[idx].numpy()
        paired = float((p * t).sum(1).mean())
        shuf = float((p * t[np.random.RandomState(0).permutation(200)]).sum(1).mean())
        meta["subjects"][sub] = {
            "train_shape": [16540, int(tr_avg.shape[1])],
            "test_shape": [int(te_n.shape[0]), int(te_n.shape[1])],
            "reps_averaged": int(reps),
            "train200_paired_cos": paired,
            "train200_shuffled_cos": shuf,
            "train200_cos_gap": paired - shuf,
        }
        print(f"[OK] {sub} exported; train200 cos_gap={paired-shuf:.4f}")

    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[OK] bridge cache ready at {out_dir}")


if __name__ == "__main__":
    main()
