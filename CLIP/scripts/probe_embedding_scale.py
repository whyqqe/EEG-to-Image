#!/usr/bin/env python
"""Measure the RAW (pre-normalisation) per-dimension scale profile of the embedding.

Why this is a separate probe
---------------------------
VICReg's variance term is a hinge on the per-dimension standard deviation: it charges
`relu(1 - std_d)` for every dimension `d`, i.e. it pushes EVERY dimension to have std
>= 1. That is the correct objective for self-supervised learning, where the embedding is
meant to USE all of its dimensions.

It is the wrong objective here, and the probe measures whether it is firing. The
concept manifold of this task occupies ~16 dimensions (the subspace sweep: r=16 captures
93-96% of the query variance, and its span also captures 94% of the image target's
variance). If VICReg succeeds in giving all 512 dimensions std ~1, then the 16 dimensions
that carry the concept signal are accompanied by 496 that carry nothing, all at the same
scale -- and because the contrast is computed on the L2-NORMALISED vector, a direction's
influence is proportional to its share of the total norm. Inflating 496 noise dimensions
to parity with the 16 signal dimensions therefore dilutes the concept direction by
roughly sqrt(512/16) = 5.7x, and it is also what fills the covariance that
`saw_whiten` has to invert -- giving it a spectrum it cannot estimate from 200 samples.

So the diagnostic is a ratio, not a value: `std` in the leading concept dimensions
against `std` in the trailing ones. A healthy representation for this task should be
CONCENTRATED (leading >> trailing). A uniform profile is the fingerprint of a variance
hinge that is fighting the geometry the retrieval needs.

Run:  python scripts/probe_embedding_scale.py --ckpts <c1> <c2> --target-subject 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from samclip import config, evaluate  # noqa: E402
from samclip.data import things_eeg  # noqa: E402
from samclip.data.targets import load_target_stack  # noqa: E402
from samclip.models import build_model  # noqa: E402


def raw_features(ckpt_path: Path, target_subject: int, mvnn: str) -> dict:
    """RAW (pre-L2-normalisation) EEG and image features from a frozen checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    channels = (config.CHANNELS_OCCIPITO_PARIETAL
                if cfg.get("channel_set") == "occipital17" else None)
    img = cfg.get("image", {}) or {}
    _, test = things_eeg.load_subject_std(target_subject, channels, mvnn=mvnn)
    targets_te = load_target_stack(img.get("feature_set", "clip_h14_multilevel"),
                                   img.get("layers"), "test")
    model = build_model(cfg, targets_te.shape[2], targets_te.shape[-1])
    model.load_state_dict(ckpt["model"])
    model.eval()
    loader = DataLoader(things_eeg.TestDataset(np.asarray(test), targets_te),
                        batch_size=200, shuffle=False, collate_fn=things_eeg.collate)
    eeg, imgs = [], []
    with torch.no_grad():
        for batch in loader:
            # `encode_eeg(..., normalize=False)` is the raw head output, i.e. the vector
            # whose per-dimension scale VICReg actually penalises.
            eeg.append(model.encode_eeg(batch["eeg"], normalize=False).cpu())
            imgs.append(model.encode_target(batch["target"], training=False,
                                            normalize=False).cpu())
    return {"eeg": torch.cat(eeg).numpy(), "img": torch.cat(imgs).numpy()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--target-subject", type=int, required=True)
    ap.add_argument("--mvnn", default="test")
    args = ap.parse_args()

    print(f"[scale] VICReg's variance hinge charges relu(1 - std_d) per dimension, i.e. it "
          f"pushes EVERY dim to std >= 1.")
    print(f"        For a task whose concept manifold is ~16-dimensional, the diagnostic is "
          f"the RATIO leading/trailing, not the level.\n")

    for c in args.ckpts:
        p = Path(c)
        tag = f"{p.parent.parent.name}/{p.parent.name}"
        f = raw_features(p, args.target_subject, args.mvnn)
        print(f"{'=' * 84}\n[{tag}]\n{'=' * 84}")
        for name in ("eeg", "img"):
            x = f[name]
            sd = x.std(axis=0)
            # Rank dimensions by their VARIANCE in the RAW space, which is the ordering
            # VICReg sees.
            s = np.sort(sd)[::-1]
            lead = s[:16].mean()
            trail = s[16:].mean()
            uniform = s.std() / max(s.mean(), 1e-12)
            print(f"  {name}: raw per-dim std  mean {s.mean():.4f}  min {s.min():.4f}  "
                  f"max {s.max():.4f}")
            print(f"       leading-16 mean {lead:.4f}   trailing mean {trail:.4f}   "
                  f"RATIO {lead / max(trail, 1e-12):.2f}x   "
                  f"cv(std over dims) {uniform:.3f}")
            print(f"       dims with std >= 1.0: {int((sd >= 1.0).sum())}/{sd.size}   "
                  f"(VICReg's hinge target)")
            if name == "eeg":
                # Normalised-space share of norm per direction is what the contrast sees.
                xn = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
                print(f"       after L2-norm: variance is spread over "
                      f"{int(np.searchsorted(np.cumsum(np.sort(xn.var(0))[::-1] / xn.var(0).sum()), 0.9) + 1)}"
                      f" dims for 90% (vs 16 for the concept manifold)")
        print()


if __name__ == "__main__":
    main()
