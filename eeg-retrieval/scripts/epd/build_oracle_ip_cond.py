#!/usr/bin/env python3
"""Write the ORACLE IP-Adapter condition: the ground-truth CLIP joint embedding.

Why this exists
---------------
The generation arms currently bracket the pipeline on one side only. `null_txt2img`
establishes the FLOOR -- 0.4962 CLIP 2-way, i.e. exactly chance, from an IP condition
built out of zeroed EEG. What is missing is the CEILING: what the same generation
stack scores when the semantic condition is CORRECT. Without it there is no way to
say whether a score of 0.786 means "the EEG tower is nearly as good as the signal
allows" or "the decoder is throwing away most of what the tower provides" -- and
those two readings call for opposite work.

The ground truth is on disk: `data/image_feature/ViT-H-14/image_test.npy` is
`(200, 1, 1024)`, the CLIP ViT-H-14 joint-space embedding of each of the 200 test
images, in the same space as the `atoms` the deployed condition is mixed from
(`export_conds.py --atoms` defaults to `image_train.npy`, also 1024-wide). So the
oracle condition is the test image's own embedding, L2-normalised because that is
what `generate_struct_inject_decode.py` does to every `--embed-npy` and what
IP-Adapter's embeddings live on.

A caveat that matters for reading the number
--------------------------------------------
This is the embedding of the image the model is being asked to reconstruct, so the
arm is an upper bound and not a fair competitor: it is "EEG replaced by a perfect
readout of the target". It is still the right number for the ceiling, because the
question it answers -- how much of the target is *reachable by this decoder at all* --
is exactly the question a ceiling is for.

Usage:
  python scripts/epd/build_oracle_ip_cond.py --out-dir outputs/sub08/epd_da2_depth8_export/conds
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

DEFAULT_TEST = "data/image_feature/ViT-H-14/image_test.npy"


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--test-features", default=DEFAULT_TEST)
    ap.add_argument("--out-dir", default="outputs/sub08/epd_da2_depth8_export/conds")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    out = root / args.out_dir / "ip_oracle_test.npy"
    if out.is_file() and not args.force:
        print(f"[SKIP] {out} exists")
        return

    x = np.load(root / args.test_features).astype(np.float32)
    if x.ndim == 3:
        x = x[:, 0]
    if x.shape != (200, 1024):
        raise SystemExit(f"[oracle-ip] expected (200, 1024), got {x.shape}")
    # Same normalisation the generator applies, done here so the artifact on disk is
    # already in the space IP-Adapter consumes and cannot be double-normalised by a
    # later edit without the change being visible.
    n = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.maximum(n, 1e-8)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, x)
    print(f"[oracle-ip] wrote {out}  shape {x.shape}  "
          f"pre-normalisation norm mean {float(n.mean()):.2f}  "
          f"post {float(np.linalg.norm(x, axis=1).mean()):.4f}")


if __name__ == "__main__":
    main()
