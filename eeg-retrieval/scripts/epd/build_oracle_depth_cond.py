#!/usr/bin/env python3
"""Build the ORACLE ControlNet-depth condition: the ground-truth depth map.

Why this exists
---------------
The `depth_cn070` arm scored below `sem_only` on CLIP/Inception under the corrected
(Pearson, paired) measurement, and the exported condition turned out to be within
r = +0.018 per-sample of the constant mean field. Two explanations fit that pair of
facts and they call for opposite next steps:

  (a) the EEG depth map is wrong, and a correct one would help. -> fix the tower.
  (b) the depth ControlNet route is what destroys semantics, at any depth map. -> 
      the whole structural branch is mis-designed and no amount of tower work fixes it.

They are separated by one arm that costs ~8 minutes: hold the semantic condition, the
scale, the seed and the whole generation stack fixed and replace ONLY the depth map
with the ground truth. If the oracle arm also loses CLIP, the route is the problem. If
it gains, the route is fine and the tower's map was simply not good enough.

Construction
------------
Deliberately identical to `export_conds.py`'s depth branch except for the source of
the map, because any other difference would make the comparison an experiment about
the pipeline rather than about the depth map:

  * source resolution 8x8 -- the resolution the loss was defined on (`--struct-scale`)
  * the SAME display range, read from the checkpoint's `_struct_target_range`, so the
    quantisation is the same fixed fit-split range and neither arm gets a per-image
    stretch that the other does not
  * the same BILINEAR upscale to 512 and the same grey-RGB PNG encoding

The ground-truth cache is per-image normalised to [0, 1] (verified: every row's min is
0 and max is 1), so it is mapped through the display range directly. A per-image
affine rescale does not change the relative layout within an image, which is all a
depth ControlNet consumes, so this is a legitimate oracle map and not a differently
scaled one.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DEFAULT_CKPT = "outputs/sub08/epd_da2_depth8_best.pt"
DEFAULT_GT = "outputs/sub08/patch_dual_targets/gt_depth/test_depth_64.npy"
DEFAULT_OUT = "outputs/sub08/epd_da2_depth8_export/spatial/cond_depth_oracle_g1"


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT,
                    help="only read for `_struct_target_range` and `_struct_scale`")
    ap.add_argument("--gt-depth", default=DEFAULT_GT)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    out = root / args.out_dir
    if (out / "199.png").is_file() and not args.force:
        print(f"[SKIP] {out} already complete")
        return

    import torch
    ck = torch.load(root / args.ckpt, map_location="cpu", weights_only=False)
    a = ck.get("args", ck)
    rng = a.get("_struct_target_range")
    scale = int(a.get("struct_scale", 8))
    if not rng:
        raise SystemExit("[oracle] checkpoint has no `_struct_target_range`; the "
                         "quantisation has to match the deployed condition's")
    lo, hi = float(rng[0]), float(rng[1])

    gt = np.load(root / args.gt_depth).astype(np.float32)
    print(f"[oracle] GT depth {gt.shape} range [{gt.min():.4f}, {gt.max():.4f}] "
          f"-> target grid {scale}x{scale}, display range [{lo:.4f}, {hi:.4f}]")

    from PIL import Image
    n = gt.shape[0]
    out.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        m = gt[i]
        if m.shape != (scale, scale):
            # Area-average on the way down rather than point-sample: a depth map has
            # no single "correct" pixel to keep, and subsampling would make the oracle
            # depend on the phase of the grid.
            m = np.asarray(Image.fromarray(m).resize((scale, scale), Image.Resampling.BILINEAR),
                           dtype=np.float32)
        u8 = np.clip((m - lo) / (hi - lo), 0.0, 1.0)
        arr = Image.fromarray((u8 * 255.0).round().astype(np.uint8), mode="L")
        arr = arr.resize((512, 512), Image.Resampling.BILINEAR)
        Image.merge("RGB", (arr, arr, arr)).save(out / f"{i:03d}.png")

    # The same provenance numbers the real export reports, so the two conditions can be
    # compared on contrast rather than only on the score they produce.
    stack = np.stack([np.asarray(Image.open(out / f"{i:03d}.png").convert("L"),
                                 dtype=np.float32) / 255.0 for i in range(n)])
    print(f"[oracle] wrote {n} PNGs -> {out}")
    print(f"[oracle] u8 mean {stack.mean():.4f} std {stack.std():.4f}, "
          f"across-sample std of the per-sample mean {stack.reshape(n, -1).mean(1).std():.4f}, "
          f"within-sample std {stack.reshape(n, -1).std(1).mean():.4f}")


if __name__ == "__main__":
    main()
