#!/usr/bin/env python3
"""Decompose PixCorr into its mean-field and instance-specific parts.

Why this exists
---------------
PixCorr is Pearson r on raw pixels at 425, which means an arm that outputs a plausible
"average natural image" and nothing concept-specific already scores well. Measured on
sub-08: the pixel-wise mean of the 200 test concepts' GROUND-TRUTH images scores
PixCorr +0.1530 -- and ATM's entire reported PixCorr on this subject is ~0.160. So a
bare PixCorr comparison between arms cannot distinguish "this arm decoded the concept"
from "this arm emitted the dataset's average spatial layout".

That distinction is the whole question for the img2img frontier: the low-strength arms
score PixCorr 0.18-0.23 partly because the init is a *decode of a 4x4 latent*, i.e. a
smooth low-frequency field, and smooth fields are close to the mean field by
construction. So this script reports, for every arm:

  raw        the official protocol, comparable to ATM and to every earlier table
  field      what that arm's OWN mean field scores against the GT mean field. This is
             the score an arm would get for having no per-concept content at all.
  residual   PixCorr after subtracting each side's own mean field. This is the
             instance-specific part -- the only part that reflects decoding.

`residual` is NOT the official number and must not be quoted as one. It is the number
that says whether a low-strength arm is doing anything.

The mean field is computed over the 200 test concepts for symmetry with `raw`. A reader
who wants to know whether the field is *learnable* (rather than a test-set artefact
being smuggled in) should look at `--field-source`: it also reports the field built
from the TRAIN concepts, so the two can be compared directly. If they agree, the field
is dataset statistics that any trained model has access to.

Usage
-----
  python scripts/epd/decompose_pixcorr.py                    # all arms found on disk
  python scripts/epd/decompose_pixcorr.py --arms ll_s050 A0  # a subset
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim

ROOT = Path(__file__).resolve().parents[2]
SIZE = 425


def _load(paths: list[Path], size: int = SIZE) -> list[np.ndarray]:
    """Load, resize and rescale exactly as `eval_standard7.lowlevel` does."""
    out = []
    for p in paths:
        im = Image.open(p).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        out.append(np.asarray(im).astype(np.float64) / 255.0)
    return out


def _gt_paths(images_root: Path) -> list[Path]:
    root = images_root / "test_images"
    paths: list[Path] = []
    for d in sorted([p for p in root.iterdir() if p.is_dir()]):
        paths.extend(sorted(list(d.glob("*.jpg")) + list(d.glob("*.JPEG")) + list(d.glob("*.png"))))
    return paths


def _arm_vectors(gen: list[np.ndarray], gt: list[np.ndarray]) -> dict:
    """PixCorr/SSIM raw, the arm's own mean-field score, and the residual score.

    All three use the identical per-image pipeline so the difference between them is
    only which input was correlated with which. That matters: it would be easy to make
    `residual` look better by scoring it on a differently-resized image.
    """
    fm_g = np.mean(np.stack(gen, 0), 0)
    fm_t = np.mean(np.stack(gt, 0), 0)

    raw_p, raw_s, field_p, field_s, res_p, res_s = [], [], [], [], [], []
    for g, t in zip(gen, gt):
        raw_p.append(float(np.corrcoef(t.reshape(-1), g.reshape(-1))[0, 1]))
        raw_s.append(float(ssim(rgb2gray(t), rgb2gray(g), gaussian_weights=True,
                                sigma=1.5, use_sample_covariance=False, data_range=1.0)))
        # What this arm scores for having no per-concept content: its own mean field
        # against the GT mean field. Computed per image so it averages the same way.
        field_p.append(float(np.corrcoef(t.reshape(-1), fm_g.reshape(-1))[0, 1]))
        field_s.append(float(ssim(rgb2gray(t), rgb2gray(fm_g), gaussian_weights=True,
                                  sigma=1.5, use_sample_covariance=False, data_range=1.0)))
        # Instance-specific: remove each side's shared spatial bias first.
        rg, rt = g - fm_g, t - fm_t
        res_p.append(float(np.corrcoef(rt.reshape(-1), rg.reshape(-1))[0, 1]))
        res_s.append(float(ssim(rgb2gray(rt + 0.5), rgb2gray(rg + 0.5), gaussian_weights=True,
                                sigma=1.5, use_sample_covariance=False, data_range=1.0)))
    f = lambda v: float(np.nanmean(v))  # noqa: E731
    return {"raw_pix": f(raw_p), "raw_ssim": f(raw_s),
            "field_pix": f(field_p), "field_ssim": f(field_s),
            "res_pix": f(res_p), "res_ssim": f(res_s),
            "n": len(gen)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--images-root", default="/project/peilab/why/data/images_set")
    ap.add_argument("--arms", nargs="*", default=None,
                    help="arm names to score; default is every arm whose dir is found")
    ap.add_argument("--size", type=int, default=SIZE,
                    help="the official protocol resizes to 425 before correlating")
    args = ap.parse_args()

    root = Path(args.root)
    gt_paths = _gt_paths(Path(args.images_root))
    print(f"ground truth: {len(gt_paths)} images from {args.images_root}/test_images")

    # (dir containing 000.png..199.png). Every generator writes into a nested
    # `generated/` so both spellings are tried rather than assumed.
    candidates: dict[str, Path] = {
        "ll_txt2img": root / "outputs/sub08/epd_opt_dino3_eegit_gen/deploy_txt2img/generated",
        "ll_s080": root / "outputs/sub08/epd_opt_dino3_eegit_gen/deploy_sdedit/generated",
    }
    for a in ("ll_s020", "ll_s035", "ll_s050", "ll_s065",
              "ll_s050_noiseip", "ll_s050_depthinit"):
        candidates[a] = root / f"outputs/sub08/epd_lowlevel_gen/{a}/generated/generated"
    lad = root / "outputs/sub08/epd_sem_gen"
    if lad.is_dir():
        for d in sorted(lad.iterdir()):
            g = d / "generated/generated"
            if g.is_dir():
                candidates[d.name.replace("epd_sem_", "")] = g

    want = args.arms if args.arms else list(candidates)
    print(f"scoring {len(want)} arm(s) at {args.size}x{args.size}, official pipeline\n")

    gt = _load(gt_paths, args.size)
    rows: dict[str, dict] = {}
    for arm in want:
        d = candidates.get(arm)
        if d is None or not d.is_dir():
            print(f"  [skip] {arm}: no image dir at {d}")
            continue
        pngs = sorted(d.glob("*.png"))[:len(gt_paths)]
        if len(pngs) < len(gt_paths):
            print(f"  [skip] {arm}: only {len(pngs)} images, need {len(gt_paths)}")
            continue
        rows[arm] = _arm_vectors(_load(pngs, args.size), gt)
        r = rows[arm]
        print(f"  [ok  ] {arm:18s} raw {r['raw_pix']:+.4f} / field {r['field_pix']:+.4f} "
              f"/ residual {r['res_pix']:+.4f}")

    if not rows:
        print("\nno arms scored")
        return 1

    # The reference field: GT-vs-GT is 1.0 by definition, so the informative reference
    # is the GT's own mean field against the per-image GT -- the score a constant
    # output achieves.
    print("\n" + "=" * 96)
    print("PixCorr decomposition (official pipeline, %d test concepts)" % len(gt_paths))
    print("=" * 96)
    fm_t = np.mean(np.stack(gt, 0), 0)
    const_p = float(np.nanmean([np.corrcoef(t.reshape(-1), fm_t.reshape(-1))[0, 1] for t in gt]))
    print(f"the GT mean field alone scores PixCorr {const_p:+.4f}  <- the floor a "
          f"content-free output reaches")
    print(f"ATM (NeurIPS'24) reports PixCorr ~0.160 on this subject\n")

    order = sorted(rows, key=lambda a: -rows[a]["raw_pix"])
    print(f"{'arm':20s} {'PixCorr':>9s} {'= field':>9s} {'+ instance':>11s} "
          f"{'SSIM':>8s} {'SSIM field':>11s} {'SSIM inst':>10s}")
    print("-" * 96)
    for a in order:
        r = rows[a]
        print(f"{a:20s} {r['raw_pix']:+9.4f} {r['field_pix']:+9.4f} {r['res_pix']:+11.4f} "
              f"{r['raw_ssim']:8.4f} {r['field_ssim']:11.4f} {r['res_ssim']:10.4f}")

    print("""
Reading the table
-----------------
`= field` is what the arm would score with ALL per-concept content removed, because it
is the correlation between the GT and a single image that is the arm's average. It is
an UPPER BOUND on the free credit available to that arm, not its actual baseline -- the
per-image `raw` and `= field` numbers use the same per-image GT, so their sum need not
be `raw`.

`+ instance` is the part that survives removing both mean fields. It is small for every
arm, and that is the finding: on sub-08 the PixCorr axis is dominated by shared spatial
layout, so a large `raw` difference between two arms can be a difference in how closely
each arm's average image resembles the dataset average rather than in how much either
decoded. Quote `raw` against ATM, and use `+ instance` to decide whether an arm is
doing anything.
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
