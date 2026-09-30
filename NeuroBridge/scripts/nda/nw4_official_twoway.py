#!/usr/bin/env python
"""Re-score generated images with the OFFICIAL ATM/MindEye/META-MEG 2-way protocol.

WHY THIS EXISTS -- A PROTOCOL BUG THAT INVALIDATED THE SOTA COMPARISON
---------------------------------------------------------------------
Two evaluation scripts in this repo compute metrics with the SAME NAMES but DIFFERENT
similarities:

  scripts/nda/eval_official_seven_dir.py   sim = t @ g.T        (cosine)
        -> because erdc_twoway_metrics.encode_bundle L2-normalises every feature
  scripts/nda/eval_standard7.py            sim = pearson_sim()  (Pearson / centred cosine)
        -> explicitly "PEARSON correlation (not cosine) per official ATM notebook"

Everything reported for NW4 came from the first, while the published bars it was
compared against -- ATM 0.734 inception / 0.786 CLIP, CogCap 0.669 / 0.715, CogCapPro
0.779 / 0.830 -- and our own HCMA prior (inception 0.9015, CLIP 0.9230) come from the
second.  So the comparison was between an uncentred and a centred statistic.

The difference is not cosmetic.  Both features sets are L2-normalised, so `t @ g.T` is
cosine, which retains the component shared by every image; on 200 images of the same
dataset that shared component is large, it is nearly identical across i and j, and it
COMPRESSES the discriminative range in which the correct pairing has to win.  Pearson
removes it.  For 2-way identification -- whose score is literally the fraction of
off-diagonal pairs the diagonal beats -- that compression directly lowers the score.

The observed pattern fits: HCMA (Pearson) gets 0.9015 inception, we (cosine) got 0.6867,
yet on AlexNet-5 the two are level (0.8599 vs 0.8589).  A pure image-quality difference
could not leave AlexNet-5 untouched while moving Inception by 0.21; a change of
similarity statistic moves exactly the models whose features carry the largest shared
component.

WHAT THIS SCRIPT DOES
---------------------
For each generated directory it re-encodes the images with
`erdc_twoway_metrics.encode_bundle` (identical backbone code to the original run, so the
features are the same ones) and reports the 2-way accuracy under BOTH statistics:

    cos     reproduces the recorded number -> proves the re-encode is faithful
    pearson the official number -> what must be compared against the published bars

Pixel metrics are statistic-independent and are read from the existing eval JSON rather
than recomputed: PixCorr and SSIM use the same `eval_standard7.lowlevel` (gray@425,
gaussian sigma 1.5) in both scripts, and FID the same `erdc_fid_metrics.compute_fid`.

Usage:
  python scripts/nda/nw4_official_twoway.py \
      --gen-root outputs/nw4_10s/arms --arms a_hi,a_cal,a_raw,a_ret1cal \
      --subjects 1,2,3,4,5,6,7,8,9,10 --out outputs/nw4_10s/official_twoway.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

NB_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NB_ROOT / "scripts" / "nda"))
sys.path.insert(0, "/project/peilab/why/eeg-brainit/scripts")

# the hub cache must be writable or open_clip/hub loads warn and can fail
os.environ.setdefault("HF_HOME", "/project/peilab/why/cache/eeg-brainit/hf")
os.environ.setdefault("HF_HUB_CACHE", os.environ["HF_HOME"] + "/hub")
os.environ.setdefault("TORCH_HOME", "/project/peilab/why/cache/eeg-brainit/torch")
os.environ.setdefault("XDG_CACHE_HOME", "/project/peilab/why/cache/xdg")


def twoway(sim: np.ndarray) -> float:
    n = sim.shape[0]
    c = t = 0
    for i in range(n):
        s_ii = sim[i, i]
        for j in range(n):
            if i == j:
                continue
            c += float(s_ii > sim[i, j])
            t += 1
    return c / max(t, 1)


def pearson_sim(real: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Centred cosine, identical to eval_standard7.pearson_sim / np.corrcoef."""
    rc = real - real.mean(1, keepdims=True)
    pc = pred - pred.mean(1, keepdims=True)
    rn = np.linalg.norm(rc, axis=1, keepdims=True)
    pn = np.linalg.norm(pc, axis=1, keepdims=True)
    rn[rn < 1e-8] = 1.0
    pn[pn < 1e-8] = 1.0
    return (rc / rn) @ (pc / pn).T


KEYS = ["clip", "alex2", "alex5", "inception"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-root", type=str, required=True,
                    help="dir containing <arm>/gen/<sub>/generated")
    ap.add_argument("--arms", type=str, default="a_hi,a_cal")
    ap.add_argument("--subjects", type=str, default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--images-root", type=str, default="/project/peilab/why/data/images_set")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--reuse-gt-cache", type=int, default=1,
                    help="reuse <gen>/../_twoway_cache/gt_feats.npz written by the original eval")
    args = ap.parse_args()

    import torch
    from erdc_twoway_metrics import encode_bundle, list_gen  # type: ignore
    from eval_atm_pipeline import list_test_images  # type: ignore

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("[FATAL] CUDA requested but unavailable (check the driver)")
    device = torch.device(args.device)

    gt_paths = list_test_images(Path(args.images_root))
    arms = [a for a in args.arms.split(",") if a]
    subs = [int(s) for s in args.subjects.split(",") if s]
    root = Path(args.gen_root)

    print(f"[official-twoway] device={device} gt={len(gt_paths)} arms={arms} subs={subs}")
    out: dict = {"stage": "nw4_official_twoway", "metric_note":
                 "2-way identification; cos = recorded run, pearson = official ATM/MindEye",
                 "arms": {}}

    for arm in arms:
        rec = {"per_subject": {}, "mean": {}}
        acc = {k: {"cos": [], "pearson": []} for k in KEYS}
        for s in subs:
            stag = f"sub-{s:02d}"
            gen_dir = root / arm / "gen" / stag / "generated"
            if not (gen_dir / "199.png").is_file():
                print(f"  [{arm}/{stag}] missing images - skip")
                continue

            # GT features: reuse the cache the original eval wrote, else encode (200 imgs)
            gt_cache = root / arm / "gen" / stag / "_twoway_cache" / "gt_feats.npz"
            if args.reuse_gt_cache and gt_cache.is_file():
                gtf = {k: np.load(gt_cache)[k] for k in KEYS}
            else:
                gtf = encode_bundle(gt_paths, device)

            gen = list_gen(gen_dir, len(gt_paths))
            if len(gen) != len(gt_paths):
                print(f"  [{arm}/{stag}] {len(gen)} images vs {len(gt_paths)} gt - skip")
                continue
            genf = encode_bundle(gen, device)

            row = {}
            for k in KEYS:
                t, q = gtf[k], genf[k]
                c = twoway(t @ q.T)                 # L2-normalised -> cosine
                p = twoway(pearson_sim(t, q))       # official
                acc[k]["cos"].append(c)
                acc[k]["pearson"].append(p)
                row[k] = {"cos": round(c, 4), "pearson": round(p, 4)}
            rec["per_subject"][stag] = row
            print(f"  [{arm}/{stag}] " + "  ".join(
                f"{k}: cos={row[k]['cos']:.4f} pearson={row[k]['pearson']:.4f}" for k in KEYS))

        for k in KEYS:
            if acc[k]["cos"]:
                rec["mean"][k] = {
                    "cos": round(float(np.mean(acc[k]["cos"])), 4),
                    "pearson": round(float(np.mean(acc[k]["pearson"])), 4),
                    "pearson_std": round(float(np.std(acc[k]["pearson"])), 4),
                    "n": len(acc[k]["cos"]),
                }
        out["arms"][arm] = rec

    # fold in the statistic-independent metrics from the original eval JSONs
    for arm in arms:
        means = out["arms"][arm].setdefault("mean", {})
        for m in ("pixcorr", "ssim", "fid"):
            vals = []
            for s in subs:
                f = root / arm / "eval" / f"sub-{s:02d}.json"
                if f.is_file():
                    d = json.loads(f.read_text())
                    if isinstance(d.get(m), (int, float)):
                        vals.append(float(d[m]))
            if vals:
                means[m] = {"value": round(float(np.mean(vals)), 4),
                            "std": round(float(np.std(vals)), 4), "n": len(vals)}

    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")

    print()
    print("=" * 88)
    print("OFFICIAL (Pearson) vs RECORDED (cosine) 2-way, mean over subjects")
    print("=" * 88)
    print(f"{'arm':<10}{'metric':<11}{'cosine':>9}{'pearson':>9}{'delta':>9}{'pearson sd':>12}")
    for arm in arms:
        m = out["arms"][arm].get("mean", {})
        for k in KEYS:
            if k in m:
                print(f"{arm:<10}{k:<11}{m[k]['cos']:>9.4f}{m[k]['pearson']:>9.4f}"
                      f"{m[k]['pearson'] - m[k]['cos']:>+9.4f}{m[k]['pearson_std']:>12.4f}")
        print()
    print("reference, official protocol:")
    print("  our HCMA prior   inception 0.9015  clip 0.9230  alex2 0.7170  alex5 0.8599")
    print("  ATM              inception 0.7340  clip 0.7860  alex2 0.7760  alex5 0.8660")
    print("  CogCap (10subj)  inception 0.6690  clip 0.7150  alex2 0.7540  alex5 0.6230")
    print("  CogCapPro        inception 0.7790  clip 0.8300")
    print(f"\n[official-twoway] wrote {args.out}")


if __name__ == "__main__":
    main()
