#!/usr/bin/env python
"""Build EEG-derived STRUCTURE conditions (depth-CLIP, edge-CLIP) for the NW5 arms.

WHY THIS FILE EXISTS
--------------------
The optimized NW config moves structure out of PIXEL space and into FEATURE space.
Our shipped chain constrains pixels (ControlNet-depth + an img2img init at strength
0.82-0.92); CogCapPro does not constrain pixels at all and instead injects depth and
edge as CLIP embeddings through IP-Adapter, at half strength on a single UNet level.
That is why its SSIM and its Inception can both be high: a feature-space structural
condition states *what structure* should be present without dictating *where every
pixel goes*, so it does not spend the generator's freedom the way a pixel constraint
does.  Reproducing the mechanism therefore needs depth/edge CLIP conditions.

The project already owns the two best EEG->structure predictors, both trained on
leak-free splits (grades only on `fit`, checkpoint selection only on `val_b`):

    depth  outputs/uck/<sub>/full/spatial/pred_depth_rgb_512/*.png        (EEG -> VAE + depth)
    init   outputs/sdedit_ll_full10/<sub>/vae_head/pred_lowlevel_rgb_512/*.png

rather than train new modality heads we encode those with the SAME image encoder the
IP-Adapter uses (OpenCLIP ViT-H/14 -- the 1024-d features in `outputs/gem/cond_cache`
are that encoder's output), and calibrate the result to the matching TRAIN bank's
concentration.  Edges are Canny over the predicted low-level image.

PROVENANCE / LIMITS -- stated up front because it decides how the numbers may be read:
  * This is a PROXY for the faithful design, which is modality heads on the S1 encoder.
    It borrows structure from two sibling models instead of predicting it jointly.
  * Arms that consume `clip_depth1024_test.npy` / `clip_edge1024_test.npy` directly (GT
    structure) measure the mechanism CEILING and do not depend on this file at all.
  * Nothing here touches test targets: the calibration reference is the TRAIN bank, and
    no GT test depth/edge is used to build the EEG-derived variants.

VALIDATION
----------
`--self-check` re-encodes the 200 GT test images with the local OpenCLIP ViT-H/14 and
compares against `cond_cache/clip_img1024_test.npy`.  If the cosine is ~1.0 the encoder
path is bit-compatible with whatever produced the cache, so the depth/edge rows written
here live in the same space and can be mixed with the semantic branch.  Run it once
before trusting any arm that uses `--eeg-deriv`.

Usage (validation first, then the real thing):
  python scripts/nda/nw5_make_struct_conds.py --self-check
  python scripts/nda/nw5_make_struct_conds.py --subject sub-08 --out-dir outputs/nw5/conds
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

os.environ.setdefault("HF_HOME", "/project/peilab/why/cache/eeg-brainit/hf")
os.environ.setdefault("HF_HUB_CACHE", os.environ["HF_HOME"] + "/hub")
os.environ.setdefault("TORCH_HOME", "/project/peilab/why/cache/eeg-brainit/torch")
os.environ.setdefault("XDG_CACHE_HOME", "/project/peilab/why/cache/xdg")
os.environ.setdefault("OPENCLIP_CACHE_DIR", os.environ["HF_HOME"] + "/open_clip")

CC = NB_ROOT / "outputs/gem/cond_cache"
DEPTH_SRC = "outputs/uck/{stag}/full/spatial/pred_depth_rgb_512"
INIT_SRC = "outputs/sdedit_ll_full10/{stag}/vae_head/pred_lowlevel_rgb_512"
IMAGES_ROOT = Path("/project/peilab/why/data/images_set")


def l2n(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)


def load_clip(device: str):
    """The IP-Adapter's image encoder, i.e. OpenCLIP ViT-H/14 (1024-d output)."""
    import open_clip
    import torch
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14", pretrained="laion2b_s32b_b79k", cache_dir=os.environ["OPENCLIP_CACHE_DIR"])
    model = model.to(device).eval()
    return model, preprocess, torch


def encode(paths: list[Path], model, preprocess, torch, device: str, batch: int = 16) -> np.ndarray:
    import torch.nn.functional as F
    from PIL import Image
    from tqdm import tqdm
    out = []
    with torch.no_grad():
        for i in tqdm(range(0, len(paths), batch), desc="encode"):
            chunk = paths[i:i + batch]
            x = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in chunk]).to(device)
            out.append(F.normalize(model.encode_image(x).float(), dim=-1).cpu())
    return torch.cat(out, dim=0).numpy()


def test_image_paths() -> list[Path]:
    root = IMAGES_ROOT / "test_images"
    paths = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        imgs = sorted(list(d.glob("*.jpg")) + list(d.glob("*.JPEG")) + list(d.glob("*.png")))
        if imgs:
            paths.append(imgs[0])
    return paths


def structure_paths(kind: str, stag: str, n: int, edge_from_init: bool) -> list[Path]:
    """Deterministic EEG-derived structure inputs, in test order."""
    import cv2
    from PIL import Image
    if kind == "depth":
        d = NB_ROOT / DEPTH_SRC.format(stag=stag)
        paths = [d / f"{i:03d}.png" for i in range(n)]
        missing = [p for p in paths if not p.is_file()]
        if missing:
            raise SystemExit(f"[FATAL] {len(missing)} depth images missing, e.g. {missing[0]}")
        return paths
    # edges: Canny over OUR predicted low-level image (EEG-derived, not from GT)
    d = NB_ROOT / INIT_SRC.format(stag=stag)
    tmp = NB_ROOT / "outputs/nw5/_edge_tmp" / stag
    tmp.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        src = d / f"{i:03d}.png"
        if not src.is_file():
            raise SystemExit(f"[FATAL] missing lowlevel {src}")
        dst = tmp / f"{i:03d}.png"
        if not dst.is_file():
            g = np.asarray(Image.open(src).convert("L"))
            e = cv2.Canny(g, 80, 160)
            Image.fromarray(np.stack([e] * 3, axis=-1)).save(dst)
        paths.append(dst)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=str, default="sub-08")
    ap.add_argument("--out-dir", type=str, default="")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--allow-cpu", type=int, default=0)
    ap.add_argument("--self-check", action="store_true",
                    help="re-encode the GT test images and compare to clip_img1024_test.npy")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    from dev_guard import pick_device
    device = pick_device(args.device, allow_cpu=bool(args.allow_cpu))

    model, preprocess, torch = load_clip(device)
    gt = test_image_paths()
    print(f"[struct-conds] encoder=OpenCLIP ViT-H/14 device={device} test_images={len(gt)}")

    # ---- self check: is this encoder the one that produced cond_cache? -------
    ref = l2n(np.load(CC / "clip_img1024_test.npy").astype(np.float32))
    mine = l2n(encode(gt, model, preprocess, torch, device, args.batch))
    cos = float((ref * mine).sum(1).mean())
    rowcos_ref = float((ref @ ref.T)[np.triu_indices(len(ref), 1)].mean())
    rowcos_mine = float((mine @ mine.T)[np.triu_indices(len(mine), 1)].mean())
    print(f"[self-check] mean cos(image CLIP) vs cond_cache = {cos:.6f}")
    print(f"[self-check] rowcos  cache={rowcos_ref:.4f}  re-encoded={rowcos_mine:.4f}")
    ok = cos > 0.99
    print(f"[self-check] {'PASS - encoder matches the cache' if ok else 'FAIL - wrong encoder'}")

    if args.self_check:
        Path("outputs/nw5").mkdir(parents=True, exist_ok=True)
        Path("outputs/nw5/self_check.json").write_text(json.dumps(
            {"encoder": "OpenCLIP ViT-H/14 laion2b_s32b_b79k", "n": len(gt),
             "mean_cos_vs_cond_cache": round(cos, 6),
             "rowcos_cache": round(rowcos_ref, 6), "rowcos_reencoded": round(rowcos_mine, 6),
             "pass": bool(ok)}, indent=2), encoding="utf-8")
        if not ok:
            raise SystemExit("[FATAL] encoder mismatch - do not use --eeg-deriv arms")
        return

    if not ok:
        raise SystemExit("[FATAL] encoder does not match cond_cache; aborting to avoid "
                         "writing structure rows in the wrong space")

    # ---- build the EEG-derived structure conditions --------------------------
    from ocf_train import calibrate_quantile  # noqa: E402
    out = Path(args.out_dir or (NB_ROOT / "outputs/nw5/conds"))
    out.mkdir(parents=True, exist_ok=True)
    stag = args.subject
    report = {"subject": stag, "encoder": "OpenCLIP ViT-H/14 laion2b_s32b_b79k",
              "self_check_cos": round(cos, 6), "variants": {}}

    for kind in ("depth", "edge"):
        paths = structure_paths(kind, stag, len(gt), edge_from_init=True)
        z = l2n(encode(paths, model, preprocess, torch, device, args.batch))
        raw_p = out / f"eeg_{kind}1024_{stag}_test.npy"
        np.save(raw_p, z.astype(np.float32))
        # calibrate concentration to the matching TRAIN bank (no test target involved)
        R = l2n(np.load(CC / f"clip_{kind}1024_train.npy").astype(np.float32))
        zc = l2n(np.asarray(calibrate_quantile(z, R, two_sided=True)[0], dtype=np.float32))
        cal_p = out / f"eeg_{kind}1024_cal_{stag}_test.npy"
        np.save(cal_p, zc.astype(np.float32))

        bt = l2n(np.load(CC / f"clip_{kind}1024_test.npy").astype(np.float32))
        rc = lambda a: float((a @ a.T)[np.triu_indices(len(a), 1)].mean())
        report["variants"][f"eeg_{kind}"] = {
            "raw": str(raw_p), "cal": str(cal_p),
            "rowcos": round(rc(z), 4), "rowcos_cal": round(rc(zc), 4),
            "bank_rowcos_test": round(rc(bt), 4), "bank_rowcos_train_ref": round(rc(R[:200]), 4),
            # how close each EEG-derived row sits to SOME real structure row
            "nn_cos_to_gt_bank": round(float((z @ bt.T).max(1).mean()), 4),
            "diag_cos_to_gt": round(float((z * bt).sum(1).mean()), 4),
        }
        print(f"[struct-conds] {kind}: rowcos {report['variants'][f'eeg_{kind}']['rowcos']:.4f} -> "
              f"cal {report['variants'][f'eeg_{kind}']['rowcos_cal']:.4f}  "
              f"nn_to_gt {report['variants'][f'eeg_{kind}']['nn_cos_to_gt_bank']:.4f}")

    Path(out / f"eeg_struct_{stag}_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print(f"[struct-conds] wrote {out}/eeg_*_{stag}_test.npy")
    print(json.dumps(report["variants"], indent=2))


if __name__ == "__main__":
    main()
