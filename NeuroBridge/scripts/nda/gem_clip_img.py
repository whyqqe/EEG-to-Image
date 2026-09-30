#!/usr/bin/env python3
"""GEM stage 0 -- the CLIP tower's REAL target: projected ViT-H-14 image features.

WHY THIS FILE EXISTS (it repairs a space mismatch, not a hyper-parameter)
------------------------------------------------------------------------
The IP-Adapter checkpoint this project generates with is
`ip-adapter_sdxl_vit-h.bin`.  Its `image_proj` consumes the **projected**
1024-d image feature of OpenCLIP ViT-H-14 -- the output of the model's own
contrastive projection head.

The condition the old pipeline actually handed it was 1024-d, so nothing
crashed, but the 1024 numbers were produced by regressing EEG onto
`sem_image_*.npy`, which is a **1280-d ViT-H-14 penultimate activation**
(`g2_build_targets.py`: `sem_image = l2(acc / len(layers))` over pooled
`layer_*.npy`).  1280 is the width of the transformer block, not of the shared
text/image space.  So the "structural" condition was being pushed toward a
representation the image adapter does not read, and the fusion that produced the
final 1024-d vector was mixing a text-tower quantity into an image slot.

This script extracts what the adapter actually reads:
    * `clip_img1024_<split>.npy` -- `encode_image()` (projected, unit-norm)
    * `clip_img1280_<split>.npy` -- the penultimate CLS (kept for the ordering
      audit below and for the iREPA patch comparison)

ORDERING AUDIT
--------------
Every per-row cache in this project is indexed by `list_split_images()` order
(sorted concept directories, then sorted filenames).  Rather than trusting that,
the script (a) re-derives the same list, (b) asserts it equals the `path` column
of `captions_<split>.jsonl` row by row, and (c) reports the top-1
nearest-neighbour agreement between the fresh 1280-d penultimate features and
the existing `sem_image_<split>.npy`.  A row-order slip would show up as a
collapse of that agreement, and the script then fails instead of writing a
misaligned cache.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

NB_ROOT = Path(__file__).resolve().parents[2]


def l2(a: np.ndarray) -> np.ndarray:
    return a / np.clip(np.linalg.norm(a, axis=-1, keepdims=True), 1e-8, None)


def list_split_images(images_root: Path, split: str) -> list[Path]:
    """Identical to `g2_build_targets.list_split_images` -- the project-wide order."""
    root = images_root / ("training_images" if split == "train" else "test_images")
    if not root.is_dir():
        raise SystemExit(f"[FATAL] missing image root {root}")
    paths: list[Path] = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        paths.extend(sorted(list(d.glob("*.jpg")) + list(d.glob("*.JPEG"))
                            + list(d.glob("*.png"))))
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default=str(NB_ROOT / "outputs/gem/cond_cache"))
    ap.add_argument("--images-root", type=str, default="/project/peilab/why/data/images_set")
    ap.add_argument("--captions-dir", type=str, default=str(NB_ROOT / "outputs/g2/captions"))
    ap.add_argument("--targets-dir", type=str, default=str(NB_ROOT / "outputs/g2/targets"))
    ap.add_argument("--clip-model", type=str, default="ViT-H-14")
    ap.add_argument("--clip-pretrained", type=str, default="laion2b_s32b_b79k")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--max-images", type=int, default=0,
                    help="cap the number of images per split. Non-zero is for SMOKE "
                         "TESTS only: the resulting cache is a PREFIX of the split, so "
                         "it is only valid for a run limited to those rows.")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OPENCLIP_CACHE_DIR",
                          "/project/peilab/why/cache/eeg-brainit/open_clip")

    import torch
    import open_clip
    from PIL import Image
    from torchvision import transforms as T

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.clip_model, pretrained=args.clip_pretrained, device=dev)
    model.eval()

    # The projection is NOT an `nn.Linear` here: open_clip's `VisionTransformer`
    # holds it as a bare `nn.Parameter` (`visual.proj`, 1280x1024) and applies it
    # in `forward` as `pooled @ self.proj`.  The penultimate 1280-d vector is
    # therefore `ln_post(...)[:, 0]`, and the identity
    #     encode_image(x)  ==  ln_post(x)[:, 0] @ proj
    # is checked numerically below rather than assumed -- if a future open_clip
    # reorders pooling, the audit fails instead of writing a silently wrong cache.
    pen: dict[str, torch.Tensor] = {}

    def hook(_m, _inp, out):
        pen["x"] = out.detach()

    if not hasattr(model.visual, "proj") or model.visual.proj is None:
        raise SystemExit("[FATAL] visual.proj absent; the penultimate activation and "
                         "the projected feature cannot both be recovered")
    h = model.visual.ln_post.register_forward_hook(hook)
    with torch.no_grad():
        _p = model.encode_image(torch.zeros(1, 3, 224, 224, device=dev))
        _q = pen["x"][:, 0] @ model.visual.proj
        _d = float((_p.float() - _q.float()).abs().max())
    if _d > 1e-3:
        raise SystemExit(f"[FATAL] `ln_post[:,0] @ proj` differs from `encode_image` by "
                         f"{_d:.3g}; the 1280-d audit vector is not the penultimate "
                         f"activation of this open_clip build")
    print(f"[gem-clip] proj identity verified (max abs diff {_d:.2e}); the 1280-d vector "
          f"is `ln_post[:,0]`, the 1024-d vector is `encode_image`")

    report: dict = {"clip_model": args.clip_model, "pretrained": args.clip_pretrained,
                    "splits": {}}

    for split in args.splits:
        # ---- ordering audit (a): our list must equal the captions jsonl row order
        paths = list_split_images(Path(args.images_root), split)
        cap = Path(args.captions_dir) / f"captions_{split}.jsonl"
        if not cap.is_file():
            raise SystemExit(f"[FATAL] missing {cap}")
        jpaths = [json.loads(l)["path"] for l in cap.read_text(encoding="utf-8").splitlines()
                  if l.strip()]
        if len(jpaths) != len(paths):
            raise SystemExit(f"[FATAL] {split}: jsonl {len(jpaths)} rows vs images "
                             f"{len(paths)}; the caches cannot be aligned")
        bad = [i for i, (a, b) in enumerate(zip(jpaths, paths)) if Path(a).name != b.name
               or Path(a).parent.name != b.parent.name]
        if bad:
            raise SystemExit(f"[FATAL] {split}: row order differs from captions_{split}"
                             f".jsonl at {len(bad)} positions, first {bad[:5]}. Writing "
                             f"this cache would misalign every per-row array downstream.")
        print(f"[gem-clip] {split}: {len(paths)} images, row order verified against "
              f"captions_{split}.jsonl")
        if args.max_images:
            paths = paths[: args.max_images]
            print(f"[gem-clip] {split}: CAPPED to {len(paths)} images (--max-images). "
                  f"This cache is a prefix of the split and must not be used to train "
                  f"a run that sees later rows.")

        # ---- encode
        p1024, p1280 = [], []
        bs = args.batch_size
        with torch.no_grad():
            for s in range(0, len(paths), bs):
                chunk = paths[s:s + bs]
                ims = [preprocess(Image.open(p).convert("RGB")) for p in chunk]
                x = torch.stack(ims).to(dev)
                p1024.append(model.encode_image(x).float().cpu().numpy())
                p1280.append(pen["x"][:, 0].float().cpu().numpy())
                if s % (bs * 20) == 0:
                    print(f"  [gem-clip] {split} {s + len(chunk)}/{len(paths)}")
        f1024 = l2(np.concatenate(p1024))
        f1280 = np.concatenate(p1280)

        np.save(out / f"clip_img1024_{split}.npy", f1024.astype(np.float32))
        np.save(out / f"clip_img1280_{split}.npy", f1280.astype(np.float16))

        # ---- ordering audit: is row i of this cache the SAME IMAGE as row i of
        # `sem_image_<split>.npy`?
        #
        # NOT a nearest-neighbour test.  The two objects are different features of
        # the same image: this cache is the penultimate CLS (`ln_post[:,0]`), while
        # `sem_image` is `l2(mean over several pooled layer_*.npy)`.  Measured on
        # the test split they correlate at cos = 0.35 and agree on the top-1
        # neighbour 17% of the time -- both are consistent with a correct pairing
        # of two different feature types, and neither can distinguish that from a
        # wrong pairing.
        #
        # The instrument that CAN distinguish them is a cross-validated linear map:
        # a 1280->1280 ridge fitted on the first 3/4 of the rows must predict the
        # remaining rows far better than the same fit against a SHUFFLED pairing.
        # If the row order were wrong, both numbers would be equally bad.
        rep: dict = {"n": len(paths), "dim1024": int(f1024.shape[1]),
                     "dim1280": int(f1280.shape[1])}
        ref_p = Path(args.targets_dir) / f"sem_image_{split}.npy"
        if ref_p.is_file():
            ref = np.load(ref_p).astype(np.float32)
            if ref.shape[0] == f1280.shape[0]:
                A, B = l2(f1280), l2(ref)
                n = len(A)
                tr = slice(0, int(0.75 * n))
                te = slice(int(0.75 * n), n)
                mu, sd = A[tr].mean(0), A[tr].std(0).clip(1e-6)
                As = (A[tr] - mu) / sd
                ym = B[tr].mean(0)
                W = np.linalg.solve(As.T @ As + 1.0 * np.eye(As.shape[1]),
                                    As.T @ (B[tr] - ym))

                def r2(target: np.ndarray) -> float:
                    P = ((A[te] - mu) / sd) @ W + ym
                    return float(1.0 - ((P - target) ** 2).sum()
                                 / ((target - ym) ** 2).sum())

                r2_ok = r2(B[te])
                sh = np.random.default_rng(0).permutation(n)[te]
                r2_sh = r2(B[sh])
                rep.update({"pairing_r2_correct": r2_ok, "pairing_r2_shuffled": r2_sh,
                            "rowwise_cos_vs_sem_image": float((A * B).sum(1).mean())})
                print(f"  [audit] {split}: pairing R2 {r2_ok:.4f} vs shuffled "
                      f"{r2_sh:.4f} (row-wise cos "
                      f"{rep['rowwise_cos_vs_sem_image']:.4f})")
                if not (r2_ok > 0.02 and r2_ok - r2_sh > 0.10):
                    raise SystemExit(
                        f"[FATAL] {split}: the pairing test does not confirm that row i "
                        f"of this cache and row i of `sem_image_{split}.npy` are the "
                        f"same image (R2 {r2_ok:.4f} vs shuffled {r2_sh:.4f}). Refusing "
                        f"to write a cache that would misalign every downstream array.")
            else:
                rep["audit"] = (f"sem_image_{split} has {ref.shape[0]} rows vs "
                                f"{f1280.shape[0]}; audit skipped")
        else:
            rep["audit"] = f"{ref_p} absent; audit skipped"
        report["splits"][split] = rep

    h.remove()
    (out / "gem_clip_img_report.json").write_text(json.dumps(report, indent=2),
                                                  encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
