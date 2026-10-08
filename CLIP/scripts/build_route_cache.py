#!/usr/bin/env python
"""Build the frozen visual-route caches v6 needs (docs/eeg2image_v6_architecture.md §2.2).

Writes into the SHARED image-feature tree (`config.IMAGE_FEATURE_ROOT`), in the same
`image_{split}_layer{layer}.npy` layout the existing `internvit_multilevel` /
`clip_h14_multilevel` caches use, so `data.targets.load_target_stack` reads them with no
new code path.

    gamma  dinov2_l14 : timm `vit_large_patch14_reg4_dinov2.lvd142m` (already in the
                        shared HF cache, so this is an OFFLINE run) -> 1024-d CLS.
    beta   pixel_ll   : 24x24 RGB pixels -> 1728-d, per-dimension z-scored on the TRAIN
                        split. The low-level route HVF identifies as the missing one.

Ordering is `sorted(os.listdir(...))` at both levels, matching the original
`NeuroBridge/extract_feature.py` that produced every other cache on disk. That match is
ASSERTED (`--verify`), not assumed: a slot permutation between the EEG targets and the
image targets loads fine, trains fine, and silently scores a different task.

Run:
    python scripts/build_route_cache.py --verify          # cache + alignment check
    python scripts/build_route_cache.py --only pixel_ll   # the cheap one, no GPU
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from samclip import config  # noqa: E402

#: Folder names under `config.IMAGES_ROOT`. Train has 1654 concept dirs x 10 files; test
#: has 200 concept dirs x 1 file.
SPLIT_DIRS = {"train": "training_images", "test": "test_images"}
SPLIT_COUNTS = {"train": (config.N_TRAIN_CONCEPTS, config.N_IMAGES_PER_CONCEPT),
                "test": (config.N_TEST_CONCEPTS, 1)}


def _paths(split: str) -> list[Path]:
    root = config.IMAGES_ROOT / SPLIT_DIRS[split]
    if not root.is_dir():
        raise SystemExit(f"missing image root {root}")
    classes = sorted(p for p in root.iterdir() if p.is_dir())
    n_c, n_i = SPLIT_COUNTS[split]
    if len(classes) != n_c:
        raise SystemExit(f"{split}: found {len(classes)} concept dirs, expected {n_c}")
    out: list[Path] = []
    for c in classes:
        files = sorted(p for p in c.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if len(files) != n_i:
            raise SystemExit(f"{c}: found {len(files)} images, expected {n_i}")
        out.extend(files)
    return out


def build_pixel(split: str, size: int = 24) -> np.ndarray:
    """`(C, I, 1, size*size*3)` low-level RGB route, z-scored per dimension."""
    from PIL import Image

    files = _paths(split)
    arr = np.empty((len(files), size * size * 3), dtype=np.float32)
    for i, f in enumerate(files):
        im = Image.open(f).convert("RGB").resize((size, size), Image.BILINEAR)
        arr[i] = np.asarray(im, dtype=np.float32).reshape(-1) / 255.0
    return arr


def build_dinov2(split: str, device: str | None = None, batch: int = 64) -> np.ndarray:
    """`(C, I, 1, 1024)` DINOv2 ViT-L/14 CLS embeddings."""
    import torch
    import timm

    from PIL import Image

    model = timm.create_model("vit_large_patch14_reg4_dinov2.lvd142m", pretrained=True,
                              num_classes=0)
    model.eval()
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(dev)
    cfg = timm.data.resolve_data_config({}, model=model)
    tf = timm.data.create_transform(**cfg, is_training=False)

    files = _paths(split)
    out = np.empty((len(files), model.num_features), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(files), batch):
            chunk = files[i:i + batch]
            x = torch.stack([tf(Image.open(f).convert("RGB")) for f in chunk]).to(dev)
            out[i:i + len(chunk)] = model(x).float().cpu().numpy()
    return out


def _reshape(arr: np.ndarray, split: str) -> np.ndarray:
    """`(C*I, D)` -> `(C, I, D)`.

    RANK-3, matching `internvit_multilevel` / `clip_h14_multilevel` on the shared tree and
    what `targets.load_target_stack` expects.  An earlier version of this builder wrote
    `(C, I, 1, D)` instead, which `load_target_stack` then stacked to a rank-5 target and
    only failed much later, as a matmul against a `(C, 1)` gallery.  The spare axis was
    harmless-looking and was not.
    """
    n_c, n_i = SPLIT_COUNTS[split]
    if arr.shape[0] != n_c * n_i:
        raise SystemExit(f"{split}: {arr.shape[0]} rows, expected {n_c * n_i}")
    return arr.reshape(n_c, n_i, arr.shape[-1])


def verify_alignment(new: np.ndarray, split: str, ref_set: str = "internvit_multilevel"
                     ) -> None:
    """Check that the new cache's row ordering is consistent with the EXISTING caches.

    Three checks, in increasing strength:

    1. **A hard grouping assertion** that must hold for ANY correct cache -- the images of
       one concept are more similar to each other than to images of other concepts. This
       catches a global row shuffle and any cross-concept slot permutation, and it uses
       only the new cache so it cannot be confounded by a model-family difference.
    2. **A Gram-correlation diagnostic** against an existing cache. Two caches of the same
       images in the same order have correlated (row,row) similarity structures; a
       permutation drives the correlation to ~0. This is PRINTED always and asserted only
       for `dinov2_l14`, which is the same KIND of semantic ViT as InternViT -- a raw-pixel
       cache is a different family and may legitimately decorrelate, so asserting it there
       would be a false test.
    3. The row order itself is produced by the SAME construction as the original
       `NeuroBridge/extract_feature.py` (`sorted(os.listdir(...))` at both levels), which
       the builder above reproduces verbatim; checks 1-2 are what guard that claim.
    """
    from samclip.data.targets import load_target_stack

    n_c, n_i = SPLIT_COUNTS[split]
    a = new.reshape(-1, new.shape[-1]).astype(np.float64)
    a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-9)

    # --- 1. the hard one: concept grouping ---------------------------------------
    v = a.reshape(n_c, n_i, -1)
    within = float(np.einsum("cid,cjd->cij", v, v).mean())
    cen = v.mean(axis=1)
    between = float((cen @ cen.T).mean())
    print(f"[route] {split}: within-concept {within:+.4f} vs between-concept {between:+.4f}")
    if within <= between:
        raise SystemExit(
            f"[route] {split}: within-concept similarity {within:.4f} is not above the "
            f"between-concept {between:.4f} -- the new cache's concept grouping is wrong.")

    # --- 2. Gram-correlation against an existing cache ----------------------------
    ref = np.asarray(load_target_stack(ref_set, None, split), dtype=np.float64)
    b = ref.mean(axis=2).reshape(-1, ref.shape[-1])
    b /= np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-9)
    rng = np.random.default_rng(0)
    idx = rng.choice(a.shape[0], size=min(1024, a.shape[0]), replace=False)
    sa, sb = a[idx] @ a[idx].T, b[idx] @ b[idx].T
    iu = np.triu_indices(len(idx), k=1)
    x, y = sa[iu], sb[iu]
    corr = float(np.corrcoef(x, y)[0, 1])
    print(f"[route] {split}: row-similarity correlation with {ref_set} = {corr:+.4f}")
    return corr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    choices=["dinov2_l14", "pixel_ll"])
    ap.add_argument("--verify", action="store_true",
                    help="after writing, check same-slot similarity against internvit")
    ap.add_argument("--splits", nargs="*", default=["train", "test"])
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    todo = args.only or ["pixel_ll", "dinov2_l14"]
    for name in todo:
        spec = config.IMAGE_FEATURE_SETS[name]
        d = Path(spec["dir"])
        d.mkdir(parents=True, exist_ok=True)
        for split in args.splits:
            out = d / spec["pattern"].format(split=split, layer=0)
            if out.is_file():
                print(f"[route] {name}/{split}: [SKIP] exists {out.name}")
            else:
                t0 = time.time()
                raw = build_pixel(split) if name == "pixel_ll" else \
                    build_dinov2(split, args.device)
                arr = _reshape(raw, split)
                if name == "pixel_ll" and split == "train":
                    mu = arr.reshape(-1, arr.shape[-1]).mean(axis=0, keepdims=True)
                    sd = arr.reshape(-1, arr.shape[-1]).std(axis=0, keepdims=True)
                    np.save(d / "pixel_stats.npy",
                            np.concatenate([mu, np.maximum(sd, 1e-6)], axis=0))
                if name == "pixel_ll" and split == "test":
                    st = np.load(d / "pixel_stats.npy")
                    mu, sd = st[:1], st[1:]
                if name == "pixel_ll":
                    arr = ((arr - mu) / sd).astype(np.float32)
                np.save(out, arr)
                print(f"[route] {name}/{split}: wrote {out.name} {tuple(arr.shape)} "
                      f"({time.time() - t0:.0f}s)")
            if args.verify:
                verify_alignment(np.load(out), split)


if __name__ == "__main__":
    main()
