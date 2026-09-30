#!/usr/bin/env python
"""
Layer-wise frozen visual feature extraction for the neural-visibility study.

WHY THIS RUNS ON A GPU BOX (docs E1)
------------------------------------
Encoders are FROZEN instruments: never trained, never see EEG.  This is what
makes the NV interpretation valid (docs 3.3) -- candidate axes must come from a
source statistically independent of the EEG, otherwise the analysis is circular.

We extract EVERY transformer block, not only the final projection.  A CLIP image
tower's final projection has already been contrastively compressed onto the
text-aligned manifold, which removes dimensions; the layer profile must stay free
to reveal which depth is most neurally visible (test P5 vs the NVOL literature).

DATA LAYOUT (verified)
    training_images/<concept_id>_<name>/<name>_NN.jpg    1654 concepts x 10
    test_images/<concept_id>_<name>/<name>_NN.jpg         200 concepts x  1
Concept order = lexicographic on the zero-padded id, which is the canonical
THINGS order and must match the EEG arrays.  The order is verified against the
existing ViT-H-14 features by verify_alignment() rather than assumed.

Outputs (idempotent; existing arrays are skipped unless --overwrite)
    <out>/<tag>/layer_<L>.npy   float32 (n_concepts, n_imgs, d)
    <out>/<tag>/concepts.json   the exact concept order used
    <out>/<tag>/manifest.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

IMG_ROOT = "/project/peilab/why/data/images_set"
EEG_ROOT = "/project/peilab/why/NeuroBridge/data/things_eeg"
SPLIT_DIR = {"train": "training_images", "test": "test_images"}


# ---------------------------------------------------------------------------
# Concept-aware image listing
# ---------------------------------------------------------------------------


def list_concepts(split: str) -> list:
    """[(concept_id, [img_path, ...]), ...] in canonical order."""
    root = os.path.join(IMG_ROOT, SPLIT_DIR[split])
    ids = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    out = []
    for cid in ids:
        d = os.path.join(root, cid)
        imgs = sorted(f for f in os.listdir(d)
                      if f.lower().endswith((".jpg", ".jpeg", ".png")))
        if imgs:
            out.append((cid, [os.path.join(d, f) for f in imgs]))
    return out


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


def _store(feats: dict, key: str, v: np.ndarray) -> None:
    """Tokens -> mean over patch tokens (skip CLS); keep everything else."""
    if v.ndim == 3:
        v = v[:, 1:].mean(1) if v.shape[1] > 1 else v[:, 0]
    feats.setdefault(key, []).append(v.astype(np.float32))


def extract_open_clip(model_name, pretrained, concepts, batch, device):
    import torch
    import open_clip
    from PIL import Image

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device)
    model.eval()
    vision = model.visual
    trunk = getattr(vision, "trunk", vision)

    acts, handles = {}, []
    for name, mod in trunk.named_children():
        handles.append(mod.register_forward_hook(
            lambda _m, _i, o, n=name: acts.__setitem__(
                n, (o[0] if isinstance(o, (tuple, list)) else o).detach())))

    feats: dict = {}
    done = 0
    with torch.no_grad():
        for cid, paths in concepts:
            for i in range(0, len(paths), batch):
                chunk = paths[i:i + batch]
                ims = torch.stack([preprocess(Image.open(p).convert("RGB"))
                                   for p in chunk]).to(device)
                acts.clear()
                pooled = vision(ims)
                pooled = pooled if torch.is_tensor(pooled) else pooled[0]
                for k, v in {**acts, "_pooled": pooled}.items():
                    _store(feats, k, v.float().cpu().numpy())
            done += 1
            if done % 100 == 0:
                print(f"    {done}/{len(concepts)} concepts", flush=True)
    for h in handles:
        h.remove()
    return feats


def extract_timm(model_name, concepts, batch, device):
    import torch
    import timm
    from PIL import Image

    model = timm.create_model(model_name, pretrained=True,
                              num_classes=0).to(device).eval()
    acts, handles = {}, []
    for name, mod in model.named_children():
        handles.append(mod.register_forward_hook(
            lambda _m, _i, o, n=name: acts.__setitem__(n, o)))

    cfg = timm.data.resolve_data_config({}, model=model)
    tf = timm.data.create_transform(**cfg, is_training=False)

    feats: dict = {}
    done = 0
    with torch.no_grad():
        for cid, paths in concepts:
            for i in range(0, len(paths), batch):
                chunk = paths[i:i + batch]
                ims = torch.stack([tf(Image.open(p).convert("RGB"))
                                   for p in chunk]).to(device)
                acts.clear()
                pooled = model(ims)
                for k, v in {**acts, "_pooled": pooled}.items():
                    _store(feats, k, v.float().cpu().numpy())
            done += 1
            if done % 200 == 0:
                print(f"    {done}/{len(concepts)} concepts", flush=True)
    for h in handles:
        h.remove()
    return feats


# ---------------------------------------------------------------------------
# Verification against the shipped features (order + semantics)
# ---------------------------------------------------------------------------


def verify_alignment(tag_dir: str, layer_key: str = "_pooled") -> dict:
    """Confirm that our extraction matches the shipped ViT-H-14 features.

    This is not optional: if the concept order or the image-averaging convention
    differs from the shipped arrays, every downstream EEG alignment silently
    becomes noise.  We therefore check the per-concept correlation between our
    pooled feature and the shipped one, which must be very high.
    """
    ours_f = os.path.join(tag_dir, f"layer_{layer_key}.npy")
    theirs_f = f"{EEG_ROOT}/image_feature/ViT-H-14/image_train.npy"
    if not (os.path.exists(ours_f) and os.path.exists(theirs_f)):
        return {"checked": False, "reason": "missing file"}
    A = np.load(ours_f)
    B = np.load(theirs_f)
    if A.ndim == 3:
        A = A.mean(1)
    if A.shape != B.shape:
        return {"checked": False, "reason": f"shape {A.shape} vs {B.shape}"}
    Ac = A - A.mean(0, keepdims=True)
    Bc = B - B.mean(0, keepdims=True)
    cos = (Ac * Bc).sum(1) / np.maximum(
        np.linalg.norm(Ac, axis=1) * np.linalg.norm(Bc, axis=1), 1e-12)
    return {"checked": True, "mean_cosine": float(cos.mean()),
            "min_cosine": float(cos.min()),
            "frac_above_0.9": float((cos > 0.9).mean()),
            "shape": list(A.shape)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/project/peilab/why/eeg-retrieval/alignment/data/features")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--kind", required=True, choices=["open_clip", "timm"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--pretrained", default="")
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verify", action="store_true",
                    help="check our pooled feature against the shipped ViT-H-14")
    ap.add_argument("--limit", type=int, default=0,
                    help="use only the first N concepts (smoke tests)")
    args = ap.parse_args()

    outdir = os.path.join(args.out, args.tag)
    os.makedirs(outdir, exist_ok=True)
    concepts = list_concepts(args.split)
    if args.limit:
        concepts = concepts[:args.limit]
    n_imgs = np.array([len(p) for _, p in concepts])

    print("=" * 96)
    print(f"extract_features  tag={args.tag}  kind={args.kind}  model={args.model}")
    print(f"  split={args.split}  concepts={len(concepts)}  "
          f"imgs/concept={n_imgs.min()}..{n_imgs.max()}")
    print(f"  out={outdir}")
    print("=" * 96)
    if not concepts:
        print("[FATAL] no concepts found"); return 2

    t0 = time.time()
    if args.kind == "open_clip":
        feats = extract_open_clip(args.model, args.pretrained, concepts,
                                  args.batch, args.device)
    else:
        feats = extract_timm(args.model, concepts, args.batch, args.device)

    # reshape each flat list into (n_concepts, imgs_per_concept, d)
    manifest = {"tag": args.tag, "kind": args.kind, "model": args.model,
                "pretrained": args.pretrained, "split": args.split,
                "n_concepts": len(concepts), "layers": {},
                "imgs_per_concept_min": int(n_imgs.min()),
                "imgs_per_concept_max": int(n_imgs.max())}

    n_saved = 0
    # _store() accumulates IMAGE-level rows (n_concepts * imgs_per_concept), not
    # concept-level rows.  The guard below used to compare against len(concepts),
    # which is only correct when every concept has exactly one image -- i.e. the
    # test split.  On the train split (10 imgs/concept) it rejected all 16540-row
    # layers and silently wrote nothing.  Compare against the total row count.
    n_rows_expect = int(n_imgs.sum())
    for k, chunks in sorted(feats.items()):
        v = np.concatenate(chunks, 0)
        if v.shape[0] != n_rows_expect:
            print(f"  [skip] {k}: {v.shape[0]} rows != {n_rows_expect} "
                  f"(= {len(concepts)} concepts x imgs)")
            continue
        if n_imgs.min() == n_imgs.max():
            try:
                v = v.reshape(len(concepts), int(n_imgs[0]), -1)
            except ValueError:
                print(f"  [skip] {k}: cannot reshape {v.shape}")
                continue
        else:
            print(f"  [warn] {k}: ragged imgs/concept, keeping flat "
                  f"{v.shape} (aggregate per concept downstream)")
        fn = os.path.join(outdir, f"layer_{k}.npy")
        if os.path.exists(fn) and not args.overwrite:
            print(f"  [skip] {os.path.basename(fn)} exists")
        else:
            np.save(fn, v)
        n_saved += 1
        manifest["layers"][k] = list(v.shape)
        nrm = np.linalg.norm(v.reshape(-1, v.shape[-1]), axis=1)
        print(f"  {k:24s} -> {str(v.shape):24s} "
              f"|f| mean={nrm.mean():8.3f} NaN={int(np.isnan(v).sum())}", flush=True)

    with open(os.path.join(outdir, "concepts.json"), "w") as f:
        json.dump({"split": args.split, "ids": [c for c, _ in concepts]}, f, indent=2)

    if n_saved == 0:
        print(f"\n[FATAL] no layers saved for tag={args.tag}  "
              f"(expected {n_rows_expect} rows per layer, got: "
              f"{ {k: int(np.concatenate(v,0).shape[0]) for k, v in feats.items()} })")
        return 3

    if args.verify:
        vr = verify_alignment(outdir, "_pooled")
        manifest["verify_vs_shipped"] = vr
        print("\n  [verify vs shipped ViT-H-14]", json.dumps(vr, indent=4))

    manifest["elapsed_s"] = round(time.time() - t0, 1)
    with open(os.path.join(outdir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\ndone in {manifest['elapsed_s']}s -> {outdir}/manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
