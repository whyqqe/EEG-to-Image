"""Generate the `depth` and `edge` conditions CogCapPro's extra branches align to.

CogCapPro has four alignment branches (image / text / depth / edge) whose targets are
`CLIP(image)`, `CLIP(caption)`, `CLIP(depth map)`, `CLIP(edge map)` respectively
(`generate_image/generator.py:225-232` names the source folders
`Image_set_Resize` / `Image_depth_set_Resize` / `Image_edge_set_Resize`). This project has
the RGB set (`data/images_set`) but not the depth or edge renders, so they are generated
here and encoded through the **same** frozen encoder the image conditions use.

Consistency with the `image` conditions is not optional: the three branches share a
`logit_scale` and are averaged into one loss, so a target space that differs by
preprocessing would show up as one branch being systematically easier to fit rather than
as an error. That is why `list_concepts` / `load_encoder` /
`gate_listing_matches_metadata` are imported from `recon.extract_clip_h14` rather than
reimplemented -- that module is what produced `clip_h14_ip_adapter`, it pins
ViT-H-14/laion2b_s32b_b79k, and it carries the listing gate that ties row order to
`image_metadata.npy`.

Depth model: `depth-anything/Depth-Anything-V2-Small-hf`, already in the cluster HF cache
(`cache/eeg-brainit/hf/hub/models--depth-anything--Depth-Anything-V2-Small-hf`).

Not included: the `text` branch. CogCapPro's text target is a per-image BLIP2 caption
encoded by the CLIP text tower, read from `weights/texts/eeg/texts_BLIP2_{mode}.npy`
(`data/eeg.py:264-269`). This project's caption files are SDXL prompts keyed by concept,
not the same object; substituting them would change what the branch is aligned to while
looking like a faithful reproduction, so the branch is omitted instead.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cogcap import config                                    # noqa: E402
from recon.extract_clip_h14 import (                          # noqa: E402
    EXPECTED_DIM,
    gate_listing_matches_metadata,
    list_concepts,
    load_encoder,
)

DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


def build_depth(device):
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    proc = AutoImageProcessor.from_pretrained(DEPTH_MODEL)
    model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL).to(device).eval()
    return proc, model


def depth_renders(paths, proc, model, device, size: int = 224):
    """Depth maps as RGB PIL images, normalised per image to the full 0-255 range.

    Per-image normalisation (not dataset-wide) because the depth branch is aligned to a
    CLIP embedding of the render, and CLIP is contrast-normalised: what a single rendered
    map conveys is its relative structure, not its absolute metric depth. A dataset-wide
    range would make a close-up and a wide shot of the same class produce near-identical
    maps.
    """
    from PIL import Image

    ims = []
    for p in paths:
        with Image.open(p) as im:
            ims.append(im.convert("RGB").resize((size, size), Image.BICUBIC))
    with torch.inference_mode():
        inputs = proc(images=ims, return_tensors="pt").to(device)
        pred = model(**inputs).predicted_depth                       # [B, h, w]
    out = []
    for i in range(pred.shape[0]):
        d = pred[i]
        d = (d - d.min()) / (d.max() - d.min() + 1e-8)
        arr = (d.clamp(0, 1) * 255.0).to(torch.uint8).cpu().numpy()
        rgb = np.repeat(arr[:, :, None], 3, axis=2)
        out.append(Image.fromarray(rgb, mode="RGB"))
    return out


def edge_renders(paths, size: int = 224):
    """Canny edges as RGB PIL images, on black, with the same thresholds CogCapPro's
    `blur_kernel_size`/edge pipeline concept assumes (8-bit, 100/200)."""
    import cv2
    from PIL import Image

    out = []
    for p in paths:
        with Image.open(p) as im:
            im = im.convert("RGB").resize((size, size), Image.BICUBIC)
        gray = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2GRAY)
        e = cv2.Canny(gray, 100, 200)
        rgb = np.repeat(e[:, :, None], 3, axis=2)
        out.append(Image.fromarray(rgb, mode="RGB"))
    return out


def extract(split, modality, encoder, preprocess, device, batch, limit, depth_ctx):
    concepts = list_concepts(split)
    if limit:
        concepts = concepts[:limit]
    items = [p for _cid, paths in concepts for p in paths]
    n_img = [len(paths) for _cid, paths in concepts]
    if len(set(n_img)) != 1:
        raise SystemExit(f"[FATAL] {split}: ragged images-per-concept {sorted(set(n_img))}")

    vecs = []
    t0 = time.time()
    for start in range(0, len(items), batch):
        chunk = items[start:start + batch]
        if modality == "depth":
            pil = depth_renders(chunk, *depth_ctx, device)
        elif modality == "edge":
            pil = edge_renders(chunk)
        else:
            raise SystemExit(f"unknown modality {modality}")
        x = torch.stack([preprocess(im) for im in pil]).to(device)
        with torch.inference_mode():
            feats = encoder.encode_image(x).float()
        vecs.append(feats.cpu().numpy())
        if start % (batch * 20) < batch or start + batch >= len(items):
            done = min(start + batch, len(items))
            rate = done / max(time.time() - t0, 1e-6)
            print(f"[{modality}:{split}] {done}/{len(items)} ({rate:.1f}/s)", flush=True)
    feats = np.concatenate(vecs, 0)
    if feats.shape[-1] != EXPECTED_DIM:
        raise SystemExit(f"[FATAL] got {feats.shape[-1]}-d, expected {EXPECTED_DIM}")
    return feats.reshape(len(concepts), n_img[0], -1), concepts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modalities", default="depth,edge")
    ap.add_argument("--splits", default="train,test")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="train concepts; smoke only")
    ap.add_argument("--limit-test", type=int, default=-1,
                    help="test concepts; -1 reuses --limit, 0 means all 200")
    ap.add_argument("--suffix", default="",
                    help="appended to both output names, e.g. _smoke")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[INFO] device={device} modalities={args.modalities}", flush=True)
    encoder, preprocess = load_encoder(device, None)

    modalities = [m for m in args.modalities.split(",") if m]
    depth_ctx = build_depth(device) if "depth" in modalities else None

    def limit_for(split: str) -> int:
        if split == "test" and args.limit_test >= 0:
            return args.limit_test
        return args.limit

    out_root = config.IMAGE_FEATURE_DIR
    manifest = {"depth_model": DEPTH_MODEL if "depth" in modalities else None,
                "encoder": "ViT-H-14/laion2b_s32b_b79k (via recon.extract_clip_h14)",
                "edge": "cv2.Canny(100,200) on the 224x224 greyscale render",
                "limits": {"train": args.limit, "test": limit_for("test")},
                "suffix": args.suffix,
                "splits": {}}
    for modality in modalities:
        d = out_root / f"cogcap_{modality}"
        d.mkdir(parents=True, exist_ok=True)
        for split in [s for s in args.splits.split(",") if s]:
            lim = limit_for(split)
            feats, concepts = extract(split, modality, encoder, preprocess, device,
                                      args.batch, lim, depth_ctx)
            gate = gate_listing_matches_metadata(split, concepts)
            np.save(d / f"{modality}_{split}{args.suffix}.npy", feats.astype(np.float32))
            manifest["splits"].setdefault(modality, {})[split] = {
                "shape": list(feats.shape), "gate_listing": gate, "limit": lim,
            }
            print(f"[ok   ] {modality}/{split} {tuple(feats.shape)} gate={gate.get('match')}",
                  flush=True)
        (d / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out_root / "cogcap_prep_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
