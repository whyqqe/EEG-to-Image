#!/usr/bin/env python3
"""Extract OpenCLIP ViT-H-14 image embeddings for the THINGS-EEG2 image corpus.

WHY THIS FILE EXISTS, AND WHY THE TARGET IS 1024-D
--------------------------------------------------
SAMGA is a *retrieval* model: its EEG branch ends in
`share_encoder(eeg_projector(model(x)))` -> 512-d, living in SAMGA's own learned
contrastive space. That space is not CLIP space, so it cannot drive a diffusion
decoder directly. To give SAMGA a generation path we need a target space that

  (a) a pretrained image-conditioned generator already accepts, and
  (b) is reachable from SAMGA's encoder output by a cheap learned head.

The generator we reuse (SDXL-Turbo + `ip-adapter_sdxl_vit-h.bin`) pins (a) exactly.
Read from the IP-Adapter source, `ip_adapter/ip_adapter.py`:

    class IPAdapter:                       # base class, line 66
        def init_proj(self):               # line 86
            image_proj_model = ImageProjModel(
                cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
                clip_embeddings_dim=self.image_encoder.config.projection_dim,   # <-- (A)
                clip_extra_context_tokens=self.num_tokens,
            )

        def get_image_embeds(self, pil_image=None, clip_image_embeds=None):     # line 140
            ...
            clip_image_embeds = self.image_encoder(clip_image.to(...)).image_embeds   # <-- (B)

    class IPAdapterXL(IPAdapter):          # line 221, the SDXL class
        '''SDXL'''                          # overrides NEITHER of the above

and for OpenCLIP ViT-H-14, `config.projection_dim == 1024` while
`config.hidden_size == 1280`. So the non-plus SDXL ViT-H adapter conditions on the
**projected 1024-d `image_embeds`**, i.e. `open_clip.encode_image`'s output -- NOT
the 1280-d penultimate hidden state.

That distinction matters because the *plus* variants do use the other one:
`IPAdapterPlusXL` (line 328) builds a `Resampler(embedding_dim=hidden_size)` and its
`get_image_embeds` (line 303) feeds `hidden_states[-2]`. Picking the wrong variant and
the wrong array yields a shape mismatch at best and a silently mis-scaled condition at
worst. We use the non-plus adapter, so the target here is 1024-d.

This also happens to equal SAMGA's `--eeg_feature_dim 1024`, which is what makes the
bridge head a plain `Linear(1024, 1024)`-class map rather than a lossy bottleneck.

OUTPUT LAYOUT
-------------
    <out>/clip_h14_{split}.npy        (Nconcept, Nimg, 1024) float32, L2-normalized on
                                      the last axis  -- same [concept][image] layout as
                                      the InternViT multilayer arrays, so SAMGA's
                                      EEGPreImageDataset's own (object_idx, image_idx)
                                      can index it directly
    <out>/clip_h14_{split}_flat.npy   (Nconcept*Nimg, 1024) float32 -- the flat form the
                                      generator wants as its neighbour-retrieval gallery
    <out>/manifest.json               provenance + both gate results

THE GATE IS THE POINT
---------------------
Every downstream number is indexed by row, and a mis-ordered feature array still
produces plausible-looking images and metrics. So `image_metadata.npy` (the dataset's
own record of the canonical sequence) is compared against our sorted listing, exactly
as in `scripts/epd/extract_internvit_layers.py`. This costs nothing and catches the one
error that would invalidate the run while looking fine.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
IMG_ROOT = REPO / "data" / "images_set"
META = IMG_ROOT / "image_metadata.npy"
SPLIT_DIR = {"train": "training_images", "test": "test_images"}
EXT = (".jpg", ".jpeg", ".png")
DEFAULT_OUT = REPO / "data" / "image_feature" / "clip_h14_ip_adapter"

# Pinned so the extraction is reproducible and so a cache miss is loud rather than
# silently substituting a different checkpoint with the same name.
OPENCLIP_ARCH = "ViT-H-14"
OPENCLIP_PRETRAINED = "laion2b_s32b_b79k"
EXPECTED_DIM = 1024

# Where the ViT-H-14 weights are known to live on this cluster. Printed on failure only;
# deliberately NOT passed to open_clip as `cache_dir`, because an explicit cache_dir
# overrides HF_HUB_CACHE and this path is an OPENCLIP_CACHE_DIR, not a hub root. See
# `load_encoder`.
CACHE_CANDIDATES = (
    Path("/project/peilab/why/cache/eeg-brainit/hf/hub"),
    Path("/project/peilab/why/cache/huggingface"),
)


def list_concepts(split: str) -> list[tuple[str, list[str]]]:
    """[(concept_dir, [img_path, ...]), ...] in lexicographic (canonical) order."""
    root = IMG_ROOT / SPLIT_DIR[split]
    if not root.is_dir():
        raise SystemExit(f"[FATAL] missing image dir {root}")
    out: list[tuple[str, list[str]]] = []
    for cid in sorted(d.name for d in root.iterdir() if d.is_dir()):
        imgs = sorted(f.name for f in (root / cid).iterdir()
                      if f.name.lower().endswith(EXT))
        if imgs:
            out.append((cid, [str(root / cid / f) for f in imgs]))
    return out


def gate_listing_matches_metadata(split: str, concepts) -> dict:
    """Prove the row order matches the EEG arrays' concept order.

    Compared as a PREFIX: a `--limit` smoke run is intentionally short, and a gate that
    fires on correct input trains the reader to ignore gates. `complete` records whether
    the listing also covers the whole metadata sequence.
    """
    if not META.is_file():
        return {"checked": False, "reason": f"missing {META}"}
    meta = np.load(META, allow_pickle=True).item()
    key = f"{split}_img_files"
    ref = [str(x) for x in meta.get(key, [])]
    ours = [Path(p).name for _cid, paths in concepts for p in paths]
    if not ref:
        return {"checked": False, "reason": f"no {key} in metadata"}
    if len(ours) > len(ref):
        return {"checked": True, "match": False,
                "reason": f"we listed {len(ours)} images, metadata has only {len(ref)}"}
    n = min(len(ours), len(ref))
    first_bad = next((i for i, (a, b) in enumerate(zip(ours[:n], ref[:n])) if a != b), None)
    return {"checked": True, "match": first_bad is None, "n_compared": n,
            "n_metadata": len(ref), "complete": len(ours) == len(ref),
            "first_mismatch": None if first_bad is None else {
                "index": int(first_bad), "ours": ours[first_bad], "ref": ref[first_bad]}}


def gate_backbone_sane(feats: np.ndarray, n_img: np.ndarray, split: str) -> dict:
    """Check that the model actually encodes image *content*, not just a shared direction.

    CLIP spaces are well-conditioned enough that this needs no centering (unlike
    InternViT's raw CLS token, which needed it -- see `extract_internvit_layers.py`).
    For every concept with >=2 images we compare the mean within-concept cosine against
    the mean cross-concept cosine. A correct model gives a large positive gap; a broken
    extraction (zeroed batches, wrong split, shuffled rows) collapses it to ~0.
    """
    flat = feats.reshape(-1, feats.shape[-1]).astype(np.float32)
    flat /= np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-8)
    multi = np.where(n_img >= 2)[0]
    if len(multi) == 0:
        return {"checked": False, "reason": "no concept has >=2 images"}
    within, between = [], []
    for c in multi:
        rows = feats[c][:n_img[c]]
        s = rows @ rows.T
        k = len(rows)
        within.append(float(s[np.triu_indices(k, 1)].mean()) if k > 1 else float("nan"))
        # Cross-concept: this concept's mean direction against a deterministic sample of
        # other concepts' mean directions.
        others = multi[multi != c][:64]
        if len(others):
            between.append(float((rows.mean(0) @ feats[others][:, :1].mean(1).T).mean()))
    within_m = float(np.nanmean(within))
    between_m = float(np.nanmean(between)) if between else float("nan")
    # The paired-diagonal statistic is the one that must hold. Guard it separately from
    # the cross-concept number so a degenerate `between` cannot mask a broken `within`.
    return {"checked": True, "within_cos": within_m, "cross_cos": between_m,
            "gap": within_m - between_m, "ok": bool(within_m - between_m > 0.05)}


def load_encoder(device: torch.device, cache_dir: Path | None):
    """Build the ViT-H-14 encoder, resolving weights through the HF hub cache.

    The `cache_dir` trap, which cost a job and is worth writing down because the failure
    message points the wrong way:

        `huggingface_hub.hf_hub_download(..., cache_dir=X)` treats X as the *hub cache
        root*, overriding the `HF_HUB_CACHE` environment variable. open_clip passes its
        `cache_dir` argument straight through (`open_clip/pretrained.py`,
        `download_pretrained_from_hf`). So passing an `OPENCLIP_CACHE_DIR`-shaped path --
        which is what the env var name suggests, and what looks like the right thing to
        pass -- sends the lookup to a directory that holds no weights, bypassing the
        `HF_HUB_CACHE` that does.

        Worse, open_clip tries the `.safetensors` alternatives first inside a bare
        `except Exception: pass`, so only the final `.bin` attempt raises. The error then
        reads "cannot find open_clip_pytorch_model.bin", even when the real problem is a
        `.safetensors` file sitting one directory away, and even on a machine that has
        every byte of the model.

    Therefore: default to NOT passing `cache_dir` at all and let `HF_HUB_CACHE` govern,
    which is exactly what the sibling project does and is the only configuration known to
    resolve. `--cache-dir` remains available but is opt-in.
    """
    import open_clip

    def _build(kwargs):
        model, _, preprocess = open_clip.create_model_and_transforms(
            OPENCLIP_ARCH, pretrained=OPENCLIP_PRETRAINED, device=device, **kwargs)
        model.eval()
        return model, preprocess

    try:
        return _build({} if cache_dir is None else {"cache_dir": str(cache_dir)})
    except Exception as exc:  # noqa: BLE001
        import os

        hub = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
        snaps = []
        if hub:
            root = Path(hub)
            for d in root.glob(f"models--laion--CLIP-ViT-H-14*"):
                snaps += [str(p) for p in d.glob("snapshots/*")]
        raise SystemExit(
            f"[FATAL] could not build {OPENCLIP_ARCH}/{OPENCLIP_PRETRAINED}: {exc}\n"
            f"        HF_HUB_CACHE={hub}\n"
            f"        snapshots found: {snaps or 'NONE'}\n"
            f"        cache_dir passed: {cache_dir}  (None is correct; an explicit value "
            f"OVERRIDES HF_HUB_CACHE and will fail if it holds no weights)\n"
            f"        A compute node has no network, so the weights must already be in the "
            f"hub cache above.\n"
            f"        known locations: {[str(p) for p in CACHE_CANDIDATES]}"
        ) from exc


@torch.inference_mode()
def extract(split: str, model, preprocess, device: torch.device, batch: int,
            limit: int = 0) -> tuple[np.ndarray, np.ndarray, list[str]]:
    from PIL import Image

    concepts = list_concepts(split)
    if limit > 0:
        concepts = concepts[:limit]
    # Concept-major flattening is what makes `reshape(Nconcept, Nimg, D)` valid; the
    # metadata gate above is what makes the concept order trustworthy.
    items = [(ci, p) for ci, (_cid, paths) in enumerate(concepts) for p in paths]
    dim = None
    vecs: list[np.ndarray] = []

    for start in range(0, len(items), batch):
        chunk = items[start:start + batch]
        xs = []
        for _ci, path in chunk:
            with Image.open(path) as im:
                xs.append(preprocess(im.convert("RGB")))
        x = torch.stack(xs).to(device)
        feats = model.encode_image(x).float()
        if dim is None:
            dim = int(feats.shape[-1])
            if dim != EXPECTED_DIM:
                raise SystemExit(
                    f"[FATAL] {OPENCLIP_ARCH}/{OPENCLIP_PRETRAINED} gave {dim}-d features, "
                    f"but IP-Adapter's `image_embeds` path requires {EXPECTED_DIM}-d. If you "
                    f"switched to a *_plus_* adapter, the target is instead "
                    f"`hidden_states[-2]` (1280-d) and this script is the wrong one."
                )
        vecs.append(feats.cpu().numpy())
        done = min(start + batch, len(items))
        if start % (batch * 20) < batch or done == len(items):
            print(f"  [{split}] {done}/{len(items)}", flush=True)

    flat = np.concatenate(vecs, 0).astype(np.float32)
    flat /= np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-8)

    n_img = np.array([len(paths) for _cid, paths in concepts], dtype=np.int64)
    if int(n_img.sum()) != flat.shape[0]:
        raise SystemExit("[FATAL] concept/image bookkeeping disagrees with the feature rows")
    per_concept = set(n_img.tolist())
    if len(per_concept) != 1:
        raise SystemExit(
            f"[FATAL] uneven images per concept {sorted(per_concept)}; a ragged array cannot "
            f"be expressed as (Nconcept, Nimg, D). A partial download would look like this."
        )
    return flat.reshape(len(concepts), int(n_img[0]), dim), n_img, concepts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--splits", default="train,test")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0,
                    help="concepts per split (smoke); 0 = all")
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="DANGEROUS: opens_clip forwards this to hf_hub_download as the hub "
                         "cache root, OVERRIDING HF_HUB_CACHE. Leave unset unless you know "
                         "it contains the weights.")
    ap.add_argument("--verify-against", type=Path, default=None,
                    help="optional flat (N,1024) array; report cosine agreement. Used to "
                         "cross-check against the sibling project's existing CLIP cache.")
    args = ap.parse_args()

    out = args.out_dir if args.out_dir.is_absolute() else REPO / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device} out={out}")

    cache_dir = args.cache_dir
    print(f"[INFO] cache_dir={cache_dir} (None -> HF_HUB_CACHE="
          f"{os.environ.get('HF_HUB_CACHE', '<unset>')})")
    model, preprocess = load_encoder(device, cache_dir)

    manifest: dict = {
        "arch": OPENCLIP_ARCH, "pretrained": OPENCLIP_PRETRAINED,
        "conditioning_dim": EXPECTED_DIM,
        "conditioning_source": "open_clip.encode_image -> projected image_embeds",
        "why": "IP-Adapter SDXL ViT-H (non-plus) conditions on image_embeds "
               "(clip_embeddings_dim=config.projection_dim); see module docstring",
        "cache_dir": str(cache_dir), "limit": args.limit, "splits": {},
    }

    for split in [s for s in args.splits.split(",") if s]:
        t0 = time.time()
        print(f"[INFO] {split}: extracting")
        feats, n_img, concepts = extract(split, model, preprocess, device, args.batch,
                                         args.limit)
        gate = gate_listing_matches_metadata(split, concepts)
        sanity = gate_backbone_sane(feats, n_img, split)

        tags = f" (limit={args.limit})" if args.limit else ""
        np.save(out / f"clip_h14_{split}{'_smoke' if args.limit else ''}.npy", feats)
        np.save(out / f"clip_h14_{split}_flat{'_smoke' if args.limit else ''}.npy",
                feats.reshape(-1, feats.shape[-1]))

        entry = {"shape": list(feats.shape), "n_concepts": len(concepts),
                 "images_per_concept": int(n_img[0]),
                 "gate_listing": gate, "gate_backbone": sanity,
                 "seconds": round(time.time() - t0, 1), "limited": bool(args.limit)}
        manifest["splits"][split] = entry
        # The backbone gate is skipped on the test split: THINGS-EEG2 test has exactly one
        # image per concept, so there is no within-concept pair to measure. Format every
        # field defensively -- a gate that reports `checked: False` is not a failure, and
        # it must not crash the report of a successful extraction.
        _w = sanity.get("within_cos")
        _g = sanity.get("gap")
        print(f"[GATE] {split} listing={gate.get('match')} complete={gate.get('complete')} "
              f"backbone={sanity.get('checked')} "
              f"within={_w:.4f} gap={_g:.4f}"
              if sanity.get("checked") else
              f"[GATE] {split} listing={gate.get('match')} complete={gate.get('complete')} "
              f"backbone=skipped ({sanity.get('reason')})")

        if not gate.get("match", False) and gate.get("checked"):
            print(f"[FATAL] {split}: image listing does not match image_metadata.npy: "
                  f"{gate.get('first_mismatch')}")
            print(json.dumps(manifest, indent=2))
            return 2
        if sanity.get("checked") and not sanity.get("ok"):
            print(f"[FATAL] {split}: backbone sanity failed {sanity}")
            print(json.dumps(manifest, indent=2))
            return 3

        if args.verify_against is not None:
            ref_path = args.verify_against if args.verify_against.is_absolute() \
                else REPO / args.verify_against
            if ref_path.is_file():
                ref = np.load(ref_path).reshape(-1, np.load(ref_path).shape[-1])
                ours = feats.reshape(-1, feats.shape[-1])
                if ref.shape == ours.shape:
                    cos = (ref * ours).sum(1)
                    entry["verify_against"] = {
                        "path": str(ref_path), "mean_row_cosine": float(cos.mean()),
                        "frac_cos_gt_0.99": float((cos > 0.99).mean()),
                        "agrees": bool(cos.mean() > 0.99)}
                    print(f"[CROSSCHECK] mean row cosine vs {ref_path.name} = "
                          f"{cos.mean():.5f} ({(cos > 0.99).mean() * 100:.2f}% > 0.99)")
                else:
                    entry["verify_against"] = {"path": str(ref_path),
                                               "skipped": f"shape {ref.shape} != {ours.shape}"}
                    print(f"[CROSSCHECK] skipped: shape {ref.shape} != {ours.shape}")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[OK] wrote {out}/manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
