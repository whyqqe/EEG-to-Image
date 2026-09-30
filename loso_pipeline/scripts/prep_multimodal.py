#!/usr/bin/env python
"""Extract the frozen multimodal targets that Stage 2 aligns the EEG encoder to.

Four teacher spaces are produced, matching the design's section 4:

  clip_image    laion CLIP ViT-H-14 image embedding, 1024-d.  Global visual
                semantics: category, object identity, overall appearance.
  clip_text     laion CLIP ViT-H-14 text embedding, 1024-d, in the *same* space as
                clip_image because it is the same model's text tower.  Two
                granularities are written: per-image BLIP2 captions (instance
                level) and per-concept template prompts (category level).  The
                design warns that category-only text supervision yields
                category-conforming but instance-agnostic generations, so both are
                kept and the weights are chosen at training time.
  dino          DINOv2-L global feature, 1024-d.  Self-supervised, so it carries
                layout/shape structure that CLIP's language supervision discards.
  vae_latent    SDXL VAE posterior mean, (4, 64, 64), scaled.  This is the
                diffusion target and the low-level appearance constraint.

Notes on reuse: the IP-Adapter that conditions the frozen diffusion backbone in
Stage 3 expects exactly the CLIP ViT-H-14 *projected image* embedding, i.e. the
same tensor as `clip_image`.  It is not recomputed here -- `clip_image` is passed
to the adapter directly.  This mirrors the reference implementation, where the
adapter's conditioning targets are produced by the IP-Adapter image encoder over
the same stimuli.

Every output is a preallocated `.npy` written through a memmap, with a sidecar
progress file, so a killed job resumes instead of restarting.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from PIL import Image

from loso import paths
from loso.data import captions as captions_mod
from loso.data import things

# SDXL's VAE was trained with this scale factor; the diffusion stage must use the
# same one or the UNet sees latents outside its training distribution.
SDXL_VAE_SCALE = 0.13025

# Canonical pixel resolution for the diffusion stage and for the pixel-space
# metrics.  The source JPEGs are 500x500, so this is a mild resize.
SD_RESOLUTION = 512
# SDXL VAE downsamples by 8.
SD_LATENT_RES = SD_RESOLUTION // 8


# --- resumable memmap writer -------------------------------------------------
class TargetWriter:
    """Append rows to a preallocated .npy, tracking which rows are already done."""

    def __init__(self, out_path: Path, n_rows: int, row_shape: tuple[int, ...],
                 dtype: np.dtype):
        self.out_path = out_path
        self.progress_path = out_path.with_suffix(out_path.suffix + ".progress.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.is_file():
            arr = np.load(out_path, mmap_mode="r")
            if arr.shape != (n_rows, *row_shape):
                raise ValueError(
                    f"{out_path} has shape {arr.shape}, expected {(n_rows, *row_shape)}. "
                    f"Delete it to rebuild."
                )
        else:
            np.lib.format.open_memmap(
                out_path, mode="w+", dtype=dtype, shape=(n_rows, *row_shape),
            )
        self.arr = np.load(out_path, mmap_mode="r+")
        self.n_rows = n_rows
        self.done: set[int] = set()
        if self.progress_path.is_file():
            payload = json.loads(self.progress_path.read_text())
            if payload.get("n_rows") == n_rows:
                self.done = set(payload.get("done", []))

    def write(self, rows: Sequence[int], values: np.ndarray) -> None:
        if len(rows) != len(values):
            raise ValueError(f"{len(rows)} indices vs {len(values)} rows")
        self.arr[list(rows)] = values
        self.done.update(int(r) for r in rows)

    def flush(self) -> None:
        self.arr.flush()
        # Written last and atomically: a truncated progress file would silently mark
        # rows as done that never reached the memmap.
        tmp = self.progress_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"n_rows": self.n_rows, "done": sorted(self.done)}))
        tmp.replace(self.progress_path)

    @property
    def n_done(self) -> int:
        return len(self.done)


# --- teacher wrappers --------------------------------------------------------
class ClipTeacher:
    """laion CLIP ViT-H-14, used for both the image and the text tower."""

    def __init__(self, device: str, dtype: torch.dtype):
        from transformers import CLIPModel, CLIPImageProcessor, CLIPProcessor

        self.device, self.dtype = device, dtype
        self.model = CLIPModel.from_pretrained(paths.CLIP_ID, torch_dtype=dtype).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.image_processor = CLIPImageProcessor.from_pretrained(paths.CLIP_ID)
        self.text_processor = CLIPProcessor.from_pretrained(paths.CLIP_ID).tokenizer

    @torch.inference_mode()
    def image_features(self, images: list[Image.Image]) -> torch.Tensor:
        px = self.image_processor(images=images, return_tensors="pt")["pixel_values"]
        out = self.model.get_image_features(pixel_values=px.to(self.device, self.dtype))
        return out.float()

    @torch.inference_mode()
    def text_features(self, texts: list[str]) -> torch.Tensor:
        tok = self.text_processor(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=77,
        )
        out = self.model.get_text_features(
            input_ids=tok["input_ids"].to(self.device),
            attention_mask=tok.get("attention_mask", torch.ones_like(tok["input_ids"])).to(self.device),
        )
        return out.float()


class DinoTeacher:
    """DINOv2-L/14 at its native resolution, exposing the CLS global feature."""

    def __init__(self, device: str, dtype: torch.dtype, size: int | None = None):
        import timm
        import timm.data

        self.device, self.dtype = device, dtype
        self.model = timm.create_model(
            paths.DINO_ID, pretrained=True, num_classes=0, dynamic_img_size=True,
        ).to(device, dtype).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        data_cfg = timm.data.resolve_data_config({}, model=self.model)
        if size is not None:
            data_cfg["input_size"] = (3, size, size)
            data_cfg["crop_pct"] = 1.0
        self.transform = timm.data.create_transform(**data_cfg)
        self.out_dim = int(getattr(self.model, "num_features", 1024))
        self.size = data_cfg["input_size"][-1]

    @torch.inference_mode()
    def features(self, images: list[Image.Image]) -> torch.Tensor:
        batch = torch.stack([self.transform(im) for im in images]).to(self.device, self.dtype)
        return self.model(batch).float()


class VaeTeacher:
    """Frozen SDXL VAE; emits the scaled posterior mean used as a diffusion target.

    Always runs in fp32.  SDXL's VAE is notorious for overflowing under fp16
    (`force_upcast=True` in the official config); encoding in half precision
    silently fills the latent with NaNs, which then poison every loss that
    touches `vae_latent`.  The cost of fp32 here is a few minutes over 16k
    images -- cheap compared to training on garbage.
    """

    def __init__(self, device: str, dtype: torch.dtype):
        from diffusers import AutoencoderKL

        self.device = device
        # Ignore the caller's dtype: VAE encode must be fp32 (see docstring).
        self.dtype = torch.float32
        self.vae = AutoencoderKL.from_pretrained(
            paths.VAE_ID, torch_dtype=torch.float32,
        ).to(device).eval()
        # Belt-and-braces: even if a future diffusers version loads a fp16
        # variant, keep the encode path in full precision.
        if hasattr(self.vae.config, "force_upcast"):
            self.vae.config.force_upcast = True
        for p in self.vae.parameters():
            p.requires_grad_(False)
        self.scale = float(getattr(self.vae.config, "scaling_factor", SDXL_VAE_SCALE))

    @torch.inference_mode()
    def latents(self, tensors: torch.Tensor) -> torch.Tensor:
        """`tensors` in [-1, 1], shape (B, 3, R, R)."""
        posterior = self.vae.encode(tensors.to(self.device, self.dtype)).latent_dist
        out = (posterior.mean * self.scale).float()
        if not torch.isfinite(out).all():
            n_bad = int((~torch.isfinite(out)).any(dim=(1, 2, 3)).sum())
            raise RuntimeError(
                f"VAE produced non-finite latents in {n_bad}/{out.shape[0]} rows; "
                f"refusing to write them.  This usually means the VAE ran in fp16."
            )
        return out


# --- image loading -----------------------------------------------------------
def load_images(records: Sequence[things.ImageRecord], resolution: int) -> torch.Tensor:
    """Load records as an (N, 3, R, R) tensor in [-1, 1]."""
    out = torch.empty(len(records), 3, resolution, resolution, dtype=torch.float32)
    for i, rec in enumerate(records):
        with Image.open(rec.path) as im:
            im = im.convert("RGB").resize((resolution, resolution), Image.BICUBIC)
            arr = torch.from_numpy(np.asarray(im, dtype=np.uint8).copy())
        out[i] = arr.permute(2, 0, 1).float().div_(127.5).sub_(1.0)
    return out


def batched(items: Sequence, size: int):
    for start in range(0, len(items), size):
        yield start, items[start:start + size]


# --- per-target drivers ------------------------------------------------------
def run_clip_image(writer, records, device, dtype, batch_size, resolution):
    teacher = ClipTeacher(device, dtype)
    for start, chunk in batched(records, batch_size):
        rows = [r.index for r in chunk]
        if all(r in writer.done for r in rows):
            continue
        images = []
        keep = []
        for rec in chunk:
            try:
                with Image.open(rec.path) as im:
                    images.append(im.convert("RGB"))
                    keep.append(rec)
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] unreadable {rec.path}: {exc}", file=sys.stderr, flush=True)
        if not images:
            continue
        feats = teacher.image_features(images).cpu().numpy().astype(np.float16)
        writer.write([r.index for r in keep], feats)
        writer.flush()
    del teacher
    gc.collect()
    torch.cuda.empty_cache()


def run_clip_text_caption(writer, records, device, dtype, batch_size, resolution,
                          captions: dict[str, str]):
    teacher = ClipTeacher(device, dtype)
    for start, chunk in batched(records, batch_size):
        rows = [r.index for r in chunk]
        if all(r in writer.done for r in rows):
            continue
        keep = [r for r in chunk if r.image_key in captions]
        if not keep:
            continue
        texts = [captions[r.image_key] for r in keep]
        feats = teacher.text_features(texts).cpu().numpy().astype(np.float16)
        writer.write([r.index for r in keep], feats)
        writer.flush()
    del teacher
    gc.collect()
    torch.cuda.empty_cache()


def run_clip_text_concept(writer, concepts, device, dtype, batch_size, resolution,
                          prompts: dict[str, list[str]]):
    """One row per concept: mean of the L2-normalised template embeddings."""
    teacher = ClipTeacher(device, dtype)
    todo = [i for i in range(len(concepts)) if i not in writer.done]
    for start, chunk_idx in batched(todo, batch_size):
        flat: list[str] = []
        owner: list[int] = []
        for i in chunk_idx:
            for p in prompts[concepts[i]]:
                flat.append(p)
                owner.append(i)
        embs = teacher.text_features(flat)
        embs = torch.nn.functional.normalize(embs, dim=-1)
        agg = torch.zeros(len(chunk_idx), embs.shape[-1])
        for j, i in enumerate(chunk_idx):
            mask = torch.tensor([o == i for o in owner])
            agg[j] = torch.nn.functional.normalize(embs[mask].mean(0), dim=-1)
        writer.write(chunk_idx, agg.numpy().astype(np.float16))
        writer.flush()
    del teacher
    gc.collect()
    torch.cuda.empty_cache()


def run_dino(writer, records, device, dtype, batch_size, resolution, dino_size):
    teacher = DinoTeacher(device, dtype, size=dino_size)
    print(f"[dino] input size {teacher.size}, out_dim {teacher.out_dim}", flush=True)
    for start, chunk in batched(records, batch_size):
        rows = [r.index for r in chunk]
        if all(r in writer.done for r in rows):
            continue
        images, keep = [], []
        for rec in chunk:
            try:
                with Image.open(rec.path) as im:
                    images.append(im.convert("RGB"))
                    keep.append(rec)
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] unreadable {rec.path}: {exc}", file=sys.stderr, flush=True)
        if not images:
            continue
        feats = teacher.features(images).cpu().numpy().astype(np.float16)
        writer.write([r.index for r in keep], feats)
        writer.flush()
    del teacher
    gc.collect()
    torch.cuda.empty_cache()


def run_vae_latent(writer, records, device, dtype, batch_size, resolution):
    teacher = VaeTeacher(device, dtype)
    print(f"[vae] scaling factor {teacher.scale} (encode dtype={teacher.dtype})",
          flush=True)
    for start, chunk in batched(records, batch_size):
        rows = [r.index for r in chunk]
        if all(r in writer.done for r in rows):
            continue
        px = load_images(chunk, SD_RESOLUTION)
        lat = teacher.latents(px).cpu().numpy().astype(np.float16)
        writer.write(rows, lat)
        writer.flush()
        if start % (batch_size * 20) == 0:
            print(f"[vae] {writer.n_done}/{writer.n_rows}", flush=True)
    del teacher
    gc.collect()
    torch.cuda.empty_cache()


def run_pixels(writer, records, device, dtype, batch_size, resolution):
    """uint8 512x512 renders, kept for pixel-space metrics (test split only)."""
    for start, chunk in batched(records, batch_size):
        rows = [r.index for r in chunk]
        if all(r in writer.done for r in rows):
            continue
        px = load_images(chunk, SD_RESOLUTION)                      # (N,3,R,R) in [-1,1]
        u8 = px.add(1).mul(127.5).clamp(0, 255).round().to(torch.uint8)
        writer.write(rows, u8.permute(0, 2, 3, 1).numpy())           # (N,R,R,3)
        writer.flush()


# --- registry ----------------------------------------------------------------
TARGETS: dict[str, dict] = {
    "clip_image": {
        "row_shape": (1024,), "dtype": np.float16, "splits": ("train", "test"),
        "fn": run_clip_image, "desc": "laion CLIP ViT-H-14 image embedding",
    },
    "clip_text_caption": {
        "row_shape": (1024,), "dtype": np.float16, "splits": ("train", "test"),
        "fn": run_clip_text_caption, "desc": "CLIP text embedding of the BLIP2 caption",
    },
    "clip_text_concept": {
        "row_shape": (1024,), "dtype": np.float16, "splits": ("train", "test"),
        "fn": run_clip_text_concept, "desc": "CLIP text embedding of concept prompts (per concept)",
        "per_concept": True,
    },
    "dino": {
        "row_shape": (1024,), "dtype": np.float16, "splits": ("train", "test"),
        "fn": run_dino, "desc": "DINOv2-L global feature",
    },
    "vae_latent": {
        "row_shape": (4, SD_LATENT_RES, SD_LATENT_RES), "dtype": np.float16,
        "splits": ("train", "test"), "fn": run_vae_latent,
        "desc": "SDXL VAE posterior mean, scaled",
    },
    "pixels": {
        "row_shape": (SD_RESOLUTION, SD_RESOLUTION, 3), "dtype": np.uint8,
        "splits": ("test",), "fn": run_pixels,
        "desc": "512x512 uint8 render, for pixel metrics",
    },
}


def load_captions(split: str) -> dict[str, str]:
    """Merge shard files into one {image_key: caption}, prompt-stripped and checked.

    Delegates to `loso.data.captions` so the generation and consumption sides share
    one stripping rule; see that module for why a prompt left in the caption text is
    a silent corruption rather than a cosmetic issue.
    """
    records = captions_mod.load_caption_records(split)
    ok, msg = captions_mod.caption_quality_report(records, split)
    print(f"[captions] {'OK' if ok else 'FAIL'} -- {msg}", flush=True)
    if not ok:
        raise SystemExit(f"[FATAL] caption store for {split} failed quality checks")
    return {k: v["caption"] for k, v in records.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--targets", nargs="+", default=list(TARGETS),
                    choices=list(TARGETS), help="which teachers to extract")
    ap.add_argument("--splits", nargs="+", default=["train", "test"],
                    choices=["train", "test"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--dino-size", type=int, default=None,
                    help="override DINOv2 input resolution (default: the model's native size)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--limit", type=int, default=0, help="debug: cap rows per target")
    args = ap.parse_args()

    paths.ensure_dirs()
    device = args.device
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    manifest: dict = {"config": {
        "clip": paths.CLIP_ID, "dino": paths.DINO_ID, "vae": paths.VAE_ID,
        "blip2": paths.BLIP2_ID, "sd_resolution": SD_RESOLUTION,
        "vae_scale": SDXL_VAE_SCALE, "dtype": args.dtype,
    }}

    for split in args.splits:
        records = things.build_index(split)
        if args.limit:
            records = records[:args.limit]
        concepts = things.concept_records(records)
        concept_prompts = json.loads(
            (paths.THINGS_DIR / f"{split}_concepts.json").read_text()
        )
        captions = load_captions(split) if "clip_text_caption" in args.targets else {}
        if "clip_text_caption" in args.targets:
            missing = [r.image_key for r in records if r.image_key not in captions]
            if missing:
                print(f"[FATAL] {len(missing)} images in {split} have no BLIP2 caption "
                      f"(e.g. {missing[:3]}).  Run scripts/gen_captions.py first.",
                      file=sys.stderr)
                return 2

        for name in args.targets:
            spec = TARGETS[name]
            if split not in spec["splits"]:
                continue
            per_concept = spec.get("per_concept", False)
            n_rows = len(concepts) if per_concept else len(records)
            out = paths.TARGET_DIR / f"{name}_{split}.npy"
            writer = TargetWriter(out, n_rows, spec["row_shape"], np.dtype(spec["dtype"]))
            if writer.n_done >= n_rows:
                print(f"[skip] {name}/{split}: already complete ({n_rows} rows)", flush=True)
                manifest[f"{name}_{split}"] = {"shape": [n_rows, *spec["row_shape"]],
                                               "complete": True}
                continue

            print(f"[{name}/{split}] {writer.n_done}/{n_rows} done -> {out}", flush=True)
            t0 = time.time()
            fn: Callable = spec["fn"]
            if per_concept:
                fn(writer, concepts, device, dtype, args.batch_size,
                   SD_RESOLUTION, concept_prompts)
            elif name == "clip_text_caption":
                fn(writer, records, device, dtype, args.batch_size,
                   SD_RESOLUTION, captions)
            elif name == "dino":
                fn(writer, records, device, dtype, args.batch_size,
                   SD_RESOLUTION, args.dino_size)
            else:
                fn(writer, records, device, dtype, args.batch_size, SD_RESOLUTION)
            writer.flush()
            print(f"[{name}/{split}] finished {writer.n_done}/{n_rows} "
                  f"in {(time.time() - t0) / 60:.1f} min", flush=True)
            manifest[f"{name}_{split}"] = {
                "shape": [n_rows, *spec["row_shape"]],
                "dtype": np.dtype(spec["dtype"]).name, "desc": spec["desc"],
                "complete": writer.n_done >= n_rows,
            }

    manifest_path = paths.TARGET_DIR / "manifest.json"
    existing = {}
    if manifest_path.is_file():
        try:
            existing = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            existing = {}
    existing.update(manifest)
    manifest_path.write_text(json.dumps(existing, indent=1))
    print(f"[manifest] -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
