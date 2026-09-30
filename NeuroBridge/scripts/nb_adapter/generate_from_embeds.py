#!/usr/bin/env python3
"""SDXL (+ IP-Adapter ViT-H) generation from CLIP-1024 embeds.

Prefers local SDXL-base; falls back to sdxl-turbo (guidance=0, few steps).
Reuses path resolvers / metrics from eeg-brainit when available.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

BRAINIT = Path("/project/peilab/why/eeg-brainit")
sys.path.insert(0, str(BRAINIT / "scripts"))
sys.path.insert(0, str(BRAINIT / "src"))

from eval_atm_pipeline import (  # type: ignore
    image_metrics,
    list_test_images,
    resolve_ip_adapter_dir,
    resolve_sdxl_model_path,
)


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-8)


def resolve_model(hub: Path) -> tuple[str, dict]:
    """Return (model_path_or_id, gen_defaults)."""
    try:
        path = resolve_sdxl_model_path(hub)
        if path.endswith("stable-diffusion-xl-base-1.0") or "stable-diffusion-xl-base-1.0" in path:
            # verify snapshot actually exists locally when offline
            if Path(path).is_dir() and (Path(path) / "model_index.json").is_file():
                return path, {"steps": 30, "guidance": 5.0, "variant": "fp16"}
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] resolve base failed: {exc}")

    turbo_root = hub / "models--stabilityai--sdxl-turbo" / "snapshots"
    if turbo_root.is_dir():
        for snap in sorted(turbo_root.iterdir(), reverse=True):
            if (snap / "model_index.json").is_file():
                print(f"[INFO] fallback SDXL-Turbo: {snap}")
                return str(snap), {"steps": 4, "guidance": 0.0, "variant": "fp16"}
    # last resort: hub id (needs network)
    return "stabilityai/sdxl-turbo", {"steps": 4, "guidance": 0.0, "variant": "fp16"}


def generate(
    embeds: np.ndarray,
    out_dir: Path,
    device: torch.device,
    steps: int,
    guidance: float,
    size: int,
    max_images: int,
    seed: int,
    ip_scale: float,
) -> list[Path]:
    from diffusers import StableDiffusionXLPipeline

    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(embeds) if max_images <= 0 else min(len(embeds), max_images)
    existing = [out_dir / f"{i:03d}.png" for i in range(n)]
    if all(p.is_file() for p in existing):
        print(f"[INFO] skip gen: {n} images already in {out_dir}")
        return existing

    hub = Path(os.environ.get("HF_HUB_CACHE", "/project/peilab/why/cache/eeg-brainit/hf/hub"))
    model_id, defaults = resolve_model(hub)
    if steps <= 0:
        steps = int(defaults["steps"])
    if guidance < 0:
        guidance = float(defaults["guidance"])

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    local_only = Path(model_id).is_dir()
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        variant=defaults.get("variant"),
        use_safetensors=True,
        local_files_only=local_only,
    ).to(device)

    ip_root = resolve_ip_adapter_dir(hub)
    if ip_root is None:
        raise FileNotFoundError("IP-Adapter missing in HF cache")
    weight_name = "ip-adapter_sdxl_vit-h.bin"
    load_kwargs = {
        "subfolder": "sdxl_models",
        "image_encoder_folder": None,
        "local_files_only": True,
    }
    try:
        pipe.load_ip_adapter(str(ip_root), weight_name=weight_name, **load_kwargs)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] vit-h load failed ({exc}); fallback ip-adapter_sdxl.bin")
        weight_name = "ip-adapter_sdxl.bin"
        pipe.load_ip_adapter(str(ip_root), weight_name=weight_name, **load_kwargs)

    pipe.set_ip_adapter_scale(float(ip_scale))
    g = torch.Generator(device=device).manual_seed(seed)
    paths: list[Path] = []
    for i in tqdm(range(n), desc="gen"):
        path = out_dir / f"{i:03d}.png"
        if path.is_file():
            paths.append(path)
            continue
        emb = torch.from_numpy(embeds[i : i + 1]).to(device=device, dtype=pipe.dtype)
        uncond = torch.zeros_like(emb)
        image_embeds = torch.cat([uncond, emb], dim=0).unsqueeze(1)
        result = pipe(
            prompt="",
            negative_prompt="",
            ip_adapter_image_embeds=[image_embeds],
            num_inference_steps=steps,
            guidance_scale=guidance,
            height=size,
            width=size,
            generator=g,
        )
        result.images[0].save(path)
        paths.append(path)

    meta = {
        "model": model_id,
        "weight_name": weight_name,
        "steps": steps,
        "guidance": guidance,
        "ip_scale": ip_scale,
        "n": n,
        "size": size,
        "seed": seed,
    }
    (out_dir / "gen_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    del pipe
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-npy", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--images-root", type=str, default="/project/peilab/why/data/images_set")
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--gen-steps", type=int, default=0, help="0 => auto by model")
    ap.add_argument("--gen-guidance", type=float, default=-1.0, help="<0 => auto by model")
    ap.add_argument("--gen-size", type=int, default=512)
    ap.add_argument("--ip-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-metrics", action="store_true")
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    cache = Path("/project/peilab/why/cache/eeg-brainit")
    os.environ.setdefault("HF_HOME", str(cache / "hf"))
    os.environ.setdefault("HF_HUB_CACHE", str(cache / "hf" / "hub"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache / "hf" / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache / "hf" / "hub"))
    os.environ.setdefault("DIFFUSERS_CACHE", str(cache / "hf" / "hub"))
    os.environ.setdefault("TORCH_HOME", str(cache / "torch"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embeds = l2(np.load(args.embed_npy).astype(np.float32))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gen_dir = out_dir / "generated"

    paths = generate(
        embeds,
        gen_dir,
        device=device,
        steps=args.gen_steps,
        guidance=args.gen_guidance,
        size=args.gen_size,
        max_images=args.max_images,
        seed=args.seed,
        ip_scale=args.ip_scale,
    )

    report: dict = {"tag": args.tag, "embed_npy": args.embed_npy, "n_gen": len(paths)}
    if not args.skip_metrics:
        gt = list_test_images(Path(args.images_root))
        n = len(paths)
        metrics = image_metrics(paths, gt[:n], device)
        report["metrics"] = metrics
        print(
            f"[METRICS] {args.tag or Path(args.embed_npy).stem}: "
            f"SSIM={metrics['ssim']:.4f} CLIP={metrics['clip_cosine']:.4f}"
        )
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[OK] {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
