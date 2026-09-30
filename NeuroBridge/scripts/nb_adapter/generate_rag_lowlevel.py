#!/usr/bin/env python3
"""Phase 3: RAG neighbor img2img + IP-Adapter ViT-H semantic conditioning."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
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


def list_train_images(images_root: Path) -> list[Path]:
    root = images_root / "training_images"
    paths: list[Path] = []
    for d in sorted([p for p in root.iterdir() if p.is_dir()]):
        imgs = sorted(list(d.glob("*.jpg")) + list(d.glob("*.JPEG")) + list(d.glob("*.png")))
        paths.extend(imgs)
    return paths


def resolve_model(hub: Path) -> tuple[str, dict]:
    try:
        path = resolve_sdxl_model_path(hub)
        if Path(path).is_dir() and (Path(path) / "model_index.json").is_file():
            return path, {"steps": 30, "guidance": 5.0, "variant": "fp16"}
    except Exception:
        pass
    turbo_root = hub / "models--stabilityai--sdxl-turbo" / "snapshots"
    if turbo_root.is_dir():
        for snap in sorted(turbo_root.iterdir(), reverse=True):
            if (snap / "model_index.json").is_file():
                return str(snap), {"steps": 4, "guidance": 0.0, "variant": "fp16"}
    return "stabilityai/sdxl-turbo", {"steps": 4, "guidance": 0.0, "variant": "fp16"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-npy", type=str, required=True)
    ap.add_argument("--neighbor-idx-npy", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--images-root", type=str, default="/project/peilab/why/data/images_set")
    ap.add_argument("--strength", type=float, default=0.5)
    ap.add_argument("--strength-npy", type=str, default="", help="per-sample img2img strength (n,)")
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--gen-steps", type=int, default=0)
    ap.add_argument("--gen-guidance", type=float, default=-1.0)
    ap.add_argument("--ip-scale", type=float, default=1.0)
    ap.add_argument("--ip-scale-npy", type=str, default="", help="per-sample IP-Adapter scale (n,)")
    ap.add_argument("--prompts-json", type=str, default="", help="list[str] prompts, one per sample")
    ap.add_argument("--prompt-scale", type=float, default=1.0, help="unused marker; prompts go to SDXL text encoder")
    ap.add_argument("--negative-prompt", type=str, default="blurry, low quality, distorted, watermark")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", type=str, default="rag_lowlevel")
    ap.add_argument("--skip-metrics", action="store_true")
    args = ap.parse_args()

    cache = Path(os.environ.get("HF_HUB_CACHE", "/project/peilab/why/cache/eeg-brainit/hf/hub"))
    os.environ.setdefault("HF_HOME", str(cache.parent))
    os.environ.setdefault("OPENCLIP_CACHE_DIR", str(cache.parent / "open_clip"))
    os.environ.setdefault("TORCH_HOME", str(cache.parent / "torch"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embeds = l2(np.load(args.embed_npy).astype(np.float32))
    strength_arr = None
    if args.strength_npy:
        strength_arr = np.load(args.strength_npy).astype(np.float32).reshape(-1)
    ip_scale_arr = None
    if args.ip_scale_npy:
        ip_scale_arr = np.load(args.ip_scale_npy).astype(np.float32).reshape(-1)
    neighbors = np.load(args.neighbor_idx_npy)
    train_paths = list_train_images(Path(args.images_root))
    prompts = None
    if args.prompts_json:
        prompts = json.loads(Path(args.prompts_json).read_text(encoding="utf-8"))
        if len(prompts) < len(embeds):
            raise ValueError(f"prompts {len(prompts)} < embeds {len(embeds)}")

    out_dir = Path(args.output_dir)
    gen_dir = out_dir / "generated"
    gen_dir.mkdir(parents=True, exist_ok=True)

    n = len(embeds) if args.max_images <= 0 else min(len(embeds), args.max_images)
    if all((gen_dir / f"{i:03d}.png").is_file() for i in range(n)):
        print(f"[INFO] skip gen, {n} images exist")
    else:
        from diffusers import StableDiffusionXLImg2ImgPipeline

        model_id, defaults = resolve_model(cache)
        steps = args.gen_steps if args.gen_steps > 0 else int(defaults["steps"])
        # with text prompts, prefer non-zero guidance if model supports it
        if args.gen_guidance >= 0:
            guidance = float(args.gen_guidance)
        elif prompts is not None and "turbo" not in str(model_id).lower():
            guidance = 5.0
        elif prompts is not None:
            guidance = max(float(defaults["guidance"]), 1.5)
        else:
            guidance = float(defaults["guidance"])
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            variant=defaults.get("variant"),
            use_safetensors=True,
            local_files_only=Path(model_id).is_dir(),
        ).to(device)

        ip_root = resolve_ip_adapter_dir(cache)
        weight_name = "ip-adapter_sdxl_vit-h.bin"
        load_kwargs = {"subfolder": "sdxl_models", "image_encoder_folder": None, "local_files_only": True}
        try:
            pipe.load_ip_adapter(str(ip_root), weight_name=weight_name, **load_kwargs)
        except Exception:
            weight_name = "ip-adapter_sdxl.bin"
            pipe.load_ip_adapter(str(ip_root), weight_name=weight_name, **load_kwargs)
        pipe.set_ip_adapter_scale(float(args.ip_scale))

        g = torch.Generator(device=device).manual_seed(args.seed)
        for i in tqdm(range(n), desc="rag-i2i"):
            path = gen_dir / f"{i:03d}.png"
            if path.is_file():
                continue
            nb = int(neighbors[i, 0] if neighbors.ndim > 1 else neighbors[i])
            init = Image.open(train_paths[nb]).convert("RGB").resize((512, 512), Image.Resampling.BICUBIC)
            emb = torch.from_numpy(embeds[i : i + 1]).to(device=device, dtype=pipe.dtype)
            if guidance > 1.0:
                uncond = torch.zeros_like(emb)
                ip_emb = torch.cat([uncond, emb], dim=0).unsqueeze(1)
            else:
                ip_emb = emb.unsqueeze(1)
            strength = float(strength_arr[i]) if strength_arr is not None else float(args.strength)
            if ip_scale_arr is not None:
                pipe.set_ip_adapter_scale(float(ip_scale_arr[i]))
            prompt = prompts[i] if prompts is not None else ""
            neg = args.negative_prompt if prompts is not None else ""
            result = pipe(
                prompt=prompt,
                negative_prompt=neg,
                image=init,
                strength=strength,
                ip_adapter_image_embeds=[ip_emb],
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=g,
            )
            result.images[0].save(path)
        del pipe
        if device.type == "cuda":
            torch.cuda.empty_cache()

    paths = sorted(gen_dir.glob("*.png"))[:n]
    report: dict = {
        "tag": args.tag,
        "embed_npy": args.embed_npy,
        "strength": args.strength,
        "strength_npy": args.strength_npy or None,
        "ip_scale": args.ip_scale,
        "ip_scale_npy": args.ip_scale_npy or None,
        "prompts_json": args.prompts_json or None,
        "n_gen": len(paths),
    }
    if not args.skip_metrics:
        gt = list_test_images(Path(args.images_root))
        metrics = image_metrics(paths, gt[:len(paths)], device)
        report["metrics"] = metrics
        print(f"[METRICS] CLIP={metrics['clip_cosine']:.4f} SSIM={metrics['ssim']:.4f}")
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[OK] {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
