#!/usr/bin/env python3
"""Low-level decoder experiments: fuse / img2img from EEG→VAE blurry RGB.

Modes:
  fuse     — MindEye2-style weighted blend of semantic PNG + low-level PNG
  img2img  — SDXL img2img init=low-level, IP=semantic embeds, optional text
"""

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
    list_test_images,
    resolve_ip_adapter_dir,
    resolve_sdxl_model_path,
)


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, required=True, choices=["fuse", "img2img"])
    ap.add_argument("--lowlevel-dir", type=str, required=True, help="{i:03d}.png EEG-VAE decode")
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--tag", type=str, default="lowlevel")
    ap.add_argument("--semantic-dir", type=str, default="", help="for fuse: existing semantic generated/")
    ap.add_argument("--fuse-alpha", type=float, default=0.80, help="weight on semantic (MindEye2 ~0.8)")
    ap.add_argument("--embed-npy", type=str, default="", help="for img2img IP embeds")
    ap.add_argument("--prompts-json", type=str, default="")
    ap.add_argument("--strength", type=float, default=0.55)
    ap.add_argument(
        "--strength-npy",
        type=str,
        default="",
        help="optional per-sample img2img strength (N,); overrides --strength when set",
    )
    ap.add_argument("--ip-scale", type=float, default=1.0)
    ap.add_argument(
        "--ip-scale-npy",
        type=str,
        default="",
        help="optional per-sample IP scale (N,)",
    )
    ap.add_argument("--gen-steps", type=int, default=30)
    ap.add_argument("--gen-guidance", type=float, default=5.0)
    ap.add_argument("--gen-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--negative-prompt", type=str, default="blurry, low quality, distorted, watermark")
    ap.add_argument("--lora-dir", type=str, default="", help="diffusers LoRA weights dir")
    ap.add_argument("--lora-scale", type=float, default=1.0)
    ap.add_argument("--skip-metrics", action="store_true")
    ap.add_argument("--images-root", type=str, default="/project/peilab/why/data/images_set")
    args = ap.parse_args()

    cache = Path(os.environ.get("HF_HUB_CACHE", "/project/peilab/why/cache/eeg-brainit/hf/hub"))
    os.environ.setdefault("HF_HOME", str(cache.parent))
    os.environ.setdefault("OPENCLIP_CACHE_DIR", str(cache.parent / "open_clip"))
    os.environ.setdefault("TORCH_HOME", str(cache.parent / "torch"))

    out = Path(args.output_dir)
    gen_dir = out / "generated"
    gen_dir.mkdir(parents=True, exist_ok=True)
    ll_dir = Path(args.lowlevel_dir)
    ll_paths = sorted(ll_dir.glob("*.png"))
    n = len(ll_paths) if args.max_images <= 0 else min(len(ll_paths), args.max_images)

    if args.mode == "fuse":
        sem_dir = Path(args.semantic_dir)
        if not sem_dir.is_dir():
            raise FileNotFoundError(sem_dir)
        for i in tqdm(range(n), desc="fuse"):
            dst = gen_dir / f"{i:03d}.png"
            if dst.is_file():
                continue
            ll = Image.open(ll_dir / f"{i:03d}.png").convert("RGB").resize(
                (args.gen_size, args.gen_size), Image.Resampling.BICUBIC
            )
            sem = Image.open(sem_dir / f"{i:03d}.png").convert("RGB").resize(
                (args.gen_size, args.gen_size), Image.Resampling.BICUBIC
            )
            # Image.blend(im1, im2, alpha) = im1*(1-a)+im2*a → put semantic as im2
            Image.blend(ll, sem, float(args.fuse_alpha)).save(dst)
    else:
        if not args.embed_npy:
            raise ValueError("--embed-npy required for img2img")
        embeds = l2(np.load(args.embed_npy).astype(np.float32))
        prompts = None
        if args.prompts_json:
            prompts = json.loads(Path(args.prompts_json).read_text(encoding="utf-8"))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not all((gen_dir / f"{i:03d}.png").is_file() for i in range(n)):
            from diffusers import StableDiffusionXLImg2ImgPipeline

            sdxl = resolve_sdxl_model_path(cache)
            dtype = torch.float16 if device.type == "cuda" else torch.float32
            pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                sdxl,
                torch_dtype=dtype,
                variant="fp16" if device.type == "cuda" else None,
                use_safetensors=True,
                local_files_only=Path(str(sdxl)).is_dir(),
            ).to(device)
            ip_root = resolve_ip_adapter_dir(cache)
            try:
                pipe.load_ip_adapter(
                    str(ip_root),
                    weight_name="ip-adapter_sdxl_vit-h.bin",
                    subfolder="sdxl_models",
                    image_encoder_folder=None,
                    local_files_only=True,
                )
            except Exception:
                pipe.load_ip_adapter(
                    str(ip_root),
                    weight_name="ip-adapter_sdxl.bin",
                    subfolder="sdxl_models",
                    image_encoder_folder=None,
                    local_files_only=True,
                )
            if args.lora_dir:
                pipe.load_lora_weights(args.lora_dir)
                pipe.fuse_lora(lora_scale=float(args.lora_scale))
                print(f"[INFO] loaded LoRA {args.lora_dir} scale={args.lora_scale}")
            strengths = None
            if args.strength_npy:
                strengths = np.load(args.strength_npy).astype(np.float32).reshape(-1)
                if len(strengths) < n:
                    raise ValueError(f"strength-npy len {len(strengths)} < n={n}")
            ip_scales = None
            if args.ip_scale_npy:
                ip_scales = np.load(args.ip_scale_npy).astype(np.float32).reshape(-1)
                if len(ip_scales) < n:
                    raise ValueError(f"ip-scale-npy len {len(ip_scales)} < n={n}")
            g = torch.Generator(device=device).manual_seed(args.seed)
            used_s = []
            for i in tqdm(range(n), desc="img2img-lowlevel"):
                path = gen_dir / f"{i:03d}.png"
                s_i = float(strengths[i]) if strengths is not None else float(args.strength)
                s_i = float(np.clip(s_i, 0.05, 0.95))
                used_s.append(s_i)
                ip_i = float(ip_scales[i]) if ip_scales is not None else float(args.ip_scale)
                pipe.set_ip_adapter_scale(ip_i)
                if path.is_file():
                    continue
                init = Image.open(ll_dir / f"{i:03d}.png").convert("RGB").resize(
                    (args.gen_size, args.gen_size), Image.Resampling.BICUBIC
                )
                emb = torch.from_numpy(embeds[i : i + 1]).to(device=device, dtype=pipe.dtype)
                if args.gen_guidance > 1.0:
                    ip_emb = torch.cat([torch.zeros_like(emb), emb], dim=0).unsqueeze(1)
                else:
                    ip_emb = emb.unsqueeze(1)
                prompt = prompts[i] if prompts is not None else ""
                neg = args.negative_prompt if prompt else ""
                img = pipe(
                    prompt=prompt or "",
                    negative_prompt=neg,
                    image=init,
                    strength=s_i,
                    ip_adapter_image_embeds=[ip_emb],
                    num_inference_steps=int(args.gen_steps),
                    guidance_scale=float(args.gen_guidance),
                    generator=g,
                ).images[0]
                img.save(path)
            if used_s:
                np.save(out / "strength_used.npy", np.asarray(used_s, dtype=np.float32))
            del pipe
            if device.type == "cuda":
                torch.cuda.empty_cache()

    report = {
        "tag": args.tag,
        "mode": args.mode,
        "lowlevel_dir": args.lowlevel_dir,
        "semantic_dir": args.semantic_dir or None,
        "fuse_alpha": args.fuse_alpha if args.mode == "fuse" else None,
        "strength": args.strength if args.mode == "img2img" else None,
        "strength_npy": args.strength_npy or None,
        "ip_scale_npy": args.ip_scale_npy or None,
        "lora_dir": args.lora_dir or None,
        "lora_scale": args.lora_scale if args.lora_dir else None,
        "n_gen": n,
    }
    if not args.skip_metrics:
        from eval_atm_pipeline import image_metrics  # type: ignore

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        paths = [gen_dir / f"{i:03d}.png" for i in range(n)]
        gt = list_test_images(Path(args.images_root))
        report["metrics"] = image_metrics(paths, gt[:n], device)
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
