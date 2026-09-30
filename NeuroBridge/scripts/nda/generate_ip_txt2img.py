#!/usr/bin/env python3
"""SDXL + IP-Adapter (+ optional LoRA) from noise — semantic baseline / Phase-B check."""

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

from eval_atm_pipeline import resolve_ip_adapter_dir, resolve_sdxl_model_path  # type: ignore


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-npy", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--tag", type=str, default="ip_txt2img")
    ap.add_argument("--prompts-json", type=str, default="")
    ap.add_argument("--ip-scale", type=float, default=1.0)
    ap.add_argument("--lora-dir", type=str, default="")
    ap.add_argument("--lora-scale", type=float, default=1.0)
    ap.add_argument("--gen-steps", type=int, default=28)
    ap.add_argument("--gen-guidance", type=float, default=5.0)
    ap.add_argument("--gen-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--negative-prompt", type=str, default="blurry, low quality, distorted, watermark")
    args = ap.parse_args()

    cache = Path(os.environ.get("HF_HUB_CACHE", "/project/peilab/why/cache/eeg-brainit/hf/hub"))
    os.environ.setdefault("HF_HOME", str(cache.parent))
    os.environ.setdefault("OPENCLIP_CACHE_DIR", str(cache.parent / "open_clip"))
    os.environ.setdefault("TORCH_HOME", str(cache.parent / "torch"))
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)

    out = Path(args.output_dir)
    gen_dir = out / "generated"
    gen_dir.mkdir(parents=True, exist_ok=True)
    embeds = l2(np.load(args.embed_npy).astype(np.float32))
    prompts = json.loads(Path(args.prompts_json).read_text(encoding="utf-8")) if args.prompts_json else None
    n = len(embeds) if args.max_images <= 0 else min(len(embeds), args.max_images)

    if all((gen_dir / f"{i:03d}.png").is_file() for i in range(n)):
        print(f"[SKIP] n={n}")
    else:
        from diffusers import StableDiffusionXLPipeline

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        sdxl = resolve_sdxl_model_path(cache)
        pipe = StableDiffusionXLPipeline.from_pretrained(
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
        pipe.set_ip_adapter_scale(float(args.ip_scale))
        if args.lora_dir:
            pipe.load_lora_weights(args.lora_dir)
            pipe.fuse_lora(lora_scale=float(args.lora_scale))
            print(f"[INFO] LoRA {args.lora_dir} scale={args.lora_scale}")

        g = torch.Generator(device=device).manual_seed(args.seed)
        for i in tqdm(range(n), desc="ip-txt2img"):
            path = gen_dir / f"{i:03d}.png"
            if path.is_file():
                continue
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
                ip_adapter_image_embeds=[ip_emb],
                num_inference_steps=int(args.gen_steps),
                guidance_scale=float(args.gen_guidance),
                height=args.gen_size,
                width=args.gen_size,
                generator=g,
            ).images[0]
            img.save(path)
        del pipe
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report = {
        "tag": args.tag,
        "lora_dir": args.lora_dir or None,
        "lora_scale": args.lora_scale if args.lora_dir else None,
        "n_gen": n,
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
