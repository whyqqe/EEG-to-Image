#!/usr/bin/env python3
"""Structure-first injection decode (ATM/CogCap style).

Modes:
  txt2img  — from noise: Depth/Canny ControlNet (timed) + IP + optional text
  img2img  — ATM-style low-level init + timed ControlNet + IP + optional text

Key knobs vs prior failed full-strength CN / post-hoc luma:
  --control-guidance-start/end  (CN only early denoising; IP owns late semantics)
  --strength                    (latent init mix via SDEdit; mild band)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

BRAINIT = Path("/project/peilab/why/eeg-brainit")
sys.path.insert(0, str(BRAINIT / "scripts"))
sys.path.insert(0, str(BRAINIT / "src"))

from eval_atm_pipeline import resolve_ip_adapter_dir, resolve_sdxl_model_path  # type: ignore


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-8)


def resolve_controlnet_path(hub: Path, control_type: str) -> str:
    name = "controlnet-canny-sdxl-1.0" if control_type == "canny" else "controlnet-depth-sdxl-1.0"
    root = hub / f"models--diffusers--{name}" / "snapshots"
    if root.is_dir():
        for snap in sorted(root.iterdir(), reverse=True):
            if (snap / "config.json").is_file():
                return str(snap)
    return f"diffusers/{name}"


def rgb_to_canny(path: Path, size: int, low: int = 80, high: int = 160) -> Image.Image:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low, high)
    return Image.fromarray(cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB))


def load_cond(path: Path, size: int, control_type: str) -> Image.Image:
    if control_type == "canny":
        return rgb_to_canny(path, size)
    return Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BICUBIC)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, required=True, choices=["txt2img", "img2img"])
    ap.add_argument("--embed-npy", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--tag", type=str, default="struct_inject")
    ap.add_argument("--prompts-json", type=str, default="")
    ap.add_argument("--control-type", type=str, default="depth", choices=["depth", "canny"])
    ap.add_argument("--cond-dir", type=str, required=True, help="per-sample cond RGB {i:03d}.png (depth or source for canny)")
    ap.add_argument("--init-dir", type=str, default="", help="img2img init RGB {i:03d}.png (Pc/LL)")
    ap.add_argument("--cn-scale", type=float, default=0.45)
    ap.add_argument("--cn-scale-npy", type=str, default="")
    ap.add_argument("--ip-scale", type=float, default=1.0)
    ap.add_argument("--ip-scale-npy", type=str, default="")
    ap.add_argument("--strength", type=float, default=0.30, help="img2img SDEdit strength")
    ap.add_argument("--strength-npy", type=str, default="")
    ap.add_argument("--control-guidance-start", type=float, default=0.0)
    ap.add_argument("--control-guidance-end", type=float, default=0.40, help="CN off after this fraction of steps")
    ap.add_argument("--gen-steps", type=int, default=28)
    ap.add_argument("--gen-guidance", type=float, default=5.0)
    ap.add_argument("--gen-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--negative-prompt", type=str, default="blurry, low quality, distorted, watermark")
    ap.add_argument("--lora-dir", type=str, default="", help="diffusers LoRA weights dir")
    ap.add_argument("--lora-scale", type=float, default=1.0)
    ap.add_argument("--skip-metrics", action="store_true")
    args = ap.parse_args()

    if args.mode == "img2img" and not args.init_dir:
        raise ValueError("--init-dir required for img2img")

    cache = Path(os.environ.get("HF_HUB_CACHE", "/project/peilab/why/cache/eeg-brainit/hf/hub"))
    os.environ.setdefault("HF_HOME", str(cache.parent))
    os.environ.setdefault("OPENCLIP_CACHE_DIR", str(cache.parent / "open_clip"))
    os.environ.setdefault("TORCH_HOME", str(cache.parent / "torch"))
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)

    out = Path(args.output_dir)
    gen_dir = out / "generated"
    gen_dir.mkdir(parents=True, exist_ok=True)
    cond_dir = Path(args.cond_dir)
    init_dir = Path(args.init_dir) if args.init_dir else None

    embeds = l2(np.load(args.embed_npy).astype(np.float32))
    prompts = json.loads(Path(args.prompts_json).read_text(encoding="utf-8")) if args.prompts_json else None
    n = len(embeds) if args.max_images <= 0 else min(len(embeds), args.max_images)
    cn_arr = np.load(args.cn_scale_npy).astype(np.float32).reshape(-1) if args.cn_scale_npy else None
    ip_arr = np.load(args.ip_scale_npy).astype(np.float32).reshape(-1) if args.ip_scale_npy else None
    s_arr = np.load(args.strength_npy).astype(np.float32).reshape(-1) if args.strength_npy else None

    if all((gen_dir / f"{i:03d}.png").is_file() for i in range(n)):
        print(f"[SKIP] gen exists n={n}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        sdxl = resolve_sdxl_model_path(cache)
        cn_path = resolve_controlnet_path(cache, args.control_type)
        print(f"[INFO] mode={args.mode} sdxl={sdxl}")
        print(f"[INFO] controlnet={cn_path} timed=[{args.control_guidance_start},{args.control_guidance_end}]")

        from diffusers import ControlNetModel
        from diffusers import StableDiffusionXLControlNetImg2ImgPipeline
        from diffusers import StableDiffusionXLControlNetPipeline

        controlnet = ControlNetModel.from_pretrained(
            cn_path, torch_dtype=dtype, local_files_only=Path(str(cn_path)).is_dir()
        )
        pipe_cls = (
            StableDiffusionXLControlNetImg2ImgPipeline
            if args.mode == "img2img"
            else StableDiffusionXLControlNetPipeline
        )
        pipe = pipe_cls.from_pretrained(
            sdxl,
            controlnet=controlnet,
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

        g = torch.Generator(device=device).manual_seed(args.seed)
        for i in tqdm(range(n), desc=f"struct-{args.mode}-{args.control_type}"):
            path = gen_dir / f"{i:03d}.png"
            if path.is_file():
                continue
            cond = load_cond(cond_dir / f"{i:03d}.png", args.gen_size, args.control_type)
            emb = torch.from_numpy(embeds[i : i + 1]).to(device=device, dtype=pipe.dtype)
            if args.gen_guidance > 1.0:
                ip_emb = torch.cat([torch.zeros_like(emb), emb], dim=0).unsqueeze(1)
            else:
                ip_emb = emb.unsqueeze(1)
            prompt = prompts[i] if prompts is not None else ""
            neg = args.negative_prompt if prompt else ""
            cn_s = float(cn_arr[i]) if cn_arr is not None else float(args.cn_scale)
            ip_s = float(ip_arr[i]) if ip_arr is not None else float(args.ip_scale)
            pipe.set_ip_adapter_scale(ip_s)
            kwargs = dict(
                prompt=prompt or "",
                negative_prompt=neg,
                controlnet_conditioning_scale=cn_s,
                control_guidance_start=float(args.control_guidance_start),
                control_guidance_end=float(args.control_guidance_end),
                ip_adapter_image_embeds=[ip_emb],
                num_inference_steps=int(args.gen_steps),
                guidance_scale=float(args.gen_guidance),
                generator=g,
            )
            if args.mode == "txt2img":
                img = pipe(
                    image=cond,
                    height=args.gen_size,
                    width=args.gen_size,
                    **kwargs,
                ).images[0]
            else:
                assert init_dir is not None
                init = Image.open(init_dir / f"{i:03d}.png").convert("RGB").resize(
                    (args.gen_size, args.gen_size), Image.Resampling.BICUBIC
                )
                s_i = float(s_arr[i]) if s_arr is not None else float(args.strength)
                s_i = float(np.clip(s_i, 0.05, 0.95))
                img = pipe(
                    image=init,
                    control_image=cond,
                    strength=s_i,
                    **kwargs,
                ).images[0]
            img.save(path)

        del pipe, controlnet
        if device.type == "cuda":
            torch.cuda.empty_cache()

    report = {
        "tag": args.tag,
        "mode": args.mode,
        "control_type": args.control_type,
        "cond_dir": args.cond_dir,
        "init_dir": args.init_dir or None,
        "cn_scale": args.cn_scale,
        "cn_scale_npy": args.cn_scale_npy or None,
        "ip_scale": args.ip_scale,
        "strength": args.strength if args.mode == "img2img" else None,
        "strength_npy": args.strength_npy or None,
        "control_guidance_start": args.control_guidance_start,
        "control_guidance_end": args.control_guidance_end,
        "gen_steps": args.gen_steps,
        "gen_guidance": args.gen_guidance,
        "lora_dir": args.lora_dir or None,
        "lora_scale": args.lora_scale if args.lora_dir else None,
        "n_gen": n,
        "method_note": "ATM latent-init / CogCap depth-CN; timed CN early + IP late; optional LoRA",
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
