#!/usr/bin/env python3
"""Two-stage refine: StageA (structure img2img) → StageB (semantic refine img2img).

If --stage-a-dir is given, skip Stage A and refine those images directly.
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

from eval_atm_pipeline import resolve_ip_adapter_dir, resolve_sdxl_model_path  # type: ignore


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-8)


def run_img2img(
    pipe,
    init: Image.Image,
    emb: torch.Tensor,
    prompt: str,
    neg: str,
    strength: float,
    steps: int,
    guidance: float,
    generator: torch.Generator,
) -> Image.Image:
    if guidance > 1.0:
        ip_emb = torch.cat([torch.zeros_like(emb), emb], dim=0).unsqueeze(1)
    else:
        ip_emb = emb.unsqueeze(1)
    return pipe(
        prompt=prompt or "",
        negative_prompt=neg if prompt else "",
        image=init,
        strength=float(strength),
        ip_adapter_image_embeds=[ip_emb],
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        generator=generator,
    ).images[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lowlevel-dir", type=str, default="", help="init for Stage A if no stage-a-dir")
    ap.add_argument("--stage-a-dir", type=str, default="", help="reuse existing structure gens as Stage A")
    ap.add_argument("--embed-npy", type=str, required=True)
    ap.add_argument("--prompts-json", type=str, default="")
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--tag", type=str, default="twostage")
    ap.add_argument("--strength-a", type=float, default=0.45)
    ap.add_argument("--strength-b", type=float, default=0.70)
    ap.add_argument("--ip-scale-a", type=float, default=1.0)
    ap.add_argument("--ip-scale-b", type=float, default=1.0)
    ap.add_argument("--gen-steps", type=int, default=28)
    ap.add_argument("--gen-guidance", type=float, default=5.0)
    ap.add_argument("--gen-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--negative-prompt", type=str, default="blurry, low quality, distorted, watermark")
    ap.add_argument("--keep-stage-a", action="store_true")
    args = ap.parse_args()

    cache = Path(os.environ.get("HF_HUB_CACHE", "/project/peilab/why/cache/eeg-brainit/hf/hub"))
    os.environ.setdefault("HF_HOME", str(cache.parent))
    os.environ.setdefault("OPENCLIP_CACHE_DIR", str(cache.parent / "open_clip"))
    os.environ.setdefault("TORCH_HOME", str(cache.parent / "torch"))

    out = Path(args.output_dir)
    gen_b = out / "generated"
    gen_a = out / "stage_a"
    gen_b.mkdir(parents=True, exist_ok=True)
    if args.keep_stage_a:
        gen_a.mkdir(parents=True, exist_ok=True)

    embeds = l2(np.load(args.embed_npy).astype(np.float32))
    prompts = json.loads(Path(args.prompts_json).read_text(encoding="utf-8")) if args.prompts_json else None
    n = len(embeds) if args.max_images <= 0 else min(len(embeds), args.max_images)

    stage_a_src = Path(args.stage_a_dir) if args.stage_a_dir else None
    low_dir = Path(args.lowlevel_dir) if args.lowlevel_dir else None
    if stage_a_src is None and low_dir is None:
        raise ValueError("need --stage-a-dir or --lowlevel-dir")

    if all((gen_b / f"{i:03d}.png").is_file() for i in range(n)):
        print(f"[SKIP] {n} stage-B images exist")
    else:
        from diffusers import StableDiffusionXLImg2ImgPipeline

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
                str(ip_root), weight_name="ip-adapter_sdxl_vit-h.bin",
                subfolder="sdxl_models", image_encoder_folder=None, local_files_only=True,
            )
        except Exception:
            pipe.load_ip_adapter(
                str(ip_root), weight_name="ip-adapter_sdxl.bin",
                subfolder="sdxl_models", image_encoder_folder=None, local_files_only=True,
            )
        g = torch.Generator(device=device).manual_seed(args.seed)

        for i in tqdm(range(n), desc="twostage"):
            out_b = gen_b / f"{i:03d}.png"
            if out_b.is_file():
                continue
            if stage_a_src is not None:
                img_a = Image.open(stage_a_src / f"{i:03d}.png").convert("RGB").resize(
                    (args.gen_size, args.gen_size), Image.Resampling.BICUBIC
                )
            else:
                init0 = Image.open(low_dir / f"{i:03d}.png").convert("RGB").resize(
                    (args.gen_size, args.gen_size), Image.Resampling.BICUBIC
                )
                emb = torch.from_numpy(embeds[i : i + 1]).to(device=device, dtype=pipe.dtype)
                prompt = prompts[i] if prompts is not None else ""
                pipe.set_ip_adapter_scale(float(args.ip_scale_a))
                img_a = run_img2img(
                    pipe, init0, emb, prompt, args.negative_prompt,
                    args.strength_a, args.gen_steps, args.gen_guidance, g,
                )
                if args.keep_stage_a:
                    img_a.save(gen_a / f"{i:03d}.png")

            emb = torch.from_numpy(embeds[i : i + 1]).to(device=device, dtype=pipe.dtype)
            prompt = prompts[i] if prompts is not None else ""
            pipe.set_ip_adapter_scale(float(args.ip_scale_b))
            img_b = run_img2img(
                pipe, img_a, emb, prompt, args.negative_prompt,
                args.strength_b, args.gen_steps, args.gen_guidance, g,
            )
            img_b.save(out_b)

        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    report = {
        "tag": args.tag,
        "mode": "twostage_refine",
        "strength_a": args.strength_a,
        "strength_b": args.strength_b,
        "stage_a_dir": args.stage_a_dir or None,
        "lowlevel_dir": args.lowlevel_dir or None,
        "n": n,
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
