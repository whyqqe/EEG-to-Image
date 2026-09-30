#!/usr/bin/env python3
"""LG-Gate decode: HCMA-S dual decode with per-sample CN scale from a Router.

Same stack as generate_hcma_s_decode.py (Depth-CN + LL-RGB SDEdit init + IP +
HCMA prompts), but cn_scale_i comes from an EEG->u router:
  cn_i = cn_min + clip(u_i,0,1) * (cn_max - cn_min)
High u (structure decodable) => strong CN; low u => weak CN (closer to sdedit-LL).

Rows:
  --u-npy u_hat_test.npy      -> router-gated (main)
  --u-npy u_true_test.npy     -> oracle-gated  (upper bound; diagnostic only)
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


def resolve_controlnet_path(hub: Path) -> str:
    root = hub / "models--diffusers--controlnet-depth-sdxl-1.0" / "snapshots"
    if root.is_dir():
        for snap in sorted(root.iterdir(), reverse=True):
            if (snap / "config.json").is_file():
                return str(snap)
    return "diffusers/controlnet-depth-sdxl-1.0"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-npy", type=str, required=True)
    ap.add_argument("--prompts-json", type=str, required=True)
    ap.add_argument("--depth-rgb-dir", type=str, required=True)
    ap.add_argument("--lowlevel-rgb-dir", type=str, required=True)
    ap.add_argument("--u-npy", type=str, required=True, help="per-sample router u in [0,1]")
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--tag", type=str, default="lg_gate")
    ap.add_argument("--cn-min", type=float, default=0.0)
    ap.add_argument("--cn-max", type=float, default=0.45)
    ap.add_argument("--ip-scale", type=float, default=1.0)
    ap.add_argument("--strength", type=float, default=0.82)
    ap.add_argument("--gen-steps", type=int, default=28)
    ap.add_argument("--gen-guidance", type=float, default=5.0)
    ap.add_argument("--gen-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--negative-prompt", type=str, default="blurry, low quality, distorted, watermark")
    args = ap.parse_args()

    cache = Path(os.environ.get("HF_HUB_CACHE", "/project/peilab/why/cache/eeg-brainit/hf/hub"))
    os.environ.setdefault("HF_HOME", str(cache.parent))
    os.environ.setdefault("OPENCLIP_CACHE_DIR", str(cache.parent / "open_clip"))
    os.environ.setdefault("TORCH_HOME", str(cache.parent / "torch"))
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embeds = l2(np.load(args.embed_npy).astype(np.float32))
    prompts = json.loads(Path(args.prompts_json).read_text(encoding="utf-8"))
    if len(prompts) < len(embeds):
        raise ValueError(f"prompts {len(prompts)} < embeds {len(embeds)}")
    u = np.load(args.u_npy).astype(np.float32).reshape(-1)
    if len(u) != len(embeds):
        raise ValueError(f"u {len(u)} != embeds {len(embeds)}")
    cn = (args.cn_min + np.clip(u, 0.0, 1.0) * (args.cn_max - args.cn_min)).astype(np.float32)
    print(f"[INFO] u mean={u.mean():.3f} std={u.std():.3f} -> cn mean={cn.mean():.3f} "
          f"range=[{cn.min():.3f},{cn.max():.3f}]")

    depth_dir = Path(args.depth_rgb_dir)
    ll_dir = Path(args.lowlevel_rgb_dir)
    out_dir = Path(args.output_dir)
    gen_dir = out_dir / "generated"
    gen_dir.mkdir(parents=True, exist_ok=True)
    n = len(embeds)

    if all((gen_dir / f"{i:03d}.png").is_file() for i in range(n)):
        print(f"[INFO] skip gen, {n} images exist")
    else:
        from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline

        sdxl = resolve_sdxl_model_path(cache)
        cn_path = resolve_controlnet_path(cache)
        print(f"[INFO] sdxl={sdxl}\n[INFO] controlnet={cn_path}\n[INFO] ip={args.ip_scale} "
              f"strength={args.strength} steps={args.gen_steps} guidance={args.gen_guidance}")
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        controlnet = ControlNetModel.from_pretrained(
            cn_path, torch_dtype=dtype, local_files_only=Path(str(cn_path)).is_dir()
        )
        pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            sdxl,
            controlnet=controlnet,
            torch_dtype=dtype,
            variant="fp16" if device.type == "cuda" else None,
            use_safetensors=True,
            local_files_only=Path(str(sdxl)).is_dir(),
        ).to(device)
        ip_root = resolve_ip_adapter_dir(cache)
        weight_name = "ip-adapter_sdxl_vit-h.bin"
        kwargs = {"subfolder": "sdxl_models", "image_encoder_folder": None, "local_files_only": True}
        try:
            pipe.load_ip_adapter(str(ip_root), weight_name=weight_name, **kwargs)
        except Exception:
            weight_name = "ip-adapter_sdxl.bin"
            pipe.load_ip_adapter(str(ip_root), weight_name=weight_name, **kwargs)
        pipe.set_ip_adapter_scale(float(args.ip_scale))

        g = torch.Generator(device=device).manual_seed(args.seed)
        for i in tqdm(range(n), desc=f"lg-gate[{args.tag}]"):
            path = gen_dir / f"{i:03d}.png"
            if path.is_file():
                continue
            dp = depth_dir / f"{i:03d}.png"
            lp = ll_dir / f"{i:03d}.png"
            if not dp.is_file():
                raise FileNotFoundError(dp)
            if not lp.is_file():
                raise FileNotFoundError(lp)
            control = Image.open(dp).convert("RGB").resize((args.gen_size, args.gen_size), Image.Resampling.BICUBIC)
            init = Image.open(lp).convert("RGB").resize((args.gen_size, args.gen_size), Image.Resampling.BICUBIC)
            emb = torch.from_numpy(embeds[i : i + 1]).to(device=device, dtype=pipe.dtype)
            if args.gen_guidance > 1.0:
                ip_emb = torch.cat([torch.zeros_like(emb), emb], dim=0).unsqueeze(1)
            else:
                ip_emb = emb.unsqueeze(1)
            prompt = prompts[i] if prompts[i] else ""
            neg = args.negative_prompt if prompt else ""
            img = pipe(
                prompt=prompt,
                negative_prompt=neg,
                image=init,
                control_image=control,
                strength=float(args.strength),
                controlnet_conditioning_scale=float(cn[i]),
                ip_adapter_image_embeds=[ip_emb],
                num_inference_steps=int(args.gen_steps),
                guidance_scale=float(args.gen_guidance),
                generator=g,
            ).images[0]
            img.save(path)

        del pipe, controlnet
        if device.type == "cuda":
            torch.cuda.empty_cache()

    np.save(out_dir / "cn_scale.npy", cn.astype(np.float32))
    report = {
        "tag": args.tag,
        "pipeline": "LG-Gate (HCMA-S + EEG Router per-sample CN)",
        "u_npy": args.u_npy,
        "cn_map": f"cn = cn_min + clip(u)*(cn_max-cn_min), [{args.cn_min},{args.cn_max}]",
        "embed_npy": args.embed_npy,
        "prompts_json": args.prompts_json,
        "depth_rgb_dir": args.depth_rgb_dir,
        "lowlevel_rgb_dir": args.lowlevel_rgb_dir,
        "ip_scale": args.ip_scale,
        "strength": args.strength,
        "gen_steps": args.gen_steps,
        "gen_guidance": args.gen_guidance,
        "n_gen": n,
        "u_mean": float(u.mean()),
        "cn_mean": float(cn.mean()),
        "note": "high u => strong CN; low u => near sdedit-LL (semantic safe)",
    }
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
