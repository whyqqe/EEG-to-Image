#!/usr/bin/env python3
"""Decoder fix: SDXL ControlNet(Canny) + IP-Adapter + optional CPA text.

Replaces turbo img2img-only decoding. Structure from retrieved neighbor Canny
(no GT). Semantics from NDA-SS ViT-H embeds via IP-Adapter.
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
from PIL import Image, ImageFilter
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


def resolve_controlnet_path(hub: Path, control_type: str = "canny") -> str:
    name = "controlnet-canny-sdxl-1.0" if control_type == "canny" else "controlnet-depth-sdxl-1.0"
    root = hub / f"models--diffusers--{name}" / "snapshots"
    if root.is_dir():
        for snap in sorted(root.iterdir(), reverse=True):
            if (snap / "config.json").is_file():
                return str(snap)
    return f"diffusers/{name}"


def to_canny(path: Path, size: int, low: int = 100, high: int = 200) -> Image.Image:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low, high)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges)


def load_depth_map(cache_dir: Path, nb: int, size: int) -> Image.Image:
    path = cache_dir / f"{nb:06d}.png"
    if not path.is_file():
        raise FileNotFoundError(f"missing depth cache {path}")
    return Image.open(path).convert("RGB").resize((size, size), Image.Resampling.BICUBIC)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-npy", type=str, required=True)
    ap.add_argument("--neighbor-idx-npy", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--images-root", type=str, default="/project/peilab/why/data/images_set")
    ap.add_argument("--prompts-json", type=str, default="")
    ap.add_argument("--cn-scale", type=float, default=0.8)
    ap.add_argument("--ip-scale", type=float, default=0.9)
    ap.add_argument("--cn-scale-npy", type=str, default="", help="per-sample ControlNet scale (SCR)")
    ap.add_argument("--ip-scale-npy", type=str, default="", help="per-sample IP scale (SCR)")
    ap.add_argument("--fuse-beta-npy", type=str, default="", help="per-sample semantic blend weight; rest=blurred neighbor")
    ap.add_argument("--enable-fuse", action="store_true")
    ap.add_argument("--control-type", type=str, default="canny", choices=["canny", "depth"])
    ap.add_argument("--depth-cache-dir", type=str, default="", help="dir with {neighbor_idx:06d}.png depth maps")
    ap.add_argument(
        "--depth-sample-dir",
        type=str,
        default="",
        help="per-test-sample depth RGB dir with {i:03d}.png (EEG-predicted or GT); overrides neighbor cache",
    )
    ap.add_argument("--gen-steps", type=int, default=30)
    ap.add_argument("--gen-guidance", type=float, default=5.0)
    ap.add_argument("--gen-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", type=str, default="cn_ip")
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--negative-prompt", type=str, default="blurry, low quality, distorted, watermark")
    ap.add_argument("--skip-metrics", action="store_true")
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()

    cache = Path(os.environ.get("HF_HUB_CACHE", "/project/peilab/why/cache/eeg-brainit/hf/hub"))
    os.environ.setdefault("HF_HOME", str(cache.parent))
    os.environ.setdefault("OPENCLIP_CACHE_DIR", str(cache.parent / "open_clip"))
    os.environ.setdefault("TORCH_HOME", str(cache.parent / "torch"))
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embeds = l2(np.load(args.embed_npy).astype(np.float32))
    neighbors = np.load(args.neighbor_idx_npy)
    train_paths = list_train_images(Path(args.images_root))
    prompts = None
    if args.prompts_json:
        prompts = json.loads(Path(args.prompts_json).read_text(encoding="utf-8"))
        if len(prompts) < len(embeds):
            raise ValueError(f"prompts {len(prompts)} < embeds {len(embeds)}")
    cn_arr = np.load(args.cn_scale_npy).astype(np.float32).reshape(-1) if args.cn_scale_npy else None
    ip_arr = np.load(args.ip_scale_npy).astype(np.float32).reshape(-1) if args.ip_scale_npy else None
    fuse_arr = np.load(args.fuse_beta_npy).astype(np.float32).reshape(-1) if args.fuse_beta_npy else None

    out_dir = Path(args.output_dir)
    gen_dir = out_dir / "generated"
    gen_dir.mkdir(parents=True, exist_ok=True)
    n = len(embeds) if args.max_images <= 0 else min(len(embeds), args.max_images)

    if all((gen_dir / f"{i:03d}.png").is_file() for i in range(n)):
        print(f"[INFO] skip gen, {n} images exist")
    else:
        from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline

        sdxl = resolve_sdxl_model_path(cache)
        cn_path = resolve_controlnet_path(cache, args.control_type)
        print(f"[INFO] sdxl={sdxl}")
        print(f"[INFO] controlnet={cn_path} type={args.control_type}")
        if args.control_type == "depth" and not args.depth_cache_dir and not args.depth_sample_dir:
            raise ValueError("--depth-cache-dir or --depth-sample-dir required for control-type=depth")
        depth_cache = Path(args.depth_cache_dir) if args.depth_cache_dir else None
        depth_sample = Path(args.depth_sample_dir) if args.depth_sample_dir else None
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        local_only = Path(str(sdxl)).is_dir() and Path(str(cn_path)).is_dir()
        controlnet = ControlNetModel.from_pretrained(
            cn_path, torch_dtype=dtype, local_files_only=local_only and Path(str(cn_path)).is_dir()
        )
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
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

        canny_cache: dict[int, Image.Image] = {}
        g = torch.Generator(device=device).manual_seed(args.seed)
        for i in tqdm(range(n), desc=f"{args.control_type}-ip-decode"):
            path = gen_dir / f"{i:03d}.png"
            if path.is_file():
                continue
            nb = int(neighbors[i, 0] if neighbors.ndim > 1 else neighbors[i])
            if args.control_type == "depth":
                if depth_sample is not None:
                    sp = depth_sample / f"{i:03d}.png"
                    if not sp.is_file():
                        raise FileNotFoundError(sp)
                    cond = Image.open(sp).convert("RGB").resize(
                        (args.gen_size, args.gen_size), Image.Resampling.BICUBIC
                    )
                else:
                    cond = load_depth_map(depth_cache, nb, args.gen_size)
            else:
                if nb not in canny_cache:
                    canny_cache[nb] = to_canny(train_paths[nb], args.gen_size)
                cond = canny_cache[nb]
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
            img = pipe(
                prompt=prompt or "",
                negative_prompt=neg,
                image=cond,
                controlnet_conditioning_scale=cn_s,
                ip_adapter_image_embeds=[ip_emb],
                num_inference_steps=int(args.gen_steps),
                guidance_scale=float(args.gen_guidance),
                height=args.gen_size,
                width=args.gen_size,
                generator=g,
            ).images[0]
            if args.enable_fuse:
                neigh_img = Image.open(train_paths[nb]).convert("RGB").resize(
                    (args.gen_size, args.gen_size), Image.Resampling.BICUBIC
                )
                blur = neigh_img.filter(ImageFilter.GaussianBlur(radius=2))
                beta = float(fuse_arr[i]) if fuse_arr is not None else 0.9
                img = Image.blend(blur, img, beta)
            img.save(path)

        del pipe, controlnet
        if device.type == "cuda":
            torch.cuda.empty_cache()

    paths = sorted(gen_dir.glob("*.png"))[:n]
    report = {
        "tag": args.tag,
        "decoder": f"SDXL ControlNet-{args.control_type} + IP-Adapter (+ optional CPA text / SCR)",
        "control_type": args.control_type,
        "depth_cache_dir": args.depth_cache_dir or None,
        "depth_sample_dir": args.depth_sample_dir or None,
        "embed_npy": args.embed_npy,
        "cn_scale": args.cn_scale,
        "ip_scale": args.ip_scale,
        "cn_scale_npy": args.cn_scale_npy or None,
        "ip_scale_npy": args.ip_scale_npy or None,
        "enable_fuse": bool(args.enable_fuse),
        "prompts_json": args.prompts_json or None,
        "gen_steps": args.gen_steps,
        "gen_guidance": args.gen_guidance,
        "n_gen": len(paths),
    }
    if not args.skip_metrics:
        gt = list_test_images(Path(args.images_root))
        metrics = image_metrics(paths, gt[: len(paths)], device)
        report["metrics"] = metrics
        print(f"[METRICS] CLIP={metrics['clip_cosine']:.4f} SSIM={metrics['ssim']:.4f} Pix={metrics['pixcorr']:.4f}")
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[OK] {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
