#!/usr/bin/env python3
"""Build Probe-Decoder supervision: CLIP(embed) after img2img decode on a train subset."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

NB_ROOT = Path(__file__).resolve().parents[2]
BRAINIT = Path("/project/peilab/why/eeg-brainit")
sys.path.insert(0, str(BRAINIT / "scripts"))
sys.path.insert(0, str(BRAINIT / "src"))

from eval_atm_pipeline import list_test_images, resolve_ip_adapter_dir, resolve_sdxl_model_path  # type: ignore


def resolve_model(hub: Path) -> tuple[str, dict]:
    """Prefer local SDXL-base; fall back to cached sdxl-turbo (offline-safe)."""
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
    raise FileNotFoundError(
        f"No local SDXL snapshot under {hub}; need stable-diffusion-xl-base-1.0 or sdxl-turbo"
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


@torch.no_grad()
def encode_clip_paths(paths: list[Path], device: torch.device, batch_size: int = 32) -> np.ndarray:
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14", pretrained="laion2b_s32b_b79k", device=device
    )
    model.eval()
    feats = []
    for i in tqdm(range(0, len(paths), batch_size), desc="clip-encode-gen"):
        batch = paths[i : i + batch_size]
        xs = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in batch]).to(device)
        fe = F.normalize(model.encode_image(xs).float(), dim=-1)
        feats.append(fe.cpu().numpy())
    return np.concatenate(feats, axis=0).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-npy", type=str, required=True, help="train ViT-H embeds (N,1024)")
    ap.add_argument("--neighbor-idx-npy", type=str, required=True, help="(N,k) train neighbor indices")
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--images-root", type=str, default="/project/peilab/why/data/images_set")
    ap.add_argument("--clip-train-npy", type=str, default="")
    ap.add_argument("--max-samples", type=int, default=512)
    ap.add_argument("--strength", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    cache = Path(os.environ.get("HF_HUB_CACHE", "/project/peilab/why/cache/eeg-brainit/hf/hub"))
    os.environ.setdefault("HF_HOME", str(cache.parent))
    os.environ.setdefault("OPENCLIP_CACHE_DIR", str(cache.parent / "open_clip"))
    os.environ.setdefault("TORCH_HOME", str(cache.parent / "torch"))

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    embeds = l2(np.load(args.embed_npy).astype(np.float32))
    clip_train_path = args.clip_train_npy or "/project/peilab/why/eeg-brainit/outputs/atm_bridge/clip_img_train_1024.npy"
    clip_train = l2(np.load(clip_train_path).astype(np.float32))
    neighbors = np.load(args.neighbor_idx_npy)
    n = min(len(embeds), args.max_samples)
    rng = np.random.RandomState(args.seed)
    indices = rng.choice(len(embeds), size=n, replace=False)
    indices = np.sort(indices)

    train_paths = list_train_images(Path(args.images_root))
    gen_dir = out / "probe_gen"
    gen_dir.mkdir(parents=True, exist_ok=True)

    from diffusers import StableDiffusionXLImg2ImgPipeline

    model_id, defaults = resolve_model(cache)
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
    pipe.set_ip_adapter_scale(1.0)

    gen_steps = int(defaults["steps"])
    gen_guidance = float(defaults["guidance"])
    g = torch.Generator(device=device).manual_seed(args.seed)
    gen_paths: list[Path] = []
    e_list, a_list = [], []
    for j, i in enumerate(tqdm(indices, desc="probe-decode")):
        out_path = gen_dir / f"{j:04d}.png"
        nb = int(neighbors[i, 0])
        init = Image.open(train_paths[nb]).convert("RGB").resize((512, 512), Image.Resampling.BICUBIC)
        emb = torch.from_numpy(embeds[i : i + 1]).to(device=device, dtype=pipe.dtype)
        ip_emb = emb.unsqueeze(1)
        if not out_path.is_file():
            img = pipe(
                prompt="",
                negative_prompt="",
                image=init,
                strength=args.strength,
                ip_adapter_image_embeds=[ip_emb],
                num_inference_steps=gen_steps,
                guidance_scale=gen_guidance,
                generator=g,
            ).images[0]
            img.save(out_path)
        gen_paths.append(out_path)
        e_list.append(embeds[i])
        a_list.append(clip_train[nb])

    clip_gen = encode_clip_paths(gen_paths, device)
    e_arr = np.stack(e_list, axis=0).astype(np.float32)
    a_arr = np.stack(a_list, axis=0).astype(np.float32)
    np.savez_compressed(
        out / "probe_supervision.npz",
        indices=indices,
        eeg_embed=e_arr,
        anchor_embed=a_arr,
        clip_gen=clip_gen,
        clip_gt=l2(np.load(args.embed_npy)[indices]),
    )
    report = {
        "n": n,
        "strength": args.strength,
        "mean_cos_gen_gt": float(np.mean(np.sum(clip_gen * l2(np.load(args.embed_npy)[indices]), axis=1))),
        "path": str(out / "probe_supervision.npz"),
    }
    (out / "probe_build_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
