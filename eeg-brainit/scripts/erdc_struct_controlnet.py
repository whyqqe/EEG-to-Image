#!/usr/bin/env python3
"""ERDC W9: Retrieval-Canny ControlNet + IP-Adapter + brain-consistency loop.

Structure path (PixCorr push):
  EEG emb → retrieve TRAIN neighbor → Canny(edge) → SDXL-ControlNet
  + IP-Adapter from bit/ATM CLIP emb → multi-hypothesis → brain pick.

No GT edges. Controls: random neighbor / misalign / cn-only / ip-only.
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
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from erdc_ras_closed_loop import (  # type: ignore
    encode_clip,
    l2,
    list_train_images,
    load_embed,
    retrieve_neighbors,
    select_and_eval,
)
from eval_atm_pipeline import (  # type: ignore
    list_test_images,
    resolve_ip_adapter_dir,
    resolve_sdxl_model_path,
)


def resolve_controlnet_path(hub: Path) -> str:
    root = hub / "models--diffusers--controlnet-canny-sdxl-1.0" / "snapshots"
    if not root.is_dir():
        raise FileNotFoundError(f"ControlNet missing under {root}")
    for snap in sorted(root.iterdir(), reverse=True):
        if (snap / "config.json").is_file():
            return str(snap)
    raise FileNotFoundError("no ControlNet snapshot with config.json")


def to_canny(path: Path, size: int, low: int = 100, high: int = 200) -> Image.Image:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low, high)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges)


def ip_embeds(emb: torch.Tensor) -> list[torch.Tensor]:
    uncond = torch.zeros_like(emb)
    return [torch.cat([uncond, emb], dim=0).unsqueeze(1)]


@torch.no_grad()
def load_cn_pipe(device: torch.device):
    from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline

    hub = Path(os.environ.get("HF_HUB_CACHE", "/project/peilab/why/cache/eeg-brainit/hf/hub"))
    sdxl = resolve_sdxl_model_path(hub)
    cn_path = resolve_controlnet_path(hub)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    controlnet = ControlNetModel.from_pretrained(cn_path, torch_dtype=dtype, local_files_only=True)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        sdxl,
        controlnet=controlnet,
        torch_dtype=dtype,
        variant="fp16" if device.type == "cuda" else None,
        use_safetensors=True,
        local_files_only=True,
    ).to(device)
    ip_root = resolve_ip_adapter_dir(hub)
    if ip_root is None:
        raise FileNotFoundError("IP-Adapter missing")
    weight_name = "ip-adapter_sdxl_vit-h.bin"
    kwargs = {"subfolder": "sdxl_models", "image_encoder_folder": None, "local_files_only": True}
    try:
        pipe.load_ip_adapter(str(ip_root), weight_name=weight_name, **kwargs)
    except Exception:
        weight_name = "ip-adapter_sdxl.bin"
        pipe.load_ip_adapter(str(ip_root), weight_name=weight_name, **kwargs)
    return pipe, sdxl, cn_path, weight_name


def candidate_specs(
    cn_scales: list[float],
    ip_scales: list[float],
    top_m: int,
    seed0: int,
    include_ip_only: bool,
    include_cn_only: bool,
) -> list[dict]:
    specs = []
    k = 0
    if include_ip_only:
        specs.append(
            {"k": k, "mode": "ip_only", "cn_scale": 0.0, "ip_scale": 1.0, "neighbor_rank": -1, "seed": seed0}
        )
        k += 1
    if include_cn_only:
        for r in range(top_m):
            for cs in cn_scales:
                specs.append(
                    {
                        "k": k,
                        "mode": "cn_only",
                        "cn_scale": float(cs),
                        "ip_scale": 0.0,
                        "neighbor_rank": int(r),
                        "seed": seed0 + 10 + k,
                    }
                )
                k += 1
    for r in range(top_m):
        for cs in cn_scales:
            for ips in ip_scales:
                specs.append(
                    {
                        "k": k,
                        "mode": "cn_ip",
                        "cn_scale": float(cs),
                        "ip_scale": float(ips),
                        "neighbor_rank": int(r),
                        "seed": seed0 + 100 + k,
                    }
                )
                k += 1
    return specs


@torch.no_grad()
def generate_bank(
    pipe,
    embeds: np.ndarray,
    neighbor_idx: np.ndarray,
    train_paths: list[Path],
    cand_dir: Path,
    specs: list[dict],
    device: torch.device,
    steps: int,
    guidance: float,
    size: int,
    n: int,
) -> None:
    cand_dir.mkdir(parents=True, exist_ok=True)
    (cand_dir / "specs.json").write_text(json.dumps(specs, indent=2), encoding="utf-8")
    canny_cache: dict[int, Image.Image] = {}
    total = n * len(specs)
    pbar = tqdm(total=total, desc="cn-gen")
    for i in range(n):
        emb = torch.from_numpy(embeds[i : i + 1]).to(device=device, dtype=pipe.dtype)
        ie = ip_embeds(emb)
        for sp in specs:
            out = cand_dir / f"{i:03d}_k{sp['k']}.png"
            if out.is_file():
                pbar.update(1)
                continue
            g = torch.Generator(device=device).manual_seed(int(sp["seed"]))
            mode = sp["mode"]
            if mode == "ip_only":
                pipe.set_ip_adapter_scale(float(sp["ip_scale"]))
                # dummy canny (zeros) with cn scale 0
                blank = Image.fromarray(np.zeros((size, size, 3), dtype=np.uint8))
                img = pipe(
                    prompt="",
                    negative_prompt="",
                    image=blank,
                    controlnet_conditioning_scale=0.0,
                    ip_adapter_image_embeds=ie,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    height=size,
                    width=size,
                    generator=g,
                ).images[0]
            else:
                nb = int(neighbor_idx[i, sp["neighbor_rank"]])
                if nb not in canny_cache:
                    canny_cache[nb] = to_canny(train_paths[nb], size)
                canny = canny_cache[nb]
                pipe.set_ip_adapter_scale(float(sp["ip_scale"]))
                img = pipe(
                    prompt="",
                    negative_prompt="",
                    image=canny,
                    controlnet_conditioning_scale=float(sp["cn_scale"]),
                    ip_adapter_image_embeds=ie if sp["ip_scale"] > 0 else ip_embeds(torch.zeros_like(emb)),
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    height=size,
                    width=size,
                    generator=g,
                ).images[0]
            img.save(out)
            pbar.update(1)
    pbar.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, default="sub-08")
    parser.add_argument("--bridge-dir", type=str, default="outputs/atm_bridge")
    parser.add_argument("--embed-source", type=str, default="bit_clip", choices=["atm", "bit_clip"])
    parser.add_argument(
        "--bit-npy",
        type=str,
        default="outputs/eval/atm_pipeline_sub08/sub-08_bit_clip_1024.npy",
    )
    parser.add_argument("--top-m", type=int, default=1)
    parser.add_argument("--cn-scales", type=str, default="0.5,0.8,1.0")
    parser.add_argument("--ip-scales", type=str, default="0.8,1.0")
    parser.add_argument("--include-ip-only", action="store_true", default=True)
    parser.add_argument("--no-ip-only", action="store_true")
    parser.add_argument("--include-cn-only", action="store_true", default=True)
    parser.add_argument("--no-cn-only", action="store_true")
    parser.add_argument(
        "--neighbor-mode",
        type=str,
        default="retrieve",
        choices=["retrieve", "random", "misalign"],
    )
    parser.add_argument("--images-root", type=str, default="/project/peilab/why/data/images_set")
    parser.add_argument("--output-dir", type=str, default="outputs/erdc/w9_cn_bit_sub08")
    parser.add_argument("--gen-steps", type=int, default=30)
    parser.add_argument("--gen-guidance", type=float, default=5.0)
    parser.add_argument("--gen-size", type=int, default=512)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--seed0", type=int, default=42)
    args = parser.parse_args()
    if args.no_ip_only:
        args.include_ip_only = False
    if args.no_cn_only:
        args.include_cn_only = False

    project = ROOT
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = project / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_root = Path("/project/peilab/why/cache/eeg-brainit")
    os.environ.setdefault("HF_HOME", str(cache_root / "hf"))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_root / "hf" / "hub"))
    os.environ.setdefault("OPENCLIP_CACHE_DIR", str(cache_root / "open_clip"))
    os.environ.setdefault("TORCH_HOME", str(cache_root / "torch"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("XFORMERS_DISABLED", "1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device} GPU={torch.cuda.get_device_name(0) if device.type=='cuda' else 'cpu'}")

    bridge = Path(args.bridge_dir)
    if not bridge.is_absolute():
        bridge = project / bridge
    if args.embed_source == "atm":
        embeds = load_embed(bridge / f"{args.subject}_test_eeg_1024.npy")
    else:
        p = Path(args.bit_npy)
        embeds = load_embed(p if p.is_absolute() else project / p)

    gallery = load_embed(bridge / "clip_img_train_1024.npy")
    train_paths = list_train_images(Path(args.images_root))
    n = len(embeds) if args.max_images <= 0 else min(len(embeds), args.max_images)
    mode = args.neighbor_mode
    if mode == "retrieve":
        neighbor_idx = retrieve_neighbors(embeds[:n], gallery, top_m=args.top_m)
    elif mode == "random":
        rng = np.random.RandomState(args.seed0 + 77)
        neighbor_idx = rng.randint(0, len(train_paths), size=(n, args.top_m))
    else:
        neighbor_idx = retrieve_neighbors(embeds[:n], gallery, top_m=args.top_m)
        perm = np.random.RandomState(2).permutation(len(train_paths))
        train_paths = [train_paths[i] for i in perm]
    np.save(out_dir / "neighbor_idx.npy", neighbor_idx)

    cn_scales = [float(x) for x in args.cn_scales.split(",") if x.strip()]
    ip_scales = [float(x) for x in args.ip_scales.split(",") if x.strip()]
    specs = candidate_specs(
        cn_scales, ip_scales, args.top_m, args.seed0, args.include_ip_only, args.include_cn_only
    )
    print(
        f"[INFO] source={args.embed_source} mode={mode} n={n} K={len(specs)} "
        f"cn={cn_scales} ip={ip_scales}"
    )

    pipe, sdxl, cn_path, weight_name = load_cn_pipe(device)
    generate_bank(
        pipe,
        embeds,
        neighbor_idx,
        train_paths,
        out_dir / "candidates",
        specs,
        device=device,
        steps=args.gen_steps,
        guidance=args.gen_guidance,
        size=args.gen_size,
        n=n,
    )
    del pipe
    if device.type == "cuda":
        torch.cuda.empty_cache()

    gt = list_test_images(Path(args.images_root))
    report = select_and_eval(
        embeds, out_dir / "candidates", specs, out_dir, gt, n, device, args.seed0
    )
    report.update(
        {
            "subject": args.subject,
            "embed_source": args.embed_source,
            "neighbor_mode": mode,
            "cn_scales": cn_scales,
            "ip_scales": ip_scales,
            "sdxl": sdxl,
            "controlnet": cn_path,
            "ip_weight": weight_name,
        }
    )
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[OK] wrote {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
