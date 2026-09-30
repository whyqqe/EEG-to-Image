#!/usr/bin/env python3
"""ERDC W1/W3: ATM/bridge embeds → SDXL + IP-Adapter with optional confidence scales.

Baselines / variants:
  B0 / fixed     : constant ip_scale
  conf           : ip_scale ∝ retrieval-margin confidence (no GT label needed)
  inv_conf       : inverted confidence (control; should hurt)
  teacher        : CLIP image embeds (generation ceiling)

Writes under --output-dir only. Offline HF cache under project.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

# Reuse path resolvers / metrics from the proven ATM pipeline.
from eval_atm_pipeline import (  # type: ignore
    image_metrics,
    list_test_images,
    resolve_ip_adapter_dir,
    resolve_sdxl_model_path,
)


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-8)


def retrieval_confidence(
    query: np.ndarray,
    gallery: np.ndarray,
    temperature: float = 0.07,
) -> dict[str, np.ndarray]:
    """Label-free confidence from retrieval geometry.

    margin = top1_sim - top2_sim over gallery (higher → more decisive)
    entropy = H(softmax(sim/τ)) / log(N)  (lower → more peaked)
    conf = 0.5 * margin_norm + 0.5 * (1 - entropy)
    """
    sim = query @ gallery.T  # (N, M)
    n, m = sim.shape
    # top-2
    part = np.partition(sim, -2, axis=1)
    top1 = part[:, -1]
    top2 = part[:, -2]
    margin = top1 - top2
    # normalize margin to [0,1] by rank within batch
    order = margin.argsort()
    margin_norm = np.empty_like(margin, dtype=np.float64)
    margin_norm[order] = np.linspace(0.0, 1.0, n)

    logits = sim / max(temperature, 1e-6)
    logits = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    p = p / p.sum(axis=1, keepdims=True).clip(min=1e-12)
    ent = -(p * np.log(p.clip(min=1e-12))).sum(axis=1) / np.log(float(m))
    conf = 0.5 * margin_norm + 0.5 * (1.0 - ent)
    conf = np.clip(conf, 0.0, 1.0).astype(np.float32)
    return {
        "conf": conf,
        "margin": margin.astype(np.float32),
        "margin_norm": margin_norm.astype(np.float32),
        "entropy": ent.astype(np.float32),
        "top1_sim": top1.astype(np.float32),
    }


def scales_from_mode(
    mode: str,
    conf: np.ndarray,
    base_scale: float,
    scale_min: float,
    scale_max: float,
) -> np.ndarray:
    if mode in ("fixed", "b0", "teacher"):
        return np.full(len(conf), base_scale, dtype=np.float32)
    if mode == "conf":
        return (scale_min + (scale_max - scale_min) * conf).astype(np.float32)
    if mode == "inv_conf":
        return (scale_min + (scale_max - scale_min) * (1.0 - conf)).astype(np.float32)
    raise ValueError(f"unknown mode={mode}")


@torch.no_grad()
def generate_with_scales(
    clip_embeds: np.ndarray,
    scales: np.ndarray,
    out_dir: Path,
    device: torch.device,
    steps: int,
    guidance: float,
    height: int,
    width: int,
    max_images: int,
    seed: int,
    source_name: str,
) -> list[Path]:
    from diffusers import StableDiffusionXLPipeline

    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(clip_embeds) if max_images <= 0 else min(len(clip_embeds), max_images)
    existing = [out_dir / f"{i:03d}.png" for i in range(n)]
    if all(p.is_file() for p in existing):
        print(f"[INFO] skip gen ({source_name}): {n} images already in {out_dir}")
        return existing

    print(f"[INFO] loading SDXL for {source_name} (n={n})...")
    hub = Path(os.environ.get("HF_HUB_CACHE", "/project/peilab/why/cache/eeg-brainit/hf/hub"))
    model_id = resolve_sdxl_model_path(hub)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        variant="fp16" if device.type == "cuda" else None,
        use_safetensors=True,
        local_files_only=True,
    ).to(device)

    ip_root = resolve_ip_adapter_dir(hub)
    weight_name = "ip-adapter_sdxl_vit-h.bin"
    load_kwargs = {
        "subfolder": "sdxl_models",
        "image_encoder_folder": None,
        "local_files_only": True,
    }
    if ip_root is None:
        raise FileNotFoundError("IP-Adapter snapshot missing under HF hub cache")
    try:
        pipe.load_ip_adapter(str(ip_root), weight_name=weight_name, **load_kwargs)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] vit-h failed ({exc}); fallback ip-adapter_sdxl.bin")
        weight_name = "ip-adapter_sdxl.bin"
        pipe.load_ip_adapter(str(ip_root), weight_name=weight_name, **load_kwargs)

    paths: list[Path] = []
    use_unsqueeze = True  # proven layout from prior ATM eval
    g = torch.Generator(device=device).manual_seed(seed)

    for i in tqdm(range(n), desc=f"erdc-gen[{source_name}]"):
        path = out_dir / f"{i:03d}.png"
        if path.is_file():
            paths.append(path)
            continue
        pipe.set_ip_adapter_scale(float(scales[i]))
        emb = torch.from_numpy(clip_embeds[i : i + 1]).to(device=device, dtype=pipe.dtype)
        uncond = torch.zeros_like(emb)
        image_embeds = torch.cat([uncond, emb], dim=0)
        if use_unsqueeze:
            image_embeds = image_embeds.unsqueeze(1)
        result = pipe(
            prompt="",
            negative_prompt="",
            ip_adapter_image_embeds=[image_embeds],
            num_inference_steps=steps,
            guidance_scale=guidance,
            height=height,
            width=width,
            generator=g,
        )
        result.images[0].save(path)
        paths.append(path)

    meta = {
        "source": source_name,
        "weight_name": weight_name,
        "steps": steps,
        "guidance": guidance,
        "model": model_id,
        "n": n,
        "scale_mean": float(np.mean(scales[:n])),
        "scale_std": float(np.std(scales[:n])),
        "scale_min": float(np.min(scales[:n])),
        "scale_max": float(np.max(scales[:n])),
    }
    (out_dir / "adapter.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    del pipe
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return paths


def load_embed(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    return l2(np.load(path).astype(np.float32))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, default="sub-08")
    parser.add_argument("--bridge-dir", type=str, default="outputs/atm_bridge")
    parser.add_argument(
        "--gallery",
        type=str,
        default="outputs/eval/atm_baseline/test_ViT-H-14_laion2b_s32b_b79k_features.npy",
        help="CLIP gallery for confidence + metrics (ViT-H/14)",
    )
    parser.add_argument(
        "--embed-source",
        type=str,
        default="atm",
        choices=["atm", "teacher", "prior", "bit_clip", "bridge_clip", "npy"],
    )
    parser.add_argument("--embed-npy", type=str, default="", help="when --embed-source=npy")
    parser.add_argument(
        "--prior-npy",
        type=str,
        default="outputs/eval/atm_pipeline_sub08/sub-08_prior_clip_1024.npy",
    )
    parser.add_argument(
        "--bit-npy",
        type=str,
        default="outputs/eval/atm_pipeline_sub08/sub-08_bit_clip_1024.npy",
    )
    parser.add_argument(
        "--bridge-npy",
        type=str,
        default="outputs/eval/atm_pipeline_sub08/sub-08_bridge_clip_1024.npy",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="fixed",
        choices=["fixed", "b0", "conf", "inv_conf", "teacher"],
    )
    parser.add_argument("--base-scale", type=float, default=1.0)
    parser.add_argument("--scale-min", type=float, default=0.35)
    parser.add_argument("--scale-max", type=float, default=1.15)
    parser.add_argument("--conf-temperature", type=float, default=0.07)
    parser.add_argument("--images-root", type=str, default="/project/peilab/why/data/images_set")
    parser.add_argument("--output-dir", type=str, default="outputs/erdc/w1_b0_sub08")
    parser.add_argument("--gen-steps", type=int, default=30)
    parser.add_argument("--gen-guidance", type=float, default=5.0)
    parser.add_argument("--gen-size", type=int, default=512)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-metrics", action="store_true")
    args = parser.parse_args()

    project = ROOT
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = project / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_root = Path("/project/peilab/why/cache/eeg-brainit")
    os.environ.setdefault("HF_HOME", str(cache_root / "hf"))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_root / "hf" / "hub"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_root / "hf" / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_root / "hf" / "hub"))
    os.environ.setdefault("DIFFUSERS_CACHE", str(cache_root / "hf" / "hub"))
    os.environ.setdefault("OPENCLIP_CACHE_DIR", str(cache_root / "open_clip"))
    os.environ.setdefault("TORCH_HOME", str(cache_root / "torch"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")
    if device.type == "cuda":
        print(f"[INFO] GPU={torch.cuda.get_device_name(0)}")

    bridge_dir = Path(args.bridge_dir)
    if not bridge_dir.is_absolute():
        bridge_dir = project / bridge_dir
    subject = args.subject

    gallery_path = Path(args.gallery)
    if not gallery_path.is_absolute():
        gallery_path = project / gallery_path
    gallery = load_embed(gallery_path)

    mode = "teacher" if args.mode == "teacher" else args.mode
    embed_source = "teacher" if mode == "teacher" else args.embed_source

    if embed_source == "atm":
        embeds = load_embed(bridge_dir / f"{subject}_test_eeg_1024.npy")
    elif embed_source == "teacher":
        embeds = load_embed(bridge_dir / "clip_img_test_1024.npy")
    elif embed_source == "prior":
        p = Path(args.prior_npy)
        embeds = load_embed(p if p.is_absolute() else project / p)
    elif embed_source == "bit_clip":
        p = Path(args.bit_npy)
        embeds = load_embed(p if p.is_absolute() else project / p)
    elif embed_source == "bridge_clip":
        p = Path(args.bridge_npy)
        embeds = load_embed(p if p.is_absolute() else project / p)
    else:
        p = Path(args.embed_npy)
        embeds = load_embed(p if p.is_absolute() else project / p)

    if embeds.shape[0] != gallery.shape[0]:
        raise RuntimeError(f"embed n={embeds.shape[0]} vs gallery n={gallery.shape[0]}")

    conf_pack = retrieval_confidence(embeds, gallery, temperature=args.conf_temperature)
    scales = scales_from_mode(
        mode if mode != "teacher" else "fixed",
        conf_pack["conf"],
        base_scale=args.base_scale,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
    )
    np.save(out_dir / "confidence.npy", conf_pack["conf"])
    np.save(out_dir / "scales.npy", scales)
    (out_dir / "confidence_stats.json").write_text(
        json.dumps(
            {
                "mode": mode,
                "embed_source": embed_source,
                "conf_mean": float(conf_pack["conf"].mean()),
                "conf_std": float(conf_pack["conf"].std()),
                "margin_mean": float(conf_pack["margin"].mean()),
                "entropy_mean": float(conf_pack["entropy"].mean()),
                "scale_mean": float(scales.mean()),
                "scale_std": float(scales.std()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"[INFO] mode={mode} source={embed_source} "
        f"conf={conf_pack['conf'].mean():.3f}±{conf_pack['conf'].std():.3f} "
        f"scale={scales.mean():.3f}±{scales.std():.3f}"
    )

    gen_dir = out_dir / "generated"
    paths = generate_with_scales(
        embeds,
        scales,
        gen_dir,
        device=device,
        steps=args.gen_steps,
        guidance=args.gen_guidance,
        height=args.gen_size,
        width=args.gen_size,
        max_images=args.max_images,
        seed=args.seed,
        source_name=f"{embed_source}_{mode}",
    )

    report: dict = {
        "subject": subject,
        "mode": mode,
        "embed_source": embed_source,
        "n_gen": len(paths),
        "gen_steps": args.gen_steps,
        "gen_guidance": args.gen_guidance,
        "confidence": json.loads((out_dir / "confidence_stats.json").read_text(encoding="utf-8")),
    }

    if not args.skip_metrics:
        gt_paths = list_test_images(Path(args.images_root))
        n = len(paths)
        metrics = image_metrics(paths, gt_paths[:n], device)
        report["metrics"] = metrics
        print(
            f"[INFO] metrics PixCorr={metrics['pixcorr']:.4f} "
            f"SSIM={metrics['ssim']:.4f} CLIP={metrics['clip_cosine']:.4f}"
        )

    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[OK] wrote {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
