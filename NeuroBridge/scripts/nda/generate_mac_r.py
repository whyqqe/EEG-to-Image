#!/usr/bin/env python3
"""MAC-R P1: Manifold-Aware Condition Routing on NDA-SS + CPA prompts.

Does NOT replace mem⊕decode or force CFM transport. Adds:
  1) Confidence router from CPA margins → text / IP / strength
  2) Stage-wise assembly: structure-preserving pass → semantic+text pass
  3) Optional MindEye-style low-level fuse with blurred neighbor
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

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


def _concept_name(prompt: str) -> str:
    p = prompt.strip()
    if p.startswith("a photo of "):
        p = p[len("a photo of ") :]
    return p.split(",")[0].strip()


def route_sample(
    margin: float,
    prompt: str,
    gate: float,
    soft_gate: float,
    fuse_beta: float,
) -> dict:
    """Confidence router: abstain on text when margin low; keep IP/structure alive."""
    name = _concept_name(prompt) if str(prompt).strip() else ""
    # gate < 0 disables abstention (light / always-text mode)
    if gate >= 0 and (margin < gate or not name):
        return {
            "prompt_a": "",
            "prompt_b": "",
            "ip_a": 0.75,
            "ip_b": 1.00,
            "strength_a": 0.28,
            "strength_b": 0.38,
            "fuse_beta": min(fuse_beta, 0.82),
            "mode": "abstain",
        }
    if soft_gate >= 0 and margin < soft_gate:
        return {
            "prompt_a": "",
            "prompt_b": f"a photo of {name}",
            "ip_a": 0.70,
            "ip_b": 0.92,
            "strength_a": 0.30,
            "strength_b": 0.42,
            "fuse_beta": fuse_beta,
            "mode": "soft",
        }
    # strong / always: mild structure pass + CPA text
    # strength_a must be >= ~1/steps for SDXL-Turbo (4 steps) or img2img collapses to 0 steps
    return {
        "prompt_a": "",
        "prompt_b": f"a photo of {name}, highly detailed",
        "ip_a": 0.55,
        "ip_b": 0.90,
        "strength_a": 0.30,
        "strength_b": 0.45,
        "fuse_beta": fuse_beta,
        "mode": "strong",
    }


def edge_boost(img: Image.Image, amount: float = 0.35) -> Image.Image:
    """Cheap structure prior: blend Canny-like edges into RGB init (no ControlNet)."""
    gray = img.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.SMOOTH)
    edges_rgb = Image.merge("RGB", (edges, edges, edges))
    return Image.blend(img, edges_rgb, amount)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-npy", type=str, required=True)
    ap.add_argument("--neighbor-idx-npy", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--images-root", type=str, default="/project/peilab/why/data/images_set")
    ap.add_argument("--prompts-json", type=str, required=True)
    ap.add_argument("--margins-npy", type=str, default="")
    ap.add_argument("--margin-gate", type=float, default=0.02)
    ap.add_argument("--soft-gate", type=float, default=0.05)
    ap.add_argument("--fuse-beta", type=float, default=0.85, help="weight on semantic image; rest = blurred neighbor")
    ap.add_argument("--enable-fuse", action="store_true")
    ap.add_argument("--edge-boost", type=float, default=0.25)
    ap.add_argument("--gen-steps", type=int, default=0)
    ap.add_argument("--gen-guidance", type=float, default=1.5)
    ap.add_argument("--negative-prompt", type=str, default="blurry, low quality, distorted, watermark")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", type=str, default="mac_r")
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--skip-metrics", action="store_true")
    args = ap.parse_args()

    cache = Path(os.environ.get("HF_HUB_CACHE", "/project/peilab/why/cache/eeg-brainit/hf/hub"))
    os.environ.setdefault("HF_HOME", str(cache.parent))
    os.environ.setdefault("OPENCLIP_CACHE_DIR", str(cache.parent / "open_clip"))
    os.environ.setdefault("TORCH_HOME", str(cache.parent / "torch"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embeds = l2(np.load(args.embed_npy).astype(np.float32))
    neighbors = np.load(args.neighbor_idx_npy)
    train_paths = list_train_images(Path(args.images_root))
    prompts = json.loads(Path(args.prompts_json).read_text(encoding="utf-8"))
    if len(prompts) < len(embeds):
        raise ValueError(f"prompts {len(prompts)} < embeds {len(embeds)}")
    if args.margins_npy:
        margins = np.load(args.margins_npy).astype(np.float32).reshape(-1)
    else:
        margins = np.full(len(embeds), 0.1, dtype=np.float32)

    out_dir = Path(args.output_dir)
    gen_dir = out_dir / "generated"
    gen_dir.mkdir(parents=True, exist_ok=True)
    n = len(embeds) if args.max_images <= 0 else min(len(embeds), args.max_images)

    route_log = []
    if all((gen_dir / f"{i:03d}.png").is_file() for i in range(n)):
        print(f"[INFO] skip gen, {n} images exist")
    else:
        from diffusers import StableDiffusionXLImg2ImgPipeline

        model_id, defaults = resolve_model(cache)
        steps = args.gen_steps if args.gen_steps > 0 else int(defaults["steps"])
        guidance = float(args.gen_guidance)
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

        g = torch.Generator(device=device).manual_seed(args.seed)

        def clamp_strength(strength: float) -> float:
            # SDXL-Turbo uses few steps; strength too low → 0 denoising steps → VAE crash
            min_s = (1.0 / max(steps, 1)) + 1e-3
            return float(min(0.99, max(float(strength), min_s)))

        def run_pass(init, emb, prompt, ip_scale, strength):
            pipe.set_ip_adapter_scale(float(ip_scale))
            if guidance > 1.0:
                ip_emb = torch.cat([torch.zeros_like(emb), emb], dim=0).unsqueeze(1)
            else:
                ip_emb = emb.unsqueeze(1)
            neg = args.negative_prompt if prompt else ""
            s = clamp_strength(strength)
            out = pipe(
                prompt=prompt or "",
                negative_prompt=neg,
                image=init,
                strength=s,
                ip_adapter_image_embeds=[ip_emb],
                num_inference_steps=steps,
                guidance_scale=guidance if prompt else min(guidance, 1.0),
                generator=g,
            )
            return out.images[0]

        for i in tqdm(range(n), desc="mac-r"):
            path = gen_dir / f"{i:03d}.png"
            if path.is_file():
                continue
            nb = int(neighbors[i, 0] if neighbors.ndim > 1 else neighbors[i])
            neigh = Image.open(train_paths[nb]).convert("RGB").resize((512, 512), Image.Resampling.BICUBIC)
            init = edge_boost(neigh, args.edge_boost) if args.edge_boost > 0 else neigh
            r = route_sample(
                float(margins[i]),
                prompts[i],
                args.margin_gate,
                args.soft_gate,
                args.fuse_beta,
            )
            route_log.append({"i": i, "margin": float(margins[i]), **{k: r[k] for k in r if k != "prompt_b"}, "prompt_b": r["prompt_b"][:80]})
            emb = torch.from_numpy(embeds[i : i + 1]).to(device=device, dtype=pipe.dtype)

            # Stage A: structure-preserving (early manifold) — no text
            mid = run_pass(init, emb, r["prompt_a"], r["ip_a"], r["strength_a"])
            # Stage B: semantic + gated concept text
            final = run_pass(mid, emb, r["prompt_b"], r["ip_b"], r["strength_b"])

            if args.enable_fuse:
                blur = neigh.filter(ImageFilter.GaussianBlur(radius=2))
                final = Image.blend(blur, final, float(r["fuse_beta"]))
            final.save(path)

        del pipe
        if device.type == "cuda":
            torch.cuda.empty_cache()
        (out_dir / "route_log.json").write_text(json.dumps(route_log, indent=2), encoding="utf-8")

    paths = sorted(gen_dir.glob("*.png"))[:n]
    report = {
        "tag": args.tag,
        "architecture": "MAC-R P1 (NDA-SS + CPA + confidence router + stage-wise assembly)",
        "embed_npy": args.embed_npy,
        "prompts_json": args.prompts_json,
        "margins_npy": args.margins_npy or None,
        "enable_fuse": bool(args.enable_fuse),
        "fuse_beta": args.fuse_beta,
        "edge_boost": args.edge_boost,
        "margin_gate": args.margin_gate,
        "soft_gate": args.soft_gate,
        "n_gen": len(paths),
    }
    if not args.skip_metrics:
        gt = list_test_images(Path(args.images_root))
        metrics = image_metrics(paths, gt[: len(paths)], device)
        report["metrics"] = metrics
        print(f"[METRICS] CLIP={metrics['clip_cosine']:.4f} SSIM={metrics['ssim']:.4f}")
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[OK] {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
