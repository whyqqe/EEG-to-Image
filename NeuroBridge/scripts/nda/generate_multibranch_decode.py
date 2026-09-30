#!/usr/bin/env python3
"""Multi-branch IP-Adapter decoder (CogCapPro-style parallel injection).

WHAT CHANGES VS generate_hcma_s_decode.py
-----------------------------------------
That script chains structure: depth enters as a ControlNet image, low-level as an
SDEdit init, and only ONE CLIP embedding drives IP-Adapter.

CogCapPro (arXiv:2603.12722) injects image + depth + edge CLIP embeddings as
PARALLEL IP-Adapter branches; their ablation shows depth+edge are what lift SSIM
(image-only 0.317 -> all 0.398). This script implements that.

MECHANISM (verified against diffusers 0.30 source)
--------------------------------------------------
We register N IP-Adapters on the UNet (`load_ip_adapter(..., weight_name=[w]*N)`),
which gives `unet.encoder_hid_proj.image_projection_layers` of length N, and pass
`ip_adapter_image_embeds` as a list of N tensors, each `(batch, 1, 1024)`.

NOTE: passing ONE adapter a `(batch, N, 1024)` tensor does NOT work -- the
projection reshapes to `(batch, num_image_text_embeds, -1)` and would mix tokens.
This is why N adapter layers are registered instead. Per-branch strength is set
through `set_ip_adapter_scale([s1, ..., sN])`.

PIPELINES
---------
  base  : StableDiffusionXLControlNetImg2ImgPipeline (28 steps, guidance 5) [+CN+init]
  turbo : StableDiffusionXLPipeline                  (5 steps,  guidance 0) [no CN/init]
Turbo follows CogCapPro's setting and is ~5x cheaper, which matters for best-of-N.

GUARD
-----
`--branch-sanity N` renders N rows with 1 branch vs all branches and reports the
mean abs pixel difference, so a silent no-op is caught instead of silently
producing identical rows.
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
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)


def resolve_turbo_path(hub: Path) -> str | None:
    root = hub / "models--stabilityai--sdxl-turbo" / "snapshots"
    if root.is_dir():
        for snap in sorted(root.iterdir(), reverse=True):
            if (snap / "model_index.json").is_file():
                return str(snap)
    return None


def resolve_controlnet_path(hub: Path) -> str:
    root = hub / "models--diffusers--controlnet-depth-sdxl-1.0" / "snapshots"
    if root.is_dir():
        for snap in sorted(root.iterdir(), reverse=True):
            if (snap / "config.json").is_file():
                return str(snap)
    return "diffusers/controlnet-depth-sdxl-1.0"


def load_pipeline(kind: str, cache: Path, n_branches: int, scales: list[float],
                  want_cn: bool, device, dtype):
    """Build the pipeline with `n_branches` parallel IP-Adapters. Returns (pipe, has_cn)."""
    if kind == "turbo":
        from diffusers import StableDiffusionXLPipeline
        path = resolve_turbo_path(cache) or "stabilityai/sdxl-turbo"
        print(f"[INFO] sdxl-turbo = {path}")
        pipe = StableDiffusionXLPipeline.from_pretrained(
            path, torch_dtype=dtype, variant="fp16" if dtype == torch.float16 else None,
            use_safetensors=True, local_files_only=Path(str(path)).is_dir(),
        ).to(device)
        has_cn = False
    elif kind == "base":
        sdxl = resolve_sdxl_model_path(cache)
        print(f"[INFO] sdxl = {sdxl}")
        if want_cn:
            from diffusers import ControlNetModel, StableDiffusionXLControlNetImg2ImgPipeline
            cn_path = resolve_controlnet_path(cache)
            print(f"[INFO] controlnet = {cn_path}")
            controlnet = ControlNetModel.from_pretrained(
                cn_path, torch_dtype=dtype, local_files_only=Path(str(cn_path)).is_dir())
            pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
                sdxl, controlnet=controlnet, torch_dtype=dtype,
                variant="fp16" if dtype == torch.float16 else None,
                use_safetensors=True, local_files_only=Path(str(sdxl)).is_dir(),
            ).to(device)
            has_cn = True
        else:
            from diffusers import StableDiffusionXLPipeline
            pipe = StableDiffusionXLPipeline.from_pretrained(
                sdxl, torch_dtype=dtype,
                variant="fp16" if dtype == torch.float16 else None,
                use_safetensors=True, local_files_only=Path(str(sdxl)).is_dir(),
            ).to(device)
            has_cn = False
    else:
        raise SystemExit(f"[FATAL] unknown --pipeline {kind}")

    ip_root = resolve_ip_adapter_dir(cache)
    kwargs = {"subfolder": "sdxl_models", "image_encoder_folder": None, "local_files_only": True}
    loaded = None
    for weight_name in ("ip-adapter_sdxl_vit-h.bin", "ip-adapter_sdxl.bin"):
        try:
            pipe.load_ip_adapter(str(ip_root), weight_name=[weight_name] * n_branches, **kwargs)
            loaded = weight_name
            break
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] {weight_name} x{n_branches} failed: {exc}")
    if loaded is None:
        raise SystemExit("[FATAL] could not load IP-Adapter weights")
    n_layers = len(pipe.unet.encoder_hid_proj.image_projection_layers)
    print(f"[INFO] ip-adapter = {loaded} x{n_branches} (registered layers = {n_layers})")
    if n_layers != n_branches:
        raise SystemExit(f"[FATAL] registered {n_layers} adapters != {n_branches} branches")
    pipe.set_ip_adapter_scale([float(s) for s in scales])
    print(f"[INFO] ip scales = {scales}")
    return pipe, has_cn


def build_ip_embeds(branches: list[np.ndarray], i: int, device, dtype,
                    do_cfg: bool) -> list[torch.Tensor]:
    """One (batch,1,D) tensor per branch.

    CFG duplicates the batch dim (2,1,D) so the pipeline can chunk it into
    negative/positive halves. Per-branch strength is handled by set_ip_adapter_scale.
    """
    out: list[torch.Tensor] = []
    for arr in branches:
        v = torch.from_numpy(arr[i:i + 1]).to(device=device, dtype=dtype).unsqueeze(0)  # (1,1,D)
        if do_cfg:
            v = torch.cat([torch.zeros_like(v), v], dim=0)  # (2,1,D)
        out.append(v)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond-npys", type=str, required=True,
                    help="comma-separated per-branch condition npy (row order must match)")
    ap.add_argument("--branch-scales", type=str, default="",
                    help="comma-separated per-branch IP strength (default all 1.0)")
    ap.add_argument("--prompts-json", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--tag", type=str, default="mb")
    ap.add_argument("--pipeline", type=str, default="turbo", choices=["turbo", "base"])
    ap.add_argument("--depth-rgb-dir", type=str, default="")
    ap.add_argument("--lowlevel-rgb-dir", type=str, default="")
    ap.add_argument("--cn-scale", type=float, default=0.40)
    ap.add_argument("--strength", type=float, default=0.82)
    ap.add_argument("--gen-steps", type=int, default=5)
    ap.add_argument("--gen-guidance", type=float, default=0.0)
    ap.add_argument("--gen-size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--negative-prompt", type=str, default="blurry, low quality, distorted, watermark")
    ap.add_argument("--branch-sanity", type=int, default=0,
                    help="if >0, render this many rows with 1 vs N branches and report pixel diff")
    ap.add_argument("--sanity-only", action="store_true",
                    help="skip normal generation; only run --branch-sanity")
    ap.add_argument("--sanity-out", type=str, default="")
    args = ap.parse_args()

    cache = Path(os.environ.get("HF_HUB_CACHE", "/project/peilab/why/cache/eeg-brainit/hf/hub"))
    os.environ.setdefault("HF_HOME", str(cache.parent))
    os.environ.setdefault("OPENCLIP_CACHE_DIR", str(cache.parent / "open_clip"))
    os.environ.setdefault("TORCH_HOME", str(cache.parent / "torch"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    paths = [p for p in args.cond_npys.split(",") if p.strip()]
    branches = [l2(np.load(p).astype(np.float32)) for p in paths]
    n_rows = len(branches[0])
    for p, b in zip(paths, branches):
        if len(b) != n_rows:
            raise SystemExit(f"[FATAL] {p} rows {len(b)} != {n_rows}")

    scales = ([float(x) for x in args.branch_scales.split(",") if x.strip()]
              if args.branch_scales else [1.0] * len(branches))
    if len(scales) != len(branches):
        raise SystemExit(f"[FATAL] {len(scales)} scales vs {len(branches)} branches")

    prompts = json.loads(Path(args.prompts_json).read_text(encoding="utf-8"))
    if len(prompts) < n_rows:
        raise SystemExit(f"[FATAL] prompts {len(prompts)} < rows {n_rows}")

    want_cn = bool(args.depth_rgb_dir) and bool(args.lowlevel_rgb_dir)
    if args.pipeline == "turbo" and want_cn:
        print("[WARN] turbo pipeline ignores ControlNet/init; falling back to text2img")
        want_cn = False

    out_dir = Path(args.output_dir)
    gen_dir = out_dir / "generated"
    gen_dir.mkdir(parents=True, exist_ok=True)
    n = n_rows if args.max_images <= 0 else min(n_rows, args.max_images)
    do_cfg = float(args.gen_guidance) > 1.0

    print(f"[INFO] pipeline={args.pipeline} branches={len(branches)} scales={scales} "
          f"rows={n} steps={args.gen_steps} guidance={args.gen_guidance} cfg={do_cfg} cn={want_cn}")

    if all((gen_dir / f"{i:03d}.png").is_file() for i in range(n)) or args.sanity_only:
        print(f"[INFO] skip gen ({'sanity-only' if args.sanity_only else f'{n} images exist'})")
    else:
        pipe, has_cn = load_pipeline(args.pipeline, cache, len(branches), scales,
                                     want_cn, device, dtype)
        g = torch.Generator(device=device).manual_seed(args.seed)
        for i in tqdm(range(n), desc=f"mb[{args.tag}]"):
            path = gen_dir / f"{i:03d}.png"
            if path.is_file():
                continue
            prompt = prompts[i] if prompts[i] else ""
            neg = args.negative_prompt if prompt else ""
            kw: dict = dict(
                prompt=prompt,
                negative_prompt=neg,
                ip_adapter_image_embeds=build_ip_embeds(branches, i, device, pipe.dtype, do_cfg),
                num_inference_steps=int(args.gen_steps),
                guidance_scale=float(args.gen_guidance),
                generator=g,
                height=int(args.gen_size),
                width=int(args.gen_size),
            )
            if has_cn:
                dp = Path(args.depth_rgb_dir) / f"{i:03d}.png"
                lp = Path(args.lowlevel_rgb_dir) / f"{i:03d}.png"
                if not dp.is_file() or not lp.is_file():
                    raise FileNotFoundError(f"{dp} / {lp}")
                kw["image"] = Image.open(lp).convert("RGB").resize(
                    (args.gen_size, args.gen_size), Image.Resampling.BICUBIC)
                kw["control_image"] = Image.open(dp).convert("RGB").resize(
                    (args.gen_size, args.gen_size), Image.Resampling.BICUBIC)
                kw["strength"] = float(args.strength)
                kw["controlnet_conditioning_scale"] = float(args.cn_scale)
            pipe(**kw).images[0].save(path)
        del pipe
        if device.type == "cuda":
            torch.cuda.empty_cache()

    sanity = None
    if args.branch_sanity > 0:
        sanity = run_branch_sanity(args, branches, scales, prompts, cache, device, dtype)
        if args.sanity_out:
            Path(args.sanity_out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.sanity_out).write_text(json.dumps(sanity, indent=2), encoding="utf-8")
        print("[sanity]", json.dumps(sanity, indent=2))

    report = {
        "tag": args.tag,
        "pipeline": f"multi-branch IP ({args.pipeline})",
        "cond_npys": paths,
        "branch_scales": scales,
        "n_branches": len(branches),
        "prompts_json": args.prompts_json,
        "gen_steps": args.gen_steps,
        "gen_guidance": args.gen_guidance,
        "cn_scale": args.cn_scale if want_cn else None,
        "strength": args.strength if want_cn else None,
        "n_gen": n,
        "branch_sanity": sanity,
        "note": "N registered IP-Adapters; per-branch strength via set_ip_adapter_scale",
    }
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2)[:1000])


def _render(pipe, args, branches, i, prompt, device, do_cfg):
    do_cfg = float(args.gen_guidance) > 1.0
    g = torch.Generator(device=device).manual_seed(args.seed)
    return np.asarray(pipe(
        prompt=prompt, negative_prompt="",
        ip_adapter_image_embeds=build_ip_embeds(branches, i, device, pipe.dtype, do_cfg),
        num_inference_steps=int(args.gen_steps),
        guidance_scale=float(args.gen_guidance),
        generator=g, height=args.gen_size, width=args.gen_size,
    ).images[0].convert("RGB"), dtype=np.float32)


def run_branch_sanity(args, branches, scales, prompts, cache, device, dtype) -> dict:
    """Render first rows with 1 branch vs all branches on ONE pipeline (scales per-branch).

    Uses a single N-adapter pipeline and zeroes the non-primary scales for the
    1-branch case, so the comparison isolates the effect of the extra branches.
    """
    n = min(int(args.branch_sanity), len(branches[0]))
    do_cfg = float(args.gen_guidance) > 1.0
    pipe, _ = load_pipeline(args.pipeline, cache, len(branches), scales, False, device, dtype)

    diffs: list[float] = []
    for i in range(n):
        pipe.set_ip_adapter_scale([1.0] + [0.0] * (len(branches) - 1))
        a = _render(pipe, args, branches, i, prompts[i] if prompts[i] else "", device, do_cfg)
        pipe.set_ip_adapter_scale([float(s) for s in scales])
        b = _render(pipe, args, branches, i, prompts[i] if prompts[i] else "", device, do_cfg)
        diffs.append(float(np.abs(a - b).mean()))
    pipe.set_ip_adapter_scale([float(s) for s in scales])

    emb_diff = float("nan")
    if len(branches) > 1:
        emb_diff = float(np.mean([
            np.abs(branches[0][i] * scales[0] - branches[-1][i] * scales[-1]).mean()
            for i in range(n)
        ]))
    del pipe
    if device.type == "cuda":
        torch.cuda.empty_cache()
    mean_diff = float(np.mean(diffs))
    return {
        "rows_checked": n,
        "pixel_diff_1branch_vs_Nbranch_mean": mean_diff,
        "pixel_diff_per_row": diffs,
        "emb_diff_first_vs_last_branch_mean": emb_diff,
        "verdict": ("BRANCHES_ACTIVE" if mean_diff > 0.5 else
                    "SUSPECT_NOOP: 1-branch and N-branch renders nearly identical"),
    }


if __name__ == "__main__":
    main()
