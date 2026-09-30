#!/usr/bin/env python3
"""ATM-aligned decode with HCMA semantics (no new claim / no SDXL FT).

Dual-stream (ATM Stage-II spirit):
  - Semantics: HCMA IP embeds + HCMA prompts  (unchanged innovation)
  - Structure: EEG→VAE low-level as img2img init (ATM-style inject)

Modes:
  sdedit  — standard SDEdit from VAE-decoded / RGB low-level (preferred; safer semantics)
  atmexact — ATM custom: scaled_latent + N(0,1), then truncated timesteps

Default guidance/steps match HCMA SDXL-base (not turbo) to avoid semantic collapse.
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


def decode_vae_latents(vae, latents: torch.Tensor, scaling: float) -> list[Image.Image]:
    # latents are stored scaled (mean * scaling)
    with torch.no_grad():
        x = (latents.float() / scaling).to(dtype=vae.dtype)
        imgs = vae.decode(x).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
    out = []
    for i in range(imgs.shape[0]):
        arr = (imgs[i].detach().permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)
        out.append(Image.fromarray(arr))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="sdedit", choices=["sdedit", "atmexact"])
    ap.add_argument("--embed-npy", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--tag", type=str, default="atm_aligned")
    ap.add_argument("--prompts-json", type=str, default="")
    ap.add_argument("--vae-latent-npy", type=str, default="", help="scaled VAE latents (N,4,64,64)")
    ap.add_argument("--lowlevel-rgb-dir", type=str, default="", help="optional RGB init {i:03d}.png")
    ap.add_argument("--strength", type=float, default=0.80, help="high=more semantic denoising")
    ap.add_argument("--ip-scale", type=float, default=1.0)
    # ---- M4 per-row arbitration.  The learned-arbitration mechanism predicts a
    # strength and an IP-Adapter scale for EVERY row from that row's own reliability,
    # so the generator has to be able to read one pair per row.  Without these two
    # the mechanism can only be evaluated as a scalar and the whole point -- that the
    # balance should vary with the neural evidence -- cannot be tested at all.
    #
    # The scalars above remain the value used when the file is absent, so every
    # existing row in the pipeline is byte-identical to before.
    ap.add_argument("--strength-npy", type=str, default="",
                    help="(N,) per-row img2img strength.  Overrides --strength.")
    ap.add_argument("--ip-scale-npy", type=str, default="",
                    help="(N,) per-row IP-Adapter scale.  Overrides --ip-scale.")
    ap.add_argument("--gen-steps", type=int, default=28)
    ap.add_argument("--gen-guidance", type=float, default=5.0)
    ap.add_argument("--gen-size", type=int, default=512)
    ap.add_argument("--scaling-factor", type=float, default=0.13025)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--negative-prompt", type=str, default="blurry, low quality, distorted, watermark")
    ap.add_argument("--device", type=str, default="",
                    help="explicit device (e.g. cuda:0). Empty = auto-detect, which is "
                         "the historical behaviour. Every other script in the chain is "
                         "driven with an explicit device, so this exists to keep the "
                         "orchestration uniform.")
    args = ap.parse_args()

    if not args.vae_latent_npy and not args.lowlevel_rgb_dir:
        raise ValueError("need --vae-latent-npy and/or --lowlevel-rgb-dir")

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
    vae_lat = np.load(args.vae_latent_npy) if args.vae_latent_npy else None
    if vae_lat is not None and len(vae_lat) < n:
        raise ValueError(f"vae latent n={len(vae_lat)} < {n}")
    rgb_dir = Path(args.lowlevel_rgb_dir) if args.lowlevel_rgb_dir else None

    # ---- M4: per-row generation parameters, loaded once and length-checked.
    str_row = np.load(args.strength_npy).astype(np.float32) if args.strength_npy else None
    ips_row = np.load(args.ip_scale_npy).astype(np.float32) if args.ip_scale_npy else None
    for _nm, _a in (("strength", str_row), ("ip-scale", ips_row)):
        if _a is not None and len(_a) < n:
            raise ValueError(f"--{_nm}-npy has {len(_a)} rows but {n} images are being "
                             f"generated; a shorter file would silently apply the "
                             f"wrong row's parameter to the tail.")
    if str_row is not None:
        print(f"[INFO] per-row strength: mean {str_row[:n].mean():.4f} "
              f"sd {str_row[:n].std():.4f} [{str_row[:n].min():.3f}, {str_row[:n].max():.3f}]")
    if ips_row is not None:
        print(f"[INFO] per-row ip_scale: mean {ips_row[:n].mean():.4f} "
              f"sd {ips_row[:n].std():.4f} [{ips_row[:n].min():.3f}, {ips_row[:n].max():.3f}]")

    if all((gen_dir / f"{i:03d}.png").is_file() for i in range(n)):
        print(f"[SKIP] n={n}")
    else:
        from diffusers import StableDiffusionXLImg2ImgPipeline, StableDiffusionXLPipeline

        # `--device` exists so the orchestrating pipeline can name the device
        # explicitly, exactly as every other script in the chain does.  Without
        # it a caller that passes `--device` gets `error: unrecognized arguments`
        # and the row silently produces zero images (which is what happened to
        # five of the eight TDM-DT rows before this was added).
        if args.device:
            device = torch.device(args.device)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        sdxl = resolve_sdxl_model_path(cache)
        print(f"[INFO] mode={args.mode} sdxl={sdxl} strength={args.strength}")

        def load_ip(pipe):
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
            return pipe

        g = torch.Generator(device=device).manual_seed(args.seed)

        if args.mode == "sdedit":
            pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                sdxl,
                torch_dtype=dtype,
                variant="fp16" if device.type == "cuda" else None,
                use_safetensors=True,
                local_files_only=Path(str(sdxl)).is_dir(),
            ).to(device)
            pipe = load_ip(pipe)
            # optional decode batch from vae latents
            init_imgs: list[Image.Image | None] = [None] * n
            if rgb_dir is not None:
                for i in range(n):
                    init_imgs[i] = Image.open(rgb_dir / f"{i:03d}.png").convert("RGB").resize(
                        (args.gen_size, args.gen_size), Image.Resampling.BICUBIC
                    )
            elif vae_lat is not None:
                vae = pipe.vae
                bs = 8
                for start in tqdm(range(0, n, bs), desc="decode-vae-init"):
                    chunk = torch.from_numpy(vae_lat[start : start + bs].astype(np.float32)).to(device)
                    imgs = decode_vae_latents(vae, chunk, args.scaling_factor)
                    for j, im in enumerate(imgs):
                        init_imgs[start + j] = im.resize(
                            (args.gen_size, args.gen_size), Image.Resampling.BICUBIC
                        )

            for i in tqdm(range(n), desc="sdedit-atm-aligned"):
                path = gen_dir / f"{i:03d}.png"
                if path.is_file():
                    continue
                assert init_imgs[i] is not None
                emb = torch.from_numpy(embeds[i : i + 1]).to(device=device, dtype=pipe.dtype)
                if args.gen_guidance > 1.0:
                    ip_emb = torch.cat([torch.zeros_like(emb), emb], dim=0).unsqueeze(1)
                else:
                    ip_emb = emb.unsqueeze(1)
                prompt = prompts[i] if prompts is not None else ""
                neg = args.negative_prompt if prompt else ""
                # M4: this row's own pair, falling back to the scalar when absent.
                _str_i = float(np.clip(str_row[i] if str_row is not None else args.strength,
                                       0.05, 0.95))
                _ips_i = float(ips_row[i] if ips_row is not None else args.ip_scale)
                # `set_ip_adapter_scale` only rewrites `scale` on the attention
                # processors that already exist -- it does not rebuild them -- so the
                # per-row call is an attribute assignment, not a pipeline reload.
                if ips_row is not None:
                    pipe.set_ip_adapter_scale(_ips_i)
                img = pipe(
                    prompt=prompt or "",
                    negative_prompt=neg,
                    image=init_imgs[i],
                    strength=_str_i,
                    ip_adapter_image_embeds=[ip_emb],
                    num_inference_steps=int(args.gen_steps),
                    guidance_scale=float(args.gen_guidance),
                    generator=g,
                ).images[0]
                img.save(path)
            del pipe
        else:
            # ATM-exact: latent + N(0,1), truncated timesteps via img2img-equivalent loop on SDXL txt2img guts
            # Practical equivalent used here: decode (latent+noise) is wrong; instead run custom step truncate
            # by using Img2Img after synthesizing an init from latent then relying on strength — for true ATM:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                sdxl,
                torch_dtype=dtype,
                variant="fp16" if device.type == "cuda" else None,
                use_safetensors=True,
                local_files_only=Path(str(sdxl)).is_dir(),
            ).to(device)
            pipe = load_ip(pipe)
            # Bind ATM generate helper from official repo
            gen_dir_atm = Path(BRAINIT) / "third_party" / "EEG_Image_decode" / "Generation"
            sys.path.insert(0, str(gen_dir_atm))
            from custom_pipeline_low_level import generate_ip_adapter_embeds  # type: ignore

            pipe.generate_ip_adapter_embeds = generate_ip_adapter_embeds.__get__(pipe, type(pipe))
            scaling = float(args.scaling_factor)
            for i in tqdm(range(n), desc="atmexact"):
                path = gen_dir / f"{i:03d}.png"
                if path.is_file():
                    continue
                emb = torch.from_numpy(embeds[i : i + 1]).to(device=device, dtype=pipe.dtype)
                prompt = prompts[i] if prompts is not None else ""
                # Our latents are already scaled; ATM helper multiplies by scaling again for low_level_latent.
                # Pass unscaled = scaled / scaling.
                assert vae_lat is not None
                ll = torch.from_numpy(vae_lat[i : i + 1].astype(np.float32)).to(device=device, dtype=pipe.dtype)
                ll_unscaled = ll / scaling
                # Official ATM often uses guidance 0 on turbo; we keep HCMA CFG for semantics.
                # Their API expects ip_adapter_embeds as (1,1024) tensor.
                out_img = pipe.generate_ip_adapter_embeds(
                    prompt=prompt or "",
                    negative_prompt=args.negative_prompt if prompt else "",
                    ip_adapter_embeds=emb,
                    num_inference_steps=int(args.gen_steps),
                    guidance_scale=float(args.gen_guidance),
                    generator=g,
                    img2img_strength=float(np.clip(args.strength, 0.05, 1.0)),
                    low_level_latent=ll_unscaled,
                    height=args.gen_size,
                    width=args.gen_size,
                ).images[0]
                out_img.save(path)
            del pipe

        if device.type == "cuda":
            torch.cuda.empty_cache()

    report = {
        "tag": args.tag,
        "mode": args.mode,
        "strength": args.strength,
        "ip_scale": args.ip_scale,
        "strength_npy": args.strength_npy or None,
        "ip_scale_npy": args.ip_scale_npy or None,
        "strength_used_mean": (float(str_row[:n].mean()) if str_row is not None
                               else float(args.strength)),
        "ip_scale_used_mean": (float(ips_row[:n].mean()) if ips_row is not None
                               else float(args.ip_scale)),
        "gen_steps": args.gen_steps,
        "gen_guidance": args.gen_guidance,
        "vae_latent_npy": args.vae_latent_npy or None,
        "lowlevel_rgb_dir": args.lowlevel_rgb_dir or None,
        "note": "ATM-aligned dual-stream decode; HCMA IP/prompts preserved; no SDXL FT; not a new method claim",
        "n_gen": n,
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
