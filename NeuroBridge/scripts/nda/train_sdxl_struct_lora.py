#!/usr/bin/env python3
"""Phase-B: SDXL UNet LoRA (fp32-safe) — blur SDEdit structure + base distill.

No IP-Adapter during training (loaded only at inference with EEG embeds).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from tqdm import tqdm

BRAINIT = Path("/project/peilab/why/eeg-brainit")
sys.path.insert(0, str(BRAINIT / "scripts"))
sys.path.insert(0, str(BRAINIT / "src"))

from eval_atm_pipeline import resolve_sdxl_model_path  # type: ignore


def list_train_images(images_root: Path) -> list[Path]:
    root = images_root / "training_images"
    paths: list[Path] = []
    for d in sorted([p for p in root.iterdir() if p.is_dir()]):
        imgs = sorted(list(d.glob("*.jpg")) + list(d.glob("*.JPEG")) + list(d.glob("*.png")))
        paths.extend(imgs)
    return paths


def encode_prompt_empty(pipe, device, dtype):
    prompt_embeds, _, pooled, _ = pipe.encode_prompt(
        prompt="",
        prompt_2=None,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
        negative_prompt=None,
    )
    return prompt_embeds.to(dtype=dtype), pooled.to(dtype=dtype)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--images-root", type=str, default="/project/peilab/why/data/images_set")
    ap.add_argument("--n-train", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-rank", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=8)
    ap.add_argument("--gen-size", type=int, default=512)
    ap.add_argument("--blur-radius", type=float, default=6.0)
    ap.add_argument("--strength-min", type=float, default=0.26)
    ap.add_argument("--strength-max", type=float, default=0.42)
    ap.add_argument("--noise-frac", type=float, default=0.30)
    ap.add_argument("--lambda-distill", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--clip-cache-npy", type=str, default="")
    ap.add_argument("--paths-cache-json", type=str, default="")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path(os.environ.get("HF_HUB_CACHE", "/project/peilab/why/cache/eeg-brainit/hf/hub"))
    os.environ.setdefault("HF_HOME", str(cache.parent))
    os.environ.setdefault("OPENCLIP_CACHE_DIR", str(cache.parent / "open_clip"))
    os.environ.setdefault("TORCH_HOME", str(cache.parent / "torch"))
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)

    paths_json = Path(args.paths_cache_json) if args.paths_cache_json else out / "train_paths.json"
    if paths_json.is_file():
        paths = [Path(p) for p in json.loads(paths_json.read_text(encoding="utf-8"))]
    else:
        all_paths = list_train_images(Path(args.images_root))
        idx = rng.choice(len(all_paths), size=min(args.n_train, len(all_paths)), replace=False)
        paths = [all_paths[int(i)] for i in idx]
        paths_json.write_text(json.dumps([str(p) for p in paths], indent=2), encoding="utf-8")
    print(f"[OK] n_train_images={len(paths)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Full fp32 UNet train — H800 80GB OK; avoids peft+fp16 NaNs
    dtype = torch.float32

    from diffusers import DDPMScheduler, StableDiffusionXLPipeline
    from peft import LoraConfig, get_peft_model_state_dict

    sdxl = resolve_sdxl_model_path(cache)
    print(f"[INFO] sdxl={sdxl} dtype={dtype}")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        sdxl,
        torch_dtype=dtype,
        use_safetensors=True,
        local_files_only=Path(str(sdxl)).is_dir(),
    ).to(device)
    pipe.vae.to(dtype=torch.float32)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    if pipe.text_encoder_2 is not None:
        pipe.text_encoder_2.requires_grad_(False)

    unet = pipe.unet
    unet.requires_grad_(False)
    unet.add_adapter(
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
    )
    unet.train()
    params = [p for p in unet.parameters() if p.requires_grad]
    print(f"[INFO] trainable params={sum(p.numel() for p in params)/1e6:.2f}M")
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-2)
    noise_sched = DDPMScheduler.from_config(pipe.scheduler.config)

    prompt_embeds, pooled = encode_prompt_empty(pipe, device, dtype)
    t_ids = torch.tensor(
        [[args.gen_size, args.gen_size, 0, 0, args.gen_size, args.gen_size]],
        device=device,
        dtype=dtype,
    )

    losses: list[float] = []
    nan_skips = 0
    pbar = tqdm(range(args.steps), desc="lora-train")
    for step in pbar:
        i = int(rng.randint(0, len(paths)))
        gt = Image.open(paths[i]).convert("RGB").resize(
            (args.gen_size, args.gen_size), Image.Resampling.BICUBIC
        )
        use_noise = random.random() < args.noise_frac

        with torch.no_grad():
            gt_t = pipe.image_processor.preprocess(gt).to(device=device, dtype=dtype)
            gt_lat = pipe.vae.encode(gt_t).latent_dist.sample() * pipe.vae.config.scaling_factor
            if use_noise:
                # standard diffusion on GT
                t_start = int(rng.randint(20, 980))
                noise = torch.randn_like(gt_lat)
                timesteps = torch.tensor([t_start], device=device, dtype=torch.long)
                noisy = noise_sched.add_noise(gt_lat, noise, timesteps)
                target = noise
            else:
                # SDEdit from blur → recover GT (epsilon that yields GT x0)
                init = gt.filter(ImageFilter.GaussianBlur(radius=args.blur_radius))
                init_t = pipe.image_processor.preprocess(init).to(device=device, dtype=dtype)
                init_lat = pipe.vae.encode(init_t).latent_dist.sample() * pipe.vae.config.scaling_factor
                strength = float(rng.uniform(args.strength_min, args.strength_max))
                t_start = max(20, min(980, int(strength * 999)))
                timesteps = torch.tensor([t_start], device=device, dtype=torch.long)
                noise = torch.randn_like(gt_lat)
                a = noise_sched.alphas_cumprod[t_start].to(device=device, dtype=dtype)
                sa, sna = a.sqrt(), (1.0 - a).sqrt()
                noisy = sa * init_lat + sna * noise
                target = (noisy - sa * gt_lat) / sna.clamp(min=1e-4)

        add_kwargs = {"time_ids": t_ids, "text_embeds": pooled}
        unet.enable_adapters()
        pred = unet(noisy, timesteps, prompt_embeds, added_cond_kwargs=add_kwargs, return_dict=False)[0]

        if step < 3:
            print(
                f"[diag] step={step} mode={'noise' if use_noise else 'blur'} t={t_start} "
                f"pred_finite={bool(torch.isfinite(pred).all())} "
                f"tgt_finite={bool(torch.isfinite(target).all())} "
                f"pred_std={float(pred.std()):.4f} tgt_std={float(target.std()):.4f}"
            )

        loss_diff = F.mse_loss(pred, target)
        loss_dist = torch.zeros((), device=device, dtype=dtype)
        if args.lambda_distill > 0:
            with torch.no_grad():
                unet.disable_adapters()
                pred_base = unet(
                    noisy, timesteps, prompt_embeds, added_cond_kwargs=add_kwargs, return_dict=False
                )[0]
                unet.enable_adapters()
            w = 1.0 - float(t_start) / 999.0
            loss_dist = F.mse_loss(pred, pred_base) * (0.5 + 0.5 * w)

        loss = loss_diff + args.lambda_distill * loss_dist
        if not torch.isfinite(loss):
            nan_skips += 1
            opt.zero_grad(set_to_none=True)
            if nan_skips <= 8:
                print(
                    f"[WARN] nan step={step} pred_nan={int((~torch.isfinite(pred)).sum())} "
                    f"tgt_nan={int((~torch.isfinite(target)).sum())}"
                )
            if nan_skips > 50 and step < 150:
                raise RuntimeError(f"too many NaN losses early ({nan_skips}); abort")
            continue

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        losses.append(float(loss.detach().cpu()))
        if step % 20 == 0:
            pbar.set_postfix(
                loss=f"{losses[-1]:.4f}",
                diff=f"{float(loss_diff.detach()):.4f}",
                dist=f"{float(loss_dist.detach()):.4f}",
                mode=("noise" if use_noise else "blur"),
                nan=nan_skips,
            )

        if (step + 1) % args.save_every == 0 or (step + 1) == args.steps:
            ckpt = out / f"lora_step{step+1:05d}"
            ckpt.mkdir(parents=True, exist_ok=True)
            unet_lora = get_peft_model_state_dict(unet)
            try:
                from diffusers.utils import convert_state_dict_to_diffusers

                unet_lora = convert_state_dict_to_diffusers(unet_lora)
            except Exception:
                pass
            StableDiffusionXLPipeline.save_lora_weights(
                str(ckpt), unet_lora_layers=unet_lora, safe_serialization=True
            )
            unet.save_pretrained(str(ckpt / "unet_peft"))
            meta = {
                "step": step + 1,
                "loss_mean_last100": float(np.mean(losses[-100:])) if losses else None,
                "lora_rank": args.lora_rank,
                "n_train": len(paths),
                "dtype": "float32",
                "train_ip": False,
                "nan_skips": nan_skips,
            }
            (ckpt / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            print(f"[SAVE] {ckpt} loss100={meta['loss_mean_last100']}")

    if not losses:
        raise RuntimeError("no successful training steps")
    final = out / f"lora_step{args.steps:05d}"
    (out / "lora_final.txt").write_text(str(final) + "\n", encoding="utf-8")
    report = {
        "pipeline": "sdxl_struct_lora_v1",
        "train_ip": False,
        "dtype": "float32",
        "lora_final": str(final),
        "steps": args.steps,
        "n_train": len(paths),
        "nan_skips": nan_skips,
        "loss_mean_last100": float(np.mean(losses[-100:])),
        "loss_curve_tail": losses[-50:],
    }
    (out / "train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    del pipe
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
