#!/usr/bin/env python3
"""ERDC W12: Official ATM-style mixed pipeline (NeurIPS'24 ATM paper).

High-level CLIP embed (ATM / prior / bit_clip) + low-level neighbor structure
via official custom_pipeline_low_level (SDXL-Turbo + IP-Adapter + img2img),
then brain-consistency closed-loop selection.

Closest reproduction of the paper decoder stack without EEG->VAE latent model.
Low-level uses retrieved TRAIN neighbor image or VAE latent (no GT leakage).
"""

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

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "third_party" / "EEG_Image_decode" / "Generation"
if str(GEN) not in sys.path:
    sys.path.insert(0, str(GEN))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from custom_pipeline_low_level import (  # type: ignore
    generate_ip_adapter_embeds,
    StableDiffusionXL_lowlevel,
)
from diffusers import DiffusionPipeline, StableDiffusionXLPipeline

from eeg_brainit.models.atm_diffusion_prior import load_atm_prior
from eval_atm_pipeline import (  # type: ignore
    image_metrics,
    list_test_images,
    resolve_ip_adapter_dir,
    resolve_sdxl_model_path,
)
from erdc_ras_closed_loop import (  # type: ignore
    encode_clip,
    list_train_images,
    load_embed,
    retrieve_neighbors,
    select_and_eval,
)


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-8)


def resolve_turbo_path(hub: Path) -> str | None:
    snap_root = hub / "models--stabilityai--sdxl-turbo" / "snapshots"
    if snap_root.is_dir():
        for snap in sorted(snap_root.iterdir(), reverse=True):
            if (snap / "model_index.json").is_file():
                return str(snap)
    return None


@torch.no_grad()
def load_official_pipe(device: torch.device, use_turbo: bool = True) -> StableDiffusionXLPipeline:
    hub = Path(os.environ.get("HF_HUB_CACHE", "/project/peilab/why/cache/eeg-brainit/hf/hub"))
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    turbo = resolve_turbo_path(hub) if use_turbo else None
    if turbo:
        print(f"[INFO] loading SDXL-Turbo: {turbo}")
        pipe = DiffusionPipeline.from_pretrained(
            turbo,
            torch_dtype=dtype,
            variant="fp16",
            use_safetensors=True,
            local_files_only=True,
        )
    else:
        model_id = resolve_sdxl_model_path(hub)
        print(f"[INFO] loading SDXL base: {model_id}")
        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            variant="fp16" if device.type == "cuda" else None,
            use_safetensors=True,
            local_files_only=True,
        )
    pipe = pipe.to(device)
    StableDiffusionXLPipeline.prepare_latents_img2img = StableDiffusionXL_lowlevel.prepare_latents_img2img
    StableDiffusionXLPipeline.prepare_latents_latent2img = StableDiffusionXL_lowlevel.prepare_latents_latent2img
    pipe.generate_ip_adapter_embeds = generate_ip_adapter_embeds.__get__(pipe)

    ip_root = resolve_ip_adapter_dir(hub)
    weight_name = "ip-adapter_sdxl_vit-h.bin"
    load_kwargs = {
        "subfolder": "sdxl_models",
        "image_encoder_folder": None,
        "local_files_only": True,
    }
    if ip_root is not None:
        try:
            pipe.load_ip_adapter(str(ip_root), weight_name=weight_name, **load_kwargs)
        except Exception:
            weight_name = "ip-adapter_sdxl.bin"
            pipe.load_ip_adapter(str(ip_root), weight_name=weight_name, **load_kwargs)
    else:
        pipe.load_ip_adapter("h94/IP-Adapter", weight_name=weight_name, **load_kwargs)
    pipe.set_ip_adapter_scale(1.0)
    return pipe


@torch.no_grad()
def prior_embeds(eeg: np.ndarray, prior_ckpt: Path, device: torch.device, steps: int, guidance: float, seed: int) -> np.ndarray:
    pipe = load_atm_prior(str(prior_ckpt), device)
    gen = torch.Generator(device=device).manual_seed(seed)
    out = pipe.generate(
        torch.from_numpy(eeg.astype(np.float32)).to(device),
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=gen,
    )
    return l2(F.normalize(out.float(), dim=-1).cpu().numpy().astype(np.float32))


def load_train_latents(path: Path) -> np.ndarray:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, torch.Tensor):
        return obj.float().numpy()
    if isinstance(obj, dict) and "image_latent" in obj:
        return obj["image_latent"].float().numpy()
    raise TypeError(f"unexpected latent type {type(obj)} keys={list(obj.keys()) if isinstance(obj, dict) else ''}")


def generate_bank(
    pipe: StableDiffusionXLPipeline,
    clip_embeds: np.ndarray,
    eeg_for_pick: np.ndarray,
    neighbor_idx: np.ndarray,
    train_paths: list[Path],
    train_latents: np.ndarray | None,
    cand_dir: Path,
    strengths: list[float],
    top_m: int,
    device: torch.device,
    steps: int,
    guidance: float,
    low_level_mode: str,
    seed0: int,
    n: int,
    sample_latents: np.ndarray | None = None,
    mix_alpha: float = 0.6,
) -> list[dict]:
    cand_dir.mkdir(parents=True, exist_ok=True)
    specs: list[dict] = []
    k = 0
    for rank in range(top_m):
        for strength in strengths:
            specs.append(
                {
                    "k": k,
                    "mode": "official_mix",
                    "strength": float(strength),
                    "neighbor_rank": rank,
                    "low_level_mode": low_level_mode,
                    "seed": seed0 + k,
                }
            )
            k += 1
    (cand_dir / "specs.json").write_text(json.dumps(specs, indent=2), encoding="utf-8")

    g_base = torch.Generator(device=device)
    for i in tqdm(range(n), desc="official-mix-gen"):
        for sp in specs:
            out_path = cand_dir / f"{i:03d}_k{sp['k']}.png"
            if out_path.is_file():
                continue
            rank = int(sp["neighbor_rank"])
            nb = int(neighbor_idx[i, rank])
            low_img = None
            low_lat = None
            if low_level_mode == "neighbor_image":
                low_img = Image.open(train_paths[nb]).convert("RGB").resize((512, 512), Image.BICUBIC)
            elif low_level_mode == "neighbor_latent" and train_latents is not None:
                low_lat = torch.from_numpy(train_latents[nb]).unsqueeze(0)
            elif low_level_mode == "pred_latent" and sample_latents is not None:
                low_lat = torch.from_numpy(sample_latents[i]).unsqueeze(0)
            elif low_level_mode == "mix_latent" and sample_latents is not None and train_latents is not None:
                pred = sample_latents[i]
                nb_lat = train_latents[nb]
                mixed = float(mix_alpha) * pred + (1.0 - float(mix_alpha)) * nb_lat
                low_lat = torch.from_numpy(mixed).unsqueeze(0)

            emb = torch.from_numpy(clip_embeds[i : i + 1]).to(device=device, dtype=pipe.dtype)
            # Official Generator4Embeds passes raw embed; CFG handled inside if guidance>0
            if guidance > 1.0:
                uncond = torch.zeros_like(emb)
                ip_emb = torch.cat([uncond, emb], dim=0)
                if ip_emb.dim() == 2:
                    ip_emb = ip_emb.unsqueeze(1)
            else:
                ip_emb = emb.unsqueeze(1) if emb.dim() == 2 else emb

            gen = g_base.manual_seed(int(sp["seed"]) + i)
            result = pipe.generate_ip_adapter_embeds(
                prompt="",
                ip_adapter_embeds=ip_emb,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=gen,
                height=512,
                width=512,
                img2img_strength=float(sp["strength"]),
                low_level_image=low_img,
                low_level_latent=low_lat,
            )
            result.images[0].save(out_path)

    return specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, default="sub-08")
    parser.add_argument("--bridge-dir", type=str, default="outputs/atm_bridge")
    parser.add_argument(
        "--embed-source",
        type=str,
        default="prior_atm",
        # "npy" reads a precomputed (N,1024) OpenCLIP ViT-H-14 array from --bit-npy.
        # Added so external encoders (e.g. the SAMGA-R bridge in the sibling
        # `eeg-retrieval` project) can drive this exact Turbo + IP-Adapter stack without
        # reimplementing generation, which is what keeps their outputs comparable to the
        # `bit_clip` and `prior_atm` rows already in the paper tables.
        choices=["atm", "prior_atm", "bit_clip", "npy"],
    )
    parser.add_argument("--bit-npy", type=str, default="outputs/eval/atm_pipeline_sub08/sub-08_bit_clip_1024.npy")
    parser.add_argument(
        "--prior-ckpt",
        type=str,
        default="checkpoints/atm_diffusion_prior/sub-08/diffusion_prior.pt",
    )
    parser.add_argument(
        "--low-level-mode",
        type=str,
        default="neighbor_image",
        choices=["neighbor_image", "neighbor_latent", "pred_latent", "mix_latent"],
    )
    parser.add_argument("--sample-latent-npy", type=str, default="", help="(N,4,64,64) per-sample predicted VAE latents")
    parser.add_argument("--mix-alpha", type=float, default=0.6, help="pred weight for mix_latent mode")
    parser.add_argument("--top-m", type=int, default=2)
    parser.add_argument("--strengths", type=str, default="0.35,0.50,0.65,0.85")
    parser.add_argument("--use-turbo", action="store_true", default=True)
    parser.add_argument("--no-turbo", action="store_true")
    parser.add_argument("--gen-steps", type=int, default=4)
    parser.add_argument("--gen-guidance", type=float, default=0.0)
    parser.add_argument("--prior-steps", type=int, default=50)
    parser.add_argument("--prior-guidance", type=float, default=5.0)
    parser.add_argument("--images-root", type=str, default="/project/peilab/why/data/images_set")
    parser.add_argument("--output-dir", type=str, default="outputs/erdc/w12_official_prior_sub08")
    parser.add_argument("--latent-npy", type=str, default="checkpoints/_hf_atm_ds/train_image_latent_512.pt")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--seed0", type=int, default=42)
    parser.add_argument("--neighbor-mode", type=str, default="retrieve", choices=["retrieve", "random"])
    args = parser.parse_args()
    if args.no_turbo:
        args.use_turbo = False
        if args.gen_steps == 4:
            args.gen_steps = 30
        if args.gen_guidance == 0.0:
            args.gen_guidance = 5.0

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
    os.environ.setdefault("XFORMERS_DISABLED", "1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    bridge = Path(args.bridge_dir)
    if not bridge.is_absolute():
        bridge = project / bridge
    subject = args.subject
    atm_eeg = load_embed(bridge / f"{subject}_test_eeg_1024.npy")

    if args.embed_source == "atm":
        clip_embeds = atm_eeg
        pick_eeg = atm_eeg
    elif args.embed_source in ("bit_clip", "npy"):
        # `npy`: an external encoder's conditioning array, path passed via --bit-npy.
        # Identical treatment to bit_clip on purpose -- same normalisation, same `pick_eeg`
        # (the array drives both the neighbour lookup and brain-consistency re-selection)
        # -- so the only thing that differs between an `npy` run and a `bit_clip` run is
        # whose embeddings went in.
        p = Path(args.bit_npy)
        if not p.is_absolute():
            p = project / p
        clip_embeds = load_embed(p)
        pick_eeg = clip_embeds
    else:
        prior_path = Path(args.prior_ckpt)
        if not prior_path.is_absolute():
            prior_path = project / prior_path
        clip_embeds = prior_embeds(
            atm_eeg, prior_path, device, args.prior_steps, args.prior_guidance, args.seed0
        )
        pick_eeg = atm_eeg

    gallery = load_embed(bridge / "clip_img_train_1024.npy")
    train_paths = list_train_images(Path(args.images_root))
    n = len(clip_embeds) if args.max_images <= 0 else min(len(clip_embeds), args.max_images)
    strengths = [float(x) for x in args.strengths.split(",") if x.strip()]

    if args.neighbor_mode == "retrieve":
        neighbor_idx = retrieve_neighbors(clip_embeds[:n], gallery, top_m=args.top_m)
    else:
        rng = np.random.RandomState(args.seed0)
        neighbor_idx = rng.randint(0, len(train_paths), size=(n, args.top_m))

    train_latents = None
    if args.low_level_mode in ("neighbor_latent", "mix_latent"):
        lat_path = Path(args.latent_npy)
        if not lat_path.is_absolute():
            lat_path = project / lat_path
        train_latents = load_train_latents(lat_path)
        print(f"[INFO] train latents shape={train_latents.shape}")

    sample_latents = None
    if args.low_level_mode in ("pred_latent", "mix_latent"):
        sp = Path(args.sample_latent_npy)
        if not sp.is_absolute():
            sp = project / sp
        if not sp.is_file():
            raise FileNotFoundError(f"missing --sample-latent-npy for {args.low_level_mode}: {sp}")
        sample_latents = np.load(sp).astype(np.float32)
        print(f"[INFO] sample latents shape={sample_latents.shape}")

    pipe = load_official_pipe(device, use_turbo=args.use_turbo)
    cand_dir = out_dir / "candidates"
    specs = generate_bank(
        pipe,
        clip_embeds[:n],
        pick_eeg[:n],
        neighbor_idx,
        train_paths,
        train_latents,
        cand_dir,
        strengths,
        args.top_m,
        device,
        args.gen_steps,
        args.gen_guidance,
        args.low_level_mode,
        args.seed0,
        n,
        sample_latents=sample_latents,
        mix_alpha=args.mix_alpha,
    )
    del pipe
    torch.cuda.empty_cache()

    np.save(out_dir / "neighbor_idx.npy", neighbor_idx)
    gt = list_test_images(Path(args.images_root))
    report = select_and_eval(pick_eeg[:n], cand_dir, specs, out_dir, gt, n, device, args.seed0)
    report["embed_source"] = args.embed_source
    report["low_level_mode"] = args.low_level_mode
    report["use_turbo"] = args.use_turbo
    report["gen_steps"] = args.gen_steps
    report["gen_guidance"] = args.gen_guidance
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[OK] wrote {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
