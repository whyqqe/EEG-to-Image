#!/usr/bin/env python3
"""Generate images from Fusion-space embeddings via Brain-HIVE Fusion Prior + SDXL."""

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

SCRIPT_DIR = Path(__file__).resolve().parent
BRAIN_HIVE = Path(os.environ.get("BRAIN_HIVE", "/project/peilab/why/Brain-HIVE"))
sys.path.insert(0, str(BRAIN_HIVE))
sys.path.insert(0, str(Path("/project/peilab/why/eeg-brainit/scripts")))

from main.models_adapter import IPAttnAdapterModel, IPProjectionModel, FusionEncoderModel  # noqa: E402
from eval_atm_pipeline import resolve_sdxl_model_path  # type: ignore


def list_train_images(images_root: Path) -> list[Path]:
    root = images_root / "training_images"
    paths: list[Path] = []
    for d in sorted([p for p in root.iterdir() if p.is_dir()]):
        imgs = sorted(list(d.glob("*.jpg")) + list(d.glob("*.JPEG")) + list(d.glob("*.png")))
        paths.extend(imgs)
    return paths


def resolve_turbo(hub: Path) -> str:
    turbo_root = hub / "models--stabilityai--sdxl-turbo" / "snapshots"
    if turbo_root.is_dir():
        for snap in sorted(turbo_root.iterdir(), reverse=True):
            if (snap / "model_index.json").is_file():
                return str(snap)
    return "stabilityai/sdxl-turbo"


@torch.no_grad()
def encode_clip_for_rerank(paths: list[Path], device: torch.device) -> np.ndarray:
    import open_clip
    from torchvision import transforms

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-H-14", pretrained="laion2b_s32b_b79k", device=device
    )
    model.eval()
    tf = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    feats = []
    for p in paths:
        x = tf(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
        f = model.encode_image(x)
        f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.float().cpu().numpy())
    return np.concatenate(feats, axis=0)


class FusionImageEncoder:
    """Encode generated RGB images into Fusion space for reranking."""

    def __init__(self, prior_path: str, device: torch.device, vae_id: str = "stabilityai/sdxl-vae"):
        from diffusers import AutoencoderKL
        import open_clip
        from torchvision import transforms

        self.device = device
        self.vae = AutoencoderKL.from_pretrained(vae_id).to(device).eval()
        self.clip_h, _, _ = open_clip.create_model_and_transforms(
            "ViT-H-14", pretrained="laion2b_s32b_b79k", device=device
        )
        self.clip_h.eval()
        self.clip_b, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device
        )
        self.clip_b.eval()
        self.fusion = FusionEncoderModel.from_pretrained(prior_path, subfolder="fusion_encoder").to(device).eval()
        self.tf_vae = transforms.Compose(
            [
                transforms.Resize(128, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(128),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.tf_clip = transforms.Compose(
            [
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    @torch.no_grad()
    def encode_pil(self, img: Image.Image) -> np.ndarray:
        rgb = img.convert("RGB")
        xv = self.tf_vae(rgb).unsqueeze(0).to(self.device)
        xc = self.tf_clip(rgb).unsqueeze(0).to(self.device)
        lat = self.vae.encode(xv).latent_dist.sample() * self.vae.config.scaling_factor
        vae_emb = lat.reshape(1, -1)
        ch = self.clip_h.encode_image(xc)
        ch = ch / ch.norm(dim=-1, keepdim=True)
        cb = self.clip_b.encode_image(xc)
        cb = cb / cb.norm(dim=-1, keepdim=True)
        z = self.fusion(
            {
                "CLIP-ViT-H-14-laion2B-s32B-b79K": ch,
                "CLIP-ViT-B-32-laion2B-s34B-b79K": cb,
                "vae": vae_emb,
            }
        )
        z = z / z.norm(dim=-1, keepdim=True)
        return z.float().cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fusion-npy", type=str, required=True)
    ap.add_argument("--prior-path", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--neighbor-idx-npy", type=str, default="")
    ap.add_argument("--nb-proj-npy", type=str, default="", help="for rerank cos(eeg, clip(gen))")
    ap.add_argument("--images-root", type=str, default="/project/peilab/why/data/images_set")
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--num-samples", type=int, default=1)
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--rerank-mode", type=str, default="fusion", choices=["fusion", "nb_proj", "blend"])
    ap.add_argument("--rerank-alpha", type=float, default=0.7, help="fusion weight in blend mode")
    ap.add_argument("--img2img-strength", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", type=str, default="nmb")
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    cache = Path(os.environ.get("HF_HUB_CACHE", "/project/peilab/why/cache/eeg-brainit/hf/hub"))
    os.environ.setdefault("OPENCLIP_CACHE_DIR", str(cache.parent / "open_clip"))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    fusion = np.load(args.fusion_npy).astype(np.float32)
    n = len(fusion) if args.max_images <= 0 else min(len(fusion), args.max_images)
    fusion = fusion[:n]

    out_dir = Path(args.output_dir)
    gen_dir = out_dir / "generated"
    gen_dir.mkdir(parents=True, exist_ok=True)

    from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline

    model_id = resolve_turbo(cache)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    common = dict(torch_dtype=dtype, variant="fp16", use_safetensors=True, local_files_only=Path(model_id).is_dir())
    pipe = StableDiffusionXLPipeline.from_pretrained(model_id, **common).to(device)
    pipe_i2i = None
    if args.img2img_strength > 0 and args.neighbor_idx_npy:
        pipe_i2i = StableDiffusionXLImg2ImgPipeline.from_pretrained(model_id, **common).to(device)

    attn_adapter = IPAttnAdapterModel.from_pretrained(args.prior_path, subfolder="attn_adapter")
    proj = IPProjectionModel.from_pretrained(args.prior_path, subfolder="proj")
    attn_adapter.bind_unet(pipe.unet)
    if pipe_i2i is not None:
        attn_adapter.bind_unet(pipe_i2i.unet)
    proj.to(device, dtype)
    attn_adapter.to(device, dtype)

    # diffusers 0.31+ rejects pipelines with unet=None/vae=None; reuse loaded pipe.
    with torch.no_grad():
        prompt_embeds, _, pooled_prompt_embeds, _ = pipe.encode_prompt(
            "",
            device=device,
            do_classifier_free_guidance=False,
            num_images_per_prompt=1,
        )

    train_paths = list_train_images(Path(args.images_root)) if args.neighbor_idx_npy else []
    neighbors = np.load(args.neighbor_idx_npy) if args.neighbor_idx_npy else None
    nb_proj = np.load(args.nb_proj_npy).astype(np.float32) if args.nb_proj_npy else None
    fusion_encoder = FusionImageEncoder(args.prior_path, device) if args.rerank and args.rerank_mode != "nb_proj" else None
    fusion_target = fusion.astype(np.float32)

    paths_out: list[Path] = []
    for i in tqdm(range(n), desc="nmb-gen"):
        out_path = gen_dir / f"{i:03d}.png"
        if out_path.is_file():
            paths_out.append(out_path)
            continue

        emb = torch.from_numpy(fusion[i : i + 1]).to(device=device, dtype=dtype)
        ip_hidden = proj(emb)

        candidates: list[tuple[float, Image.Image]] = []
        for s in range(args.num_samples if args.rerank else 1):
            g = torch.Generator(device=device).manual_seed(args.seed + i * 17 + s)
            if args.img2img_strength > 0 and pipe_i2i is not None and neighbors is not None:
                nb = int(neighbors[i, 0])
                init = Image.open(train_paths[nb]).convert("RGB").resize((512, 512), Image.Resampling.BICUBIC)
                img = pipe_i2i(
                    image=init,
                    strength=float(args.img2img_strength),
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    cross_attention_kwargs={"ip_hidden_states": ip_hidden.expand(1, -1, -1)},
                    num_inference_steps=args.steps,
                    guidance_scale=0.0,
                    generator=g,
                ).images[0]
            else:
                img = pipe(
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    cross_attention_kwargs={"ip_hidden_states": ip_hidden.expand(1, -1, -1)},
                    num_inference_steps=args.steps,
                    guidance_scale=0.0,
                    height=512,
                    width=512,
                    generator=g,
                ).images[0]
            score = 0.0
            if args.rerank:
                fusion_score = 0.0
                nb_score = 0.0
                if fusion_encoder is not None:
                    z = fusion_encoder.encode_pil(img)
                    fusion_score = float((z * fusion_target[i : i + 1]).sum())
                if args.rerank_mode == "nb_proj" and nb_proj is not None:
                    tmp = gen_dir / f"_tmp_{i}_{s}.png"
                    img.save(tmp)
                    clip_f = encode_clip_for_rerank([tmp], device)
                    nb_score = float((clip_f * nb_proj[i : i + 1]).sum())
                    tmp.unlink(missing_ok=True)
                elif args.rerank_mode == "nb_proj" and nb_proj is None:
                    nb_score = fusion_score
                if args.rerank_mode == "fusion":
                    score = fusion_score
                elif args.rerank_mode == "nb_proj":
                    score = nb_score
                else:
                    score = args.rerank_alpha * fusion_score + (1.0 - args.rerank_alpha) * (
                        nb_score if nb_proj is not None else fusion_score
                    )
            candidates.append((score, img))

        if args.rerank and len(candidates) > 1:
            candidates.sort(key=lambda x: -x[0])
        candidates[0][1].save(out_path)
        paths_out.append(out_path)

    meta = {
        "tag": args.tag,
        "fusion_npy": args.fusion_npy,
        "n": n,
        "steps": args.steps,
        "num_samples": args.num_samples,
        "rerank": args.rerank,
        "rerank_mode": args.rerank_mode,
        "rerank_alpha": args.rerank_alpha,
        "img2img_strength": args.img2img_strength,
    }
    (out_dir / "metrics.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[OK] {gen_dir} n={n}")


if __name__ == "__main__":
    main()
