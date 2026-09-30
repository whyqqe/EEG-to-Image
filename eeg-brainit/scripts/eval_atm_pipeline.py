#!/usr/bin/env python3
"""End-to-end ATM evaluation for sub-08.

1) Retrieval suite: raw ATM / prior CLIP / distill S1 bridge_clip / S3 bit_clip
2) Optional SDXL+IP-Adapter generation from atm / prior / bit_clip embeds
3) Image metrics vs THINGS-EEG2 test GT (PixCorr / SSIM / CLIP cosine)

Writes only under --output-dir. Uses project HF cache (offline-friendly).
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
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from eeg_brainit.models.atm_bridge import AtmBrainITPipeline
from eeg_brainit.models.atm_diffusion_prior import load_atm_prior
from eeg_brainit.utils.metrics import pixel_correlation, ssim_simple


def retrieval_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    sim = pred @ target.T
    ranks = []
    top1 = top5 = 0
    n = sim.shape[0]
    for i in range(n):
        order = np.argsort(-sim[i])
        rank = int(np.where(order == i)[0][0]) + 1
        ranks.append(rank)
        if rank == 1:
            top1 += 1
        if rank <= 5:
            top5 += 1
    ranks = np.asarray(ranks)
    paired = float((pred * target).sum(1).mean())
    shuffled = float(
        (pred * target[np.random.RandomState(0).permutation(n)]).sum(1).mean()
    )
    return {
        "n": int(n),
        "top1": float(top1 / n),
        "top5": float(top5 / n),
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(ranks.mean()),
        "paired_cos": paired,
        "shuffled_cos": shuffled,
        "cos_gap": paired - shuffled,
    }


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-8)


def list_test_images(images_root: Path) -> list[Path]:
    root = images_root / "test_images"
    paths = []
    for d in sorted([p for p in root.iterdir() if p.is_dir()]):
        imgs = sorted(list(d.glob("*.jpg")) + list(d.glob("*.JPEG")) + list(d.glob("*.png")))
        if not imgs:
            raise FileNotFoundError(d)
        paths.append(imgs[0])
    if len(paths) != 200:
        raise RuntimeError(f"Expected 200 test images, got {len(paths)}")
    return paths


@torch.no_grad()
def encode_head(
    model: AtmBrainITPipeline,
    eeg: np.ndarray,
    key: str,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    model.eval()
    outs = []
    x = torch.from_numpy(eeg.astype(np.float32))
    for i in range(0, len(x), batch_size):
        out = model(x[i : i + batch_size].to(device))
        outs.append(F.normalize(out[key].float(), dim=-1).cpu().numpy())
    return np.concatenate(outs, axis=0).astype(np.float32)


def load_bridge_model(ckpt: Path, project: Path, device: torch.device) -> AtmBrainITPipeline:
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    # Ensure BIT path is on for bit_clip eval when present in weights.
    if any(k.startswith("bit.") for k in ck["model"].keys()):
        cfg.setdefault("atm", {})["use_bit"] = True
    model = AtmBrainITPipeline.from_config(cfg, project_root=str(project)).to(device)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    print(f"[INFO] loaded {ckpt.name} missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()
    return model


def resolve_sdxl_model_path(hub_cache: Path | None = None) -> str:
    """Prefer a complete local snapshot directory over repo-id lookup."""
    hubs = []
    if hub_cache is not None:
        hubs.append(Path(hub_cache))
    for key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        v = os.environ.get(key)
        if v:
            hubs.append(Path(v))
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        hubs.append(Path(hf_home) / "hub")
    hubs.append(Path("/project/peilab/why/cache/eeg-brainit/hf/hub"))

    seen: set[str] = set()
    for hub in hubs:
        hub = hub.resolve() if hub.exists() else hub
        key = str(hub)
        if key in seen:
            continue
        seen.add(key)
        snap_root = hub / "models--stabilityai--stable-diffusion-xl-base-1.0" / "snapshots"
        if not snap_root.is_dir():
            continue
        for snap in sorted(snap_root.iterdir(), reverse=True):
            if (snap / "model_index.json").is_file():
                print(f"[INFO] using local SDXL snapshot: {snap}")
                return str(snap)
    # Fall back to repo id (requires network or complete hub cache).
    return "stabilityai/stable-diffusion-xl-base-1.0"


def resolve_ip_adapter_dir(hub_cache: Path | None = None) -> Path | None:
    hubs = []
    if hub_cache is not None:
        hubs.append(Path(hub_cache))
    for key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        v = os.environ.get(key)
        if v:
            hubs.append(Path(v))
    hubs.append(Path("/project/peilab/why/cache/eeg-brainit/hf/hub"))
    for hub in hubs:
        snap_root = hub / "models--h94--IP-Adapter" / "snapshots"
        if not snap_root.is_dir():
            continue
        for snap in sorted(snap_root.iterdir(), reverse=True):
            if (snap / "sdxl_models" / "ip-adapter_sdxl_vit-h.bin").is_file():
                return snap
            if (snap / "sdxl_models" / "ip-adapter_sdxl.bin").is_file():
                return snap
    return None


@torch.no_grad()
def generate_images_sdxl(
    clip_embeds: np.ndarray,
    out_dir: Path,
    device: torch.device,
    steps: int = 30,
    guidance: float = 5.0,
    ip_scale: float = 1.0,
    height: int = 512,
    width: int = 512,
    max_images: int = 0,
    seed: int = 42,
    source_name: str = "clip",
) -> list[Path]:
    from diffusers import StableDiffusionXLPipeline

    out_dir.mkdir(parents=True, exist_ok=True)
    # Resume if partial run already wrote images (preempt-friendly).
    existing = sorted(out_dir.glob("*.png"))
    n = len(clip_embeds) if max_images <= 0 else min(len(clip_embeds), max_images)
    if len(existing) >= n:
        print(f"[INFO] skip gen ({source_name}): already have {len(existing)} images in {out_dir}")
        return existing[:n]

    print(f"[INFO] loading SDXL pipeline for {source_name}...")
    hub = Path(os.environ.get("HF_HUB_CACHE", "/project/peilab/why/cache/eeg-brainit/hf/hub"))
    model_id = resolve_sdxl_model_path(hub)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        variant="fp16" if device.type == "cuda" else None,
        use_safetensors=True,
        local_files_only=True,
    )
    pipe = pipe.to(device)

    # We feed CLIP embeds directly via ip_adapter_image_embeds, so skip image_encoder.
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
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] vit-h adapter failed ({exc}); trying ip-adapter_sdxl.bin")
            weight_name = "ip-adapter_sdxl.bin"
            pipe.load_ip_adapter(str(ip_root), weight_name=weight_name, **load_kwargs)
    else:
        pipe.load_ip_adapter("h94/IP-Adapter", weight_name=weight_name, **load_kwargs)
    pipe.set_ip_adapter_scale(ip_scale)
    print(f"[INFO] IP-Adapter weight={weight_name} scale={ip_scale}")

    paths: list[Path] = []
    g = torch.Generator(device=device).manual_seed(seed)

    def _call_with_embeds(image_embeds: torch.Tensor):
        return pipe(
            prompt="",
            negative_prompt="",
            ip_adapter_image_embeds=[image_embeds],
            num_inference_steps=steps,
            guidance_scale=guidance,
            height=height,
            width=width,
            generator=g,
        )

    # Discover embed layout on the first sample (CFG expects concat([neg, pos], dim=0)).
    use_unsqueeze = False
    layout_resolved = False
    for i in tqdm(range(n), desc=f"sdxl-gen[{source_name}]"):
        path = out_dir / f"{i:03d}.png"
        if path.is_file():
            paths.append(path)
            continue
        emb = torch.from_numpy(clip_embeds[i : i + 1]).to(device=device, dtype=pipe.dtype)
        uncond = torch.zeros_like(emb)
        image_embeds = torch.cat([uncond, emb], dim=0)
        if use_unsqueeze:
            image_embeds = image_embeds.unsqueeze(1)
        if not layout_resolved:
            try:
                result = _call_with_embeds(image_embeds)
            except Exception:
                image_embeds = torch.cat([uncond, emb], dim=0).unsqueeze(1)
                result = _call_with_embeds(image_embeds)
                use_unsqueeze = True
            layout_resolved = True
            print(f"[INFO] IP-Adapter embed layout unsqueeze={use_unsqueeze}")
        else:
            result = _call_with_embeds(image_embeds)
        result.images[0].save(path)
        paths.append(path)
    (out_dir / "adapter.json").write_text(
        json.dumps(
            {
                "source": source_name,
                "weight_name": weight_name,
                "ip_scale": ip_scale,
                "steps": steps,
                "model": model_id,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    del pipe
    torch.cuda.empty_cache()
    return paths


@torch.no_grad()
def image_metrics(
    gen_paths: list[Path],
    gt_paths: list[Path],
    device: torch.device,
) -> dict:
    import open_clip
    from torchvision import transforms

    assert len(gen_paths) <= len(gt_paths)
    n = len(gen_paths)
    to_tensor = transforms.Compose(
        [
            transforms.Resize((256, 256), antialias=True),
            transforms.ToTensor(),
        ]
    )
    pixs, ssims, clips = [], [], []
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14", pretrained="laion2b_s32b_b79k", device=device
    )
    model.eval()
    for i in tqdm(range(n), desc="img-metrics"):
        g = Image.open(gen_paths[i]).convert("RGB")
        t = Image.open(gt_paths[i]).convert("RGB")
        gt = to_tensor(t).unsqueeze(0).to(device)
        gp = to_tensor(g).unsqueeze(0).to(device)
        pixs.append(pixel_correlation(gp, gt))
        ssims.append(ssim_simple(gp, gt))
        # CLIP cosine of generated vs GT image
        gx = preprocess(g).unsqueeze(0).to(device)
        tx = preprocess(t).unsqueeze(0).to(device)
        ge = F.normalize(model.encode_image(gx).float(), dim=-1)
        te = F.normalize(model.encode_image(tx).float(), dim=-1)
        clips.append(float((ge * te).sum()))
    return {
        "n": n,
        "pixcorr": float(np.mean(pixs)),
        "ssim": float(np.mean(ssims)),
        "clip_cosine": float(np.mean(clips)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, default="sub-08")
    parser.add_argument("--bridge-dir", type=str, default="outputs/atm_bridge")
    parser.add_argument(
        "--prior-ckpt",
        type=str,
        default="checkpoints/atm_diffusion_prior/sub-08/diffusion_prior.pt",
    )
    parser.add_argument(
        "--s1-ckpt",
        type=str,
        default="outputs/atm_distill_s1_sub08/checkpoints/atm_stage1_best.pt",
    )
    parser.add_argument(
        "--s2-ckpt",
        type=str,
        default="outputs/atm_distill_s3_sub08/checkpoints/atm_stage3_best.pt",
        help="Prefer distill S3 bit_clip checkpoint (legacy name --s2-ckpt).",
    )
    parser.add_argument(
        "--s3-ckpt",
        type=str,
        default="",
        help="Optional alias for distill S3; overrides --s2-ckpt when set.",
    )
    parser.add_argument("--images-root", type=str, default="/project/peilab/why/data/images_set")
    parser.add_argument("--output-dir", type=str, default="outputs/eval/atm_pipeline_sub08")
    parser.add_argument("--prior-steps", type=int, default=50)
    parser.add_argument("--prior-guidance", type=float, default=5.0)
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument(
        "--gen-sources",
        type=str,
        default="prior,atm,bit_clip",
        help="Comma list among: prior,atm,bridge_clip,bit_clip",
    )
    parser.add_argument("--gen-steps", type=int, default=30)
    parser.add_argument("--gen-guidance", type=float, default=5.0)
    parser.add_argument("--gen-size", type=int, default=512)
    parser.add_argument("--max-images", type=int, default=0, help="0=all 200")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.s3_ckpt:
        args.s2_ckpt = args.s3_ckpt

    project = ROOT
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = project / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_root = Path("/project/peilab/why/cache/eeg-brainit")
    os.environ.setdefault("HF_HOME", str(Path("/project/peilab/why/cache/eeg-brainit/hf")))
    os.environ.setdefault("OPENCLIP_CACHE_DIR", str(cache_root / "open_clip"))
    os.environ.setdefault("TORCH_HOME", str(cache_root / "torch"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")
    if device.type == "cuda":
        print(f"[INFO] GPU={torch.cuda.get_device_name(0)}")

    bridge_dir = Path(args.bridge_dir)
    if not bridge_dir.is_absolute():
        bridge_dir = project / bridge_dir
    subject = args.subject

    eeg = l2(np.load(bridge_dir / f"{subject}_test_eeg_1024.npy").astype(np.float32))
    img_feat = l2(
        np.load(
            project / "outputs/eval/atm_baseline/test_ViT-H-14_laion2b_s32b_b79k_features.npy"
        ).astype(np.float32)
    )
    gt_paths = list_test_images(Path(args.images_root))
    print(f"[INFO] eeg={eeg.shape} img={img_feat.shape}")

    report: dict = {
        "subject": subject,
        "chance_top1": 1.0 / float(eeg.shape[0]),
        "retrieval": {},
    }

    # 1) raw ATM
    report["retrieval"]["raw_atm"] = retrieval_metrics(eeg, img_feat)
    print(
        f"[INFO] raw_atm top1={report['retrieval']['raw_atm']['top1']*100:.2f}% "
        f"top5={report['retrieval']['raw_atm']['top5']*100:.2f}%"
    )

    # 2) diffusion prior CLIP
    prior_path = Path(args.prior_ckpt)
    if not prior_path.is_absolute():
        prior_path = project / prior_path
    pipe = load_atm_prior(str(prior_path), device)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    prior_out = pipe.generate(
        torch.from_numpy(eeg).to(device),
        num_inference_steps=args.prior_steps,
        guidance_scale=args.prior_guidance,
        generator=gen,
    )
    prior_np = l2(F.normalize(prior_out.float(), dim=-1).cpu().numpy().astype(np.float32))
    np.save(out_dir / f"{subject}_prior_clip_1024.npy", prior_np)
    report["retrieval"]["atm_prior"] = retrieval_metrics(prior_np, img_feat)
    print(
        f"[INFO] atm_prior top1={report['retrieval']['atm_prior']['top1']*100:.2f}% "
        f"top5={report['retrieval']['atm_prior']['top5']*100:.2f}%"
    )

    # 3) S1 / S3 heads (legacy flag name --s2-ckpt points at distill S3 by default)
    emb_bank: dict[str, np.ndarray] = {
        "atm": eeg,
        "prior": prior_np,
    }
    for tag, ckpt_arg, key in (
        ("bridge_s1", args.s1_ckpt, "bridge_clip"),
        ("bit_s3", args.s2_ckpt, "bit_clip"),
    ):
        ckpt = Path(ckpt_arg)
        if not ckpt.is_absolute():
            ckpt = project / ckpt
        if not ckpt.is_file():
            print(f"[WARN] skip {tag}: missing {ckpt}")
            continue
        model = load_bridge_model(ckpt, project, device)
        keys = ["atm_emb", "bridge_clip"] + (["bit_clip"] if key == "bit_clip" else [])
        for k in keys:
            try:
                emb = encode_head(model, eeg, k, device)
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] {tag}/{k}: {exc}")
                continue
            m = retrieval_metrics(emb, img_feat)
            report["retrieval"][f"{tag}_{k}"] = m
            print(f"[INFO] {tag}_{k} top1={m['top1']*100:.2f}% top5={m['top5']*100:.2f}%")
            if tag == "bridge_s1" and k == "bridge_clip":
                emb_bank["bridge_clip"] = emb
                np.save(out_dir / f"{subject}_bridge_clip_1024.npy", emb)
            if tag == "bit_s3" and k == "bit_clip":
                emb_bank["bit_clip"] = emb
                np.save(out_dir / f"{subject}_bit_clip_1024.npy", emb)
        del model
        torch.cuda.empty_cache()

    # 4) optional image generation + metrics (multi-source)
    if not args.skip_generate:
        wanted = [s.strip() for s in args.gen_sources.split(",") if s.strip()]
        report["generation"] = {}
        for src in wanted:
            if src not in emb_bank:
                msg = f"missing embeds for gen source={src}"
                print(f"[WARN] {msg}")
                report["generation"][src] = {"error": msg}
                continue
            gen_dir = out_dir / "generated" / src
            try:
                gen_paths = generate_images_sdxl(
                    emb_bank[src],
                    gen_dir,
                    device,
                    steps=args.gen_steps,
                    guidance=args.gen_guidance,
                    height=args.gen_size,
                    width=args.gen_size,
                    max_images=args.max_images,
                    seed=args.seed,
                    source_name=src,
                )
                metrics = image_metrics(gen_paths, gt_paths, device)
                report["generation"][src] = metrics
                print(
                    f"[INFO] generation[{src}] n={metrics['n']} "
                    f"pixcorr={metrics['pixcorr']:.4f} ssim={metrics['ssim']:.4f} "
                    f"clip={metrics['clip_cosine']:.4f}"
                )
            except Exception as exc:  # noqa: BLE001
                report["generation"][src] = {"error": str(exc)}
                print(f"[ERROR] generation[{src}] failed: {exc}")

    # Attach official ATM gen metrics if present (same protocol).
    official = project / "outputs/eval/atm_official_gen_sub08/metrics.json"
    if official.is_file():
        try:
            report["official_atm_gen"] = json.loads(official.read_text(encoding="utf-8")).get(
                "metrics"
            )
        except Exception:  # noqa: BLE001
            pass

    out_json = out_dir / "metrics.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("========== ATM PIPELINE EVAL ==========")
    for k, m in report["retrieval"].items():
        print(
            f"{k:24s} Top-1={m['top1']*100:6.2f}% Top-5={m['top5']*100:6.2f}% "
            f"med={m['median_rank']:5.1f}"
        )
    if isinstance(report.get("generation"), dict):
        for src, g in report["generation"].items():
            if "error" in g:
                print(f"generation[{src:8s}] ERROR: {g['error']}")
            else:
                print(
                    f"generation[{src:8s}] PixCorr={g['pixcorr']:.4f} SSIM={g['ssim']:.4f} "
                    f"CLIP={g['clip_cosine']:.4f} (n={g['n']})"
                )
    if report.get("official_atm_gen"):
        g = report["official_atm_gen"]
        print(
            f"official_atm_gen         PixCorr={g['pixcorr']:.4f} SSIM={g['ssim']:.4f} "
            f"CLIP={g['clip_cosine']:.4f} (n={g['n']})"
        )
    print(f"[OK] wrote {out_json}")


if __name__ == "__main__":
    main()
