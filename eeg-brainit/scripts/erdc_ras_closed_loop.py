#!/usr/bin/env python3
"""ERDC W8: Retrieval-Augmented Structure (RAS) + brain-consistency loop.

Motivation (literature gap):
  ENIGMA/ATM/SGDM show PixCorr~0.14–0.25 needs low-level / structure cues.
  Our W7 closed-loop wins semantics (CLIP) but PixCorr lags official ATM (~0.11 vs 0.16).
  CogCap stacks *image-side* multimodal teachers; SGDM uses ControlNet structure maps.
  We instead retrieve a train-set visual neighbor from *EEG evidence*, use it as
  img2img structural init, keep bit/ATM CLIP as IP-Adapter semantic condition,
  then closed-loop pick by cos(eeg, CLIP(gen)).

No GT leakage: gallery is TRAIN images only.
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
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

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
        if not imgs:
            raise FileNotFoundError(d)
        paths.extend(imgs)
    if len(paths) != 16540:
        raise RuntimeError(f"Expected 16540 train images, got {len(paths)}")
    return paths


def load_embed(path: Path) -> np.ndarray:
    return l2(np.load(path).astype(np.float32))


def retrieve_neighbors(
    query: np.ndarray,
    gallery: np.ndarray,
    top_m: int,
) -> np.ndarray:
    """Return (N, M) indices of top-M gallery neighbors."""
    sim = query @ gallery.T
    return np.argsort(-sim, axis=1)[:, :top_m]


@torch.no_grad()
def load_pipes(device: torch.device):
    from diffusers import StableDiffusionXLImg2ImgPipeline, StableDiffusionXLPipeline

    hub = Path(os.environ.get("HF_HUB_CACHE", "/project/peilab/why/cache/eeg-brainit/hf/hub"))
    model_id = resolve_sdxl_model_path(hub)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    common = dict(
        torch_dtype=dtype,
        variant="fp16" if device.type == "cuda" else None,
        use_safetensors=True,
        local_files_only=True,
    )
    pipe = StableDiffusionXLPipeline.from_pretrained(model_id, **common).to(device)
    pipe_i2i = StableDiffusionXLImg2ImgPipeline.from_pretrained(model_id, **common).to(device)

    ip_root = resolve_ip_adapter_dir(hub)
    if ip_root is None:
        raise FileNotFoundError("IP-Adapter missing")
    weight_name = "ip-adapter_sdxl_vit-h.bin"
    kwargs = {"subfolder": "sdxl_models", "image_encoder_folder": None, "local_files_only": True}
    for p in (pipe, pipe_i2i):
        try:
            p.load_ip_adapter(str(ip_root), weight_name=weight_name, **kwargs)
        except Exception:
            weight_name = "ip-adapter_sdxl.bin"
            p.load_ip_adapter(str(ip_root), weight_name=weight_name, **kwargs)
    return pipe, pipe_i2i, model_id, weight_name


def ip_embeds(emb: torch.Tensor) -> list[torch.Tensor]:
    uncond = torch.zeros_like(emb)
    ie = torch.cat([uncond, emb], dim=0).unsqueeze(1)
    return [ie]


@torch.no_grad()
def generate_bank(
    pipe,
    pipe_i2i,
    embeds: np.ndarray,
    neighbor_idx: np.ndarray,
    train_paths: list[Path],
    cand_dir: Path,
    device: torch.device,
    strengths: list[float],
    steps: int,
    guidance: float,
    size: int,
    ip_scale: float,
    seed0: int,
    n: int,
    include_ip_only: bool,
) -> list[dict]:
    cand_dir.mkdir(parents=True, exist_ok=True)
    specs: list[dict] = []
    k = 0
    if include_ip_only:
        specs.append({"k": k, "mode": "ip_only", "strength": 0.0, "neighbor_rank": -1, "seed": seed0})
        k += 1
    for r, _ in enumerate(range(neighbor_idx.shape[1])):
        for s in strengths:
            specs.append(
                {
                    "k": k,
                    "mode": "ras_i2i",
                    "strength": float(s),
                    "neighbor_rank": int(r),
                    "seed": int(seed0 + 10 + k),
                }
            )
            k += 1
    (cand_dir / "specs.json").write_text(json.dumps(specs, indent=2), encoding="utf-8")

    pipe.set_ip_adapter_scale(ip_scale)
    pipe_i2i.set_ip_adapter_scale(ip_scale)
    total = n * len(specs)
    pbar = tqdm(total=total, desc="ras-gen")
    for i in range(n):
        emb = torch.from_numpy(embeds[i : i + 1]).to(device=device, dtype=pipe.dtype)
        ie = ip_embeds(emb)
        for sp in specs:
            out_path = cand_dir / f"{i:03d}_k{sp['k']}.png"
            if out_path.is_file():
                pbar.update(1)
                continue
            g = torch.Generator(device=device).manual_seed(int(sp["seed"]))
            if sp["mode"] == "ip_only":
                img = pipe(
                    prompt="",
                    negative_prompt="",
                    ip_adapter_image_embeds=ie,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    height=size,
                    width=size,
                    generator=g,
                ).images[0]
            else:
                nb = int(neighbor_idx[i, sp["neighbor_rank"]])
                init = Image.open(train_paths[nb]).convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
                img = pipe_i2i(
                    prompt="",
                    negative_prompt="",
                    image=init,
                    strength=float(sp["strength"]),
                    ip_adapter_image_embeds=ie,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    generator=g,
                ).images[0]
            img.save(out_path)
            pbar.update(1)
    pbar.close()
    return specs


@torch.no_grad()
def encode_clip(paths: list[Path], device: torch.device, batch_size: int = 32) -> np.ndarray:
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14", pretrained="laion2b_s32b_b79k", device=device
    )
    model.eval()
    feats = []
    for i in tqdm(range(0, len(paths), batch_size), desc="clip-encode"):
        batch = paths[i : i + batch_size]
        xs = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in batch]).to(device)
        fe = F.normalize(model.encode_image(xs).float(), dim=-1)
        feats.append(fe.cpu().numpy().astype(np.float32))
    return np.concatenate(feats, axis=0)


def select_and_eval(
    eeg: np.ndarray,
    cand_dir: Path,
    specs: list[dict],
    out_dir: Path,
    gt_paths: list[Path],
    n: int,
    device: torch.device,
    seed: int,
) -> dict:
    k = len(specs)
    flat = [cand_dir / f"{i:03d}_k{sp['k']}.png" for i in range(n) for sp in specs]
    feat_path = cand_dir / "clip_feats_flat.npy"
    if feat_path.is_file() and np.load(feat_path).shape[0] == n * k:
        flat_feats = np.load(feat_path)
    else:
        flat_feats = encode_clip(flat, device)
        np.save(feat_path, flat_feats)
    feats = flat_feats.reshape(n, k, -1)
    scores = np.einsum("nd,nkd->nk", eeg[:n], feats)
    np.save(cand_dir / "brain_scores.npy", scores.astype(np.float32))

    rng = np.random.RandomState(seed)
    picks = {
        "first": np.zeros(n, dtype=np.int64),
        "random": rng.randint(0, k, size=n),
        "brain": scores.argmax(1),
    }
    # oracle diagnostic
    gt_feat_path = cand_dir / "gt_clip_feats.npy"
    if gt_feat_path.is_file() and np.load(gt_feat_path).shape[0] == n:
        gt_feats = np.load(gt_feat_path)
    else:
        gt_feats = encode_clip(gt_paths[:n], device)
        np.save(gt_feat_path, gt_feats)
    oracle = np.einsum("nd,nkd->nk", gt_feats, feats)
    picks["oracle"] = oracle.argmax(1)

    report: dict = {"n": n, "k": k, "specs": specs, "selection": {}}
    for name, idx in picks.items():
        odir = out_dir / f"selected_{name}"
        odir.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            src = cand_dir / f"{i:03d}_k{int(idx[i])}.png"
            dst = odir / f"{i:03d}.png"
            if not dst.is_file():
                Image.open(src).save(dst)
        metrics = image_metrics([odir / f"{i:03d}.png" for i in range(n)], gt_paths[:n], device)
        report["selection"][name] = {
            "metrics": metrics,
            "mean_brain_score": float(scores[np.arange(n), idx].mean()),
            "pick_hist": {str(j): int((idx == j).sum()) for j in range(k)},
        }
        print(
            f"[INFO] pick={name:6s} CLIP={metrics['clip_cosine']:.4f} "
            f"Pix={metrics['pixcorr']:.4f} brainS={scores[np.arange(n), idx].mean():.4f}"
        )
    base = report["selection"]["first"]["metrics"]
    for name, blk in report["selection"].items():
        m = blk["metrics"]
        blk["delta_vs_first"] = {
            "clip": float(m["clip_cosine"] - base["clip_cosine"]),
            "pixcorr": float(m["pixcorr"] - base["pixcorr"]),
            "ssim": float(m["ssim"] - base["ssim"]),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, default="sub-08")
    parser.add_argument("--bridge-dir", type=str, default="outputs/atm_bridge")
    parser.add_argument(
        "--embed-source",
        type=str,
        default="bit_clip",
        choices=["atm", "bit_clip", "bridge_clip", "prior"],
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
        "--prior-npy",
        type=str,
        default="outputs/eval/atm_pipeline_sub08/sub-08_prior_clip_1024.npy",
    )
    parser.add_argument("--top-m", type=int, default=2)
    parser.add_argument("--strengths", type=str, default="0.45,0.60,0.75")
    parser.add_argument("--include-ip-only", action="store_true", default=True)
    parser.add_argument("--no-ip-only", action="store_true")
    parser.add_argument("--ip-scale", type=float, default=1.0)
    parser.add_argument("--images-root", type=str, default="/project/peilab/why/data/images_set")
    parser.add_argument("--output-dir", type=str, default="outputs/erdc/w8_ras_bit_sub08")
    parser.add_argument("--gen-steps", type=int, default=30)
    parser.add_argument("--gen-guidance", type=float, default=5.0)
    parser.add_argument("--gen-size", type=int, default=512)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--seed0", type=int, default=42)
    parser.add_argument(
        "--neighbor-mode",
        type=str,
        default="retrieve",
        choices=["retrieve", "random", "misalign"],
        help="retrieve=EEG→CLIP neighbors; random=random train images; "
        "misalign=retrieve indices but break feature↔image alignment (VALID negative control)",
    )
    parser.add_argument(
        "--shuffle-gallery",
        action="store_true",
        help="DEPRECATED broken control (permutes gallery+paths together). Prefer --neighbor-mode misalign/random.",
    )
    args = parser.parse_args()
    if args.no_ip_only:
        args.include_ip_only = False

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
    subject = args.subject

    if args.embed_source == "atm":
        embeds = load_embed(bridge / f"{subject}_test_eeg_1024.npy")
    elif args.embed_source == "bit_clip":
        p = Path(args.bit_npy)
        embeds = load_embed(p if p.is_absolute() else project / p)
    elif args.embed_source == "bridge_clip":
        p = Path(args.bridge_npy)
        embeds = load_embed(p if p.is_absolute() else project / p)
    else:
        p = Path(args.prior_npy)
        embeds = load_embed(p if p.is_absolute() else project / p)

    gallery = load_embed(bridge / "clip_img_train_1024.npy")
    train_paths = list_train_images(Path(args.images_root))
    if args.shuffle_gallery:
        # kept for back-compat but warn: this is NOT a valid control
        print("[WARN] --shuffle-gallery is broken (same neighbors). Use --neighbor-mode misalign/random.")
        perm = np.random.RandomState(0).permutation(len(gallery))
        gallery = gallery[perm]
        train_paths = [train_paths[i] for i in perm]

    n = len(embeds) if args.max_images <= 0 else min(len(embeds), args.max_images)
    strengths = [float(x) for x in args.strengths.split(",") if x.strip()]
    mode = args.neighbor_mode
    if mode == "retrieve":
        neighbor_idx = retrieve_neighbors(embeds[:n], gallery, top_m=args.top_m)
    elif mode == "random":
        rng = np.random.RandomState(args.seed0 + 99)
        neighbor_idx = rng.randint(0, len(train_paths), size=(n, args.top_m))
        print("[INFO] neighbor-mode=random (VALID negative control)")
    else:  # misalign: correct CLIP retrieval indices, wrong image files
        neighbor_idx = retrieve_neighbors(embeds[:n], gallery, top_m=args.top_m)
        perm = np.random.RandomState(1).permutation(len(train_paths))
        train_paths = [train_paths[i] for i in perm]
        print("[INFO] neighbor-mode=misalign (VALID negative control: broken feature↔image)")
    np.save(out_dir / "neighbor_idx.npy", neighbor_idx)
    print(f"[INFO] source={args.embed_source} mode={mode} n={n} top_m={args.top_m} strengths={strengths}")

    pipe, pipe_i2i, model_id, weight_name = load_pipes(device)
    specs = generate_bank(
        pipe,
        pipe_i2i,
        embeds,
        neighbor_idx,
        train_paths,
        out_dir / "candidates",
        device=device,
        strengths=strengths,
        steps=args.gen_steps,
        guidance=args.gen_guidance,
        size=args.gen_size,
        ip_scale=args.ip_scale,
        seed0=args.seed0,
        n=n,
        include_ip_only=args.include_ip_only,
    )
    del pipe, pipe_i2i
    if device.type == "cuda":
        torch.cuda.empty_cache()

    gt = list_test_images(Path(args.images_root))
    report = select_and_eval(
        embeds, out_dir / "candidates", specs, out_dir, gt, n, device, args.seed0
    )
    report.update(
        {
            "subject": subject,
            "embed_source": args.embed_source,
            "neighbor_mode": mode,
            "top_m": args.top_m,
            "strengths": strengths,
            "shuffle_gallery": bool(args.shuffle_gallery),
            "model": model_id,
            "ip_weight": weight_name,
        }
    )
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[OK] wrote {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
