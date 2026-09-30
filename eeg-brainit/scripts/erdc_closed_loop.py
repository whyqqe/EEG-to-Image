#!/usr/bin/env python3
"""ERDC closed-loop (Stage-3): multi-hypothesis generation + brain-consistency pick.

Fastest post-W3 path after naive conf→scale failed:
  For each EEG sample, generate K candidates (seed / mild scale grid),
  re-encode with OpenCLIP ViT-H/14, score cos(eeg_emb, clip(gen)),
  select argmax. Controls: first-of-K, random-of-K, oracle-vs-GT.

No training. Offline HF cache. Writes under --output-dir only.
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


def load_embed(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    return l2(np.load(path).astype(np.float32))


def resolve_eeg_embeds(
    project: Path,
    subject: str,
    embed_source: str,
    bridge_dir: Path,
    args: argparse.Namespace,
) -> np.ndarray:
    if embed_source == "atm":
        return load_embed(bridge_dir / f"{subject}_test_eeg_1024.npy")
    if embed_source == "teacher":
        return load_embed(bridge_dir / "clip_img_test_1024.npy")
    mapping = {
        "prior": args.prior_npy,
        "bit_clip": args.bit_npy,
        "bridge_clip": args.bridge_npy,
        "npy": args.embed_npy,
    }
    if embed_source not in mapping or not mapping[embed_source]:
        raise ValueError(f"need path for embed_source={embed_source}")
    p = Path(mapping[embed_source])
    return load_embed(p if p.is_absolute() else project / p)


def candidate_specs(k: int, scale_grid: list[float], seed0: int) -> list[dict]:
    """K candidates: cycle mild scales × seed offsets (keeps scale near 1.0)."""
    specs = []
    for i in range(k):
        specs.append(
            {
                "k": i,
                "scale": float(scale_grid[i % len(scale_grid)]),
                "seed": int(seed0 + i),
            }
        )
    return specs


@torch.no_grad()
def load_sdxl(device: torch.device):
    from diffusers import StableDiffusionXLPipeline

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
    if ip_root is None:
        raise FileNotFoundError("IP-Adapter snapshot missing")
    weight_name = "ip-adapter_sdxl_vit-h.bin"
    load_kwargs = {
        "subfolder": "sdxl_models",
        "image_encoder_folder": None,
        "local_files_only": True,
    }
    try:
        pipe.load_ip_adapter(str(ip_root), weight_name=weight_name, **load_kwargs)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] vit-h failed ({exc}); fallback")
        weight_name = "ip-adapter_sdxl.bin"
        pipe.load_ip_adapter(str(ip_root), weight_name=weight_name, **load_kwargs)
    return pipe, model_id, weight_name


@torch.no_grad()
def generate_candidates(
    pipe,
    embeds: np.ndarray,
    specs: list[dict],
    cand_dir: Path,
    device: torch.device,
    steps: int,
    guidance: float,
    size: int,
    n: int,
) -> None:
    cand_dir.mkdir(parents=True, exist_ok=True)
    (cand_dir / "specs.json").write_text(json.dumps(specs, indent=2), encoding="utf-8")
    use_unsqueeze = True
    total = n * len(specs)
    done = 0
    pbar = tqdm(total=total, desc="gen-candidates")
    for i in range(n):
        emb = torch.from_numpy(embeds[i : i + 1]).to(device=device, dtype=pipe.dtype)
        uncond = torch.zeros_like(emb)
        image_embeds = torch.cat([uncond, emb], dim=0)
        if use_unsqueeze:
            image_embeds = image_embeds.unsqueeze(1)
        for sp in specs:
            path = cand_dir / f"{i:03d}_k{sp['k']}.png"
            if path.is_file():
                done += 1
                pbar.update(1)
                continue
            pipe.set_ip_adapter_scale(float(sp["scale"]))
            g = torch.Generator(device=device).manual_seed(int(sp["seed"]))
            out = pipe(
                prompt="",
                negative_prompt="",
                ip_adapter_image_embeds=[image_embeds],
                num_inference_steps=steps,
                guidance_scale=guidance,
                height=size,
                width=size,
                generator=g,
            )
            out.images[0].save(path)
            done += 1
            pbar.update(1)
    pbar.close()


@torch.no_grad()
def encode_images_clip(
    paths: list[Path],
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
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


def pick_and_export(
    eeg: np.ndarray,
    cand_dir: Path,
    out_dirs: dict[str, Path],
    specs: list[dict],
    n: int,
    gt_paths: list[Path],
    device: torch.device,
    seed: int,
) -> dict:
    k = len(specs)
    # flatten paths in order i, then k
    flat_paths: list[Path] = []
    for i in range(n):
        for sp in specs:
            flat_paths.append(cand_dir / f"{i:03d}_k{sp['k']}.png")
            if not flat_paths[-1].is_file():
                raise FileNotFoundError(flat_paths[-1])

    feat_path = cand_dir / "clip_feats_flat.npy"
    if feat_path.is_file() and feat_path.stat().st_size > 0:
        flat_feats = np.load(feat_path)
        if flat_feats.shape[0] != n * k:
            flat_feats = encode_images_clip(flat_paths, device)
            np.save(feat_path, flat_feats)
    else:
        flat_feats = encode_images_clip(flat_paths, device)
        np.save(feat_path, flat_feats)

    feats = flat_feats.reshape(n, k, -1)
    # brain consistency: cos(eeg, gen_clip)
    scores = np.einsum("nd,nkd->nk", eeg[:n], feats)  # (n,k)
    np.save(cand_dir / "brain_scores.npy", scores.astype(np.float32))

    # oracle: cos(gen_clip, gt_clip) — diagnostic only
    gt_feat_path = cand_dir / "gt_clip_feats.npy"
    if gt_feat_path.is_file():
        gt_feats = np.load(gt_feat_path)
        if gt_feats.shape[0] != n:
            gt_feats = encode_images_clip(gt_paths[:n], device)
            np.save(gt_feat_path, gt_feats)
    else:
        gt_feats = encode_images_clip(gt_paths[:n], device)
        np.save(gt_feat_path, gt_feats)
    oracle = np.einsum("nd,nkd->nk", gt_feats, feats)

    rng = np.random.RandomState(seed)
    picks = {
        "first": np.zeros(n, dtype=np.int64),
        "random": rng.randint(0, k, size=n),
        "brain": scores.argmax(axis=1),
        "oracle": oracle.argmax(axis=1),
    }

    report: dict = {"n": n, "k": k, "specs": specs, "selection": {}}
    for name, idx in picks.items():
        odir = out_dirs[name]
        odir.mkdir(parents=True, exist_ok=True)
        sel_scores = scores[np.arange(n), idx]
        for i in range(n):
            src = cand_dir / f"{i:03d}_k{int(idx[i])}.png"
            dst = odir / f"{i:03d}.png"
            if not dst.is_file():
                Image.open(src).save(dst)
        metrics = image_metrics(
            [odir / f"{i:03d}.png" for i in range(n)],
            gt_paths[:n],
            device,
        )
        report["selection"][name] = {
            "metrics": metrics,
            "mean_brain_score": float(sel_scores.mean()),
            "pick_hist": {str(j): int((idx == j).sum()) for j in range(k)},
        }
        print(
            f"[INFO] pick={name:6s} CLIP={metrics['clip_cosine']:.4f} "
            f"Pix={metrics['pixcorr']:.4f} brainS={sel_scores.mean():.4f}"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, default="sub-08")
    parser.add_argument("--bridge-dir", type=str, default="outputs/atm_bridge")
    parser.add_argument(
        "--embed-source",
        type=str,
        default="bit_clip",
        choices=["atm", "bit_clip", "bridge_clip", "prior", "teacher", "npy"],
    )
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
    parser.add_argument("--embed-npy", type=str, default="")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument(
        "--scale-grid",
        type=str,
        default="0.9,1.0,1.1,1.0",
        help="Comma scales cycled over K (stay near 1.0; no confidence downweight)",
    )
    parser.add_argument("--seed0", type=int, default=42)
    parser.add_argument("--images-root", type=str, default="/project/peilab/why/data/images_set")
    parser.add_argument("--output-dir", type=str, default="outputs/erdc/w6_loop_bit_sub08")
    parser.add_argument("--gen-steps", type=int, default=30)
    parser.add_argument("--gen-guidance", type=float, default=5.0)
    parser.add_argument("--gen-size", type=int, default=512)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--skip-generate", action="store_true")
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
    os.environ.setdefault("XFORMERS_DISABLED", "1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device} GPU={torch.cuda.get_device_name(0) if device.type=='cuda' else 'cpu'}")

    bridge_dir = Path(args.bridge_dir)
    if not bridge_dir.is_absolute():
        bridge_dir = project / bridge_dir
    embeds = resolve_eeg_embeds(project, args.subject, args.embed_source, bridge_dir, args)
    n = len(embeds) if args.max_images <= 0 else min(len(embeds), args.max_images)
    scale_grid = [float(x) for x in args.scale_grid.split(",") if x.strip()]
    specs = candidate_specs(args.k, scale_grid, args.seed0)
    print(f"[INFO] source={args.embed_source} n={n} K={args.k} specs={specs}")

    cand_dir = out_dir / "candidates"
    if not args.skip_generate:
        pipe, model_id, weight_name = load_sdxl(device)
        generate_candidates(
            pipe,
            embeds,
            specs,
            cand_dir,
            device=device,
            steps=args.gen_steps,
            guidance=args.gen_guidance,
            size=args.gen_size,
            n=n,
        )
        del pipe
        if device.type == "cuda":
            torch.cuda.empty_cache()
        (out_dir / "adapter.json").write_text(
            json.dumps(
                {
                    "model": model_id,
                    "weight_name": weight_name,
                    "steps": args.gen_steps,
                    "guidance": args.gen_guidance,
                    "embed_source": args.embed_source,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    gt_paths = list_test_images(Path(args.images_root))
    out_dirs = {
        "first": out_dir / "selected_first",
        "random": out_dir / "selected_random",
        "brain": out_dir / "selected_brain",
        "oracle": out_dir / "selected_oracle",
    }
    report = pick_and_export(
        embeds,
        cand_dir,
        out_dirs,
        specs,
        n,
        gt_paths,
        device,
        seed=args.seed0,
    )
    report["subject"] = args.subject
    report["embed_source"] = args.embed_source
    report["gen_steps"] = args.gen_steps

    # Deltas vs first (single-sample baseline within same candidate bank)
    base = report["selection"]["first"]["metrics"]
    for name, blk in report["selection"].items():
        m = blk["metrics"]
        blk["delta_vs_first"] = {
            "clip": float(m["clip_cosine"] - base["clip_cosine"]),
            "pixcorr": float(m["pixcorr"] - base["pixcorr"]),
            "ssim": float(m["ssim"] - base["ssim"]),
        }

    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[OK] wrote {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
