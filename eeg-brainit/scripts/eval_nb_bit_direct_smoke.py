#!/usr/bin/env python3
"""Smoke: original NeuroBOLT tokens → BiT → SDXL image generation.

Uses THINGS-EEG2 sub-08 tokens extracted with NeuroBOLT ``glb.pth``, then
RoiToBitBridge + pretrained Brain-IT → CLIP emb → SDXL+IP-Adapter.

Modes:
  as_trained_atm_skip — checkpoint residual (ATM skip floor)
  pure_nb_bit         — residual_mode=none (NeuroBOLT→BiT only)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Avoid broken xformers/triton import chain on this cluster image.
os.environ.setdefault("XFORMERS_DISABLED", "1")
os.environ.setdefault("HF_HOME", "/project/peilab/why/cache/eeg-brainit/hf")
os.environ.setdefault("HF_HUB_CACHE", "/project/peilab/why/cache/eeg-brainit/hf/hub")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from eeg_brainit.data.neurobolt_dataset import NeuroBoltBridgeDataset
from eeg_brainit.models.neurobolt_bridge import NeuroBoltBrainITPipeline


def _retrieval(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    sim = pred @ target.T
    n = sim.shape[0]
    top1 = top5 = 0
    ranks = []
    for i in range(n):
        order = np.argsort(-sim[i])
        rank = int(np.where(order == i)[0][0]) + 1
        ranks.append(rank)
        top1 += int(rank == 1)
        top5 += int(rank <= 5)
    return {
        "n": n,
        "top1": top1 / n,
        "top5": top5 / n,
        "median_rank": float(np.median(ranks)),
        "chance_top1": 1.0 / n,
    }


def list_test_images(images_root: Path) -> list[Path]:
    root = images_root / "test_images"
    paths = []
    for d in sorted([p for p in root.iterdir() if p.is_dir()]):
        imgs = sorted(list(d.glob("*.jpg")) + list(d.glob("*.JPEG")) + list(d.glob("*.png")))
        if imgs:
            paths.append(imgs[0])
    return paths


def resolve_sdxl_model_path(hub: Path) -> str:
    snap_root = hub / "models--stabilityai--stable-diffusion-xl-base-1.0" / "snapshots"
    for snap in sorted(snap_root.iterdir(), reverse=True):
        if (snap / "model_index.json").is_file():
            return str(snap)
    return "stabilityai/stable-diffusion-xl-base-1.0"


def resolve_ip_adapter_dir(hub: Path) -> Path | None:
    snap_root = hub / "models--h94--IP-Adapter" / "snapshots"
    if not snap_root.is_dir():
        return None
    for snap in sorted(snap_root.iterdir(), reverse=True):
        if (snap / "sdxl_models" / "ip-adapter_sdxl_vit-h.bin").is_file():
            return snap
        if (snap / "sdxl_models" / "ip-adapter_sdxl.bin").is_file():
            return snap
    return None


def _disable_broken_xformers() -> None:
    """Cluster xformers+triton crashes on import; force diffusers to skip it."""
    import diffusers.utils.import_utils as iu

    iu._xformers_available = False
    # If something already imported attention_processor with xformers=True, clear cache.
    for name in list(sys.modules):
        if name.startswith("diffusers.models.attention_processor") or name.startswith(
            "diffusers.loaders.ip_adapter"
        ):
            del sys.modules[name]


@torch.no_grad()
def generate_images_sdxl(
    clip_embeds: np.ndarray,
    out_dir: Path,
    device: torch.device,
    steps: int = 20,
    height: int = 512,
    width: int = 512,
    max_images: int = 8,
    seed: int = 42,
    source_name: str = "clip",
) -> list[Path]:
    _disable_broken_xformers()
    from diffusers import StableDiffusionXLPipeline

    out_dir.mkdir(parents=True, exist_ok=True)
    n = min(len(clip_embeds), max_images) if max_images > 0 else len(clip_embeds)
    hub = Path(os.environ["HF_HUB_CACHE"])
    model_id = resolve_sdxl_model_path(hub)
    print(f"[INFO] loading SDXL for {source_name} from {model_id}")
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
    load_kwargs = {"subfolder": "sdxl_models", "image_encoder_folder": None, "local_files_only": True}
    if ip_root is None:
        raise FileNotFoundError("IP-Adapter not found under HF_HUB_CACHE")
    try:
        pipe.load_ip_adapter(str(ip_root), weight_name=weight_name, **load_kwargs)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] {exc}; fallback ip-adapter_sdxl.bin")
        weight_name = "ip-adapter_sdxl.bin"
        pipe.load_ip_adapter(str(ip_root), weight_name=weight_name, **load_kwargs)
    pipe.set_ip_adapter_scale(1.0)

    paths: list[Path] = []
    g = torch.Generator(device=device).manual_seed(seed)
    use_unsqueeze = False
    layout_ok = False
    for i in tqdm(range(n), desc=f"sdxl[{source_name}]"):
        path = out_dir / f"{i:03d}.png"
        if path.is_file():
            paths.append(path)
            continue
        emb = torch.from_numpy(clip_embeds[i : i + 1]).to(device=device, dtype=pipe.dtype)
        uncond = torch.zeros_like(emb)
        image_embeds = torch.cat([uncond, emb], dim=0)
        if use_unsqueeze:
            image_embeds = image_embeds.unsqueeze(1)

        def _run(ie):
            return pipe(
                prompt="",
                negative_prompt="",
                ip_adapter_image_embeds=[ie],
                num_inference_steps=steps,
                guidance_scale=5.0,
                height=height,
                width=width,
                generator=g,
            )

        if not layout_ok:
            try:
                result = _run(image_embeds)
            except Exception:
                image_embeds = torch.cat([uncond, emb], dim=0).unsqueeze(1)
                result = _run(image_embeds)
                use_unsqueeze = True
            layout_ok = True
        else:
            result = _run(image_embeds)
        result.images[0].save(path)
        paths.append(path)
    del pipe
    torch.cuda.empty_cache()
    return paths


@torch.no_grad()
def _encode(model, tokens, atm, device, key: str, bs: int = 64) -> np.ndarray:
    model.eval()
    outs = []
    tok = torch.from_numpy(tokens.astype(np.float32))
    atm_t = torch.from_numpy(atm.astype(np.float32))
    for i in range(0, len(tok), bs):
        out = model(tok[i : i + bs].to(device), atm_t[i : i + bs].to(device))
        outs.append(F.normalize(out[key].float(), dim=-1).cpu().numpy())
    return np.concatenate(outs, 0).astype(np.float32)


def _load_model(ckpt: Path, project: Path, device: torch.device, residual_mode: str):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    nb = cfg.setdefault("neurobolt", {})
    nb["use_bit"] = True
    nb["residual_mode"] = residual_mode
    model = NeuroBoltBrainITPipeline.from_config(cfg, project_root=str(project)).to(device)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    print(
        f"[INFO] loaded {ckpt.name} residual_mode={model.residual_mode} "
        f"missing={len(missing)} unexpected={len(unexpected)} "
        f"bit_res_scale={float(model.bit_res_scale):.4f}"
    )
    return model


def _make_grid(gen_paths: list[Path], gt_paths: list[Path], out_path: Path, n: int) -> None:
    cells = []
    for i in range(min(n, len(gen_paths), len(gt_paths))):
        g = Image.open(gen_paths[i]).convert("RGB").resize((256, 256))
        t = Image.open(gt_paths[i]).convert("RGB").resize((256, 256))
        row = Image.new("RGB", (512, 256))
        row.paste(t, (0, 0))
        row.paste(g, (256, 0))
        cells.append(row)
    canvas = Image.new("RGB", (512, 256 * len(cells)), (255, 255, 255))
    for i, row in enumerate(cells):
        canvas.paste(row, (0, i * 256))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f"[INFO] wrote grid {out_path} (left=GT right=gen)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="sub-08")
    parser.add_argument(
        "--neurobolt-dir",
        default="/project/peilab/why/cache/things_eeg2_b2/neurobolt",
    )
    parser.add_argument("--bridge-dir", default="outputs/atm_bridge")
    parser.add_argument("--ckpt", default="outputs/nb_bit_s3_sub08/checkpoints/nb_stage3_best.pt")
    parser.add_argument("--output-dir", default="outputs/eval/nb_bit_direct_smoke")
    parser.add_argument("--max-images", type=int, default=8)
    parser.add_argument("--gen-steps", type=int, default=20)
    parser.add_argument("--gen-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    bridge_dir = Path(args.bridge_dir)
    if not bridge_dir.is_absolute():
        bridge_dir = ROOT / bridge_dir
    ckpt = Path(args.ckpt)
    if not ckpt.is_absolute():
        ckpt = ROOT / ckpt

    ds = NeuroBoltBridgeDataset(
        args.neurobolt_dir,
        bridge_dir,
        args.subject,
        split="test",
        teacher_img=ROOT / "outputs/eval/atm_baseline/test_ViT-H-14_laion2b_s32b_b79k_features.npy",
    )
    tokens = np.stack([ds[i]["fmri_tokens"].numpy() for i in range(len(ds))])
    atm = np.stack([ds[i]["atm_emb"].numpy() for i in range(len(ds))])
    img = np.stack([ds[i]["clip_emb"].numpy() for i in range(len(ds))])
    print(f"[INFO] NeuroBOLT tokens={tokens.shape} (glb.pth offline) atm={atm.shape}")

    report: dict = {
        "pipeline": "NeuroBOLT(glb tokens) → RoiToBitBridge → BiT → SDXL+IP-Adapter",
        "subject": args.subject,
        "ckpt": str(ckpt),
        "retrieval": {"atm": _retrieval(atm, img)},
        "generation": {},
    }
    print(
        f"[INFO] ATM top1={report['retrieval']['atm']['top1']*100:.2f}% "
        f"chance={report['retrieval']['atm']['chance_top1']*100:.2f}%"
    )

    gt_paths = list_test_images(Path("/project/peilab/why/data/images_set"))
    modes = [
        ("as_trained_atm_skip", "atm_skip"),
        ("pure_nb_bit", "none"),
    ]

    for tag, residual in modes:
        model = _load_model(ckpt, ROOT, device, residual)
        emb = _encode(model, tokens, atm, device, "bit_clip")
        ret = _retrieval(emb, img)
        report["retrieval"][tag] = ret
        print(f"[INFO] retrieval[{tag}] top1={ret['top1']*100:.2f}% top5={ret['top5']*100:.2f}%")
        np.save(out_dir / f"{args.subject}_{tag}_1024.npy", emb)

        gen_dir = out_dir / "generated" / tag
        for p in gen_dir.glob("*.png") if gen_dir.is_dir() else []:
            p.unlink()
        paths = generate_images_sdxl(
            emb,
            gen_dir,
            device,
            steps=args.gen_steps,
            height=args.gen_size,
            width=args.gen_size,
            max_images=args.max_images,
            seed=args.seed,
            source_name=tag,
        )
        _make_grid(paths, gt_paths, out_dir / f"grid_{tag}.png", args.max_images)
        report["generation"][tag] = {"n": len(paths), "dir": str(gen_dir), "grid": str(out_dir / f"grid_{tag}.png")}
        del model
        torch.cuda.empty_cache()

    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[OK] wrote {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
