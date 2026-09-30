#!/usr/bin/env python3
"""Phase-2 smoke: NOD GT fMRI ROI → CLIP → SDXL (+ optional Phase-1 pred cascade).

Establishes the image-generation ceiling when fMRI is ground-truth (classmean /
stimvar ROIs), using a thin MLP into CLIP ViT-H/14 + IP-Adapter SDXL.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("XFORMERS_DISABLED", "1")
os.environ.setdefault("HF_HOME", "/project/peilab/why/cache/eeg-brainit/hf")
os.environ.setdefault("HF_HUB_CACHE", "/project/peilab/why/cache/eeg-brainit/hf/hub")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


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


def _split_by_image_id(image_ids: list[str], val_frac: float, seed: int):
    rng = random.Random(seed)
    uniq = sorted(set(image_ids))
    rng.shuffle(uniq)
    n_val = max(1, int(len(uniq) * val_frac)) if len(uniq) > 1 else 0
    val_set = set(uniq[:n_val])
    train = [i for i, x in enumerate(image_ids) if x not in val_set]
    val = [i for i, x in enumerate(image_ids) if x in val_set]
    return train, val


class FmriToClip(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 1024, hidden: int = 512, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x.float()), dim=-1)


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
    import diffusers.utils.import_utils as iu

    iu._xformers_available = False
    for name in list(sys.modules):
        if name.startswith("diffusers.models.attention_processor") or name.startswith("diffusers.loaders.ip_adapter"):
            del sys.modules[name]


@torch.no_grad()
def generate_images_sdxl(
    clip_embeds: np.ndarray,
    out_dir: Path,
    device: torch.device,
    steps: int = 20,
    size: int = 512,
    max_images: int = 8,
    seed: int = 42,
    tag: str = "gt",
) -> list[Path]:
    _disable_broken_xformers()
    from diffusers import StableDiffusionXLPipeline

    out_dir.mkdir(parents=True, exist_ok=True)
    n = min(len(clip_embeds), max_images) if max_images > 0 else len(clip_embeds)
    hub = Path(os.environ["HF_HUB_CACHE"])
    model_id = resolve_sdxl_model_path(hub)
    print(f"[INFO] loading SDXL for {tag} from {model_id}")
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
        raise FileNotFoundError("IP-Adapter missing under HF_HUB_CACHE")
    kwargs = {"subfolder": "sdxl_models", "image_encoder_folder": None, "local_files_only": True}
    try:
        pipe.load_ip_adapter(str(ip_root), weight_name="ip-adapter_sdxl_vit-h.bin", **kwargs)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] {exc}; fallback ip-adapter_sdxl.bin")
        pipe.load_ip_adapter(str(ip_root), weight_name="ip-adapter_sdxl.bin", **kwargs)
    pipe.set_ip_adapter_scale(1.0)

    paths: list[Path] = []
    g = torch.Generator(device=device).manual_seed(seed)
    use_unsqueeze = False
    layout_ok = False
    for i in tqdm(range(n), desc=f"sdxl[{tag}]"):
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
                height=size,
                width=size,
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


def _make_grid(gen_paths: list[Path], gt_paths: list[Path], out_path: Path) -> None:
    cells = []
    for g, t in zip(gen_paths, gt_paths):
        gi = Image.open(g).convert("RGB").resize((256, 256))
        ti = Image.open(t).convert("RGB").resize((256, 256))
        row = Image.new("RGB", (512, 256))
        row.paste(ti, (0, 0))
        row.paste(gi, (256, 0))
        cells.append(row)
    canvas = Image.new("RGB", (512, 256 * len(cells)), (255, 255, 255))
    for i, row in enumerate(cells):
        canvas.paste(row, (0, i * 256))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f"[INFO] wrote grid {out_path}")


def _resolve_stim(images_root: Path, image_id: str) -> Path | None:
    for ext in (".JPEG", ".jpg", ".png", ".jpeg"):
        p = images_root / f"{image_id}{ext}"
        if p.is_file():
            return p
    hits = list(images_root.glob(f"{image_id}.*"))
    return hits[0] if hits else None


def load_subject_pack(pairs_root: Path, clip_dir: Path, subject: str):
    z = np.load(pairs_root / subject / "pairs.npz")
    meta = json.loads((pairs_root / subject / "pairs_meta.json").read_text(encoding="utf-8"))
    index = json.loads((clip_dir / "index.json").read_text(encoding="utf-8"))
    emb = np.load(clip_dir / "embeddings.npy").astype(np.float32)
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)
    fmri = z["fmri_roi"].astype(np.float32)
    eeg = z["eeg"].astype(np.float32)
    ids = [t["image_id"] for t in meta["trials"]]
    if len(ids) != len(fmri):
        raise RuntimeError("trials/fmri length mismatch")
    keep = [i for i, iid in enumerate(ids) if iid in index]
    if not keep:
        raise RuntimeError(f"no CLIP overlap for {subject}")
    fmri = fmri[keep]
    eeg = eeg[keep]
    ids = [ids[i] for i in keep]
    clip = np.stack([emb[index[i]] for i in ids], axis=0)
    ch_names = meta.get("ch_names") or [f"C{i}" for i in range(eeg.shape[1])]
    print(f"[INFO] {subject} n={len(ids)} fmri={fmri.shape} eeg={eeg.shape} clip_miss={len(meta['trials'])-len(ids)}")
    return fmri, clip, ids, ch_names, eeg


@torch.no_grad()
def predict_fmri_phase1(eeg: np.ndarray, ch_names: list[str], ckpt: Path, device: torch.device) -> np.ndarray:
    from eeg_brainit.models.neurobolt_eeg2fmri import NeuroBoltEEG2fMRI

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ck.get("cfg", {})
    mcfg = cfg.get("eeg2fmri", {})
    model = NeuroBoltEEG2fMRI(
        ch_names=ch_names,
        num_rois=int(mcfg.get("num_rois", 64)),
        clip_dim=int(mcfg.get("clip_dim", 1024)),
        hidden=int(mcfg.get("hidden", 512)),
        dropout=float(mcfg.get("dropout", 0.3)),
        use_clip_head=False,
        glb_ckpt=str(mcfg.get("glb_ckpt", "checkpoints/neurobolt/glb.pth")),
        patch_size=int(mcfg.get("patch_size", 200)),
        win_level=int(mcfg.get("win_level", 1)),
        unfreeze_last_n_blocks=0,
        train_mss=False,
        use_mss=bool(mcfg.get("use_mss", False)),
        train_patch_embed=False,
        heads_only=True,
        head_depth=int(mcfg.get("head_depth", 1)),
    ).to(device)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    print(f"[INFO] loaded Phase-1 {ckpt.name} missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()
    outs = []
    x = torch.from_numpy(eeg)
    bs = 64
    for i in range(0, len(x), bs):
        out = model(x[i : i + bs].to(device))
        outs.append(out["fmri_pred"].float().cpu().numpy())
    return np.concatenate(outs, 0).astype(np.float32)


def train_adapter(
    fmri: np.ndarray,
    clip: np.ndarray,
    train_idx: list[int],
    val_idx: list[int],
    device: torch.device,
    epochs: int = 40,
    lr: float = 1e-3,
    batch_size: int = 64,
) -> tuple[FmriToClip, dict]:
    model = FmriToClip(fmri.shape[1], clip.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    best = -1e9
    best_cos = -1e9
    best_state = None
    history = []
    tr_f = torch.from_numpy(fmri[train_idx])
    tr_c = torch.from_numpy(clip[train_idx])
    va_f = torch.from_numpy(fmri[val_idx]).to(device)
    va_c = torch.from_numpy(clip[val_idx]).to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(tr_f))
        loss_sum = 0.0
        n_seen = 0
        for i in range(0, len(perm), batch_size):
            idx = perm[i : i + batch_size]
            pred = model(tr_f[idx].to(device))
            tgt = F.normalize(tr_c[idx].to(device), dim=-1)
            loss = 1.0 - (pred * tgt).sum(-1).mean()
            # mild batch NCE
            logits = pred @ tgt.T / 0.07
            nce = F.cross_entropy(logits, torch.arange(len(pred), device=device))
            total = loss + 0.2 * nce
            opt.zero_grad(set_to_none=True)
            total.backward()
            opt.step()
            loss_sum += float(total) * len(pred)
            n_seen += len(pred)
        model.eval()
        with torch.no_grad():
            pred = model(va_f)
            tgt = F.normalize(va_c, dim=-1)
            cos = float((pred * tgt).sum(-1).mean())
            ret = _retrieval(pred.cpu().numpy(), tgt.cpu().numpy())
        row = {"epoch": epoch, "train_loss": loss_sum / max(n_seen, 1), "val_cos": cos, **{f"val_{k}": v for k, v in ret.items()}}
        history.append(row)
        print(
            f"[adapter {epoch:03d}] loss={row['train_loss']:.4f} cos={cos:.4f} "
            f"top1={ret['top1']*100:.2f}% top5={ret['top5']*100:.2f}%"
        )
        if ret["top1"] > best:
            best = ret["top1"]
            best_cos = cos
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val_top1": best, "best_val_cos": best_cos, "history": history}


@torch.no_grad()
def encode_fmri(model: FmriToClip, fmri: np.ndarray, device: torch.device, bs: int = 256) -> np.ndarray:
    model.eval()
    outs = []
    x = torch.from_numpy(fmri)
    for i in range(0, len(x), bs):
        outs.append(model(x[i : i + bs].to(device)).cpu().numpy())
    return np.concatenate(outs, 0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="sub-01")
    parser.add_argument("--target-mode", default="classmean", choices=["classmean", "stimvar"])
    parser.add_argument("--pairs-root", default="")
    parser.add_argument("--clip-dir", default="data/nod/processed/clip_vit_h14")
    parser.add_argument("--images-root", default="data/nod/raw/ds005811/stimuli/ImageNet")
    parser.add_argument("--phase1-ckpt", default="outputs/nod_eeg2fmri/neurobolt_classmean_v3/checkpoints/best.pt")
    parser.add_argument("--output-dir", default="outputs/eval/nod_phase2_gt_fmri2image")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--max-images", type=int, default=8)
    parser.add_argument("--gen-steps", type=int, default=20)
    parser.add_argument("--gen-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-cascade", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pairs_root = Path(args.pairs_root) if args.pairs_root else ROOT / "data/nod/processed" / args.target_mode
    clip_dir = ROOT / args.clip_dir if not Path(args.clip_dir).is_absolute() else Path(args.clip_dir)
    images_root = ROOT / args.images_root if not Path(args.images_root).is_absolute() else Path(args.images_root)
    out_dir = ROOT / args.output_dir / f"{args.subject}_{args.target_mode}"
    out_dir.mkdir(parents=True, exist_ok=True)

    fmri, clip, ids, ch_names, eeg = load_subject_pack(pairs_root, clip_dir, args.subject)
    train_idx, val_idx = _split_by_image_id(ids, args.val_frac, args.seed)
    print(f"[INFO] split train={len(train_idx)} val={len(val_idx)} device={device}")

    adapter, adapt_meta = train_adapter(fmri, clip, train_idx, val_idx, device, epochs=args.epochs)
    torch.save({"model": adapter.state_dict(), "in_dim": fmri.shape[1], "meta": adapt_meta}, out_dir / "fmri2clip.pt")

    gt_clip_val = encode_fmri(adapter, fmri[val_idx], device)
    ret_gt = _retrieval(gt_clip_val, clip[val_idx])
    print(f"[INFO] GT-fMRI→CLIP val top1={ret_gt['top1']*100:.2f}% top5={ret_gt['top5']*100:.2f}%")

    # teacher CLIP retrieval floor on same split (identity)
    ret_teacher = _retrieval(clip[val_idx], clip[val_idx])
    report = {
        "phase": "Phase-2 GT fMRI → CLIP → SDXL",
        "subject": args.subject,
        "target_mode": args.target_mode,
        "adapter": adapt_meta,
        "retrieval": {"gt_fmri_to_clip": ret_gt, "teacher_identity": ret_teacher},
        "generation": {},
    }

    # generate on first max_images of val set
    gen_ids = [ids[i] for i in val_idx[: args.max_images]]
    gen_fmri = fmri[val_idx[: args.max_images]]
    gen_clip = encode_fmri(adapter, gen_fmri, device)
    gt_paths = []
    for iid in gen_ids:
        p = _resolve_stim(images_root, iid)
        if p is None:
            raise FileNotFoundError(f"missing stimulus {iid} under {images_root}")
        gt_paths.append(p)

    paths = generate_images_sdxl(
        gen_clip,
        out_dir / "generated" / "gt_fmri",
        device,
        steps=args.gen_steps,
        size=args.gen_size,
        max_images=args.max_images,
        seed=args.seed,
        tag="gt_fmri",
    )
    _make_grid(paths, gt_paths, out_dir / "grid_gt_fmri.png")
    report["generation"]["gt_fmri"] = {"n": len(paths), "grid": str(out_dir / "grid_gt_fmri.png"), "image_ids": gen_ids}

    if not args.skip_cascade:
        ckpt = ROOT / args.phase1_ckpt if not Path(args.phase1_ckpt).is_absolute() else Path(args.phase1_ckpt)
        if ckpt.is_file():
            pred = predict_fmri_phase1(eeg, ch_names, ckpt, device)
            # z-score pred like GT packs (already zscored in cache); keep as-is model output
            pred_clip_val = encode_fmri(adapter, pred[val_idx], device)
            ret_pred = _retrieval(pred_clip_val, clip[val_idx])
            report["retrieval"]["phase1_pred_fmri_to_clip"] = ret_pred
            print(f"[INFO] Phase1-pred→CLIP val top1={ret_pred['top1']*100:.2f}% top5={ret_pred['top5']*100:.2f}%")
            pred_clip_gen = encode_fmri(adapter, pred[val_idx[: args.max_images]], device)
            paths_p = generate_images_sdxl(
                pred_clip_gen,
                out_dir / "generated" / "phase1_pred",
                device,
                steps=args.gen_steps,
                size=args.gen_size,
                max_images=args.max_images,
                seed=args.seed,
                tag="phase1_pred",
            )
            _make_grid(paths_p, gt_paths, out_dir / "grid_phase1_pred.png")
            report["generation"]["phase1_pred"] = {
                "n": len(paths_p),
                "grid": str(out_dir / "grid_phase1_pred.png"),
                "ckpt": str(ckpt),
            }
        else:
            print(f"[WARN] Phase-1 ckpt missing: {ckpt}")

    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[OK] wrote {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
