#!/usr/bin/env python3
"""Paper-standard two-way identification (MindEye / Brain-Diffuser / Ozcelik).

For each GT i and each other reconstruction j≠i:
  correct if corr(emb(GT_i), emb(recon_i)) > corr(emb(GT_i), emb(recon_j))
Average over all j for each i, then mean over i. Chance = 50%.

Also reports:
  - paired CLIP cosine (mean)
  - retrieval-style N-way Top-1 of recon→GT in CLIP space (optional context)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


def list_gen(gen_dir: Path) -> list[Path]:
    paths = sorted(gen_dir.glob("*.png"))
    return paths or sorted(gen_dir.glob("*.jpg"))


def list_test_images(images_root: Path) -> list[Path]:
    root = images_root / "test_images"
    paths: list[Path] = []
    for d in sorted([p for p in root.iterdir() if p.is_dir()]):
        imgs = sorted(list(d.glob("*.jpg")) + list(d.glob("*.JPEG")) + list(d.glob("*.png")))
        paths.extend(imgs)
    return paths


@torch.no_grad()
def encode_clip_vitl(paths: list[Path], device: torch.device, batch_size: int = 16) -> np.ndarray:
    """MindEye uses CLIP ViT-L/14 final layer for 2-way."""
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai", device=device
    )
    model.eval()
    embs = []
    for i in tqdm(range(0, len(paths), batch_size), desc="clip-L14"):
        batch = paths[i : i + batch_size]
        imgs = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in batch]).to(device)
        z = model.encode_image(imgs).float()
        z = F.normalize(z, dim=-1)
        embs.append(z.cpu().numpy())
    return np.concatenate(embs, axis=0).astype(np.float32)


def two_way_identification(gt_emb: np.ndarray, recon_emb: np.ndarray) -> dict:
    """Pearson-corr based 2-way as in Ozcelik/MindEye (on already L2-normalized vectors,
    Pearson corr of centered vectors ≈ cosine of centered; we follow common practice of
    using Pearson correlation on the embedding vectors).
    """
    n = gt_emb.shape[0]
    assert recon_emb.shape[0] == n

    # center per-vector then cosine ≡ pearson for 1D vectors
    def pearson_mat(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a = a - a.mean(axis=1, keepdims=True)
        b = b - b.mean(axis=1, keepdims=True)
        a = a / np.linalg.norm(a, axis=1, keepdims=True).clip(1e-8)
        b = b / np.linalg.norm(b, axis=1, keepdims=True).clip(1e-8)
        return a @ b.T

    # For each GT i: corr(GT_i, recon_k) for all k
    # MindEye: corr(GT_i, recon_i) vs corr(GT_i, recon_j)
    sim = pearson_mat(gt_emb, recon_emb)  # (N,N) rows=GT, cols=recon
    per_i = []
    for i in range(n):
        s_ii = sim[i, i]
        others = [1.0 if s_ii > sim[i, j] else 0.0 for j in range(n) if j != i]
        per_i.append(float(np.mean(others)))
    acc = float(np.mean(per_i))
    # also cosine paired
    cos = float(np.mean((gt_emb * recon_emb).sum(axis=1)))
    # N-way: argmax_j corr(GT_i, recon_j) == i  (identification of which recon matches GT)
    # More common for recon: argmax over GT for each recon
    top1_recon_to_gt = float((sim.argmax(axis=0) == np.arange(n)).mean())
    top1_gt_to_recon = float((sim.argmax(axis=1) == np.arange(n)).mean())
    return {
        "clip_2way": acc,
        "clip_2way_pct": 100.0 * acc,
        "clip_cosine_paired": cos,
        "nway_top1_gt_to_recon": top1_gt_to_recon,
        "nway_top1_recon_to_gt": top1_recon_to_gt,
        "chance_2way": 0.5,
        "n": n,
        "protocol": "MindEye/Ozcelik: for each GT i, avg over j≠i of 1[corr(GT_i,recon_i)>corr(GT_i,recon_j)]; CLIP ViT-L/14",
    }


def two_way_retrieval(eeg: np.ndarray, img: np.ndarray) -> dict:
    """EEG↔image retrieval 2-way (chance 50%): for each i, avg over j≠i of
    1[cos(eeg_i,img_i) > cos(eeg_i,img_j)].
    """
    eeg = eeg / np.linalg.norm(eeg, axis=1, keepdims=True).clip(1e-8)
    img = img / np.linalg.norm(img, axis=1, keepdims=True).clip(1e-8)
    sim = eeg @ img.T
    n = len(eeg)
    per = []
    for i in range(n):
        s_ii = sim[i, i]
        per.append(float(np.mean([1.0 if s_ii > sim[i, j] else 0.0 for j in range(n) if j != i])))
    nway = float((sim.argmax(1) == np.arange(n)).mean())
    return {
        "retrieval_2way": float(np.mean(per)),
        "retrieval_2way_pct": 100.0 * float(np.mean(per)),
        "retrieval_nway_top1": nway,
        "n": n,
        "chance_2way": 0.5,
        "chance_nway": 1.0 / n,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dirs", type=str, required=True, help="comma-separated tag=path or just paths")
    ap.add_argument("--images-root", type=str, default="/project/peilab/why/data/images_set")
    ap.add_argument("--output-json", type=str, required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--eeg-embed-npy", type=str, default="", help="optional: also eval retrieval 2-way")
    ap.add_argument("--img-gallery-npy", type=str, default="", help="clean RN50 projected or raw gallery")
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    cache = Path("/project/peilab/why/cache/eeg-brainit")
    os.environ.setdefault("HF_HOME", str(cache / "hf"))
    os.environ.setdefault("HF_HUB_CACHE", str(cache / "hf" / "hub"))
    os.environ.setdefault("OPENCLIP_CACHE_DIR", str(cache / "open_clip"))
    os.environ.setdefault("TORCH_HOME", str(cache / "torch"))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    gt_paths = list_test_images(Path(args.images_root))

    results = []
    gt_emb = None
    for item in [x.strip() for x in args.gen_dirs.split(",") if x.strip()]:
        if "=" in item:
            tag, path = item.split("=", 1)
        else:
            path = item
            tag = Path(path).parent.name if Path(path).name == "generated" else Path(path).name
        gen_dir = Path(path)
        if gen_dir.name != "generated" and (gen_dir / "generated").is_dir():
            gen_dir = gen_dir / "generated"
        gens = list_gen(gen_dir)
        n = min(len(gens), len(gt_paths))
        if n < 2:
            print(f"[WARN] skip {tag}: n={n}")
            continue
        if gt_emb is None or gt_emb.shape[0] != n:
            print(f"[INFO] encoding GT n={n}")
            gt_emb = encode_clip_vitl(gt_paths[:n], device, args.batch_size)
        print(f"[INFO] encoding recon [{tag}]")
        recon_emb = encode_clip_vitl(gens[:n], device, args.batch_size)
        metrics = two_way_identification(gt_emb, recon_emb)
        metrics["tag"] = tag
        metrics["gen_dir"] = str(gen_dir)
        results.append(metrics)
        print(f"[OK] {tag}: CLIP-2way={metrics['clip_2way_pct']:.2f}% cosine={metrics['clip_cosine_paired']:.4f}")

    retrieval = None
    if args.eeg_embed_npy and args.img_gallery_npy:
        eeg = np.load(args.eeg_embed_npy).astype(np.float32)
        img = np.load(args.img_gallery_npy).astype(np.float32)
        if img.ndim > 2:
            # (200,1,D) or similar
            while img.ndim > 2:
                img = img.mean(axis=1) if img.shape[1] <= 10 else img.reshape(img.shape[0], -1)
                if img.ndim > 2:
                    img = img.mean(axis=tuple(range(1, img.ndim - 1)))
        # if dims differ, skip retrieval (need same space)
        if eeg.shape[-1] == img.shape[-1] and len(eeg) == len(img):
            retrieval = two_way_retrieval(eeg, img)
            print(f"[OK] retrieval 2-way={retrieval['retrieval_2way_pct']:.2f}% nway={100*retrieval['retrieval_nway_top1']:.1f}%")
        else:
            retrieval = {"error": f"shape mismatch eeg{eeg.shape} img{img.shape}"}

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metric": "two-way identification",
        "authority": "Standard in MindEye / MindEye2 / Brain-Diffuser / Ozcelik&VanRullen; chance=50%. Complementary to N-way Top-1 (harder, chance=1/N).",
        "backbone": "open_clip ViT-L-14 openai",
        "generation_2way": results,
        "retrieval_2way": retrieval,
        "best_2way": max(results, key=lambda r: r["clip_2way"]) if results else None,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"[OK] wrote {out}")


if __name__ == "__main__":
    main()
