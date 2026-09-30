#!/usr/bin/env python3
"""Class consistency: classify generated images into THINGS concepts via CLIP text gallery."""

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


def list_gen(d: Path) -> list[Path]:
    return sorted(d.glob("*.png")) or sorted(d.glob("*.jpg"))


@torch.no_grad()
def encode_images(paths: list[Path], device: torch.device, bs: int = 16) -> np.ndarray:
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14", pretrained="laion2b_s32b_b79k", device=device
    )
    model.eval()
    embs = []
    for i in tqdm(range(0, len(paths), bs), desc="img"):
        batch = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths[i : i + bs]]).to(device)
        z = F.normalize(model.encode_image(batch).float(), dim=-1)
        embs.append(z.cpu().numpy())
    return np.concatenate(embs, 0).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dirs", type=str, required=True, help="tag=path,...")
    ap.add_argument("--text-concept-npy", type=str, required=True)
    ap.add_argument("--concepts-json", type=str, default="")
    ap.add_argument("--output-json", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    cache = Path("/project/peilab/why/cache/eeg-brainit")
    os.environ.setdefault("HF_HOME", str(cache / "hf"))
    os.environ.setdefault("HF_HUB_CACHE", str(cache / "hf" / "hub"))
    os.environ.setdefault("OPENCLIP_CACHE_DIR", str(cache / "open_clip"))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    text = np.load(args.text_concept_npy).astype(np.float32)
    text = text / np.linalg.norm(text, axis=1, keepdims=True).clip(1e-8)
    concepts = json.loads(Path(args.concepts_json).read_text()) if args.concepts_json else None
    gt = np.arange(text.shape[0])

    results = []
    for item in [x.strip() for x in args.gen_dirs.split(",") if x.strip()]:
        tag, path = item.split("=", 1) if "=" in item else (Path(item).name, item)
        gdir = Path(path)
        if gdir.name != "generated" and (gdir / "generated").is_dir():
            gdir = gdir / "generated"
        paths = list_gen(gdir)
        n = min(len(paths), text.shape[0])
        if n < 2:
            continue
        img = encode_images(paths[:n], device, args.batch_size)
        sim = img @ text[:n].T
        pred = sim.argmax(1)
        top1 = float((pred == gt[:n]).mean())
        # top5 class
        top5 = float(np.mean([gt[i] in np.argsort(sim[i])[-5:] for i in range(n)]))
        # confusion samples
        wrong = [int(i) for i in range(n) if pred[i] != gt[i]][:20]
        samples = []
        if concepts is not None:
            for i in wrong[:12]:
                samples.append({"i": i, "gt": concepts[i], "pred": concepts[int(pred[i])]})
        results.append({
            "tag": tag,
            "class_top1": top1,
            "class_top5": top5,
            "n": n,
            "error_samples": samples,
        })
        print(f"[OK] {tag}: class_top1={100*top1:.1f}% top5={100*top5:.1f}%")

    payload = {
        "metric": "CLIP ViT-H text-concept classification of generated images",
        "results": results,
        "best": max(results, key=lambda r: r["class_top1"]) if results else None,
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
