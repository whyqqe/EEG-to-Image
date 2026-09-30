#!/usr/bin/env python3
"""Extract CLIP text embeddings for THINGS concepts (semantic teachers).

Uses short visible-description templates (MindAlign-style, no LLM required):
  - "a photo of a {concept}"
  - "a photo of {concept}"
Saves concept-level and flat per-image arrays aligned with image folder order.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def concept_from_dirname(name: str) -> str:
    # 00001_aardvark -> aardvark; 00007_air_conditioner -> air conditioner
    m = re.match(r"^\d+_(.+)$", name)
    raw = m.group(1) if m else name
    return raw.replace("_", " ").strip()


def list_concepts(images_root: Path, split: str) -> list[tuple[str, str, int]]:
    """Return list of (dirname, concept_phrase, n_images)."""
    root = images_root / f"{split}_images"
    out = []
    for d in sorted([p for p in root.iterdir() if p.is_dir()]):
        imgs = sorted(list(d.glob("*.jpg")) + list(d.glob("*.JPEG")) + list(d.glob("*.png")))
        if not imgs:
            raise FileNotFoundError(f"no images in {d}")
        out.append((d.name, concept_from_dirname(d.name), len(imgs)))
    return out


def templates_for(concept: str) -> list[str]:
    # article heuristic
    art = "an" if concept[:1].lower() in "aeiou" else "a"
    return [
        f"a photo of {art} {concept}",
        f"a photo of {concept}",
        f"Describe only what is directly visible in the image of {concept} in one short sentence",
    ]


@torch.no_grad()
def encode_texts(
    texts: list[str],
    model,
    tokenizer,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    feats = []
    for i in tqdm(range(0, len(texts), batch_size), desc="clip-text"):
        batch = texts[i : i + batch_size]
        tokens = tokenizer(batch).to(device)
        emb = model.encode_text(tokens)
        emb = F.normalize(emb.float(), dim=-1)
        feats.append(emb.cpu().numpy())
    return np.concatenate(feats, axis=0).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-root", type=str, default="/project/peilab/why/data/images_set")
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--model", type=str, default="ViT-H-14")
    ap.add_argument("--pretrained", type=str, default="laion2b_s32b_b79k")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--splits", type=str, default="training,test")
    args = ap.parse_args()

    os.environ.setdefault("OPENCLIP_CACHE_DIR", "/project/peilab/why/cache/eeg-brainit/open_clip")
    os.environ.setdefault("HF_HUB_CACHE", "/project/peilab/why/cache/eeg-brainit/hf/hub")

    import open_clip

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, _, _ = open_clip.create_model_and_transforms(
        args.model, pretrained=args.pretrained, device=device
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(args.model)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = {"model": args.model, "pretrained": args.pretrained, "splits": {}}

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        concepts = list_concepts(Path(args.images_root), split)
        # Encode 3 templates per concept, then mean-pool to one text emb
        all_tmpls: list[str] = []
        for _, phrase, _ in concepts:
            all_tmpls.extend(templates_for(phrase))
        tmpl_emb = encode_texts(all_tmpls, model, tokenizer, device, args.batch_size)
        tmpl_emb = tmpl_emb.reshape(len(concepts), 3, -1).mean(axis=1)
        tmpl_emb = tmpl_emb / np.linalg.norm(tmpl_emb, axis=1, keepdims=True).clip(1e-8)

        # Flat per-image (repeat concept emb for each image)
        flat = []
        phrases = []
        for i, (_, phrase, n_img) in enumerate(concepts):
            flat.append(np.repeat(tmpl_emb[i : i + 1], n_img, axis=0))
            phrases.extend([phrase] * n_img)
        flat_arr = np.concatenate(flat, axis=0).astype(np.float32)

        name = "train" if split == "training" else split
        split_dir = out / name
        split_dir.mkdir(parents=True, exist_ok=True)
        np.save(split_dir / "text_concept_clip.npy", tmpl_emb.astype(np.float32))
        np.save(split_dir / "text_flat_clip.npy", flat_arr)
        meta = {
            "n_concepts": len(concepts),
            "n_flat": int(flat_arr.shape[0]),
            "dim": int(flat_arr.shape[1]),
            "templates": templates_for("<concept>"),
            "concepts_head": [c[1] for c in concepts[:5]],
        }
        (split_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        # also save concept phrases for auditing
        (split_dir / "concept_phrases.json").write_text(
            json.dumps([c[1] for c in concepts], indent=2), encoding="utf-8"
        )
        report["splits"][name] = meta
        print(f"[OK] {name} concept={tmpl_emb.shape} flat={flat_arr.shape}")

    (out / "clip_text_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
