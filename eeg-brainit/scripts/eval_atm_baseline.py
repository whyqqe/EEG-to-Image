#!/usr/bin/env python3
"""ATM pretrained-feature retrieval baseline on THINGS-EEG2 (safe, read-only inputs).

Uses locally cached ATM EEG embeddings + OpenCLIP ViT-H/14 image features.
Does not modify ATM/NICE checkpoints or source data; only writes under --output-dir.
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


def retrieval_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    """pred/target L2-normalized (N, D); N-way identification."""
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
    return {
        "n": int(n),
        "top1": float(top1 / n),
        "top5": float(top5 / n),
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(ranks.mean()),
    }


def list_test_images(images_root: Path) -> list[Path]:
    """One image per concept folder, sorted by folder name (THINGS-EEG2 test order)."""
    root = images_root / "test_images"
    if not root.is_dir():
        raise FileNotFoundError(root)
    paths: list[Path] = []
    for d in sorted([p for p in root.iterdir() if p.is_dir()]):
        imgs = sorted(list(d.glob("*.jpg")) + list(d.glob("*.JPEG")) + list(d.glob("*.png")))
        if not imgs:
            raise FileNotFoundError(f"No image in {d}")
        paths.append(imgs[0])
    if len(paths) != 200:
        raise RuntimeError(f"Expected 200 test images, got {len(paths)}")
    return paths


@torch.no_grad()
def encode_images(
    paths: list[Path],
    model,
    preprocess,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    embs = []
    for i in tqdm(range(0, len(paths), batch_size), desc="clip-encode"):
        batch = paths[i : i + batch_size]
        x = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in batch], dim=0).to(device)
        feat = model.encode_image(x)
        feat = F.normalize(feat.float(), dim=-1)
        embs.append(feat.cpu().numpy().astype(np.float32))
    return np.concatenate(embs, axis=0)


def maybe_verify_train_teacher(
    atm_dir: Path,
    images_root: Path,
    model,
    preprocess,
    device: torch.device,
) -> dict | None:
    """Spot-check that OpenCLIP tag matches cached ViT-H-14 train teachers."""
    feat_path = atm_dir / "ViT-H-14_features_train.pt"
    train_root = images_root / "training_images"
    if not feat_path.is_file() or not train_root.is_dir():
        return None
    obj = torch.load(feat_path, map_location="cpu", weights_only=False)
    img_feat = obj["img_features"] if isinstance(obj, dict) else obj
    img_feat = F.normalize(img_feat.float(), dim=-1)

    # First concept folder, first image -> index 0 in ATM packing (1654*10).
    concepts = sorted([p for p in train_root.iterdir() if p.is_dir()])
    if not concepts:
        return None
    imgs = sorted(
        list(concepts[0].glob("*.jpg"))
        + list(concepts[0].glob("*.JPEG"))
        + list(concepts[0].glob("*.png"))
    )
    if not imgs:
        return None
    with torch.no_grad():
        x = preprocess(Image.open(imgs[0]).convert("RGB")).unsqueeze(0).to(device)
        pred = F.normalize(model.encode_image(x).float(), dim=-1).cpu()
    cos0 = float((pred[0] * img_feat[0]).sum())
    # Also try mean over first 10 images of concept 0 vs feature block
    return {
        "probe_image": str(imgs[0]),
        "cosine_to_cached_index0": cos0,
        "match_likely": bool(cos0 > 0.9),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--atm-dir",
        type=str,
        default="/project/peilab/why/eeg-to-image/checkpoints/atm",
    )
    parser.add_argument(
        "--images-root",
        type=str,
        default="/project/peilab/why/data/images_set",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/outputs/eval/atm_baseline",
    )
    parser.add_argument("--subjects", type=str, nargs="*", default=None)
    parser.add_argument("--clip-model", type=str, default="ViT-H-14")
    parser.add_argument("--clip-pretrained", type=str, default="laion2b_s32b_b79k")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--verify-teacher", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    project = Path("/project/peilab/why/eeg-brainit")
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = project / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    atm_dir = Path(args.atm_dir)
    images_root = Path(args.images_root)
    cache_root = Path("/project/peilab/why/cache/eeg-brainit")
    os.environ.setdefault("HF_HOME", str(cache_root / "hf"))
    os.environ.setdefault("OPENCLIP_CACHE_DIR", str(cache_root / "open_clip"))
    os.environ.setdefault("TORCH_HOME", str(cache_root / "torch"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    test_paths = list_test_images(images_root)
    if args.smoke:
        test_paths = test_paths[:16]
        print("[INFO] smoke mode: first 16 test images only")

    feat_cache = out_dir / f"test_{args.clip_model}_{args.clip_pretrained}_features.npy"
    paths_cache = out_dir / "test_image_paths.json"

    import open_clip

    candidate_tags = [args.clip_pretrained]
    # Fallbacks commonly used with ViT-H-14 in ATM-style pipelines.
    for tag in ("laion2b_s32b_b79k", "laion2b_s32b_b79k_fp16", "metaclip_fullcc"):
        if tag not in candidate_tags:
            candidate_tags.append(tag)

    model = preprocess = None
    chosen_tag = None
    verify = None
    for tag in candidate_tags:
        print(f"[INFO] trying OpenCLIP {args.clip_model} / {tag}")
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                args.clip_model, pretrained=tag, device=device
            )
        except Exception as exc:  # noqa: BLE001 - try next tag
            print(f"[WARN] cannot create model with tag={tag}: {exc}")
            continue
        model.eval()
        chosen_tag = tag
        if args.verify_teacher and not args.smoke:
            verify = maybe_verify_train_teacher(atm_dir, images_root, model, preprocess, device)
            print(f"[INFO] teacher verify ({tag}): {verify}")
            if verify is not None and verify.get("match_likely"):
                break
            # If verify inconclusive, still keep first creatable tag but continue searching better.
            if verify is not None and verify.get("cosine_to_cached_index0", 0) > 0.5:
                break
        else:
            break

    if model is None or chosen_tag is None:
        raise RuntimeError(f"Failed to create OpenCLIP model from tags={candidate_tags}")
    args.clip_pretrained = chosen_tag
    print(f"[INFO] using clip pretrained tag={chosen_tag}")

    if feat_cache.is_file() and paths_cache.is_file() and not args.smoke:
        img_feat = np.load(feat_cache)
        with paths_cache.open("r", encoding="utf-8") as f:
            cached_paths = json.load(f)
        if cached_paths != [str(p) for p in test_paths]:
            raise RuntimeError("Cached test paths mismatch; delete cache or regenerate")
        print(f"[INFO] loaded cached test image features {img_feat.shape}")
    else:
        img_feat = encode_images(test_paths, model, preprocess, device, batch_size=args.batch_size)
        if not args.smoke:
            np.save(feat_cache, img_feat)
            with paths_cache.open("w", encoding="utf-8") as f:
                json.dump([str(p) for p in test_paths], f, indent=2)
            print(f"[OK] wrote {feat_cache}")

    img_feat = img_feat / np.linalg.norm(img_feat, axis=1, keepdims=True).clip(min=1e-8)

    subjects = args.subjects or [f"sub-{i:02d}" for i in range(1, 11)]
    per = {}
    tops1, tops5, meds = [], [], []
    for sub in subjects:
        eeg_path = atm_dir / f"ATM_S_eeg_features_{sub}_test.pt"
        if not eeg_path.is_file():
            raise FileNotFoundError(eeg_path)
        eeg = torch.load(eeg_path, map_location="cpu", weights_only=False)
        if isinstance(eeg, dict):
            # tolerate wrapped formats
            for k in ("eeg", "features", "emb", "data"):
                if k in eeg and torch.is_tensor(eeg[k]):
                    eeg = eeg[k]
                    break
        eeg = eeg.float().numpy()
        if args.smoke:
            eeg = eeg[: len(test_paths)]
        if eeg.shape[0] != img_feat.shape[0]:
            raise RuntimeError(f"{sub}: eeg {eeg.shape} vs img {img_feat.shape}")
        eeg = eeg / np.linalg.norm(eeg, axis=1, keepdims=True).clip(min=1e-8)
        # Dim check / project if needed (should both be 1024 for H/14)
        if eeg.shape[1] != img_feat.shape[1]:
            raise RuntimeError(
                f"Dim mismatch eeg {eeg.shape[1]} vs img {img_feat.shape[1]}; "
                "wrong CLIP pretrained tag?"
            )
        m = retrieval_metrics(eeg.astype(np.float32), img_feat.astype(np.float32))
        paired = float((eeg * img_feat).sum(1).mean())
        shuffled = float((eeg * img_feat[np.random.RandomState(0).permutation(len(eeg))]).sum(1).mean())
        m["paired_cos"] = paired
        m["shuffled_cos"] = shuffled
        m["cos_gap"] = paired - shuffled
        per[sub] = m
        tops1.append(m["top1"])
        tops5.append(m["top5"])
        meds.append(m["median_rank"])
        print(
            f"[INFO] {sub}: top1={m['top1']*100:.2f}% top5={m['top5']*100:.2f}% "
            f"med={m['median_rank']:.1f} cos_gap={m['cos_gap']:.4f}"
        )

    summary = {
        "atm_dir": str(atm_dir),
        "images_root": str(images_root),
        "clip_model": args.clip_model,
        "clip_pretrained": args.clip_pretrained,
        "n_test_images": int(img_feat.shape[0]),
        "feature_dim": int(img_feat.shape[1]),
        "teacher_verify": verify,
        "chance_top1": 1.0 / float(img_feat.shape[0]),
        "macro_top1": float(np.mean(tops1)),
        "macro_top5": float(np.mean(tops5)),
        "macro_median_rank": float(np.mean(meds)),
        "per_subject": per,
        "note": (
            "ATM EEG embeddings are used as-is (already CLIP-aligned). "
            "Test image features are encoded with OpenCLIP; order follows sorted test_images/*."
        ),
    }
    out_json = out_dir / ("metrics_smoke.json" if args.smoke else "metrics.json")
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("========== ATM BASELINE ==========")
    print(
        f"macro Top-1={summary['macro_top1']*100:.2f}%  "
        f"Top-5={summary['macro_top5']*100:.2f}%  "
        f"medRank={summary['macro_median_rank']:.1f}  "
        f"(chance={summary['chance_top1']*100:.2f}%)"
    )
    print(f"[OK] wrote {out_json}")


if __name__ == "__main__":
    main()
