#!/usr/bin/env python3
"""Prepare trial-level THINGS-EEG2 manifest for EEG-Brain-IT.

Reads official packed subject files:
  Preprocessed_data_250Hz/sub-XX/preprocessed_eeg_{training,test}.npy
    dict with 'preprocessed_eeg_data' shaped (N, R, 63, T)

Averages repetitions -> (N, 63, T), writes compact per-subject arrays, and
pairs each trial with the matching image via images_set/image_metadata.npy.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm


def load_eeg_dict(path: Path) -> dict:
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        obj = obj.item() if obj.shape == () else obj
    if not isinstance(obj, dict) or "preprocessed_eeg_data" not in obj:
        raise ValueError(f"Unexpected EEG file format: {path}")
    return obj


def build_image_index(images_dir: Path) -> dict[str, Path]:
    idx: dict[str, Path] = {}
    for split_name in ("training_images", "test_images"):
        root = images_dir / split_name
        if not root.is_dir():
            continue
        for p in root.rglob("*.jpg"):
            idx[p.name] = p.resolve()
        for p in root.rglob("*.JPEG"):
            idx[p.name] = p.resolve()
    return idx


def parse_subjects(spec: str) -> list[str]:
    """Accept 'sub-01', '1', '1-3', 'sub-01,sub-02'."""
    spec = spec.strip()
    if not spec:
        return []
    out: list[str] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part and not part.startswith("sub"):
            a, b = part.split("-", 1)
            for i in range(int(a), int(b) + 1):
                out.append(f"sub-{i:02d}")
        elif part.startswith("sub-"):
            out.append(part)
        else:
            out.append(f"sub-{int(part):02d}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eeg-dir",
        type=Path,
        default=Path("/project/peilab/why/data/Preprocessed_data_250Hz"),
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("/project/peilab/why/data/images_set"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("/project/peilab/why/data/images_set/image_metadata.npy"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed/things-eeg2"),
        help="Where to write per-subject (N,C,T) arrays",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/manifest.jsonl"),
    )
    parser.add_argument("--subjects", type=str, default="1-10")
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-train-per-subject",
        type=int,
        default=0,
        help="Optional cap for quick POC (0 = all training trials)",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    subjects = parse_subjects(args.subjects)
    meta = np.load(args.metadata, allow_pickle=True).item()
    train_files: list[str] = list(meta["train_img_files"])
    test_files: list[str] = list(meta["test_img_files"])
    image_index = build_image_index(args.images_dir)
    print(f"[INFO] subjects={subjects}")
    print(f"[INFO] metadata train={len(train_files)} test={len(test_files)}")
    print(f"[INFO] image index size={len(image_index)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for subject in subjects:
        subj_dir = args.eeg_dir / subject
        train_src = subj_dir / "preprocessed_eeg_training.npy"
        test_src = subj_dir / "preprocessed_eeg_test.npy"
        if not train_src.is_file() or not test_src.is_file():
            print(f"[WARN] missing EEG for {subject}, skip")
            continue

        out_subj = args.out_dir / subject
        out_subj.mkdir(parents=True, exist_ok=True)
        train_dst = out_subj / "train_eeg.npy"
        test_dst = out_subj / "test_eeg.npy"

        for split_name, src, dst, img_files in (
            ("train", train_src, train_dst, train_files),
            ("test", test_src, test_dst, test_files),
        ):
            if dst.is_file():
                eeg = np.load(dst, mmap_mode="r")
                print(f"[SKIP] {dst} shape={eeg.shape}")
            else:
                raw = load_eeg_dict(src)["preprocessed_eeg_data"]
                data = np.asarray(raw, dtype=np.float32)
                if data.ndim == 4:
                    # (N, R, C, T) -> average repetitions
                    data = data.mean(axis=1)
                if data.ndim != 3:
                    raise ValueError(f"Expected (N,C,T) after avg, got {data.shape} from {src}")
                if data.shape[0] != len(img_files):
                    raise ValueError(
                        f"Trial/image count mismatch for {subject} {split_name}: "
                        f"eeg={data.shape[0]} images={len(img_files)}"
                    )
                np.save(dst, data)
                eeg = data
                print(f"[OK] wrote {dst} shape={tuple(eeg.shape)}")

            n = int(eeg.shape[0])
            indices = list(range(n))
            if split_name == "train" and args.max_train_per_subject > 0:
                indices = indices[: args.max_train_per_subject]

            if split_name == "train":
                random.shuffle(indices)
                n_val = max(1, int(len(indices) * args.val_ratio)) if len(indices) > 20 else 0
                val_set = set(indices[:n_val])
            else:
                val_set = set()

            for trial_idx in tqdm(indices, desc=f"{subject}-{split_name}", leave=False):
                img_name = img_files[trial_idx]
                img_path = image_index.get(img_name)
                if img_path is None:
                    raise FileNotFoundError(f"Image not found for {img_name}")
                if split_name == "test":
                    split = "test"
                elif trial_idx in val_set:
                    split = "val"
                else:
                    split = "train"
                records.append(
                    {
                        "id": f"{subject}_{split_name}-{trial_idx:05d}",
                        "eeg": str(dst.resolve()),
                        "trial_index": int(trial_idx),
                        "image": str(img_path),
                        "subject": subject,
                        "split": split,
                    }
                )

    with args.out.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    from collections import Counter

    counts = Counter(r["split"] for r in records)
    print(f"[OK] Wrote {len(records)} records -> {args.out} ({dict(counts)})")


if __name__ == "__main__":
    main()
