#!/usr/bin/env python3
"""Train EEG-Brain-IT with staged fine-tuning."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from eeg_brainit.data import SyntheticSmokeDataset, ThingsEEG2Dataset
from eeg_brainit.models import EEGBrainITPipeline
from eeg_brainit.training import Trainer
from eeg_brainit.utils.config import ensure_dirs, load_config


def deep_update(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def build_loaders(cfg: dict):
    data_cfg = cfg.get("data", {})
    root = Path(cfg.get("project_root", "."))
    manifest = root / data_cfg.get("manifest", "data/processed/manifest.jsonl")
    use_smoke = bool(data_cfg.get("use_smoke_if_missing", True))

    teacher_raw = data_cfg.get("teacher_dir")
    teacher_dir = None
    if teacher_raw:
        teacher_dir = Path(teacher_raw)
        if not teacher_dir.is_absolute():
            teacher_dir = root / teacher_dir
        if not (teacher_dir / "embeddings.npy").is_file():
            raise FileNotFoundError(
                f"CLIP teacher cache missing under {teacher_dir}; "
                "run scripts/precompute_clip_teachers.py first"
            )

    subjects = data_cfg.get("subjects")
    if subjects is not None and isinstance(subjects, str):
        subjects = [subjects]

    if manifest.is_file() and manifest.stat().st_size > 0:
        train_ds = ThingsEEG2Dataset(
            manifest=manifest,
            root=root,
            split="train",
            image_size=int(data_cfg.get("image_size", 224)),
            n_fft=int(data_cfg.get("spectrogram", {}).get("n_fft", 64)),
            hop_length=int(data_cfg.get("spectrogram", {}).get("hop_length", 16)),
            target_time=int(data_cfg.get("spectrogram", {}).get("target_time", 64)),
            expected_channels=int(data_cfg.get("num_channels", 63)),
            teacher_dir=teacher_dir,
            subjects=subjects,
        )
        val_ds = ThingsEEG2Dataset(
            manifest=manifest,
            root=root,
            split="val",
            image_size=int(data_cfg.get("image_size", 224)),
            n_fft=int(data_cfg.get("spectrogram", {}).get("n_fft", 64)),
            hop_length=int(data_cfg.get("spectrogram", {}).get("hop_length", 16)),
            target_time=int(data_cfg.get("spectrogram", {}).get("target_time", 64)),
            expected_channels=int(data_cfg.get("num_channels", 63)),
            teacher_dir=teacher_dir,
            subjects=subjects,
        )
        print(
            f"[INFO] Loaded manifest {manifest}: train={len(train_ds)} val={len(val_ds)} "
            f"subjects={subjects} teacher={teacher_dir}"
        )
    elif use_smoke:
        n = int(data_cfg.get("smoke_samples", 32))
        print(f"[WARN] Manifest missing; using SyntheticSmokeDataset(n={n})")
        with_clip = bool(
            cfg.get("clip_align", {}).get("enabled", False)
            or cfg.get("direct_clip", {}).get("enabled", False)
        )
        train_ds = SyntheticSmokeDataset(n=n, with_clip=with_clip)
        val_ds = SyntheticSmokeDataset(n=max(4, n // 4), with_clip=with_clip)
    else:
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    bs = int(cfg.get("train", {}).get("batch_size", 8))
    nw = int(cfg.get("train", {}).get("num_workers", 4))
    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        drop_last=True,
        pin_memory=pin,
        persistent_workers=nw > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=pin,
        persistent_workers=nw > 0,
    )
    return train_loader, val_loader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--override", type=str, nargs="*", default=[], help="Optional extra yaml overlays")
    parser.add_argument("--stage", type=int, default=None)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a previous stage checkpoint (.pt) to warm-start weights",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    for ov in args.override:
        cfg = deep_update(cfg, load_config(ov))
    if args.stage is not None:
        cfg.setdefault("train", {})["stage"] = args.stage
    resume = args.resume or cfg.get("train", {}).get("resume")

    project_root = Path(cfg.get("project_root", Path.cwd()))
    ensure_dirs(project_root / cfg.get("output_dir", "outputs/default"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")
    if device.type == "cuda":
        print(f"[INFO] GPU={torch.cuda.get_device_name(0)}")

    model = EEGBrainITPipeline.from_config(cfg, project_root=project_root)
    if resume:
        ckpt_path = Path(resume)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"--resume checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        print(
            f"[INFO] Resumed weights from {ckpt_path} "
            f"(missing={len(missing)} unexpected={len(unexpected)})"
        )

    train_loader, val_loader = build_loaders(cfg)
    stage = int(cfg.get("train", {}).get("stage", 1))
    trainer = Trainer(model, train_loader, val_loader, cfg, device=device, stage=stage)
    trainer.fit()


if __name__ == "__main__":
    main()
