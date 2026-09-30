#!/usr/bin/env python3
"""Download official ATM Generation assets (VAE latents, optional SDXL-Turbo)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path("/project/peilab/why/cache/eeg-brainit/hf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-turbo", action="store_true")
    parser.add_argument("--skip-latents", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", str(CACHE))
    from huggingface_hub import hf_hub_download, snapshot_download

    out_ds = ROOT / "checkpoints" / "_hf_atm_ds"
    out_ds.mkdir(parents=True, exist_ok=True)

    if not args.skip_latents:
        for name in ["train_image_latent_512.pt", "test_image_latent_512.pt"]:
            dst = out_ds / name
            if dst.is_file() and dst.stat().st_size > 1_000_000:
                print(f"[SKIP] {dst}")
                continue
            path = hf_hub_download(
                repo_id="LidongYang/EEG_Image_decode",
                repo_type="dataset",
                filename=name,
                local_dir=str(out_ds),
                local_dir_use_symlinks=False,
                cache_dir=str(CACHE),
            )
            print(f"[OK] {name} -> {path}")

    if not args.skip_turbo:
        try:
            snap = snapshot_download(
                "stabilityai/sdxl-turbo",
                cache_dir=str(CACHE / "hub"),
                local_files_only=False,
            )
            print(f"[OK] sdxl-turbo -> {snap}")
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] sdxl-turbo download failed: {exc}")

    print("[OK] official ATM assets ready")


if __name__ == "__main__":
    main()
