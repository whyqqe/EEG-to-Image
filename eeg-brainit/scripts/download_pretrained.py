#!/usr/bin/env python3
"""Safely download pretrained ATM/NICE assets into eeg-brainit/checkpoints.

- Writes only under /project/peilab/why/eeg-brainit/checkpoints and cache/
- Does not delete or overwrite existing source files under eeg-to-image/
- Skips files that already exist with matching size when possible
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files


ROOT = Path("/project/peilab/why/eeg-brainit")
CACHE = Path("/project/peilab/why/cache/eeg-brainit/hf")
ATM_SRC = Path("/project/peilab/why/eeg-to-image/checkpoints/atm")


def ensure_link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return "exists"
    try:
        os.symlink(src, dst)
        return "symlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def download_nice(out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = [
        "pretrain_Enc_eeg_cls.pth",
        "pretrain_Proj_eeg_cls.pth",
        "pretrain_Proj_img_cls.pth",
        "README.md",
    ]
    done = []
    for name in files:
        path = hf_hub_download(
            repo_id="eeyhsong/NICE",
            repo_type="model",
            filename=name,
            local_dir=str(out_dir),
            local_dir_use_symlinks=False,
            cache_dir=str(CACHE),
        )
        done.append(path)
        print(f"[OK] NICE {name} -> {path}")
    return done


def download_atm_diffusion_priors(out_dir: Path, subjects: list[str]) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    done = []
    for sub in subjects:
        rel = f"fintune_ckpts/{sub}/diffusion_prior.pt"
        local = out_dir / sub / "diffusion_prior.pt"
        if local.is_file() and local.stat().st_size > 1_000_000:
            print(f"[SKIP] exists {local}")
            done.append(str(local))
            continue
        path = hf_hub_download(
            repo_id="LidongYang/EEG_Image_decode",
            repo_type="dataset",
            filename=rel,
            local_dir=str(out_dir.parent / "_hf_atm_ds"),
            local_dir_use_symlinks=False,
            cache_dir=str(CACHE),
        )
        local.parent.mkdir(parents=True, exist_ok=True)
        if Path(path).resolve() != local.resolve():
            shutil.copy2(path, local)
        done.append(str(local))
        print(f"[OK] ATM prior {sub} -> {local} ({local.stat().st_size} bytes)")
    return done


def link_atm_features(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    if not ATM_SRC.is_dir():
        raise FileNotFoundError(ATM_SRC)
    for src in sorted(ATM_SRC.glob("*.pt")):
        dst = out_dir / src.name
        report[src.name] = ensure_link_or_copy(src, dst)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="*", default=[f"sub-{i:02d}" for i in range(1, 11)])
    parser.add_argument("--skip-nice", action="store_true")
    parser.add_argument("--skip-atm-prior", action="store_true")
    parser.add_argument("--skip-link-features", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", str(CACHE))
    ckpt = ROOT / "checkpoints"
    manifest = {"root": str(ckpt)}

    if not args.skip_link_features:
        manifest["atm_features"] = link_atm_features(ckpt / "atm")
        print(f"[OK] linked/copied ATM features -> {ckpt / 'atm'}")

    if not args.skip_nice:
        manifest["nice"] = download_nice(ckpt / "nice")

    if not args.skip_atm_prior:
        manifest["atm_diffusion_priors"] = download_atm_diffusion_priors(
            ckpt / "atm_diffusion_prior", list(args.subjects)
        )

    out_json = ckpt / "pretrained_manifest.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[OK] wrote {out_json}")


if __name__ == "__main__":
    main()
