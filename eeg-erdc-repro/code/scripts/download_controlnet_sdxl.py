#!/usr/bin/env python3
"""Download SDXL ControlNet (canny) into project HF cache for ERDC structure branch."""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download

CACHE = Path("/project/peilab/why/cache/eeg-brainit/hf")
HUB = CACHE / "hub"


def main() -> None:
    os.environ["HF_HOME"] = str(CACHE)
    os.environ["HF_HUB_CACHE"] = str(HUB)
    HUB.mkdir(parents=True, exist_ok=True)
    # Diffusers SDXL ControlNet canny (commonly used)
    repo = "diffusers/controlnet-canny-sdxl-1.0"
    print(f"[INFO] downloading {repo} ...")
    path = snapshot_download(
        repo_id=repo,
        cache_dir=str(HUB),
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.bin",
            "*.txt",
            "*.model",
        ],
        local_files_only=False,
    )
    print(f"[OK] ControlNet -> {path}")
    marker = Path("/project/peilab/why/eeg-brainit/outputs/erdc/controlnet_canny_ready.txt")
    marker.write_text(f"{path}\n", encoding="utf-8")
    print(f"[OK] wrote {marker}")


if __name__ == "__main__":
    main()
