#!/usr/bin/env python3
"""Download SDXL base + ensure IP-Adapter weights into project HF cache.

Writes only under /project/peilab/why/cache/eeg-brainit/hf/
Does not touch /home caches.
"""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download


CACHE_ROOT = Path("/project/peilab/why/cache/eeg-brainit/hf")
HUB = CACHE_ROOT / "hub"


def _env() -> None:
    os.environ["HF_HOME"] = str(CACHE_ROOT)
    os.environ["HF_HUB_CACHE"] = str(HUB)
    os.environ["TRANSFORMERS_CACHE"] = str(HUB)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(HUB)
    # Avoid accidental writes to full /home
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)


def main() -> None:
    _env()
    HUB.mkdir(parents=True, exist_ok=True)

    print("[INFO] downloading stabilityai/stable-diffusion-xl-base-1.0 (fp16 + configs)...")
    sdxl = snapshot_download(
        repo_id="stabilityai/stable-diffusion-xl-base-1.0",
        cache_dir=str(HUB),
        allow_patterns=[
            "model_index.json",
            "scheduler/**",
            "tokenizer/**",
            "tokenizer_2/**",
            "text_encoder/config.json",
            "text_encoder/model.fp16.safetensors",
            "text_encoder_2/config.json",
            "text_encoder_2/model.fp16.safetensors",
            "unet/config.json",
            "unet/diffusion_pytorch_model.fp16.safetensors",
            "vae/config.json",
            "vae/diffusion_pytorch_model.fp16.safetensors",
        ],
        ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*onnx*", "*openvino*"],
        local_files_only=False,
    )
    print(f"[OK] SDXL snapshot -> {sdxl}")
    idx = Path(sdxl) / "model_index.json"
    if not idx.is_file():
        raise FileNotFoundError(f"missing model_index.json under {sdxl}")

    print("[INFO] downloading h94/IP-Adapter sdxl_models...")
    ip = snapshot_download(
        repo_id="h94/IP-Adapter",
        cache_dir=str(HUB),
        allow_patterns=[
            "sdxl_models/ip-adapter_sdxl_vit-h.bin",
            "sdxl_models/ip-adapter_sdxl.bin",
            "sdxl_models/*.json",
        ],
        local_files_only=False,
    )
    print(f"[OK] IP-Adapter snapshot -> {ip}")
    vit_h = Path(ip) / "sdxl_models" / "ip-adapter_sdxl_vit-h.bin"
    if not vit_h.is_file():
        raise FileNotFoundError(vit_h)
    print("[OK] SDXL + IP-Adapter ready for offline generation")


if __name__ == "__main__":
    main()
