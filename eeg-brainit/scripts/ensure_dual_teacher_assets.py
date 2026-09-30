#!/usr/bin/env python3
"""Verify Dual-Teacher assets; download any missing checkpoints.

Required for Dual-Teacher training:
  - NOD classmean pairs + CLIP ViT-H/14 embeddings
  - fMRI→CLIP teacher (fmri2clip.pt)
  - SDXL base + IP-Adapter (optional generation)

Also repairs broken NeuroBOLT glb.pth / labram-base.pth symlinks used by
physical-cascade contrasts.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path("/project/peilab/why/eeg-brainit")
CACHE = Path("/project/peilab/why/cache/eeg-brainit/hf")
HUB = CACHE / "hub"


def _env() -> None:
    os.environ["HF_HOME"] = str(CACHE)
    os.environ["HF_HUB_CACHE"] = str(HUB)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(HUB)
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)


def _ok(path: Path, min_bytes: int = 1) -> bool:
    try:
        return path.is_file() and path.stat().st_size >= min_bytes
    except OSError:
        return False


def ensure_file(path: Path, min_bytes: int = 1) -> dict:
    return {"path": str(path), "ok": _ok(path, min_bytes), "size": path.stat().st_size if path.is_file() else 0}


def download_neurobolt_glb(dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if _ok(dst, 1_000_000):
        return "exists"
    # remove broken symlink
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    src = hf_hub_download(
        repo_id="ssssssup/NeuroBOLT",
        filename="checkpoints/glb.pth",
        cache_dir=str(HUB),
    )
    shutil.copy2(src, dst)
    return "downloaded"


def download_labram(dst: Path) -> str:
    """LaBraM-base from official release mirror if available."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if _ok(dst, 1_000_000):
        return "exists"
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    # Community mirrors used by NeuroBOLT users
    candidates = [
        ("935963004/LaBraM", "checkpoints/labram-base.pth"),
        ("BrokenR201/LaBraM", "labram-base.pth"),
    ]
    last_err = None
    for repo, filename in candidates:
        try:
            src = hf_hub_download(repo_id=repo, filename=filename, cache_dir=str(HUB))
            shutil.copy2(src, dst)
            return f"downloaded:{repo}"
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            print(f"[WARN] LaBraM from {repo}: {exc}")
    # Fallback: try raw github release via huggingface dataset mirrors — skip if none
    raise RuntimeError(f"failed to download labram-base.pth ({last_err})")


def ensure_sdxl_ip() -> dict:
    report = {}
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
        local_files_only=False,
    )
    report["sdxl"] = sdxl
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
    report["ip_adapter"] = ip
    vit_h = Path(ip) / "sdxl_models" / "ip-adapter_sdxl_vit-h.bin"
    report["ip_adapter_vit_h_ok"] = vit_h.is_file()
    return report


def main() -> None:
    _env()
    HUB.mkdir(parents=True, exist_ok=True)
    report: dict = {"required": {}, "optional": {}, "downloads": {}}

    required = {
        "pairs_sub01": ROOT / "data/nod/processed/classmean/sub-01/pairs.npz",
        "clip_embeddings": ROOT / "data/nod/processed/clip_vit_h14_all/embeddings.npy",
        "clip_index": ROOT / "data/nod/processed/clip_vit_h14_all/index.json",
        "fmri_teacher": ROOT / "outputs/eval/nod_phase2_gt_fmri2image/sub-01_classmean/fmri2clip.pt",
        "phase1_phys_cascade": ROOT / "outputs/nod_eeg2fmri/neurobolt_classmean_v3/checkpoints/best.pt",
        "stage_a_decoder": ROOT / "outputs/nod_cascade/neurobolt_bit_v1/decoder_best.pt",
    }
    for k, p in required.items():
        report["required"][k] = ensure_file(p, 1000)
        print(f"[{'OK' if report['required'][k]['ok'] else 'MISSING'}] {k}: {p}")

    # SDXL + IP-Adapter
    print("[INFO] ensuring SDXL + IP-Adapter...")
    report["downloads"]["sdxl_ip"] = ensure_sdxl_ip()
    print("[OK] SDXL/IP-Adapter")

    # Repair NeuroBOLT glb
    glb = ROOT / "checkpoints/neurobolt/glb.pth"
    print("[INFO] ensuring NeuroBOLT glb.pth...")
    try:
        report["downloads"]["glb"] = download_neurobolt_glb(glb)
        report["optional"]["glb"] = ensure_file(glb, 1_000_000)
        print(f"[OK] glb.pth ({report['downloads']['glb']})")
    except Exception as exc:  # noqa: BLE001
        report["downloads"]["glb_error"] = str(exc)
        print(f"[WARN] glb.pth: {exc}")

    # LaBraM (optional for Dual-Teacher; needed if re-finetuning NeuroBOLT)
    labram = ROOT / "checkpoints/neurobolt/labram-base.pth"
    print("[INFO] ensuring labram-base.pth...")
    try:
        report["downloads"]["labram"] = download_labram(labram)
        report["optional"]["labram"] = ensure_file(labram, 1_000_000)
        print(f"[OK] labram-base.pth ({report['downloads']['labram']})")
    except Exception as exc:  # noqa: BLE001
        report["downloads"]["labram_error"] = str(exc)
        print(f"[WARN] labram-base.pth: {exc}")

    missing_req = [k for k, v in report["required"].items() if not v["ok"]]
    report["ready_for_dual_teacher"] = len(missing_req) == 0
    report["missing_required"] = missing_req

    out = ROOT / "checkpoints/dual_teacher_assets.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"ready": report["ready_for_dual_teacher"], "missing": missing_req}, indent=2))
    print(f"[OK] wrote {out}")
    if missing_req:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
