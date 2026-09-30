#!/usr/bin/env python3
"""Forward smoke test for the EEG-Brain-IT fusion pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from eeg_brainit.data import SyntheticSmokeDataset
from eeg_brainit.models import EEGBrainITPipeline
from eeg_brainit.utils.config import load_config
from eeg_brainit.utils.freeze import count_trainable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--stage", type=int, default=1)
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg.get("project_root", Path.cwd()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device} torch={torch.__version__} cuda_built={torch.version.cuda}")
    if device.type == "cuda":
        print(f"[INFO] GPU={torch.cuda.get_device_name(0)}")

    model = EEGBrainITPipeline.from_config(cfg, project_root=root).to(device)
    model.apply_stage(args.stage)
    print(f"[INFO] trainable={count_trainable(model):,}")

    ds = SyntheticSmokeDataset(n=2)
    batch = ds[0]
    spec = batch["spectrogram"].unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(spec)

    for k, v in out.items():
        if torch.is_tensor(v):
            print(f"  {k}: {tuple(v.shape)} dtype={v.dtype}")
    assert out["brain_tokens"].shape[1] == int(cfg.get("bit", {}).get("num_brain_tokens", 128))
    assert out["clip_tokens"].shape[1] == int(cfg.get("bit", {}).get("num_query_tokens", 256))
    print("[OK] smoke test passed")


if __name__ == "__main__":
    main()
