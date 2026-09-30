#!/usr/bin/env python3
"""Phase-4 scaffold: align THINGS-EEG and THINGS-fMRI in CLIP space.

Heterogeneous subjects / same stimulus family. Full training loops once
THINGS-fMRI features path is configured in the config.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/things_align.yaml")
    args = parser.parse_args()
    out = ROOT / "outputs" / "things_align"
    out.mkdir(parents=True, exist_ok=True)
    status = {
        "status": "scaffold",
        "config": args.config,
        "todo": [
            "Point configs/things_align.yaml at THINGS-fMRI betas/features",
            "Reuse ATM EEG embeds from outputs/atm_bridge/",
            "InfoNCE EEG↔image and fMRI↔image (CLIP teachers already cached)",
            "Optional EEG↔fMRI retrieval in shared space",
        ],
        "refs": ["NeuroBind", "BrainFLORA", "InfFusion 2025 neural foundation models"],
    }
    path = out / "STATUS.json"
    path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(f"[OK] wrote {path}")
    print("[WAIT] Configure THINGS-fMRI paths before training.")


if __name__ == "__main__":
    main()
