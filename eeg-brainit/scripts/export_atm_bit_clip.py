#!/usr/bin/env python3
"""Export bit_clip (or bridge_clip) test embeddings from an ATM-bridge checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from eeg_brainit.models.atm_bridge import AtmBrainITPipeline


@torch.no_grad()
def encode(model: AtmBrainITPipeline, eeg: np.ndarray, key: str, device: torch.device, bs: int = 256) -> np.ndarray:
    model.eval()
    outs = []
    x = torch.from_numpy(eeg.astype(np.float32))
    for i in range(0, len(x), bs):
        out = model(x[i : i + bs].to(device))
        outs.append(F.normalize(out[key].float(), dim=-1).cpu().numpy())
    return np.concatenate(outs, axis=0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--subject", type=str, default="sub-08")
    parser.add_argument("--bridge-dir", type=str, default="outputs/atm_bridge")
    parser.add_argument("--head", type=str, default="bit_clip", choices=["bit_clip", "bridge_clip", "atm_emb"])
    parser.add_argument("--output-npy", type=str, required=True)
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = ROOT / ckpt
    bridge = Path(args.bridge_dir)
    if not bridge.is_absolute():
        bridge = ROOT / bridge

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    if any(k.startswith("bit.") for k in ck["model"].keys()):
        cfg.setdefault("atm", {})["use_bit"] = True
    model = AtmBrainITPipeline.from_config(cfg, project_root=str(ROOT)).to(device)
    model.load_state_dict(ck["model"], strict=False)
    stage = int(ck.get("stage", ck["cfg"].get("train", {}).get("stage", 3)))
    model.apply_stage(stage)

    eeg = np.load(bridge / f"{args.subject}_test_eeg_1024.npy").astype(np.float32)
    emb = encode(model, eeg, args.head, device)
    out = Path(args.output_npy)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, emb)
    meta = {
        "checkpoint": str(ckpt),
        "subject": args.subject,
        "head": args.head,
        "shape": list(emb.shape),
        "stage": stage,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[OK] wrote {out} shape={emb.shape}")


if __name__ == "__main__":
    main()
