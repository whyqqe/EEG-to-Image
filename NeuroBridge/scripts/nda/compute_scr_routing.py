#!/usr/bin/env python3
"""Structure-Credible Routing (SCR): per-sample ControlNet / IP scales.

c_s from EEG↔neighbor CLIP alignment (+ optional CPA margin).
High c_s → stronger ControlNet (structure); low c_s → weaker CN, stronger IP.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def l2(x: np.ndarray) -> np.ndarray:
    return (x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-8)).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eeg-embed-npy", type=str, required=True, help="NDA-SS / ViT-H EEG embeds (N,D)")
    ap.add_argument("--neighbor-idx-npy", type=str, required=True)
    ap.add_argument("--neighbor-clip-train-npy", type=str, required=True, help="train gallery CLIP (M,D)")
    ap.add_argument("--margins-npy", type=str, default="", help="CPA margins (N,)")
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--cn-min", type=float, default=0.35)
    ap.add_argument("--cn-max", type=float, default=1.0)
    ap.add_argument("--ip-min", type=float, default=0.85)
    ap.add_argument("--ip-max", type=float, default=1.0)
    ap.add_argument("--fuse-beta-min", type=float, default=0.80, help="semantic weight when c_s high → more blur mix uses lower beta")
    ap.add_argument("--fuse-beta-max", type=float, default=1.0)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    eeg = l2(np.load(args.eeg_embed_npy).astype(np.float32))
    neigh = np.load(args.neighbor_idx_npy)
    if neigh.ndim > 1:
        neigh0 = neigh[:, 0].astype(np.int64)
    else:
        neigh0 = neigh.astype(np.int64)
    gal = l2(np.load(args.neighbor_clip_train_npy).astype(np.float32))
    if gal.ndim == 3:
        gal = gal[:, 0]

    nb_clip = gal[neigh0]
    # dims may differ (512 vs 1024) — if so, only use margin routing
    if eeg.shape[-1] != nb_clip.shape[-1]:
        align = np.zeros(len(eeg), dtype=np.float32)
        align_note = f"dim mismatch eeg{eeg.shape[-1]} vs neigh{nb_clip.shape[-1]}; align=0, rely on margins"
    else:
        align = (eeg * nb_clip).sum(axis=1).astype(np.float32)
        align_note = "cos(EEG, neighbor CLIP)"

    if args.margins_npy:
        margins = np.load(args.margins_npy).astype(np.float32).reshape(-1)
    else:
        margins = np.zeros(len(eeg), dtype=np.float32)

    # normalize align and margin to [0,1] via ranks
    def rank01(x: np.ndarray) -> np.ndarray:
        order = np.argsort(np.argsort(x))
        return (order / max(len(x) - 1, 1)).astype(np.float32)

    a01 = rank01(align) if eeg.shape[-1] == nb_clip.shape[-1] else np.full(len(eeg), 0.5, np.float32)
    m01 = rank01(margins) if args.margins_npy else np.full(len(eeg), 0.5, np.float32)
    c_s = (0.6 * a01 + 0.4 * m01).astype(np.float32)

    cn = (args.cn_min + c_s * (args.cn_max - args.cn_min)).astype(np.float32)
    # high structure confidence → slightly lower IP to let CN speak
    ip = (args.ip_max - c_s * (args.ip_max - args.ip_min)).astype(np.float32)
    # high c_s → allow a bit more low-level neighbor blur in optional fuse (lower semantic beta)
    fuse_beta = (args.fuse_beta_max - c_s * (args.fuse_beta_max - args.fuse_beta_min)).astype(np.float32)

    np.save(out / "align_cos.npy", align)
    np.save(out / "c_s.npy", c_s)
    np.save(out / "cn_scale.npy", cn)
    np.save(out / "ip_scale.npy", ip)
    np.save(out / "fuse_beta.npy", fuse_beta)

    report = {
        "method": "Structure-Credible Routing (SCR)",
        "align": align_note,
        "c_s": "0.6*rank(align)+0.4*rank(CPA margin)",
        "cn_range": [args.cn_min, args.cn_max],
        "ip_range": [args.ip_min, args.ip_max],
        "c_s_mean": float(c_s.mean()),
        "c_s_std": float(c_s.std()),
        "cn_mean": float(cn.mean()),
        "ip_mean": float(ip.mean()),
        "align_mean": float(align.mean()) if align.size else None,
        "n": int(len(c_s)),
    }
    (out / "scr_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
