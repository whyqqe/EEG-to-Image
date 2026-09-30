#!/usr/bin/env python3
"""Map structure confidence → mild img2img strength (high conf → keep layout)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--u-npy", type=str, required=True, help="higher = more structure confidence")
    ap.add_argument("--output-npy", type=str, required=True)
    ap.add_argument("--s-min", type=float, default=0.20, help="strength when confidence is high")
    ap.add_argument("--s-max", type=float, default=0.36, help="strength when confidence is low")
    ap.add_argument("--report-json", type=str, default="")
    args = ap.parse_args()

    u = np.load(args.u_npy).astype(np.float32).reshape(-1)
    # rank-normalize to [0,1]
    order = np.argsort(u)
    ranks = np.empty_like(u, dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, num=len(u), dtype=np.float32)
    # high rank → low strength (preserve low-level layout)
    s = args.s_max - ranks * (args.s_max - args.s_min)
    s = s.astype(np.float32)
    Path(args.output_npy).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_npy, s)
    rep = {
        "n": int(len(s)),
        "s_min": args.s_min,
        "s_max": args.s_max,
        "strength_mean": float(s.mean()),
        "strength_std": float(s.std()),
        "note": "high structure confidence → lower img2img strength (keep pixel layout)",
    }
    if args.report_json:
        Path(args.report_json).write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
