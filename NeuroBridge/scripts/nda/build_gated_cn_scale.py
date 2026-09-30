#!/usr/bin/env python3
"""Map structure confidence → ControlNet scale (high conf → stronger early CN)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--u-npy", type=str, required=True)
    ap.add_argument("--output-npy", type=str, required=True)
    ap.add_argument("--cn-min", type=float, default=0.25)
    ap.add_argument("--cn-max", type=float, default=0.55)
    ap.add_argument("--report-json", type=str, default="")
    args = ap.parse_args()

    u = np.load(args.u_npy).astype(np.float32).reshape(-1)
    order = np.argsort(u)
    ranks = np.empty_like(u, dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, num=len(u), dtype=np.float32)
    cn = args.cn_min + ranks * (args.cn_max - args.cn_min)
    cn = cn.astype(np.float32)
    Path(args.output_npy).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_npy, cn)
    rep = {
        "n": int(len(cn)),
        "cn_min": args.cn_min,
        "cn_max": args.cn_max,
        "cn_mean": float(cn.mean()),
        "note": "high structure confidence → higher early ControlNet scale",
    }
    if args.report_json:
        Path(args.report_json).write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
