#!/usr/bin/env python3
"""Build gated prompts + fuse/IP scales from clean margins and u_str (disk-light npys only)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts-json", type=str, required=True, help="always-on prompts (200)")
    ap.add_argument("--margins-npy", type=str, required=True)
    ap.add_argument("--u-str-npy", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--margin-gate", type=float, default=0.02)
    ap.add_argument("--soft-gate", type=float, default=0.05)
    ap.add_argument("--cn-min", type=float, default=0.35)
    ap.add_argument("--cn-max", type=float, default=0.60)
    ap.add_argument("--ip-min", type=float, default=0.85)
    ap.add_argument("--ip-max", type=float, default=1.0)
    ap.add_argument("--fuse-beta-min", type=float, default=0.75)
    ap.add_argument("--fuse-beta-max", type=float, default=1.0)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    prompts = json.loads(Path(args.prompts_json).read_text(encoding="utf-8"))
    margins = np.load(args.margins_npy).astype(np.float32).reshape(-1)
    u = np.load(args.u_str_npy).astype(np.float32).reshape(-1)
    assert len(prompts) == len(margins) == len(u)

    gated = []
    for p, m in zip(prompts, margins):
        # prompts may already be full sentences; gate by margin only
        if m < args.margin_gate:
            gated.append("")
        elif m < args.soft_gate:
            # shorten if detailed
            name = p.replace("a photo of ", "").replace(", highly detailed", "").strip()
            gated.append(f"a photo of {name}" if name else "")
        else:
            gated.append(p)

    # rank u_str → scales
    ranks = u.argsort().argsort().astype(np.float32) / max(len(u) - 1, 1)
    cn = args.cn_min + ranks * (args.cn_max - args.cn_min)
    ip = args.ip_max - ranks * (args.ip_max - args.ip_min)
    # high structure → more neighbor-blur mix (lower beta)
    fuse = args.fuse_beta_max - ranks * (args.fuse_beta_max - args.fuse_beta_min)

    (out / "prompts_gated.json").write_text(json.dumps(gated, indent=2), encoding="utf-8")
    np.save(out / "cn_scale.npy", cn.astype(np.float32))
    np.save(out / "ip_scale.npy", ip.astype(np.float32))
    np.save(out / "fuse_beta.npy", fuse.astype(np.float32))
    report = {
        "n": len(prompts),
        "gated_coverage": float(np.mean([bool(x) for x in gated])),
        "margin_gate": args.margin_gate,
        "soft_gate": args.soft_gate,
        "cn_range": [args.cn_min, args.cn_max],
        "ip_range": [args.ip_min, args.ip_max],
        "fuse_beta_range": [args.fuse_beta_min, args.fuse_beta_max],
    }
    (out / "routing_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
