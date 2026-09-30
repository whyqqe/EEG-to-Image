#!/usr/bin/env python3
"""Keep a small random subset of generated PNGs; delete the rest."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", type=str, required=True)
    ap.add_argument("--keep", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report-json", type=str, default="")
    args = ap.parse_args()

    gen_dir = Path(args.gen_dir)
    paths = sorted(gen_dir.glob("*.png"))
    n = len(paths)
    if n <= args.keep:
        print(f"[prune] keep all n={n}")
        kept = [p.name for p in paths]
    else:
        rng = random.Random(args.seed)
        keep_paths = set(rng.sample(paths, args.keep))
        deleted = 0
        kept = []
        for p in paths:
            if p in keep_paths:
                kept.append(p.name)
            else:
                p.unlink()
                deleted += 1
        print(f"[prune] deleted={deleted} kept={len(kept)} from {n}")

    report = {"gen_dir": str(gen_dir), "n_before": n, "kept": sorted(kept), "keep": args.keep}
    out = Path(args.report_json) if args.report_json else gen_dir.parent / "pruned_samples.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
