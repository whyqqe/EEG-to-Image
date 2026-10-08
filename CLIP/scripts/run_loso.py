#!/usr/bin/env python
"""Run Stage 1 over all 10 LOSO folds and aggregate the inter-subject table.

`--submit` writes one sbatch job per fold instead of running them on the login node
(the login node has no GPU -- see AGENTS.md §3.1).

Run:  python scripts/run_loso.py --config configs/loso_s1.yaml --submit
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from samclip import config  # noqa: E402
from samclip.utils import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(config.CONFIGS / "loso_s1.yaml"))
    ap.add_argument("--subjects", type=int, nargs="*", default=config.all_subjects())
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--sbatch", default=str(config.ROOT / "slurm" / "10_stage1.sbatch"))
    ap.add_argument("--tag-prefix", default="loso")
    args = ap.parse_args()

    cfg = load_config(args.config)
    for s in args.subjects:
        out_dir = config.OUTPUTS / "stage1" / f"{args.tag_prefix}_sub{s:02d}"
        if args.submit:
            cmd = ["sbatch", f"--export=ALL,TARGET_SUBJECT={s},OUT_DIR={out_dir}",
                   args.sbatch]
            print(" ".join(cmd))
            subprocess.run(cmd, check=True)
        else:
            from samclip.train import train_stage_a
            c = dict(cfg)
            c.update({"target_subject": s, "out_dir": str(out_dir)})
            result = train_stage_a(c)
            (out_dir / "result.json").write_text(json.dumps(result, indent=2))
            print(f"[loso] sub-{s:02d} -> {result['last']}")


if __name__ == "__main__":
    main()
