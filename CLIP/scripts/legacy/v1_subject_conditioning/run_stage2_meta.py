#!/usr/bin/env python
"""Stage 2: episodic meta-training of the hypernetwork (new-subject conditioning).

Run:  python scripts/run_stage2_meta.py --config configs/meta_s2.yaml \
        --stage1-dir outputs/stage1/sub-01
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from samclip import config  # noqa: E402
from samclip.train import train_stage2  # noqa: E402
from samclip.utils import dump_config, load_config, merge_cli_overrides  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(config.CONFIGS / "meta_s2.yaml"))
    ap.add_argument("--stage1-dir", required=True)
    ap.add_argument("--target-subject", type=int, default=None)
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--meta-lr", type=float, default=None)
    ap.add_argument("--k-support", type=int, default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    cfg = merge_cli_overrides(load_config(args.config), {
        "stage1_dir": args.stage1_dir,
        "target_subject": args.target_subject,
        "episodes": args.episodes,
        "meta_lr": args.meta_lr,
        "k_support": args.k_support,
        "out_dir": args.out_dir,
    })
    if args.out_dir is None:
        cfg["out_dir"] = str(config.OUTPUTS / "stage2" / Path(args.stage1_dir).name)
    dump_config(cfg, Path(cfg["out_dir"]))
    print("[stage2] config:\n" + json.dumps(cfg, indent=2, default=str))
    result = train_stage2(cfg)
    print("[stage2] result:", json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
