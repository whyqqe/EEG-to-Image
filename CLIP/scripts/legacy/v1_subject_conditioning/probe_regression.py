#!/usr/bin/env python
"""Do you still need the calibrated z scale, and does `routed_sr` still hurt?

This started as a 2x2 that attributed the Stage-1 regression of job 637180 to
`conditioning.z_norm: unit` (image contrast held at ln(72) = 4.2767 for 12k steps, test
top-1 pinned at the initialisation value, subject dependence rising) rather than to
`target_fusion: routed`. That attribution is settled and the `unit` value is now a hard
error; `SubjectConditioner.normalize_z` carries the evidence.

What is still open is whether the CALIBRATED scale ('mark', which normalises z to the
embedding table's own init scale 0.02*sqrt(d_z) instead of sqrt(d_z)) is load-bearing
once `z_source: support` is the default, and whether `routed_sr` still costs anything. So
the same 2x2 now crosses `target_fusion` with `{mark, none}`:

  * `mark` is the shipped default.
  * `none` is the honest ablation of the fix. Under `z_source: support` it should go dead,
    because the raw support encoder runs at |z| ~ 4.5 and diverges to 12 -- which is the
    same regime as the `unit` arm, and `scripts/diag_zscale.py` shows exactly this
    signature and its reversal when |z| is brought to ~0.14.

If a `none` arm trains normally, the calibrated scale is not load-bearing in this regime
and the default should be revisited.

    python scripts/probe_regression.py --out-dir outputs/probe_regression
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from samclip import config  # noqa: E402
from samclip.utils import load_config  # noqa: E402

# (tag, target_fusion, z_norm)
ARMS = [
    ("a__routed_sr__zmmark", "routed_sr", "mark"),
    ("b__routed_sr__zmnone", "routed_sr", "none"),
    ("c__routed__zmmark", "routed", "mark"),
    ("d__routed__zmnone", "routed", "none"),
]

STEP_RE = re.compile(r"epoch (\d+) step (\d+)/\d+ loss ([\d.]+) (\{.*\})")
EPOCH_RE = re.compile(r"epoch (\d+) \| loss ([\d.]+) \| test top1 ([\d.]+) "
                      r"top5 ([\d.]+) meanrank ([\d.]+)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(config.CONFIGS / "loso_sub08.yaml"))
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--log-every", type=int, default=400)
    ap.add_argument("--seeds", type=int, nargs="*", default=[2025])
    ap.add_argument("--out-dir", default=str(config.OUTPUTS / "probe_regression"))
    args = ap.parse_args()

    base = load_config(args.config)
    summary: dict[str, dict] = {}
    for seed in args.seeds:
        for tag, fusion, z_norm in ARMS:
            arm = f"{tag}__seed{seed}"
            cfg = dict(base)
            cond = dict(cfg.get("conditioning", {}) or {})
            cond["z_norm"] = z_norm
            cfg.update({
                "conditioning": cond,
                "target_fusion": fusion,
                "seed": seed,
                "epochs": args.epochs,
                "log_every": args.log_every,
                "out_dir": str(Path(args.out_dir) / arm),
            })
            print(f"\n{'#' * 78}\n# {arm}: target_fusion={fusion} z_norm={z_norm}\n"
                  f"{'#' * 78}", flush=True)

            from samclip.train import train_stage1
            result = train_stage1(cfg)
            summary[arm] = {"target_fusion": fusion, "z_norm": z_norm,
                            "top1": result["last"]["top1"],
                            "mean_rank": result["last"]["mean_rank"]}

    out = Path(args.out_dir) / "regression_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print("\n[probe] summary (a dead arm sits at img ~ ln(72) = 4.2767):")
    for arm, r in summary.items():
        print(f"  {arm:<34} top1 {r['top1']:>5.2f}  meanrank {r['mean_rank']:>6.1f}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
