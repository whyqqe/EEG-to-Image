#!/usr/bin/env python
"""Does conditioning Stage 1 on a support set train, and does the moments anchor help?

This is the arm that decides the architecture. Three runs, one epoch each on the real
sub-08 LOSO fold, differing only in how `z_s` is produced:

  * `ids`              -- free `nn.Embedding` table, `z_norm: none`. The control: this is
                          the configuration job 636661 used and it reaches ~5.0 top-1
                          after one epoch (vs 0.50 at initialisation).
  * `support/anchor=off` -- `z_s = net(S)` from a label-free support set. The user's
                          proposal, without the moments channel.
  * `support/anchor=stats` -- `z_s = Linear([mean_c(S) ; std_c(S)]) + net(S)`. The new
                          default, and the literature's answer (see `SupportSetEncoder`).

The failure this is checking for is not subtle: a dead run pins `img` at `ln(72) = 4.2767`
and never leaves the initialisation top-1 of 0.50, which is exactly what the `z_norm:
unit` arm did for 12k steps. So one epoch discriminates cleanly.

  python scripts/probe_subject_z.py --out-dir outputs/probe_subject
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from samclip import config  # noqa: E402
from samclip.utils import load_config  # noqa: E402

# (tag, z_source, support_anchor, z_norm)
#
# All arms use `z_norm: mark`, the calibrated scale. `none` was used in the first version
# of this probe and produced TWO dead arms -- not because the support path cannot train,
# but because the raw support-encoder output sits at |z| ~ 4.5 and diverges to 12, and the
# FiLM modulation rate scales with |z|. `scripts/diag_zscale.py` isolated that (freezing
# the support set did nothing; shrinking |z| revived the run and flipped the subject
# dependence term back to falling). See SubjectConditioner.normalize_z.
#
# `support_anchor` is `none` by default because this probe measured no benefit from
# 'stats' (B 8.00/23.00 vs C 5.50/17.00 top-1/top-5, trajectories superimposed); both are
# kept so the ablation stays runnable.
ARMS = [
    ("A__ids__anchor_stats", "ids", "stats", "mark"),
    ("B__support__noanchor", "support", "none", "mark"),
    ("C__support__stats", "support", "stats", "mark"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(config.CONFIGS / "loso_sub08.yaml"))
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--log-every", type=int, default=400)
    ap.add_argument("--k-support-stage1", type=int, default=10)
    ap.add_argument("--seeds", type=int, nargs="*", default=[2025])
    ap.add_argument("--out-dir", default=str(config.OUTPUTS / "probe_subject"))
    args = ap.parse_args()

    base = load_config(args.config)
    summary: dict[str, dict] = {}
    for seed in args.seeds:
        for tag, z_source, anchor, z_norm in ARMS:
            arm = f"{tag}__seed{seed}"
            cfg = dict(base)
            cond = dict(cfg.get("conditioning", {}) or {})
            cond["support_anchor"] = anchor
            cond["z_norm"] = z_norm
            cfg.update({
                "conditioning": cond,
                "z_source": z_source,
                "k_support_stage1": args.k_support_stage1,
                "seed": seed,
                "epochs": args.epochs,
                "log_every": args.log_every,
                "out_dir": str(Path(args.out_dir) / arm),
            })
            print(f"\n{'#' * 78}\n# {arm}: z_source={z_source} anchor={anchor} "
                  f"z_norm={z_norm}\n{'#' * 78}", flush=True)

            from samclip.train import train_stage1
            result = train_stage1(cfg)
            summary[arm] = {"z_source": z_source, "anchor": anchor, "z_norm": z_norm,
                            "top1": result["last"]["top1"],
                            "mean_rank": result["last"]["mean_rank"]}

    out = Path(args.out_dir) / "probe_subject_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print("\n[probe] summary (a dead arm sits at img ~ ln(72) = 4.2767 and top1 0.50):")
    for arm, r in summary.items():
        print(f"  {arm:<32} top1 {r['top1']:>5.2f}  meanrank {r['mean_rank']:>6.1f}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
