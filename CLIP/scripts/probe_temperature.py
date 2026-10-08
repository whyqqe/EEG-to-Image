#!/usr/bin/env python
"""A/B probe for the InfoNCE temperature collapse, on the real LOSO fold.

Why this exists
---------------
The Stage-1 run on `target_fusion: routed` + `z_norm: unit` was *completely* dead:
test top-1 sat at its initialisation value (0.50) for 6 consecutive epochs while `img`
stayed pinned at `ln(72) = 4.2755` and `cross` at `4.15`. The total loss crept down
only through the non-contrastive terms (`var` 0.97 -> 0.14), so it read as a plateau
rather than a failure.

Hypothesis: `effective_scale` clamped only ABOVE (`max=100.0`), so nothing stopped
Adam from driving `logit_scale` to -inf. At scale ~0 every logit is 0, so the contrast
equals `ln(N)` exactly -- the best value reachable *without representing anything* --
and because the encoder's gradient is proportional to the scale it stops receiving
signal entirely. The other terms do not depend on the temperature, so they keep
training and mask the failure.

This runs `--steps` real steps per arm and reports the effective scale, so the claim is
checked against the number that actually decides it rather than inferred from the loss.

  python scripts/probe_temperature.py --out-dir outputs/probe            # via Slurm
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from samclip import config  # noqa: E402
from samclip.losses import contrastive  # noqa: E402
from samclip.utils import load_config  # noqa: E402

ARMS = [
    # (tag, target_fusion, SCALE_MIN)
    ("A_floor_on__routed", "routed", 1.0),
    ("B_floor_off__routed", "routed", 0.0),
    ("C_floor_on__routed_sr", "routed_sr", 1.0),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(config.CONFIGS / "loso_sub08.yaml"))
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--seeds", type=int, nargs="*", default=[2025],
                    help="one run per seed; a single seed cannot tell a collapse from an "
                         "unlucky init, and the collapse is a coin-flip on the sign of "
                         "the temperature's own gradient")
    ap.add_argument("--out-dir", default=str(config.OUTPUTS / "probe"))
    args = ap.parse_args()

    base = load_config(args.config)
    summary = {}
    for seed in args.seeds:
        for tag, fusion, scale_min in ARMS:
            arm = f"{tag}__seed{seed}"
            cfg = dict(base)
            cfg.update({
                "target_fusion": fusion,
                "seed": seed,
                "epochs": 1,
                "debug_steps": args.steps,
                "log_every": args.log_every,
                "out_dir": str(Path(args.out_dir) / arm),
            })
            # `effective_scale` reads the module global at call time, so this is
            # genuinely the floor the run enforces -- the only difference between A/B.
            contrastive.SCALE_MIN = scale_min
            print(f"\n{'=' * 70}\n[probe] {arm}: target_fusion={fusion} "
                  f"SCALE_MIN={scale_min}\n{'=' * 70}", flush=True)

            from samclip.train import train_stage1
            result = train_stage1(cfg)

            scales = {}
            ck = Path(cfg["out_dir"]) / "last.pt"
            if ck.exists():
                import torch
                blob = torch.load(ck, map_location="cpu", weights_only=False)
                scales = {k: round(float(v["logit_scale"]), 3)
                          for k, v in (blob.get("crit") or {}).items()}
            summary[arm] = {"top1": result["last"]["top1"],
                            "mean_rank": result["last"]["mean_rank"],
                            "crit_logit_scale": scales}
            print(f"[probe] {arm} -> top1 {result['last']['top1']:.2f} "
                  f"meanrank {result['last']['mean_rank']:.1f} "
                  f"logit_scale {scales}", flush=True)

    out = Path(args.out_dir) / "probe_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print("\n[probe] summary:\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
