#!/usr/bin/env python
"""A/B/C probe: which visual-target fusion actually helps? (plan doc M2 -> M3)

The three arms answer a chain of questions, in the order they should be asked:

  ``mean``      do the K CLIP layers carry complementary information at all? No learned
                fusion, so a loss here is a statement about the TARGET STACK, not about
                routing capacity.
  ``routed``    can a learned global blend beat equal weighting? It initialises at
                uniform, so it can only improve on ``mean`` -- an arm that matches it
                means the router is not earning its parameters.
  ``routed_sr`` SAMGA's subject-aware residual, dropped at inference. This is the arm
                with a real trade-off, and the one to watch: it makes the TRAINING target
                a function of the row's subject, which rewards the encoder for encoding
                subject identity -- the opposite of what `dec`/`mmd` want. Under v1's much
                stronger decorrelation pressure it measurably backfired (`dec` climbed
                0.22 -> 0.48 in the first epochs). v2's pressure is far lighter
                (dec 0.2 -> 0.05), so the trade-off is genuinely different and this is a
                live question rather than a settled one.

WHAT TO READ, NOT JUST THE TOP-1
--------------------------------
Three numbers, because the top-1 alone cannot tell these arms apart:

  * ``dec`` at the end of the run. If it climbs while `cross` falls, the `routed_sr`
    arm is buying target fit with subject invariance -- and the deployed model has the
    residual dropped, so it cannot spend what was bought.
  * the learned LAYER WEIGHTS (`LayerRouter.layer_weights`, logged by `run_eval` into the
    report). Which of the five CLIP layers the model chose to trust is the interpretable
    content of the whole structured-target design. A blend that stays near uniform is
    evidence that the "layers carry complementary information" hypothesis did not hold
    here, even if the top-1 improved.
  * the temperature scale. `sc_img` pinned at its floor is the signature of the collapse
    described at `contrastive.SCALE_MIN`, and it looks exactly like a plateau.

One epoch is enough to separate arms for the LAYER-WEIGHT and `dec` questions; use
--epochs 3+ before believing a top-1 ordering.

  python scripts/probe_target_fusion.py --out-dir outputs/probe_fusion
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from samclip import config  # noqa: E402
from samclip.utils import load_config  # noqa: E402

ARMS = [
    ("a_uniform", "mean"),
    ("b_routed", "routed"),
    ("c_routed_sr", "routed_sr"),
]

#: The tail of `Trainer`'s step log: `... loss 3.6261 {'img': 3.1, 'dec': 0.02, ...}`.
PARTS_RE = re.compile(r"loss [\d.]+ (\{.*\})\s*$")


class PartsRecorder(logging.Handler):
    """Collect the per-step `parts` dicts emitted by `samclip.train`.

    A log handler rather than a file scrape, deliberately: the step lines go to whatever
    the run's stdout is (an sbatch `.out`, a terminal, a pipe), so parsing a file only
    works when the caller happens to have redirected stdout somewhere the probe can
    find. Hooking the logger works regardless of destination, and it is the only
    tamper-free way to read `dec` -- re-deriving it after the fact would mean keeping
    the model in memory and re-running the forward pass.

    The numbers reported are therefore the ones the run actually logged, which is the
    point: `dec` is the quantity `routed_sr` is suspected of pushing up, and reading it
    from the training log is how a "the target residual costs subject invariance" claim
    would be checked.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.steps: list[dict] = []

    def emit(self, record: logging.LogRecord) -> None:
        m = PARTS_RE.search(record.getMessage())
        if not m:
            return
        try:
            self.steps.append(json.loads(m.group(1).replace("'", '"')))
        except json.JSONDecodeError:
            return

    def final(self, key: str) -> float | None:
        for s in reversed(self.steps):
            if key in s:
                return round(float(s[key]), 4)
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(config.CONFIGS / "loso_sub08.yaml"))
    ap.add_argument("--epochs", type=int, default=1,
                    help="one epoch separates the layer-weight and `dec` questions; use "
                         ">=3 before believing a top-1 ordering")
    ap.add_argument("--log-every", type=int, default=400)
    ap.add_argument("--seeds", type=int, nargs="*", default=[2025, 2026],
                    help="one run per seed. The sub-08 reference cell is 2025, and a "
                         "single seed cannot separate a fusion effect from an init")
    ap.add_argument("--out-dir", default=str(config.OUTPUTS / "probe_fusion"))
    args = ap.parse_args()

    base = load_config(args.config)
    summary: dict[str, dict] = {}
    for seed in args.seeds:
        for tag, fusion in ARMS:
            arm = f"{tag}__seed{seed}"
            cfg = dict(base)
            cfg.update({
                "target_fusion": fusion,
                "seed": seed,
                "epochs": args.epochs,
                "log_every": args.log_every,
                "out_dir": str(Path(args.out_dir) / arm),
            })
            print(f"\n{'#' * 78}\n# {arm}: target_fusion={fusion}\n{'#' * 78}",
                  flush=True)

            from samclip.train import train_stage_a
            recorder = PartsRecorder()
            train_logger = logging.getLogger("samclip.train")
            train_logger.addHandler(recorder)
            try:
                result = train_stage_a(cfg)
            finally:
                # Detach even on failure: a handler left attached to a module-level
                # logger keeps growing across arms and would blend one arm's `dec`
                # trajectory into the next one's.
                train_logger.removeHandler(recorder)

            # The learned blend is the interpretable part: a `routed` arm that lands on
            # near-uniform weights has not shown the layers are complementary.
            import torch
            ck = Path(cfg["out_dir"]) / "last.pt"
            weights = None
            if ck.exists():
                blob = torch.load(ck, map_location="cpu", weights_only=False)
                w = blob["model"].get("target_router.w")
                if w is not None:
                    weights = [round(float(v), 3) for v in torch.softmax(w, dim=0)]
            summary[arm] = {
                "target_fusion": fusion,
                "top1": result["last"]["top1"],
                "top5": result["last"]["top5"],
                "mean_rank": result["last"]["mean_rank"],
                "layer_weights": weights,
                "dec_start": recorder.steps[0].get("dec") if recorder.steps else None,
                "dec_end": recorder.final("dec"),
                "cross_end": recorder.final("cross"),
            }
            print(f"[probe] {arm} -> top1 {result['last']['top1']:.2f} "
                  f"top5 {result['last']['top5']:.2f} "
                  f"meanrank {result['last']['mean_rank']:.1f} "
                  f"layer_weights {weights} "
                  f"dec {summary[arm]['dec_start']} -> {summary[arm]['dec_end']}",
                  flush=True)

    out = Path(args.out_dir) / "fusion_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print("\n[probe] summary (watch `dec` for the routed_sr arm, and whether the "
          "learned layer weights left uniform):")
    for arm, r in summary.items():
        print(f"  {arm:<28} top1 {r['top1']:>5.2f}  meanrank {r['mean_rank']:>6.1f}  "
              f"layers {r['layer_weights']}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
