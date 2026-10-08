#!/usr/bin/env python
"""Submit the LOSO pipeline for one target subject as a chained Slurm DAG.

    cache ──▶ stageA ──▶ eval

The v1 DAG had a Stage-2 (hypernetwork) branch and a second eval that compared the two
checkpoints side by side. Both are gone with the conditioning paradigm -- see
`samclip.models.subject_conditioning`. There is one training job and one evaluation job
per seed, and the evaluation job reports the full geometric-calibration ladder
(raw cosine -> SAW -> CSLS -> recovery) rather than a conditioning axis.

Each stage is a separate allocation with its own log, and the dependency is `afterok`,
so a failed training job stops its eval instead of letting it run on a half-written
checkpoint.

Run:
  python scripts/submit_pipeline.py --target-subject 8 --recovery
  python scripts/submit_pipeline.py --target-subject 8 --dry-run
  python scripts/submit_pipeline.py --target-subject 8 --epochs 2 --debug-steps 5 \
      --out-tag smoke      # cheap "does the whole DAG run" submit
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from samclip import config  # noqa: E402
from samclip.utils import load_config  # noqa: E402


def submit(sbatch: Path, exports: dict[str, str], depend_on: str | None = None,
           dry_run: bool = False) -> str | None:
    """Submit one job; return its Slurm job id (None on dry-run)."""
    cmd = ["sbatch"]
    if depend_on:
        cmd.append(f"--dependency=afterok:{depend_on}")
    export_pairs = ",".join(f"{k}={v}" for k, v in exports.items() if v is not None)
    cmd.append(f"--export=ALL,{export_pairs}")
    cmd.append(str(sbatch))
    print("  " + " ".join(cmd))
    if dry_run:
        return None
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise SystemExit(f"sbatch failed for {sbatch}:\n{proc.stderr.strip()}")
    # "Submitted batch job 12345"
    job_id = proc.stdout.strip().split()[-1]
    return job_id


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-subject", type=int, required=True)
    ap.add_argument("--config", default=None,
                    help="default: configs/loso_sub<NN>.yaml when it exists, else "
                         "configs/loso_s1.yaml. The fold-specific file states the "
                         "source-subject list explicitly instead of leaving it to be "
                         "inferred from `target_subject`.")
    ap.add_argument("--mvnn", default="train",
                    help="training-side MVNN setting ('train' on sources). The eval "
                         "stage mirrors it to the target's 'test' split.")
    ap.add_argument("--channel-set", default="all63")
    ap.add_argument("--seeds", type=int, nargs="*", default=[2025],
                    help="the sub-08 reference cell is seed 2025; the protocol wants "
                         ">=3 seeds for the mean/std table")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--debug-steps", type=int, default=None,
                    help="cap optimizer steps per epoch. For a cheap DAG smoke use "
                         "`--epochs 2 --debug-steps 5`. `stage1_epochs` is then derived "
                         "automatically (see below) so the run still CROSSES the "
                         "Stage-1 -> Stage-2 boundary instead of skipping it. NOTE "
                         "`--epochs 0` is NOT a smoke: it returns before the training "
                         "loop, so it validates the data path and eval but never "
                         "assembles a loss")
    ap.add_argument("--stage1-epochs", type=int, default=None,
                    help="override `schedule.stage1_epochs`. Required to be "
                         "1 <= stage1_epochs < epochs; on a short run it is derived from "
                         "`--epochs` automatically rather than left to fail on the GPU "
                         "(see `--epochs`). Pass 0 to force the schedule OFF for a "
                         "DAG-only smoke that does not test the phase boundary.")
    ap.add_argument("--out-tag", default=None,
                    help="override the output directory tag. Use one per ablation arm, "
                         "PAIRED WITH --config (this flag only renames the output, it "
                         "does not select the objective). The eval report is keyed by "
                         "the checkpoint's parent directory name, so two arms sharing a "
                         "tag would silently overwrite each other in the report.")
    ap.add_argument("--recovery", action="store_true",
                    help="apply SCORE coordinate recovery in the eval stage")
    ap.add_argument("--min-landmark-rate", type=float, default=None)
    ap.add_argument("--dump-ranks", action="store_true",
                    help="also write per-concept ranks for paired arm comparisons")
    ap.add_argument("--no-cache", action="store_true",
                    help="skip stage 0 when the caches are known to be current")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    s = args.target_subject
    tag = args.out_tag or f"sub-{s:02d}"
    root = config.ROOT
    slurm = root / "slurm"

    if args.config is None:
        fold_cfg = root / "configs" / f"loso_sub{s:02d}.yaml"
        args.config = str(fold_cfg.relative_to(root)) if fold_cfg.is_file() \
            else "configs/loso_s1.yaml"

    # The schedule guard in `CoarseToFine.from_cfg` refuses `stage1_epochs >= epochs`,
    # which is CORRECT -- a 20-epoch Stage 1 inside `--epochs 1` is a mislabelled
    # single-stage run. But it means the once-documented smoke recipe
    # (`--epochs 1 --debug-steps 5`) could not run on any scheduled config, and the
    # failure only surfaced on a GPU node. Derive a VALID schedule here instead, and say
    # so loudly, rather than letting the job die after queueing.
    stage1_epochs = args.stage1_epochs
    if args.epochs is not None:
        sched = (load_config(args.config).get("schedule") or {})
        cfg_s1 = int(sched.get("stage1_epochs", 0) or 0)
        if stage1_epochs is None and sched.get("coarse_to_fine") and cfg_s1 >= args.epochs:
            if args.epochs < 2:
                stage1_epochs = 0
                print(f"[submit] WARNING: --epochs {args.epochs} cannot contain a "
                      f"Stage 1 + Stage 2, so the schedule is DISABLED for this run and "
                      f"the phase boundary will NOT be tested. Use --epochs 2 for a "
                      f"smoke that does.")
            else:
                stage1_epochs = max(1, args.epochs // 2)
                print(f"[submit] NOTE: config stage1_epochs={cfg_s1} >= epochs="
                      f"{args.epochs}; deriving stage1_epochs={stage1_epochs} so the run "
                      f"crosses the Stage-1 -> Stage-2 boundary instead of failing the "
                      f"guard on the GPU node.")

    print(f"[submit] LOSO pipeline for sub-{s:02d} | tag={tag} | seeds={args.seeds} "
          f"| recovery={args.recovery}")

    cache_dep = None
    if not args.no_cache:
        print("[submit] stage 0: cache")
        cache_dep = submit(slurm / "00_cache.sbatch",
                           {"TARGET_SUBJECT": str(s), "MVNN": args.mvnn,
                            "CHANNEL_SET": args.channel_set},
                           dry_run=args.dry_run)

    for seed in args.seeds:
        suffix = "" if len(args.seeds) == 1 else f"_seed{seed}"
        s1_dir = root / "outputs" / "stage1" / f"{tag}{suffix}"
        eval_out = root / "outputs" / "eval" / f"{tag}{suffix}.json"

        print(f"[submit] stage A: train shared encoder (seed {seed}) -> {s1_dir}")
        s1 = submit(slurm / "10_stage1.sbatch",
                    {"TARGET_SUBJECT": str(s), "CONFIG": args.config,
                     "OUT_DIR": str(s1_dir), "SEED": str(seed), "MVNN": args.mvnn,
                     "CHANNEL_SET": args.channel_set,
                     "EPOCHS": str(args.epochs) if args.epochs is not None else None,
                     "DEBUG_STEPS": str(args.debug_steps)
                     if args.debug_steps else None,
                     "STAGE1_EPOCHS": str(stage1_epochs) if stage1_epochs else None,
                     "NO_SCHEDULE": "1" if stage1_epochs == 0 else None},
                    depend_on=cache_dep, dry_run=args.dry_run)

        print(f"[submit] stage C: evaluate {s1_dir / 'last.pt'} -> {eval_out}")
        submit(slurm / "30_eval.sbatch",
               {"TARGET_SUBJECT": str(s), "CKPT": str(s1_dir / "last.pt"),
                "OUT": str(eval_out), "MVNN": "test",
                "RECOVERY": "1" if args.recovery else None,
                "MIN_LANDMARK_RATE": str(args.min_landmark_rate)
                if args.min_landmark_rate is not None else None,
                "DUMP_RANKS": "1" if args.dump_ranks else None},
               depend_on=s1, dry_run=args.dry_run)

    print("\n[submit] done. Follow with:  squeue -u $USER")


if __name__ == "__main__":
    main()
