#!/usr/bin/env python
"""Stage A: train the shared cross-subject EEG encoder for one LOSO fold.

Single stage -- there is no Stage 2 (hypernetwork) and no mapping network. See the
module docstring of `samclip.train` for why each of those was removed.

Run (GPU node, via Slurm):  python scripts/run_stage1.py --config configs/loso_s1.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from samclip import config  # noqa: E402
from samclip.train import train_stage_a  # noqa: E402
from samclip.utils import dump_config, load_config, merge_cli_overrides  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(config.CONFIGS / "loso_s1.yaml"))
    ap.add_argument("--target-subject", type=int, default=None)
    ap.add_argument("--source-subjects", type=int, nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--mvnn", choices=["off", "train", "test"], default=None)
    ap.add_argument("--channel-set", choices=["all63", "occipital17"], default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--debug-steps", type=int, default=None,
                    help="cap optimizer steps per epoch. For a cheap end-to-end smoke "
                         "use `--epochs 2 --stage1-epochs 1 --debug-steps 5`: that "
                         "exercises the real data path, the training loop AND the "
                         "Stage-1 -> Stage-2 boundary (lr drop, encoder freeze) in a few "
                         "minutes. NOTE `--epochs 0` alone is NOT that test -- it returns "
                         "before the loop and runs no training step")
    ap.add_argument("--stage1-epochs", type=int, default=None,
                    help="override `schedule.stage1_epochs` (nested key, so it cannot go "
                         "through the flat shallow-merge override path). Must satisfy "
                         "1 <= stage1_epochs < epochs. REQUIRED for a short smoke on any "
                         "config with `coarse_to_fine: true`: the guard in "
                         "`CoarseToFine.from_cfg` correctly refuses `stage1_epochs=20` "
                         "inside `--epochs 1`, which made the old documented smoke "
                         "recipe (`--epochs 1 --debug-steps 5`) unrunnable on v4")
    ap.add_argument("--no-schedule", action="store_true",
                    help="set `schedule.coarse_to_fine: false` for this run. Use only "
                         "for a DAG smoke that must not pay for two phases; it means the "
                         "phase boundary is NOT tested")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--soft-plan-alpha", type=float, default=None,
                    help="v9: FGW structure weight for the soft-plan term. Its own flag "
                         "because `soft_plan.alpha` is a NESTED key and the override path in "
                         "`merge_cli_overrides` is a flat shallow merge -- the same reason "
                         "`--stage1-epochs` exists. 0.0 reproduces v8's term exactly.")
    ap.add_argument("--soft-plan-weight", type=float, default=None,
                    help="v9: override `soft_plan.weight` (nested key, same reason as above)")
    ap.add_argument("--v12-distill", type=float, default=None,
                    help="v12: override `v12.metric_distill` (nested key; 0.0 is the PAIRED "
                         "TWIN of the v12 arm and reproduces the recipe without the metric "
                         "terms bit-for-bit -- the twin of a training term cannot be built "
                         "any other way)")
    ap.add_argument("--v12-consistency", type=float, default=None,
                    help="v12: override `v12.metric_consistency` (nested key, same reason)")
    ap.add_argument("--v12-r-use", type=int, default=None,
                    help="v12: override `v12.r_use` (repetitions per selected row)")
    ap.add_argument("--reps-weight", type=float, default=None,
                    help="concept: override `concept.reps_weight` (the T2' cross-trial term). "
                         "0.0 keeps the repetition loading (which v12 needs) but turns the term "
                         "off, so a v12 twin is not confounded by T2'.")
    ap.add_argument("--reps-group", choices=["row", "stimulus"], default=None,
                    help="concept: override `concept.reps_group`. `stimulus` is the documented "
                         "fix to the anti-T1 row-grouped arm (sub-08 42.0 -> 30.0).")
    args = ap.parse_args()

    cfg = merge_cli_overrides(load_config(args.config), {
        "target_subject": args.target_subject,
        "source_subjects": args.source_subjects,
        "epochs": args.epochs,
        "lr": args.lr,
        "mvnn": args.mvnn,
        "channel_set": args.channel_set,
        "out_dir": args.out_dir,
        "seed": args.seed,
        "debug_steps": args.debug_steps,
    })

    # A fold-free recipe is a legal config (`configs/g3_loso_k20.yaml` deliberately nulls
    # these so that a missing override cannot silently inherit another fold's subject --
    # `v4.yaml` carries `target_subject: 8`). Without this check the failure surfaces three
    # lines later as `TypeError: unsupported format string passed to NoneType.__format__`
    # from an f-string in a tag, which does not say what to do. Say it here instead.
    if cfg.get("target_subject") is None or not cfg.get("source_subjects"):
        raise SystemExit(
            f"config {args.config} has no fold identity (target_subject="
            f"{cfg.get('target_subject')!r}, source_subjects={cfg.get('source_subjects')!r}). "
            "Pass BOTH --target-subject and --source-subjects; a config that nulls them is "
            "a fold-free recipe, not a runnable fold.")

    if args.stage1_epochs is not None or args.no_schedule:
        # `schedule` is a nested mapping and `merge_cli_overrides` is intentionally a
        # SHALLOW merge, so a nested key has to be handled here or it would silently do
        # nothing -- the exact class of no-op override this project keeps getting bitten
        # by (see the `unknown schedule key(s)` guard in `CoarseToFine.from_cfg`).
        sched = dict(cfg.get("schedule", {}) or {})
        if args.stage1_epochs is not None:
            sched["stage1_epochs"] = int(args.stage1_epochs)
        if args.no_schedule:
            sched["coarse_to_fine"] = False
        cfg["schedule"] = sched

    if args.soft_plan_alpha is not None or args.soft_plan_weight is not None:
        # v9. `soft_plan` is also a nested mapping, so the same shallow-merge caveat applies:
        # without this block `--soft-plan-alpha` would be accepted and do nothing, and the
        # sweep would train v8 thirty times under a v9 name. The guard in the sbatch's
        # preflight is the second line of defence; this is the first.
        sp = dict(cfg.get("soft_plan", {}) or {})
        if args.soft_plan_alpha is not None:
            sp["alpha"] = float(args.soft_plan_alpha)
        if args.soft_plan_weight is not None:
            sp["weight"] = float(args.soft_plan_weight)
        cfg["soft_plan"] = sp

    if args.v12_distill is not None or args.v12_consistency is not None \
            or args.v12_r_use is not None:
        # v12. `v12` is a nested mapping, so the same shallow-merge caveat applies as for
        # `schedule` and `soft_plan`: without this block a `--v12-*` flag would be accepted
        # and silently do nothing, and the twin arm would train the base recipe while being
        # reported as the treatment. The PAIRED TWIN is exactly `--v12-distill 0
        # --v12-consistency 0`, which is why the weights are settable (not just the block).
        v = dict(cfg.get("v12", {}) or {})
        if args.v12_distill is not None:
            v["metric_distill"] = float(args.v12_distill)
        if args.v12_consistency is not None:
            v["metric_consistency"] = float(args.v12_consistency)
        if args.v12_r_use is not None:
            v["r_use"] = int(args.v12_r_use)
        # `enabled` is implied by a positive weight; leaving it false while a weight is set
        # would make the term inert and the run a silent no-op reported as a treatment.
        if float(v.get("metric_distill", 0.0)) > 0 or float(v.get("metric_consistency", 0.0)) > 0:
            v["enabled"] = True
        cfg["v12"] = v

    if args.reps_weight is not None or args.reps_group is not None:
        # `concept` is the third nested mapping the shallow `merge_cli_overrides` cannot reach.
        # `reps_weight: 0` with `enabled: true` is the deliberate configuration that loads the
        # repetitions (so v12 can consume them) while leaving the T2' term inert -- which is
        # what makes a v12 twin a one-variable experiment.
        c = dict(cfg.get("concept", {}) or {})
        if args.reps_weight is not None:
            c["reps_weight"] = float(args.reps_weight)
        if args.reps_group is not None:
            c["reps_group"] = str(args.reps_group)
        if "enabled" not in c:
            c["enabled"] = True
        cfg["concept"] = c
    tag = args.tag or f"sub{cfg['target_subject']:02d}"
    if args.out_dir is None:
        # Kept as `stage1/` rather than `stage_a/`: the directory name is the identity of
        # a result in every existing Slurm log, checkpoint path and eval report, and the
        # rename would orphan all of them to buy nothing. The stage is Stage A in prose.
        cfg["out_dir"] = str(config.OUTPUTS / "stage1" / tag)
    Path(cfg["out_dir"]).mkdir(parents=True, exist_ok=True)
    dump_config(cfg, Path(cfg["out_dir"]))
    print("[stageA] config:\n" + json.dumps(cfg, indent=2, default=str))

    result = train_stage_a(cfg)
    (Path(cfg["out_dir"]) / "result.json").write_text(json.dumps(result, indent=2))
    print("[stageA] result:", json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
