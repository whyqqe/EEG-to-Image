#!/usr/bin/env python
"""Stage 2 CLI: multi-modal alignment with subject invariance, LOSO on one subject.

Example (smoke test):
  python scripts/train_align.py --test-subject sub-08 --epochs 1 --max-steps-per-epoch 20 \
      --batch-size 128 --eval-triv --eval-every 1 --num-workers 2
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields

from loso import paths
from loso.models.eeg_encoder import EncoderConfig
from loso.models.heads import HeadConfig
from loso.losses.align import LossWeights
from loso.train.align import AlignConfig, run


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-subject", default="sub-08")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--warmup-epochs", type=int, default=3)
    ap.add_argument("--adv-warmup-epochs", type=int, default=5)
    ap.add_argument("--topk", type=int, default=10, help="soft-label neighbours")
    ap.add_argument("--norm-source", default="train_subjects",
                    choices=["train_subjects", "per_subject_train"])
    ap.add_argument("--amp-dtype", default="bf16", choices=["bf16", "fp16", "none"])
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--eval-triv", action="store_true",
                    help="evaluate on 2,000 test trials instead of all 16,000")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps-per-epoch", type=int, default=0)
    ap.add_argument("--segments", type=int, default=64,
                    help="number of tokens the time head aligns to VAE patches")
    ap.add_argument("--sampler", default=AlignConfig.sampler,
                    choices=["grouped", "shuffled"],
                    help="'grouped' puts whole same-image groups in each batch so the "
                         "soft label has same-image peers; 'shuffled' is uniform")
    ap.add_argument("--gradient-budget-steps", type=int, default=0,
                    help="steps of the first epoch that measure each loss term's share "
                         "of the encoder gradient (one backward per term; 0 = off)")
    ap.add_argument("--eval-avg-reps", type=int, nargs="+",
                    default=list(AlignConfig.eval_avg_reps),
                    help="test protocols to report: N averages N of the 80 repetitions "
                         "per test concept (1 = single-trial headline, 80 = high SNR)")
    ap.add_argument("--run-tag", default="",
                    help="sub-directory under align/<subject>/; use for previews so a "
                         "short run cannot overwrite the canonical checkpoint")
    ap.add_argument("--avg-trials", dest="avg_trials", action="store_true",
                    default=AlignConfig.avg_trials,
                    help="average the 4 train repetitions per image (UCK/SAMGA recipe; "
                         "4x fewer steps/epoch)")
    ap.add_argument("--no-avg-trials", dest="avg_trials", action="store_false")
    ap.add_argument("--mem-k", type=int, default=AlignConfig.mem_k,
                    help="top-k for the UCK memory retrieve term")
    ap.add_argument("--proj-layers", type=int, default=HeadConfig.proj_layers,
                    help="projector depth; 1 is a linear map (UCK/SAMGA style)")

    # architecture
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--branch-width", type=int, default=64)
    ap.add_argument("--d-inv", type=int, default=512)
    ap.add_argument("--d-sub", type=int, default=128)
    ap.add_argument("--adapter-blocks", type=int, default=2)
    ap.add_argument("--proj-hidden", type=int, default=HeadConfig.proj_hidden,
                    help="projector hidden width; the head/trunk capacity ratio is "
                         "asserted in scripts/smoke_align.py")
    ap.add_argument("--vicreg-gamma", type=float, default=HeadConfig.vicreg_gamma,
                    help="VICReg variance hinge target; measured, not guessed -- see "
                         "scripts/measure_z_scale.py")

    # Loss weights.  Defaults are read off `LossWeights` rather than repeated here:
    # a literal default in this file is a second source of truth, and a stale copy
    # silently restores the old weights no matter what the dataclass says.  That is
    # exactly how the previous run kept `time`/`trial` at 3.0 while the dataclass had
    # moved to 1.0.
    defaults = LossWeights()
    for name in [f.name for f in fields(LossWeights)]:
        ap.add_argument(f"--w-{name}", type=float, default=getattr(defaults, name),
                        dest=f"w_{name.replace('-', '_')}",
                        help=f"weight for the {name} term (default {getattr(defaults, name)})")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    paths.ensure_dirs()

    cfg = AlignConfig(
        test_subject=args.test_subject,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, warmup_epochs=args.warmup_epochs,
        adv_warmup_epochs=args.adv_warmup_epochs, topk=args.topk,
        norm_source=args.norm_source, amp_dtype=args.amp_dtype,
        num_workers=args.num_workers, eval_every=args.eval_every,
        eval_trials=2000 if args.eval_triv else 0,
        log_every=args.log_every, seed=args.seed,
        max_steps_per_epoch=args.max_steps_per_epoch,
        sampler=args.sampler,
        gradient_budget_steps=args.gradient_budget_steps,
        eval_avg_reps=tuple(args.eval_avg_reps),
        run_tag=args.run_tag,
        avg_trials=args.avg_trials,
        mem_k=args.mem_k,
        enc=EncoderConfig(
            d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
            branch_width=args.branch_width, d_inv=args.d_inv, d_sub=args.d_sub,
            adapter_blocks=args.adapter_blocks,
        ),
        head=HeadConfig(n_time_patches=args.segments,
                        proj_hidden=args.proj_hidden,
                        proj_layers=args.proj_layers,
                        vicreg_gamma=args.vicreg_gamma),
        weights=LossWeights(
            **{f.name: getattr(args, f"w_{f.name}") for f in fields(LossWeights)}
        ),
    )
    print("[align] loss weights: " + json.dumps(cfg.weights.to_dict()), flush=True)
    run(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
