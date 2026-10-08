#!/usr/bin/env python
"""Prebuild the EEG and image-target caches a fold needs, then exit.

Why this is its own pipeline stage
----------------------------------
Training must not be the first thing to discover a missing or stale cache: a Slurm
job that dies 20 minutes in because one subject's `_std.npy` predates a
preprocessing change is a wasted allocation. This walks every subject the fold
needs, materialises the caches, and asserts the per-subject standardisation landed.

MVNN split is per-ROLE, and that asymmetry is the protocol (see
`things_eeg.load_subject_std`): a source subject is whitened by its own labelled
**train** residuals, while the held-out subject is whitened by its own **test**
residuals, because its train split is exactly what LOSO excludes from training.

Run:  python scripts/build_cache.py --target-subject 8 --mvnn train
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from samclip import config  # noqa: E402
from samclip.data import things_eeg, targets as target_mod  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-subject", type=int, required=True,
                    help="the LOSO held-out subject; it is the ONLY subject allowed "
                         "to use the 'test' MVNN split")
    ap.add_argument("--source-subjects", type=int, nargs="*", default=None)
    ap.add_argument("--mvnn", choices=["off", "train"], default="train",
                    help="'train' enables MVNN on the source subjects' train split and "
                         "the target's test split; 'off' is the no-whitening ablation")
    ap.add_argument("--channel-set", choices=["all63", "occipital17"], default="all63")
    ap.add_argument("--feature-set", default="internvit_multilevel")
    ap.add_argument("--target-layers", type=int, nargs="*", default=None)
    ap.add_argument("--skip-targets", action="store_true")
    args = ap.parse_args()

    sources = args.source_subjects or [s for s in config.all_subjects()
                                       if s != args.target_subject]
    channels = (config.CHANNELS_OCCIPITO_PARIETAL
                if args.channel_set == "occipital17" else None)
    print(f"[cache] target=sub-{args.target_subject:02d} sources={sources} "
          f"mvnn={args.mvnn} channels={args.channel_set}", flush=True)

    for s in sources + [args.target_subject]:
        # The target's flagged role is what selects the MVNN fit split. Deriving it
        # from list position (as an earlier version did) silently whitens the wrong
        # subject whenever the target is not last -- which is every fold but one.
        role = "test" if s == args.target_subject else "train"
        split = role if args.mvnn != "off" else "off"
        t0 = time.time()
        things_eeg.load_subject_std(s, channels, mvnn=split)
        print(f"[cache] sub-{s:02d} mvnn={split:5s} ok ({time.time() - t0:.0f}s)",
              flush=True)

    if not args.skip_targets:
        for split in ("train", "test"):
            arr = target_mod.load_target_stack(args.feature_set, args.target_layers,
                                               split)
            print(f"[cache] targets {split}: {tuple(arr.shape)}", flush=True)

    print("[cache] done", flush=True)


if __name__ == "__main__":
    main()
