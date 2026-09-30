#!/usr/bin/env python
"""Build every MVNN cache a LOSO sweep needs, before the sweep starts.

Two reasons this is a separate step rather than a lazy side effect of training:

  * The whitener is fitted per subject from a 4 GiB raw file, and a ten-fold LOSO
    sweep over ten subjects touches each subject once. Done inside the training job
    it adds minutes of single-threaded preprocessing to a GPU allocation, once per
    fold, for the same answer every time.
  * More importantly, it makes the preprocessing auditable. A fold that fits its own
    whitener is a fold whose whitener nobody has looked at; this prints all of them
    side by side, so a subject whose `lam` or `cond` is an outlier is visible before
    it becomes an unexplained number in a table.

Which split each role uses is decided by `load_loso`, not here -- this just
materialises what that will ask for.

Run:  python scripts/epd/build_mvnn_cache.py --subjects 1 2 3 4 5 6 7 8 9 10
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from epd import config                                  # noqa: E402
from epd.data import load_subject_std                   # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 11)))
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--channels", default="all",
                    help="'all' for the 63-channel protocol, or 'occipito_parietal'")
    ap.add_argument("--shrinkage", default="lw", choices=["lw", "fixed"])
    ap.add_argument("--max-cond", type=int, default=0)
    args = ap.parse_args()

    channels = None if args.channels == "all" else config.CHANNELS_OCCIPITO_PARIETAL
    cache = config.OUTPUTS / "cache"
    t_all = time.time()
    rows = []
    for s in args.subjects:
        for split in args.splits:
            # `load_loso` calls this same function with these same arguments, so the
            # files written here are the files the fold will hit.
            t0 = time.time()
            arrs = load_subject_std(s, channels, cache, mvnn=split,
                                    mvnn_shrinkage=args.shrinkage,
                                    mvnn_max_cond=args.max_cond, verbose=True)
            n = int(arrs[0 if split == "train" else 1].shape[0])
            meta_p = cache / (f"mvnn_W_sub{s:02d}_"
                              f"{'all63' if channels is None else f'{len(channels)}ch'}"
                              f"_{split}_{args.shrinkage}.json")
            meta = json.loads(meta_p.read_text()) if meta_p.is_file() else {}
            rows.append((s, split, n, meta.get("lam"), meta.get("cond"),
                         round(time.time() - t0, 1)))
            del arrs
        print()

    print(f"{'sub':>4} {'split':>6} {'concepts':>9} {'lam':>7} {'cond':>7} {'s':>6}")
    for s, split, n, lam, cond, secs in rows:
        lam_s = f"{lam:.4f}" if lam is not None else "cached"
        cond_s = f"{cond:.1f}" if cond is not None else "-"
        print(f"{s:>4} {split:>6} {n:>9} {lam_s:>7} {cond_s:>7} {secs:>6}")
    print(f"\n{len(rows)} caches in {time.time() - t_all:.0f}s -> {cache}")

    lams = [r[3] for r in rows if r[3] is not None]
    conds = [r[4] for r in rows if r[4] is not None]
    if lams:
        print(f"lam  min {min(lams):.4f} max {max(lams):.4f}")
    if conds:
        print(f"cond min {min(conds):.1f} max {max(conds):.1f}")
        if max(conds) > 500:
            print("[warn ] a condition number above 500 means the whitener is "
                  "amplifying a direction the data cannot estimate")


if __name__ == "__main__":
    main()
