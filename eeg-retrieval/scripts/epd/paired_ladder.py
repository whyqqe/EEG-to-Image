#!/usr/bin/env python3
"""Pair every generation arm on the 200 concepts and report the shifts with CIs.

Why a script rather than the shell block
----------------------------------------
`run_epd_da2.sh` can only pair arms that live in ITS OWN metrics directory, and the
arms that decide the architecture are spread across three runs that were submitted
separately (the deployed ladder in `*_depth8_metrics`, the ground-truth-depth control
in `*_oracle_metrics`, the semantic ceiling in `*_ceiling_metrics`). The decisive
comparisons are exactly the cross-directory ones -- "is a GROUND-TRUTH depth map any
better than the EEG one", "does a perfect IP condition still lose to the depth
ControlNet" -- so the pairing has to be done over all three at once.

The arms are paired: same 200 test concepts, same order, same generation seed. Each
arm's `<arm>_seven_persample.json` holds `q_i`, the fraction of the other 199 gallery
entries that concept i's ground truth beats under the official (Pearson) similarity.
`mean(q)` is exactly the reported two-way accuracy, so a paired bootstrap over the
200 per-concept differences is the right test and it is far tighter than treating the
two arms' Wilson intervals as independent (which would overstate the width ~14x, and
would be why every structural arm "looks indistinguishable").

Two tests are reported because they fail differently: the bootstrap CI is sensitive
to the size of the shift, the sign test only to how often its direction flips. With
~25 comparisons here, a single `*` at p~0.04 is what chance produces; the signs and
the CIs have to agree before a claim is made.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from epd.stats import paired_bootstrap, sign_test

METRIC_DIRS = [
    "outputs/sub08/epd_da2_depth8_metrics",
    "outputs/sub08/epd_da2_depth8_oracle_metrics",
    "outputs/sub08/epd_da2_depth8_ceiling_metrics",
]

PAIRS = [
    ("oracle_cn070", "depth_cn070", "GT depth vs EEG depth, same route, CN 0.70"),
    ("oracle_cn035", "depth_cn035", "GT depth vs EEG depth, same route, CN 0.35"),
    ("depth_cn070", "sem_only", "EEG depth CN 0.70 vs semantic only"),
    ("depth_cn035", "sem_only", "EEG depth CN 0.35 vs semantic only"),
    ("oracle_cn070", "sem_only", "GT depth CN 0.70 vs semantic only"),
    ("oracle_cn035", "sem_only", "GT depth CN 0.35 vs semantic only"),
    ("ip_oracle_cn070", "ip_oracle", "GT depth on top of a PERFECT IP condition"),
    ("ip_oracle", "sem_only", "the semantic ceiling vs deployed"),
    ("null_cn070", "null_txt2img", "structural condition with NO EEG"),
    ("null_txt2img", "sem_only", "zero EEG vs deployed (the floor)"),
]

TWKEYS = [("clip", "CLIP"), ("inception", "Inception"), ("alex5", "AlexNet(5)"),
          ("alex2", "AlexNet(2)")]


def load_q(root: Path, dirs: list[str], arm: str, key: str):
    for d in dirs:
        p = root / d / f"{arm}_seven_persample.json"
        if not p.is_file():
            continue
        q = json.loads(p.read_text(encoding="utf-8")).get("q", {}).get(key)
        if q is not None:
            return np.asarray(q, dtype=np.float64)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--dirs", nargs="+", default=METRIC_DIRS)
    ap.add_argument("--resamples", type=int, default=10000)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    n_boot = args.resamples

    print("## Paired tests over the 200 concepts (bootstrap 95% CI, "
          f"{n_boot} resamples)")
    print()
    print("| comparison | metric | delta | 95% CI | sign p | verdict |")
    print("|---|---|---:|---|---:|---|")
    notes: list[str] = []
    for a, b, label in PAIRS:
        for key, name in TWKEYS:
            qa, qb = load_q(root, args.dirs, a, key), load_q(root, args.dirs, b, key)
            if qa is None or qb is None or qa.shape != qb.shape:
                continue
            bs = paired_bootstrap(qa, qb, n_resamples=n_boot)
            p = sign_test(qa - qb)["p"]
            if bs["excludes_zero"]:
                verdict = "shifted" if abs(bs["delta"]) > 0.02 else "shifted (small)"
            elif p < 0.01 and abs(bs["delta"]) > 0.02:
                verdict = "shifted (sign only)"
            else:
                verdict = "indistinguishable"
            print(f"| {label} | {name} | {bs['delta']:+.3f} | "
                  f"[{bs['lo']:+.3f}, {bs['hi']:+.3f}] | {p:.3f} | {verdict} |")
            if a in ("oracle_cn070", "ip_oracle_cn070") and b in ("depth_cn070", "ip_oracle") \
                    and key in ("clip", "inception"):
                notes.append((label, name, bs["delta"], bs["lo"], bs["hi"]))

    print()
    print("### The comparisons the architecture turns on")
    print()
    for label, name, m, lo, hi in notes:
        same = lo <= 0.0 <= hi
        tail = ("NO detectable difference up to the CI, i.e. the quality of the "
                "structural prediction is not what limits the arm" if same
                else "a real difference")
        print(f"- **{label}** on {name}: {m:+.3f} (CI [{lo:+.3f}, {hi:+.3f}]) -> {tail}")
    print()
    print("A `*`-free row is not a null result: it means the shift, if any, is smaller "
          "than this run can resolve (the unpaired threshold for a 200-way score is "
          "~10 points; pairing is what buys the resolution to see 0.02 here).")


if __name__ == "__main__":
    main()
