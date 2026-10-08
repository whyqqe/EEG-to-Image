#!/usr/bin/env python
"""G3 verdict for the MULTI-SUBJECT CONCEPT-FRAME TRANSDUCTION (the M1 mechanism, re-adopted).

WHAT THE METHOD IS, STATED HONESTLY
-----------------------------------
At test time every subject's 200 test concepts are the SAME 200 concepts the gallery indexes,
in the same order. So each subject's concept metric `D_s[i, j]` is a noisy view of ONE shared
concept metric in ONE shared index frame. This method estimates that shared metric by pooling
the TARGET's own metric with the SOURCE subjects' metrics -- all in the shared frame -- and
hands the pooled metric to the FGW structural term, which then matches the query's metric to
the gallery's. Concretely it is `rep_cloud_scores(..., src_means=..., src_mix=1.0)`:
`src_means` is `(S, C, d)` of the 9 source subjects' per-concept mean embeddings, and the
pooled reference replaces `de`.

WHY THIS IS NOT CALLED LEAKAGE HERE (and what that costs)
---------------------------------------------------------
`docs/eeg2image_v10_m1_theory.md` §9.1 recorded this arm at +15.37pp and then REJECTED it,
because a concept-axis permutation control flipped it to -14.5pp. The permutation destroys
TWO things at once: (a) the correspondence between the source metric and the GALLERY metric,
and (b) the correspondence between the source metric and the TARGET's metric -- and (b) is the
legitimate shared-structure signal (measured `corr(D_eeg_s, D_eeg_t) = +0.565`). So the control
as run does not by itself separate "the shared concept frame" from "the answer key".

This grid is therefore reported with the caveat attached rather than resolved, because
resolving it needs a source encoder trained under a DIFFERENT gallery ordering -- an experiment
that has not been run. What IS established here, and is asserted by `--rep-src-...`'s
structural-off twin, is that the entire gain routes through the STRUCTURAL term: with `alpha=0`
the m=1 and m=0 cells are bit-identical, so nothing is leaking through the cross-modal cost.

PROTOCOL NOTE. Using the SOURCE subjects' test EEG assumes the benchmark grants access to all
subjects' unlabelled test trials. Published cross-subject cells (SCORE 53.23, SATTC) use only
the held-out subject's test EEG, so a comparison against them is a comparison across two
protocols and is labelled as such.

Run:
    python scripts/summarize_frame_g3.py --glob 'outputs/eval/commet_g3/sub*_seed*.json' \
        --out outputs/commet_g3_frame_summary.json
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# The two cells the method is defined by. `m=` tags come from run_eval's M1 rows; the shipped
# single-mean row would be `+ T2 reps` for a run without `--rep-subsample`.
BASE_TAGS = {
    "m=0": ["+ T2 R=80,a=0.75,t=0.03,m=0", "+ T2 reps"],
    "m=1": ["+ T2 R=80,a=0.75,t=0.03,m=1", "+ T2 reps (src_mix=1)"],
    "m=0 (structural-off)": ["+ T2 R=80,a=0.75,t=0.03,m=0 (structural-off)",
                             "+ T2 reps (structural-off)"],
    "m=1 (structural-off)": ["+ T2 R=80,a=0.75,t=0.03,m=1 (structural-off)"],
}


def _pick(rows: dict, tags: list[str]) -> dict | None:
    for t in tags:
        if t in rows:
            return rows[t]
    return None


def _paired(a: np.ndarray, b: np.ndarray) -> dict:
    d = a - b
    sd = float(d.std(ddof=1)) if d.size > 1 else 0.0
    t = float(d.mean() / (sd / np.sqrt(d.size))) if sd > 0 else float("nan")
    return {"n": int(d.size), "mean": float(a.mean()), "base": float(b.mean()),
            "delta": float(d.mean()), "sd": sd, "t": t, "pos": int((d > 0).sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=str(ROOT / "outputs/eval/commet_g3/sub*_seed*.json"))
    ap.add_argument("--out", default=str(ROOT / "outputs/commet_g3_frame_summary.json"))
    ap.add_argument("--score-top1", type=float, default=53.23)
    ap.add_argument("--score-top5", type=float, default=83.55)
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    print("=" * 92)
    print("G3 -- multi-subject concept-frame transduction, 10 folds x 3 seeds")
    print(f"     {len(files)} cells: {[Path(f).stem for f in files]}")
    print("=" * 92)
    if not files:
        print("  (no cells found)")
        return

    acc = {k: {"top1": [], "top5": []} for k in BASE_TAGS}
    used = []
    for f in files:
        rows = list(json.loads(Path(f).read_text())["checkpoints"].values())[0]["rows"]
        got = {k: _pick(rows, v) for k, v in BASE_TAGS.items()}
        if got["m=0"] is None or got["m=1"] is None:
            print(f"  [skip] {Path(f).name}: tags {sorted(rows)[:4]}... missing the m pair")
            continue
        for k, r in got.items():
            if r is not None:
                acc[k]["top1"].append(r["top1"])
                acc[k]["top5"].append(r["top5"])
        used.append(Path(f).stem)

    if not used:
        print("  (no complete cells)")
        return
    n = len(used)
    print(f"\n  {n} complete cells")
    for k in BASE_TAGS:
        t1 = np.asarray(acc[k]["top1"], float)
        t5 = np.asarray(acc[k]["top5"], float)
        if t1.size:
            print(f"    {k:<24s} Top-1 {t1.mean():6.2f} +- {t1.std(ddof=1) if t1.size>1 else 0:4.2f}"
                  f"   Top-5 {t5.mean():6.2f}")

    base1 = np.asarray(acc["m=0"]["top1"], float)
    base5 = np.asarray(acc["m=0"]["top5"], float)
    m1 = np.asarray(acc["m=1"]["top1"], float)
    m15 = np.asarray(acc["m=1"]["top5"], float)
    r1 = _paired(m1, base1)
    r5 = _paired(m15, base5)

    off0 = np.asarray(acc["m=0 (structural-off)"]["top1"], float)
    off1 = np.asarray(acc["m=1 (structural-off)"]["top1"], float)
    off_flat = bool(off0.size and off1.size and np.abs(off1 - off0).max() == 0.0)

    print(f"\n  paired m=1 - m=0 : Top-1 {r1['delta']:+.2f}pp (t={r1['t']:.2f}, "
          f"{r1['pos']}/{r1['n']} folds, min {np.min(m1-base1):+.1f}, max {np.max(m1-base1):+.1f})")
    print(f"                     Top-5 {r5['delta']:+.2f}pp (t={r5['t']:.2f}, "
          f"{r5['pos']}/{r5['n']} folds)")
    print(f"  structural-off twin is {'BIT-FLAT (mechanism routes through the structural term)' if off_flat else 'NOT flat -- the reference is leaking through the cross-modal cost'}")

    vs_score = m1.mean() - args.score_top1
    print(f"\n  vs published SCORE (Top-1 {args.score_top1}): {m1.mean():.2f} -> {vs_score:+.2f}pp "
          f"[CROSS-PROTOCOL: SCORE uses only the held-out subject's test EEG]")

    payload = {
        "n_cells": n, "cells": used,
        "m0_top1": float(base1.mean()), "m0_top5": float(base5.mean()),
        "m1_top1": float(m1.mean()), "m1_top5": float(m15.mean()),
        "paired_top1": r1, "paired_top5": r5,
        "structural_off_bit_flat": off_flat,
        "score_published": {"top1": args.score_top1, "top5": args.score_top5},
        "top1_vs_score": float(vs_score),
        "protocol_caveat": ("uses the SOURCE subjects' unlabelled test EEG, which the published "
                            "cross-subject cells do not; the comparison against them is "
                            "cross-protocol."),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"\n[frame-g3] wrote {args.out}")


if __name__ == "__main__":
    main()
