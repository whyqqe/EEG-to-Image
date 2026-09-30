#!/usr/bin/env python
"""
Time-resolved split-half reliability of the EEG, independent of any image model.

WHY THIS SETTLES THE QUESTION
    The A1 gate found held-out canonical correlation ~0.27 in the pre-stimulus
    window (p < 0.001 against a 1200-sample permutation null).  That could be
      (a) genuine concept-specific EEG structure before stimulus onset, or
      (b) a CCA-specific artefact (overfitting on structure that is shared across
          concepts but not concept-specific).
    This script discriminates without using CLIP at all.  It splits the
    REPETITIONS 5/5, builds a concept x concept RDM from each half, and
    correlates the two RDMs.  That measures reproducible CONCEPT-SPECIFIC
    structure and nothing else.

    If the pre-window is reliable, the pre-window signal is real in the data.
    If it is unreliable while CCA still finds it, the CCA result is an artefact
    and the gate is telling us to distrust that component.

Also reports the F-ratio (across-concept vs within-concept variance), which is
the same statement in parametric form, and the implied noise ceiling.

Run on train (T=10, halves of 5).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_rsca import rdm, spearman_brown, upper_tri  # noqa: E402

EEG_ROOT = "/project/peilab/why/NeuroBridge/data/things_eeg"
ONSET = 50          # index of t = 0 s


def load_eeg(sub, which="train"):
    a = np.load(f"{EEG_ROOT}/preprocessed_eeg/{sub}/{which}.npy")
    if a.ndim == 5:
        a = a.mean(2)
    return a.astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/project/peilab/why/eeg-retrieval/alignment/outputs/timereliab")
    ap.add_argument("--subs", nargs="+", default=["sub-01"])
    ap.add_argument("--win", type=int, default=10, help="sliding window width (samples)")
    ap.add_argument("--step", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("=" * 104)
    print("Time-resolved split-half reliability (CLIP-free)")
    print(f"  reps split 5/5, sliding window {args.win} samples ({args.win/250*1000:.0f} ms), "
          f"step {args.step}  |  onset at sample {ONSET}")
    print("=" * 104)

    out = {"subs": {}}
    for sub in args.subs:
        E = load_eeg(sub)
        n, T, ch, nt = E.shape
        h1, h2 = E[:, :T // 2], E[:, T // 2:]
        out["subs"][sub] = {"curves": []}
        print(f"\n--- {sub}  n={n} T={T} ch={ch} nt={nt} ---")
        print(f"  {'ctr_s':>7} {'reliab_raw':>11} {'spear_brown':>12} {'F_ratio':>9} "
              f"{'sig_var_frac':>13}")

        rows = []
        for lo in range(0, nt - args.win + 1, args.step):
            hi = lo + args.win
            a = h1[:, :, :, lo:hi].reshape(n, -1)
            b = h2[:, :, :, lo:hi].reshape(n, -1)
            r = float(spearmanr(upper_tri(rdm(a)), upper_tri(rdm(b))).correlation)
            sb = spearman_brown(r)

            # parametric equivalent: across-concept vs within-concept variance
            m = E[:, :, :, lo:hi].mean(1).reshape(n, -1)          # (n, d)
            grand = m.mean(0, keepdims=True)
            res = (E[:, :, :, lo:hi].reshape(n, T, -1)
                   - m[:, None, :]).reshape(-1, m.shape[1])
            var_between = float(((m - grand) ** 2).mean())
            var_within = float((res ** 2).mean()) / max(T, 1)
            F = var_between / max(var_within, 1e-30)
            sig = max(0.0, (var_between - var_within) / max(var_between, 1e-30))

            ctr = -0.2 + (lo + args.win / 2) / 250
            rows.append({"lo": lo, "ctr_s": ctr, "reliab_raw": r,
                         "spearman_brown": sb, "F_ratio": F, "sig_var_frac": sig})
            if lo % (args.step * 4) == 0 or lo < 60:
                print(f"  {ctr:+7.3f} {r:>11.4f} {sb:>12.4f} {F:>9.3f} {sig:>13.4f}")

        out["subs"][sub]["curves"] = rows

        pre = [x for x in rows if x["ctr_s"] < 0.0]
        post = [x for x in rows if x["ctr_s"] >= 0.0]
        pre_r = float(np.mean([x["spearman_brown"] for x in pre])) if pre else float("nan")
        post_r = float(np.mean([x["spearman_brown"] for x in post])) if post else float("nan")
        peak = max(rows, key=lambda x: x["spearman_brown"])
        out["subs"][sub]["pre_mean_sb"] = pre_r
        out["subs"][sub]["post_mean_sb"] = post_r
        out["subs"][sub]["peak"] = peak

        print(f"\n  mean split-half reliability:")
        print(f"    PRE  (t<0)  = {pre_r:+.4f}")
        print(f"    POST (t>=0) = {post_r:+.4f}")
        print(f"    ratio POST/PRE = {post_r/max(pre_r,1e-9):+.2f}" if pre_r > 0 else
              f"    PRE <= 0: reported as {pre_r:+.4f}")
        print(f"    peak at t={peak['ctr_s']:+.3f}s  sb={peak['spearman_brown']:.4f}  "
              f"F={peak['F_ratio']:.2f}")

    # --------------------------- verdict ---------------------------------
    pre_all = [out["subs"][s]["pre_mean_sb"] for s in out["subs"]]
    post_all = [out["subs"][s]["post_mean_sb"] for s in out["subs"]]
    pm, qm = float(np.nanmean(pre_all)), float(np.nanmean(post_all))
    out["verdict"] = {
        "pre_mean_sb": pm, "post_mean_sb": qm,
        "pre_is_reliable": bool(pm > 0.05),
        "interpretation": (
            "PRE-WINDOW IS RELIABLE: the data genuinely carries reproducible "
            "concept-specific structure before onset. Most likely cause is "
            "temporal leakage in preprocessing (zero-phase filtering and/or "
            "epoch extraction), which makes -0.2..0 s NOT a valid pre-stimulus "
            "control. The shuffled-EEG control must become the primary negative "
            "control, and the A1 gate must be revised accordingly."
            if pm > 0.05 else
            "PRE-WINDOW IS UNRELIABLE: no reproducible concept-specific structure "
            "before onset, yet CCA reports rho~0.27 there. That is a CCA-specific "
            "artefact, meaning the screening does NOT remove all overfitting at "
            "this (n, D, K) and the estimator needs a stricter control."
        ),
    }

    with open(os.path.join(args.out, "timereliab.json"), "w") as f:
        json.dump(out, f, indent=2)

    v = out["verdict"]
    print("\n" + "=" * 104)
    print(f"VERDICT  pre={v['pre_mean_sb']:+.4f}  post={v['post_mean_sb']:+.4f}")
    print(f"  pre-window reliable? {v['pre_is_reliable']}")
    for line in v["interpretation"].replace(". ", ".\n  ").split("\n"):
        print(f"  {line}")
    print(f"\n  wrote {os.path.join(args.out, 'timereliab.json')}")
    print("=" * 104)
    return 0


if __name__ == "__main__":
    sys.exit(main())
