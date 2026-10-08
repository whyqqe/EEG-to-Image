#!/usr/bin/env python
"""probe_rep_dose -- is the +4.0pp from the repetition cloud real, and is this its curve?

WHAT TRIGGERED THIS. `probe_c2_levers.py` found that neither repetition operator beats the
banked T1 rung on its own (rep-whiten+CSLS 29.50, rep-mean-scores 25.00, vs T1 best 35.50),
but FUSING the rep-cloud estimate with the T1 best gives **39.50 / 73.00** -- +4.0pp Top-1
and +7.0pp Top-5 over 35.50, i.e. the conservative end of the pre-registered M4 band.

A single fusion win is not evidence. Fusion of two score matrices can gain for boring
reasons -- a normalisation quirk, or simply averaging two noisy copies of the SAME estimate
-- and the mechanism claimed here is specific and falsifiable:

    the repetition cloud (C*R rows) estimates the query-set mean and covariance from R times
    as many samples as the 200-row averaged query does, so its whitening/recovery is a
    second, LESS VARIANT estimate of the same operator. Fusing the two cancels estimation
    variance, and the size of the cancellation must grow with R.

So the decisive readout is the DOSE-RESPONSE in R, not the headline number. Three controls
run beside it, each of which can kill the claim on its own:

  C1  `fuse(T1best, random scores)`            -- if this also gains, `fuse_scores` is doing
                                                  something generic and the whole row is void.
  C2  `fuse(T1best, T1best again)`             -- the same-query control. This MUST come back
                                                  at 35.50; it is the null for "averaging two
                                                  estimates of one operator".
  C3  `fuse(T1best, T1-without-recovery)`      -- is the gain 'recovery twice', or does it need
                                                  the repetition statistics?

And the same R used twice must not be compared to itself: each R is drawn from several
random repetition subsets, so the curve carries a spread and not just a point. A curve that
rises in R while C1/C2 stay flat is the mechanism; a curve that is flat in R is an artefact
of the fusion normalisation and must be reported as such.

Run (seconds, CPU, on cached features from probe_c2_levers.py):
    python scripts/probe_rep_dose.py --cache outputs/probe/c2_feats_v5a1k20.npz \
        --out outputs/probe/rep_dose_v5a1k20.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from samclip import calibration  # noqa: E402


def rep_estimate(z_reps: np.ndarray, g: np.ndarray, k: int, rho: float) -> np.ndarray:
    """The T2 operator, now `calibration.rep_cloud_scores` -- the single implementation.

    It used to be a copy of the code inlined in `probe_c2_levers`, annotated "verbatim
    from `probe_c2_levers`". Kept as a one-line wrapper because the dose sweep needs a
    function of the rep cloud that hands `rep_cloud_scores` a subset.
    """
    scores, _ = calibration.rep_cloud_scores(z_reps, g, k=k, rho=rho)
    return scores


def subset(z_reps: np.ndarray, r: int, rng: np.random.Generator) -> np.ndarray:
    """`r` repetitions out of R, chosen at random per concept.

    Randomly chosen rather than the first `r` on purpose: the repetitions are ordered by
    acquisition, and taking a prefix would confound "number of repetitions" with "position
    in the block", which is exactly the kind of nuisance this project has been bitten by."""
    C, R, d = z_reps.shape
    if r >= R:
        return z_reps
    idx = np.stack([rng.choice(R, size=r, replace=False) for _ in range(C)])
    return np.take_along_axis(z_reps, idx[:, :, None], axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--csls-k", type=int, default=10)
    ap.add_argument("--rho", type=float, default=0.1)
    ap.add_argument("--rs", type=int, nargs="+", default=[1, 2, 5, 10, 20, 40, 80])
    ap.add_argument("--draws", type=int, default=5,
                    help="random repetition subsets per R (R=full uses one draw)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    c = np.load(args.cache)
    q, g, z_reps = c["q"], c["g"], c["z_reps"]
    C, R, d = z_reps.shape
    rng = np.random.default_rng(args.seed)
    print(f"[dose] C={C} R={R} d={d}  gallery={g.shape}")

    def fused(t2: np.ndarray) -> dict:
        t1, _ = calibration.calibrate(q, g, k=args.csls_k, rho=args.rho,
                                      whiten=True, csls=True, recovery=True)
        return calibration.report_with_scores(calibration.fuse_scores([t1, t2]))

    # ---- controls -------------------------------------------------------------------
    t1_best, _ = calibration.calibrate(q, g, k=args.csls_k, rho=args.rho,
                                       whiten=True, csls=True, recovery=True)
    controls = {"T1 best (no fusion)": calibration.report_with_scores(t1_best)}
    controls["C2 fuse(T1best, T1best)"] = fused(t1_best)
    t1_no_rec, _ = calibration.calibrate(q, g, k=args.csls_k, rho=args.rho,
                                         whiten=True, csls=True)
    controls["C3 fuse(T1best, T1 w/o recovery)"] = fused(t1_no_rec)
    rnd = rng.standard_normal(t1_best.shape)
    controls["C1 fuse(T1best, random)"] = fused(rnd)
    rnd2 = rng.standard_normal(t1_best.shape)
    controls["C1b fuse(T1best, random2)"] = fused(rnd2)

    print(f"\n{'control':<40} {'Top-1':>7} {'Top-5':>7}")
    print("-" * 56)
    for n, m in controls.items():
        print(f"{n:<40} {m['top1']:>7.2f} {m['top5']:>7.2f}")

    # ---- dose-response in R ----------------------------------------------------------
    curve: dict = {}
    print(f"\n{'R':>4} {'draws':>6} {'T2 alone':>10} {'fused Top-1':>13} {'fused Top-5':>13}")
    print("-" * 56)
    for r in args.rs:
        if r > R:
            continue
        n_draw = 1 if r >= R else args.draws
        alone, t1s, t5s = [], [], []
        for _ in range(n_draw):
            zz = subset(z_reps, r, rng)
            t2 = rep_estimate(zz, g, args.csls_k, args.rho)
            alone.append(calibration.report_with_scores(t2)["top1"])
            f = fused(t2)
            t1s.append(f["top1"])
            t5s.append(f["top5"])
        curve[str(r)] = {"n_draws": n_draw,
                         "t2_alone_top1": [round(float(x), 3) for x in alone],
                         "fused_top1_mean": float(np.mean(t1s)),
                         "fused_top1_std": float(np.std(t1s)),
                         "fused_top5_mean": float(np.mean(t5s))}
        print(f"{r:>4} {n_draw:>6} {np.mean(alone):>10.2f} "
              f"{np.mean(t1s):>10.2f}±{np.std(t1s):<4.2f} {np.mean(t5s):>13.2f}")

    verdict = ("DOSE-RESPONSE PRESENT: the fused gain grows with R -> the repetition cloud is "
               "buying estimation variance, not a normalisation artefact"
               if curve.get(str(R), {}).get("fused_top1_mean", 0)
               > curve.get("1", {}).get("fused_top1_mean", 1e9) + 1.0
               else "NO DOSE-RESPONSE: flat in R -> do not claim a repetition mechanism")
    print(f"\n{verdict}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"cache": args.cache, "controls": controls, "dose_curve": curve,
             "verdict": verdict}, indent=2, default=str))
        print(f"[dose] wrote {args.out}")


if __name__ == "__main__":
    main()
