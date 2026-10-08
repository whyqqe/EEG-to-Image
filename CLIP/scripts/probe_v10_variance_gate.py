#!/usr/bin/env python
"""probe_v10_variance_gate -- is the FGW structural gain GATED BY REPETITION COUNT?

WHY THIS AND NOT A NEW METHOD. The deployed v10 headline is the T2 repetition-cloud row with
the FGW structural term: +3.58pp over the same row without it, 10/10 subjects. That number
was measured at R=80 repetitions per concept. SCORE -- the cross-subject SOTA we compare
against -- does not use a repetition cloud at all. So before any of this is claimed as a win,
one question has to be answered and it is not a matter of opinion:

    is the +3.58pp bought by the STRUCTURAL TERM, or by the fact that we hand the operator
    R times more test information than the baseline sees?

Those two are separable. If the gain is structural, it must survive as R shrinks (the term
sees a good enough estimate of the target geometry at modest R) -- and the curve in R is the
pre-registered readout, decided before the runs, not after:

  * gain FALLS with R  -> the gain is riding on target-side estimation variance. Reportable,
    but it must be reported WITH the rep count and the baseline must be given the same
    information before the comparison is called fair.
  * gain FLAT in R     -> the gain is not test-information-dependent, the comparison to a
    repetition-free baseline is fair, and the structural term is doing the work.

A flat baseline row (`alpha=0`, same operator, same R) runs beside every point so that a
change in the CURVE cannot be confused with a change in the LEVEL.

COST. Pure CPU, seconds per cache: the caches already hold the encoded repetition clouds
(`outputs/probe/whiten_feats_sub*_k20_seed*.npz`, 30 of them = 10 subjects x 3 seeds), so
nothing here touches a GPU or the raw EEG. That is deliberate -- this is a PREREQUISITE
check, and a prerequisite that needs a queue is a prerequisite that gets skipped.

FIDELITY GATE. `--expect` re-derives one known cell (sub01/seed2025/R=80/alpha=0.75) and
aborts if it does not match the banked eval JSON, because a probe that silently computes a
different operator than the deployed one is worse than no probe.

Usage:
    python scripts/probe_v10_variance_gate.py --out outputs/probe/v10_variance_gate.json
"""
from __future__ import annotations

import argparse
import glob
import re
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from samclip import calibration  # noqa: E402


def make_recovery(alpha: float, tau: float, iters: int):
    """Exactly the wrapper `run_eval._recovery_fn_for` builds for `--recovery-operator fgw`.

    Copied on purpose rather than imported: importing `run_eval` would execute its module
    level and pull in torch, and this probe must stay runnable on a CPU node with no GPU.
    The fidelity gate below is what keeps the copy honest.
    """
    def fn(q, g, k=10, rho=0.1, min_landmark_rate=0.0, **kw):
        return calibration.subspace_soft_recovery(
            q, g, k=k, rho=rho, rank=None, tau=tau, iters=iters,
            hard_landmarks=False, min_landmarks=8,
            alpha=float(kw.pop("alpha", alpha)),
            fgw_outer=int(kw.pop("fgw_outer", 10)))
    fn.__name__ = f"fgw_recovery(tau={tau:g},alpha={alpha:g})"
    return fn


def subset(z_reps: np.ndarray, r: int, rng: np.random.Generator, mode: str) -> np.ndarray:
    """`r` of R repetitions per concept. `random` avoids confounding R with block position."""
    C, R, d = z_reps.shape
    if r >= R:
        return z_reps
    if mode == "prefix":
        return z_reps[:, :r]
    idx = np.stack([rng.choice(R, size=r, replace=False) for _ in range(C)])
    return np.take_along_axis(z_reps, idx[:, :, None], axis=1)


def top1(scores: np.ndarray) -> float:
    return 100.0 * float(np.mean(np.argmax(scores, axis=1) == np.arange(scores.shape[0])))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="outputs/probe/whiten_feats_sub*_k20_seed*.npz")
    ap.add_argument("--alpha", type=float, default=0.75)
    ap.add_argument("--tau", type=float, default=0.01)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--csls-k", type=int, default=10)
    ap.add_argument("--rho", type=float, default=0.1)
    ap.add_argument("--rs", type=int, nargs="+", default=[80, 40, 20, 10, 5, 1])
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--mode", choices=["random", "prefix"], default="random")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--expect", type=float, default=59.5,
                    help="banked Top-1 for sub01/seed2025 at R=full, alpha>0 (fidelity gate)")
    ap.add_argument("--expect-tol", type=float, default=0.51)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    paths = sorted(glob.glob(str(ROOT / args.glob)))
    if not paths:
        raise SystemExit(f"no caches matched {args.glob}")

    # STALENESS GUARD. The caches are keyed by checkpoint *directory name* only
    # (`whiten_feats_sub01_k20_seed2025.npz` <- any arm's `sub01_k20_seed2025/`), so a cache
    # built from an older arm is silently a different model. That is not hypothetical: the
    # caches under `outputs/probe/` were written at 18:34 while the v8 checkpoints they appear
    # to name were written at 23:24 the same day, and this probe reported 54.50 where the
    # deployed operator gives 59.50. A probe that quietly measures another model is worse than
    # no probe, so it refuses to run against a cache older than the checkpoint it claims.
    stale = []
    for p in paths:
        m = re.search(r"whiten_feats_(sub\d+_k20_seed\d+)\.npz$", str(p))
        if not m:
            continue
        name = m.group(1)
        cks = list((ROOT / "outputs/stage1/v8").glob(f"{name}/last.pt"))
        if cks and Path(p).stat().st_mtime < cks[0].stat().st_mtime:
            stale.append((Path(p).name, cks[0].parent.name))
    if stale:
        raise SystemExit(
            "[gate] STALE CACHES: " + ", ".join(f"{c} < {k}" for c, k in stale[:5])
            + f" ({len(stale)} total). These were built from an older arm than the v8 "
              "checkpoints they are named after, so they are a DIFFERENT model. Rebuild them "
              "or run the GPU sweep (slurm/v10_sweep.sbatch), which embeds fresh.")

    print(f"[gate] {len(paths)} caches, alpha={args.alpha} tau={args.tau} mode={args.mode} "
          f"rs={args.rs} draws={args.draws}")

    rec_hi = make_recovery(args.alpha, args.tau, args.iters)
    rec_lo = make_recovery(0.0, args.tau, args.iters)

    # ---- fidelity gate: one known cell, before any curve is trusted ------------------
    if args.expect is not None:
        c = np.load(paths[0])
        s_hi, _ = calibration.rep_cloud_scores(c["z_reps"], c["g"], k=args.csls_k, rho=args.rho,
                                               recovery_fn=rec_hi)
        got = top1(s_hi)
        ok = abs(got - args.expect) <= args.expect_tol
        print(f"[gate] {Path(paths[0]).name} R=full alpha={args.alpha}: got {got:.2f}, "
              f"banked {args.expect:.2f} -> {'OK' if ok else 'MISMATCH'}")
        if not ok:
            raise SystemExit("[gate] FIDELITY FAILED -- the probe is not the deployed operator. "
                             "Fix this before reading the curve; a wrong curve is worse than none.")

    # ---- the curve -------------------------------------------------------------------
    # per-run paired differences, so the headline is a paired statistic and not a difference
    # of two independently-noisy means.
    per_run: list[dict] = []
    for p in paths:
        c = np.load(p)
        z_reps, g = c["z_reps"], c["g"]
        C, R, _ = z_reps.shape
        rng = np.random.default_rng(args.seed)
        rec: dict = {"cache": Path(p).name, "curve": {}}
        for r in args.rs:
            if r > R:
                continue
            n_draw = 1 if r >= R else args.draws
            hi, lo = [], []
            for _ in range(n_draw):
                zz = subset(z_reps, r, rng, args.mode)
                s_hi, _ = calibration.rep_cloud_scores(zz, g, k=args.csls_k, rho=args.rho,
                                                       recovery_fn=rec_hi)
                s_lo, _ = calibration.rep_cloud_scores(zz, g, k=args.csls_k, rho=args.rho,
                                                       recovery_fn=rec_lo)
                hi.append(top1(s_hi))
                lo.append(top1(s_lo))
            rec["curve"][str(r)] = {"hi": float(np.mean(hi)), "lo": float(np.mean(lo)),
                                    "gain": float(np.mean(hi) - np.mean(lo)),
                                    "n_draws": n_draw}
        per_run.append(rec)

    print(f"\n{'R':>5} {'alpha>0':>9} {'alpha=0':>9} {'gain':>8} {'n_pos':>7}")
    print("-" * 42)
    agg: dict = {}
    for r in args.rs:
        g_hi = np.array([x["curve"][str(r)]["hi"] for x in per_run if str(r) in x["curve"]])
        g_lo = np.array([x["curve"][str(r)]["lo"] for x in per_run if str(r) in x["curve"]])
        if not len(g_hi):
            continue
        gain = g_hi - g_lo
        t = (gain.mean() / (gain.std(ddof=1) / np.sqrt(len(gain)))
             if gain.std(ddof=1) > 0 else float("nan"))
        agg[str(r)] = {"top1_hi_mean": float(g_hi.mean()), "top1_hi_sd": float(g_hi.std(ddof=1)),
                       "top1_lo_mean": float(g_lo.mean()), "top1_lo_sd": float(g_lo.std(ddof=1)),
                       "gain_mean": float(gain.mean()), "gain_sd": float(gain.std(ddof=1)),
                       "gain_t": float(t), "n_pos": int((gain > 0).sum()), "n": len(gain)}
        print(f"{r:>5} {g_hi.mean():>9.2f} {g_lo.mean():>9.2f} {gain.mean():>+8.2f} "
              f"{int((gain > 0).sum()):>3}/{len(gain)}  (t={t:.2f})")

    # ---- pre-registered verdict ------------------------------------------------------
    if len(agg) >= 2:
        rs = sorted((int(k) for k in agg), reverse=True)
        g_full, g_min = agg[str(rs[0])]["gain_mean"], agg[str(rs[-1])]["gain_mean"]
        drop = g_full - g_min
        if drop >= 2.0:
            verdict = (f"GATED: gain falls {g_full:+.2f} -> {g_min:+.2f} as R goes "
                       f"{rs[0]} -> {rs[-1]}. The structural term rides on target-side "
                       f"estimation variance; the rep count must accompany any claim and the "
                       f"baseline must be given equal test information before the comparison "
                       f"to SCORE is called fair.")
        elif abs(drop) < 1.0:
            verdict = (f"FLAT: gain stays {g_full:+.2f} -> {g_min:+.2f} as R goes "
                       f"{rs[0]} -> {rs[-1]}. The gain is NOT bought with extra test "
                       f"information, so the comparison against a repetition-free baseline "
                       f"is fair and the structural term is doing the work.")
        else:
            verdict = (f"INTERMEDIATE: gain moves {g_full:+.2f} -> {g_min:+.2f} as R goes "
                       f"{rs[0]} -> {rs[-1]}. Partial dependence; report the curve, not a "
                       f"single number.")
    else:
        verdict = "INSUFFICIENT R POINTS"
    print(f"\n{verdict}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"alpha": args.alpha, "tau": args.tau, "mode": args.mode, "rs": args.rs,
             "draws": args.draws, "n_caches": len(paths), "aggregate": agg,
             "per_run": per_run, "verdict": verdict}, indent=2))
        print(f"[gate] wrote {args.out}")


if __name__ == "__main__":
    main()
