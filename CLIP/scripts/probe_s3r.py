#!/usr/bin/env python
"""probe_s3r -- does the recovery operator's FLAT +3.62 come from the estimator?

WHY THIS PROBE EXISTS, AND WHY IT IS FREE. The 30-run G3 grid measures the recovery rung as
`+3.62 +- 1.72` Top-1, and that gain is FLAT: over 30 runs `corr(raw, gain) = -0.19` and
`corr(landmark_rate, gain) = -0.21`. A gain that does not respond to encoder quality or to
landmark rate is an estimator defect, not a representation defect -- and that is why every
training-side attempt to raise the landmark rate (T2', T2'', SCORE's source-only episode)
failed to move the score. They were pushing a quantity this operator does not consume.

So the question is well-posed and it is answered on EXISTING checkpoints: no training, one
CPU pass over the 30 G3 runs. The features are cached by `probe_whiten_variants.fold_features`
under `outputs/probe/whiten_feats_*.npz`, so a second question about the same fold costs
seconds.

WHAT IS MEASURED. A 2x2 over the operator's two diagnosed defects (see
`calibration.subspace_soft_recovery`), so each half is ATTRIBUTED rather than assumed:

    variant            matching   fitting subspace
    deployed           hard       full d=64        <- must reproduce +3.62
    soft_only          soft       full d=64        <- defect 1: ~42 landmarks, use all 200
    subspace_only      hard       top-16           <- defect 2: 2016 params, ~120 matter
    s3r                soft       top-16           <- both

Each variant is reported at BOTH rows the operator feeds -- `+ CSLS + recovery` (the T1 rung)
and `+ T1(CSLS + recovery) + T2 reps` (the headline, because `rep_cloud_scores` calls the same
operator) -- plus `+ T2 reps` on its own. The headline is the number that matters: G3's is
45.53 and SCORE's is 53.23.

Run (CPU, minutes):
    python scripts/probe_s3r.py --subjects 1 2 3 4 5 6 7 8 9 10 --seeds 2025 2026 2027 \
        --out outputs/probe/s3r.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from samclip import calibration  # noqa: E402
import probe_whiten_variants as pwv  # noqa: E402


def _variant(tag: str, rank: int):
    """`(name, kwargs)` for `calibration.subspace_soft_recovery`, or None for the deployed op."""
    common = dict(rank=rank)
    table = {
        "deployed": None,
        "soft_only": dict(hard_landmarks=False, rank=None),
        "subspace_only": dict(hard_landmarks=True, rank=rank),
        "s3r": dict(hard_landmarks=False, rank=rank),
    }
    if tag not in table:
        raise SystemExit(f"unknown variant {tag!r}; have {sorted(table)}")
    spec = table[tag]
    return None if spec is None else {**common, **spec}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, nargs="*", default=list(range(1, 11)))
    ap.add_argument("--seeds", type=int, nargs="*", default=[2025, 2026, 2027])
    ap.add_argument("--stage1-root", default="outputs/stage1/g3")
    ap.add_argument("--rank", type=int, default=16,
                    help="signal-subspace dimension; 16 is the project's own measurement of "
                         "the concept manifold, so it is the default rather than a tuned value")
    ap.add_argument("--tau", type=float, default=0.05,
                    help="Sinkhorn temperature; swept by --tau-grid instead of assumed")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--rho", type=float, default=0.1)
    ap.add_argument("--csls-k", type=int, default=10)
    ap.add_argument("--mvnn", default=None)
    ap.add_argument("--variants", nargs="*",
                    default=["deployed", "soft_only", "subspace_only", "s3r"])
    ap.add_argument("--tau-grid", type=float, nargs="*", default=None,
                    help="if given, run `s3r` at each tau (subjects must be a small subset)")
    ap.add_argument("--out", default="outputs/probe/s3r.json")
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    HEAD = "+ T1(CSLS + recovery) + T2 reps"
    out: dict = {"rank": args.rank, "tau": args.tau, "rho": args.rho,
                 "csls_k": args.csls_k, "per_fold": {}}

    for s in args.subjects:
        for seed in args.seeds:
            ckpt = Path(args.stage1_root) / f"sub{s:02d}_k20_seed{seed}" / "last.pt"
            if not ckpt.is_file():
                print(f"[s3r] fold {s} seed {seed}: missing {ckpt}")
                continue
            z_reps, g = pwv.fold_features(ckpt, s, args, device)
            res = {}

            # --- reference rows on the DEPLOYED operator (must reproduce the banked grid)
            base_t1, _ = calibration.calibrate(
                z_reps.mean(axis=1), g, csls=True, recovery=True,
                k=args.csls_k, rho=args.rho)
            base_t2, _ = calibration.rep_cloud_scores(
                z_reps, g, k=args.csls_k, rho=args.rho)
            t1_fuse = calibration.fuse_scores([base_t1, base_t2])
            res["deployed"] = {
                "+ CSLS + recovery": calibration.report_with_scores(base_t1)["top1"],
                "+ T2 reps": calibration.report_with_scores(base_t2)["top1"],
                HEAD: calibration.report_with_scores(t1_fuse)["top1"],
            }

            tags = list(args.variants)
            if args.tau_grid:
                tags += [f"s3r_tau{t:g}" for t in args.tau_grid]
            for tag in tags:
                if tag == "deployed":
                    continue
                if tag.startswith("s3r_tau"):
                    kw = {"hard_landmarks": False, "rank": args.rank,
                          "tau": float(tag.split("tau")[1]), "iters": args.iters}
                else:
                    spec = _variant(tag, args.rank)
                    kw = {**spec, "tau": args.tau, "iters": args.iters}

                def rec(qq, gg, *, k, rho, min_landmark_rate=0.0, _kw=kw):
                    return calibration.subspace_soft_recovery(
                        qq, gg, k=k, rho=rho, **_kw)

                t1, _ = calibration.calibrate(
                    z_reps.mean(axis=1), g, csls=True, recovery=True,
                    k=args.csls_k, rho=args.rho, recovery_fn=rec)
                t2, _ = calibration.rep_cloud_scores(
                    z_reps, g, k=args.csls_k, rho=args.rho, recovery_fn=rec)
                fuse = calibration.fuse_scores([t1, t2])
                res[tag] = {
                    "+ CSLS + recovery": calibration.report_with_scores(t1)["top1"],
                    "+ T2 reps": calibration.report_with_scores(t2)["top1"],
                    HEAD: calibration.report_with_scores(fuse)["top1"],
                }
            out["per_fold"][f"{s}_{seed}"] = res
            line = "  ".join(f"{t}={res[t][HEAD]:5.1f}" for t in res)
            print(f"[s3r] sub{s:02d} seed{seed}  headline row: {line}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))

    # ------------------------------------------------------------------ summary
    import statistics as st
    runs = list(out["per_fold"].values())
    print("\n" + "=" * 78)
    print(f"{'variant':<20} {'+CSLS+rec':>10} {'+T2 reps':>10} {HEAD:>26}")
    print("-" * 78)
    tags = list(runs[0].keys())
    for tag in tags:
        a = st.mean([r[tag]["+ CSLS + recovery"] for r in runs])
        b = st.mean([r[tag]["+ T2 reps"] for r in runs])
        c = st.mean([r[tag][HEAD] for r in runs])
        print(f"{tag:<20} {a:>10.2f} {b:>10.2f} {c:>26.2f}")
    print("-" * 78)
    dep = st.mean([r["deployed"][HEAD] for r in runs])
    for tag in tags:
        if tag == "deployed":
            continue
        d = st.mean([r[tag][HEAD] for r in runs]) - dep
        print(f"  {tag:<18} headline delta vs deployed: {d:+.2f} pp")
    print(f"\nn = {len(runs)} runs (fold x seed).  G3 banked headline (10-fold x 3-seed "
          f"mean): 45.53.  SCORE: 53.23.")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
