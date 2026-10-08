#!/usr/bin/env python
"""probe_fgw -- does the CONCEPT METRIC improve the coupling? (test of the unified theory)

WHERE THIS COMES FROM. Two honest measurements, both on frozen banked features:

  * `analyze_rigidity.py` §A, under 5-fold CV, shows the RIGID model class is NOT the
    bottleneck: orthogonal residual 0.885 vs unconstrained linear 0.922 -- **relaxing
    orthogonality makes it worse**, and a capacity-capped smooth fit only recovers ~12% (0.777).
    The first rigidity probe's "90x gap" was pure overfitting (1088 basis functions for 200
    concepts) and is not evidence. So the fix is NOT a more expressive map.
  * `analyze_rigidity.py` §B/C shows what IS true: alignment residual predicts subject accuracy
    at `r = -0.74` (orth) / `-0.77` (linear), and the concept metric is substantially
    SUBJECT-INVARIANT -- `corr(D_eeg_s, D_img) = +0.615`, `corr(D_eeg_s, D_eeg_t) = +0.565`
    over 45 pairs, against a chance baseline of -0.003.

Put together: the bottleneck is CORRESPONDENCE QUALITY, not model class -- which is the same
conclusion the S3R result reached independently (42 hard landmarks -> 465 soft, +2.80pp, with
the model class held fixed). And there exists a subject-invariant signal (the concept metric)
that the current coupling does not use at all.

WHY THE CURRENT COUPLING IGNORES IT. `subspace_soft_recovery` builds its cost purely from
CROSS-domain similarity (`csls_scores(z_e, z_i)`). But a coupling π is a correspondence, and a
correspondence should preserve INTRA-domain structure: if concepts i,k are close in subject s's
EEG geometry, their matched images j,l should be close in the concept geometry. That statement
uses no labels and no cross-domain similarity -- it is Fused Gromov-Wasserstein's structural
term:

    FGW(pi) = (1-a)*<pi, C>  +  a*sum_ijkl pi_ij pi_kl (D_eeg_ik - D_img_jl)^2

FGW is the UNIFICATION the project needs, because every existing mechanism is a term of it:
  * `L_mmd` / OT alone   -> 0th order: matches marginals, no correspondence
  * InfoNCE / CSLS       -> 1st order: the anchored term `<pi, C>`, cross-modal supervision
  * Procrustes/recovery  -> a rigid post-hoc fit through the coupling
  * the GW term          -> 2nd order: the manifold structure, which nothing currently uses
and the entropic solver is the SAME Sinkhorn iteration already implemented and property-tested.

WHAT THIS SCRIPT DECIDES. Whether the structural term actually improves the coupling, measured
by PLAN ACCURACY (`trace(pi)`) and top-1 retrieval, as a function of `a`. `a = 0` must reproduce
the S3R coupling exactly, so the comparison is against a verified baseline rather than a
reimplementation. This is the same deployment-only, no-training evaluation that produced the
+2.80pp operator result, and it can fail: at `a = 1` the coupling is pure GW and, if the concept
metric were subject-specific, accuracy would collapse.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FEAT = "outputs/probe/rigid_feats_sub{:02d}_seed{}.npz"


def sinkhorn_cost(cost: np.ndarray, tau: float, iters: int = 60, eps: float = 1e-9):
    """Entropic OT plan for a COST matrix. Row-stabilised, unit row mass -- same discipline
    (and the same two bug fixes) as `calibration._sinkhorn_plan`, so the two solvers agree."""
    k = np.exp(-(cost - cost.min(axis=1, keepdims=True)) / max(tau, eps))
    u = np.ones(k.shape[0]); v = np.ones(k.shape[1])
    for _ in range(iters):
        v = 1.0 / (k.T @ u + eps)
        u = 1.0 / (k @ v + eps)
    p = (u[:, None] * k) * v[None, :]
    return p / max(p.sum(), eps)


def std(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / max(x.std(), 1e-12)


def fgw_coupling(C, De, Di, alpha: float, tau: float,
                 outer: int = 25, inner: int = 60, eps: float = 1e-9):
    """Entropic Fused Gromov-Wasserstein by proximal/conditional gradient.

    The GW gradient factorises, which is why this is cheap:
        grad_ij = sum_kl pi_kl (De_ik - Di_jl)^2
                = [De^2 @ rowsum(pi)]_ij + [Di^2 @ colsum(pi)^T]_ij - 2 [De @ pi @ Di^T]_ij
    (`De`/`Di` here are the squared distance matrices when the L2 ground metric is used, so this
    applies the 2 factor once.) Both cost blocks are standardised to unit scale first: `C` is a
    CSLS-corrected cosine and the GW gradient is a squared-distance sum, and without a common
    scale `alpha` would be measuring the units rather than the trade-off.
    """
    n, m = C.shape
    Cs = std(C)
    pi = sinkhorn_cost((1.0 - alpha) * Cs, tau=tau)
    for _ in range(outer):
        r = pi.sum(1, keepdims=True)
        c = pi.sum(0, keepdims=True)
        grad = De @ r + (Di @ c.T) - 2.0 * (De @ pi @ Di.T)
        Ceff = (1.0 - alpha) * Cs + alpha * std(grad)
        pi = sinkhorn_cost(Ceff, tau=tau, iters=inner)
    return pi


def _pca_rank(X: np.ndarray, rank: int) -> np.ndarray:
    """Project onto the top-`rank` principal directions.

    THIS IS THE MANIFOLD-AWARE PART. The concept manifold measures at ~8-16 intrinsic
    dimensions inside a 64-dimensional ambient space (`analyze_rigidity.py`, participation
    ratio), so ~48-56 of the 64 directions are noise as far as the geometry is concerned. A
    Euclidean distance computed in the full ambient space is therefore dominated by the noise
    directions, which makes the structural cost `(D_eeg - D_img)^2` a noisy estimate of the
    manifold metric the theory actually wants. Truncating to the signal subspace before taking
    distances is the cheapest de-noising that uses the measured manifold dimension, and it
    deliberately does NOT discard coordinates from the coupling itself -- only from the
    distance estimate -- so this cannot be the "we threw away signal" failure the low-rank
    concept-frame family (G-a) was falsified for.
    """
    if rank <= 0 or rank >= min(X.shape):
        return X
    Xc = X - X.mean(0, keepdims=True)
    u, s, vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ vt[:rank].T


def _geo(X: np.ndarray, rank: int = 0) -> np.ndarray:
    """Standardised squared-distance matrix, optionally in the truncated signal subspace."""
    Y = _pca_rank(X, rank)
    Y = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-9)
    return std((Y @ Y.T - 1.0) ** 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, nargs="*", default=[4, 7, 8, 1, 10])
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--alphas", type=float, nargs="*",
                    default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--geo-rank", type=int, default=0,
                    help="truncate the distance estimate to this many principal directions "
                         "(0 = full ambient space; the measured manifold is ~8-16)")
    ap.add_argument("--out", default="outputs/probe/fgw.json")
    args = ap.parse_args()

    out = {"tau": args.tau, "per_subject": {}}
    print(f"FGW coupling test  (tau={args.tau}, no training, frozen banked features)")
    print("=" * 84)
    print(f"{'subject':<9}{'alpha':>7}{'plan_acc':>10}{'top1':>8}{'top5':>8}   note")
    print("-" * 84)
    for s in args.subjects:
        f = Path(FEAT.format(s, args.seed))
        if not f.is_file():
            print(f"sub{s:02d}: missing {f}")
            continue
        z = np.load(f)
        ze = z["z_e"].astype(np.float64)
        zi = z["z_i"].astype(np.float64)
        nrm = lambda x: x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)
        en, inn = nrm(ze), nrm(zi)
        sim = en @ inn.T
        # CSLS correction, matching the deployed ladder's hubness handling
        k = 10
        r_g = np.sort(sim, axis=1)[:, -k:].mean(1, keepdims=True)
        r_q = np.sort(sim, axis=0)[-k:, :].mean(0, keepdims=True)
        C = -(2.0 * sim - r_g - r_q)
        # intra-domain squared distances (the concept metric), optionally de-noised by the
        # measured manifold dimension
        De = _geo(en, args.geo_rank)
        Di = _geo(inn, args.geo_rank)
        rec = {"alpha": {}}
        best = None
        for a in args.alphas:
            pi = fgw_coupling(C, De, Di, a, tau=args.tau)
            acc = float(np.trace(pi))
            t1 = float(np.mean(pi.argmax(1) == np.arange(pi.shape[0])) * 100)
            order = np.argsort(-pi, axis=1)
            t5 = float(np.mean([np.arange(pi.shape[0])[i] in order[i, :5]
                                for i in range(pi.shape[0])]) * 100)
            rec["alpha"][f"{a:g}"] = {"plan_acc": acc, "top1": t1, "top5": t5}
            tag = ""
            if abs(a) < 1e-12:
                tag = "<- current S3R coupling (baseline)"
            if best is None or t1 > best[1]:
                best = (a, t1)
                tag += "  <= best here" if a != 0 else ""
            print(f"sub{s:02d}{a:>10.2f}{acc:>10.4f}{t1:>8.2f}{t5:>8.2f}   {tag}")
        rec["best_alpha"] = best[0]
        base = rec["alpha"]["0"]["top1"]
        rec["gain_at_best"] = best[1] - base
        out["per_subject"][f"sub{s:02d}"] = rec
        print(f"{'':<9}{'':>7}best alpha={best[0]:g}  gain vs baseline {best[1] - base:+.2f} pp "
              f"(baseline {base:.2f})")
        print("-" * 84)

    # aggregate over the alphas that all subjects share
    print("\nAGGREGATE (mean over subjects)")
    print(f"{'alpha':>8}{'plan_acc':>11}{'top1':>9}{'delta vs a=0':>15}")
    for a in args.alphas:
        key = f"{a:g}"
        rows = [v["alpha"][key] for v in out["per_subject"].values() if key in v["alpha"]]
        if not rows:
            continue
        base = np.mean([v["alpha"]["0"]["top1"] for v in out["per_subject"].values()])
        m = np.mean([r["top1"] for r in rows])
        print(f"{a:>8.2f}{np.mean([r['plan_acc'] for r in rows]):>11.4f}{m:>9.2f}"
              f"{m - base:>+15.2f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
