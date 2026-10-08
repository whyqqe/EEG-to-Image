#!/usr/bin/env python
"""Signal-detection analysis: is the concept information PRESENT but mis-weighted, or absent?

The design question this decides
--------------------------------
`saw_whiten` is one member of a family: reweight the query's principal directions by some
`w_j`, then retrieve by cosine. Whitening uses `w_j = 1/sqrt(lambda_j)`, i.e. it assumes
the query's variance ordering is the nuisance ordering. The subspace sweep showed the
query's top-16 directions also carry 94% of the IMAGE cloud's variance, which is evidence
AGAINST that assumption at the same time as the +5pp whitening gain is evidence FOR it.
Aggregate statistics cannot resolve that contradiction, and the redesign depends on which
side is right:

  * If the information is PRESENT but mis-weighted, the fix is a better metric -- a
    learned/constrained reweighting, ideally trained into the encoder. The architecture
    stays and the objective changes.
  * If the information is ABSENT from the representation, no metric can recover it and
    the encoder itself has to change.

So the family is bounded directly. For direction `j` let `rho_j = corr_i(q_i . u_j,
g_i . u_j)` -- how much the i-th query's coordinate along `u_j` tracks its OWN target's
coordinate along `u_j`, across the 200 concepts. Then:

  * `rho_j^2` is that direction's concept-explained variance fraction.
  * The signal-to-noise-optimal diagonal weight is `w_j = rho_j^2 / lambda_j` -- this is
    the standard linear-discriminant weighting, and whitening's `1/lambda_j` is the
    special case of it that assumes every direction has the SAME `rho^2`.
  * `sum_j lambda_j rho_j^2 / sum_j lambda_j` is the fraction of the query variance that
    is concept-covariant at all.

`rho_j` uses labels, so the resulting score is an ORACLE and not a deployable method. That
is the point: it is an upper bound for the whole diagonal-reweighting family, which
contains whitening, CSLS's local scaling at the diagonal level, and every covariance-based
subject adaptation. If the oracle is not materially above what whitening already gets,
then no improvement to the metric exists in this representation and the encoder must
change.

Run:  python scripts/probe_signal_weights.py --ckpts <c1> <c2> --target-subject 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from samclip import calibration  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_subspace_alignment import extract, rung  # noqa: E402


def top1_from_scores(scores: np.ndarray) -> float:
    return float((np.argmax(scores, axis=1) == np.arange(scores.shape[0])).mean() * 100.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--target-subject", type=int, required=True)
    ap.add_argument("--mvnn", default="test")
    ap.add_argument("--csls-k", type=int, default=10)
    args = ap.parse_args()

    for c in args.ckpts:
        p = Path(c)
        tag = f"{p.parent.parent.name}/{p.parent.name}"
        feats = extract(p, args.target_subject, args.mvnn)
        q, g = np.asarray(feats["eeg"], dtype=np.float64), np.asarray(feats["img"], dtype=np.float64)
        n = q.shape[0]
        idx = np.arange(n)

        qn = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8)
        gn = g / np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-8)

        # Principal directions of the QUERY cloud. Centering only defines the basis; the
        # projections below are on the raw vectors so the analysis stays about the
        # embedding and not about a mean offset.
        mu = qn.mean(axis=0, keepdims=True)
        qc = qn - mu
        vals, U = np.linalg.eigh(qc.T @ qc / max(n - 1, 1))
        order = np.argsort(-vals)
        vals, U = np.clip(vals[order], 0.0, None), U[:, order]
        a = qc @ U                                   # (n, D) query coords
        b = gn @ U                                   # (n, D) target coords

        def corr(col_a: np.ndarray, col_b: np.ndarray) -> float:
            sa, sb = col_a.std(), col_b.std()
            if sa < 1e-12 or sb < 1e-12:
                return 0.0
            return float(np.mean((col_a - col_a.mean()) * (col_b - col_b.mean())) / (sa * sb))

        rho2 = np.array([corr(a[:, j], b[:, j]) ** 2 for j in range(U.shape[1])])
        # The image coordinate carries no variance in some tail directions; those have
        # rho undefined and contribute nothing either way.
        var_b = b.var(axis=0)
        rho2[var_b < 1e-12] = 0.0

        lam_sum = float(vals.sum())
        concept_share = float((vals * rho2).sum() / max(lam_sum, 1e-12))

        print(f"\n{'=' * 92}\n[{tag}]  n={n}  d={q.shape[1]}\n{'=' * 92}")
        base = rung(qn, gn, args.csls_k)
        wh = calibration.saw_whiten(qn, shrink=0.1)[0]
        wh_r = calibration.report_with_scores(calibration.csls_scores(wh, gn, k=args.csls_k))
        print(f"  achieved rungs   raw {base['raw']:6.2f}   +CSLS {base['csls']:6.2f}   "
              f"saw_whiten raw {rung(wh, gn, args.csls_k)['raw']:6.2f}   "
              f"whiten+CSLS {wh_r['top1']:6.2f}")

        print(f"\n  variance that is CONCEPT-COVARIANT: {concept_share * 100:.2f}% "
              f"of the query cloud's total")
        # The offset is what the ablation below says is doing the work, so it is worth a
        # number rather than an adjective. `offset_frac` is the norm of the query cloud's
        # mean relative to the mean row norm: ~1 means the embedding is dominated by a
        # per-subject constant, ~0 means the rows are already centered on the subject.
        mu_q, mu_g = qn.mean(0), gn.mean(0)
        nq, ng = float(np.linalg.norm(qn, axis=1).mean()), float(np.linalg.norm(gn, axis=1).mean())
        print(f"  offset ||mean(q)||/mean||q|| = {float(np.linalg.norm(mu_q)) / max(nq, 1e-12):.3f}"
              f"    ||mean(g)||/mean||g|| = {float(np.linalg.norm(mu_g)) / max(ng, 1e-12):.3f}"
              f"    cos(mean_q, mean_g) = "
              f"{float(mu_q @ mu_g / max(float(np.linalg.norm(mu_q) * np.linalg.norm(mu_g)), 1e-12)):+.3f}")
        print(f"  {'band':>14} | {'lam share':>9} {'rho^2 mean':>10} {'lam*rho2 share':>15}")
        for lo, hi, name in [(0, 4, "dims 1-4"), (4, 16, "dims 5-16"),
                             (16, 64, "dims 17-64"), (64, 199, "dims 65-199"),
                             (199, len(vals), "dims 200+ (null)")]:
            hi = min(hi, len(vals))
            if lo >= hi:
                continue
            ls = vals[lo:hi].sum() / max(lam_sum, 1e-12)
            rs = rho2[lo:hi].mean()
            ss = (vals[lo:hi] * rho2[lo:hi]).sum() / max(lam_sum, 1e-12)
            print(f"  {name:>14} | {ls:>9.3f} {rs:>10.4f} {ss:>15.3f}")

        # ---- component ablation: is the rung a MEAN correction or a COVARIANCE one? ---
        # This is the measurement the whole `saw_whiten` narrative rests on, and it is
        # easy to get wrong in a way that never shows up: `saw_whiten` does BOTH a
        # centering and an inverse-covariance rescaling, so its gain has always been
        # attributed to "subject-adaptive whitening" without the two being separated.
        # They imply opposite redesigns -- an additive per-subject offset is a translation
        # to be removed by construction, a covariance difference needs a metric change --
        # so they are measured apart.
        #
        # The `shrink` sweep must go through `calibration.saw_whiten` itself rather than a
        # local reimplementation: its `shrink` mixes toward a SCALED identity
        # (`shrink * trace/d * I`), which sets the effective eigenvalue floor several
        # orders of magnitude above `lambda_max/max_cond`. An earlier version of this
        # block floored at `lambda_max/1e3` and reported full whitening at 8.5 where the
        # real function reports 19.0 -- the two are not the same estimator, and the
        # difference is exactly the tail amplification the shrink exists to prevent.
        qm = qn - qn.mean(0, keepdims=True)
        gm = gn - gn.mean(0, keepdims=True)
        print(f"\n  {'metric':>44} {'raw':>7} {'+CSLS':>7}")
        rows = [
            ("raw cosine", qn, gn),
            ("center QUERY only", qm, gn),
            ("center GALLERY only", qn, gm),
            ("center BOTH", qm, gm),
        ]
        for shr in (0.0, 0.1, 0.3, 0.7):
            rows.append((f"query centered + saw_whiten(shrink={shr})",
                         calibration.saw_whiten(qn, shrink=shr)[0], gn))
        for name, zq, zg in rows:
            r = rung(zq, zg, args.csls_k)
            print(f"  {name:>44} {r['raw']:>7.2f} {r['csls']:>7.2f}")
        print("  A large gap between `center QUERY only` and the whitening rows, with the "
              "whitening rows no better,")
        print("  means the rung is a TRANSLATION correction and the covariance part "
              "contributes nothing.")

        # ---- the SNR-optimal diagonal metric, as the bound on the whole family -------
        floor = float(vals.max()) / 1e3
        lam_safe = np.clip(vals, max(floor, 1e-12), None) + 1e-8

        def score_with(score_weight: np.ndarray) -> float:
            # A diagonal metric is applied to BOTH sides, so the SCORE weight is w^2;
            # whitening is `w = 1/sqrt(lam)` (score weight `1/lam`) and the SNR-optimal
            # weight is `w = |rho|/sqrt(lam)` (score weight `rho^2/lam`). Whitening is
            # thus exactly the member of this family that assumes every direction has the
            # SAME `rho^2`; the oracle is the family's ceiling.
            w = np.sqrt(np.clip(score_weight, 0.0, None))
            zq, zg = (qc @ U) * w, (gn @ U) * w
            return rung(zq, zg, args.csls_k)["raw"]

        oracle = rho2 / lam_safe
        top16 = np.argsort(-rho2)[:16]
        w_top16 = np.zeros_like(oracle)
        w_top16[top16] = oracle[top16]
        print(f"\n  {'ORACLE (uses labels; bounds the whole family)':>44} {'raw':>7}")
        print(f"  {'centered + w=|rho|/sqrt(lam), all dims':>44} {score_with(oracle):>7.2f}")
        print(f"  {'same, restricted to top-16 by rho^2':>44} {score_with(w_top16):>7.2f}")
        print("  Compare the ORACLE rows against `center QUERY only` above. If the oracle "
              "is not materially higher,")
        print("  then no diagonal reweighting in this representation can help and the "
              "encoder is what has to change.")


if __name__ == "__main__":
    main()
