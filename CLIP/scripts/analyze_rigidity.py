#!/usr/bin/env python
"""analyze_rigidity -- capacity-controlled rigidity analysis + the two tests that decide the theory.

The first pass (`probe_rigidity.py`) produced a `resid_smooth = 0.009` that CANNOT be believed
and must not be quoted: the "smooth" arm used 1024 random Fourier features on 200 concepts, so
it had `(64 + 1024) * 64 = 69632` free parameters against `200 * 64 = 12800` constraints. A
residual near zero from an underdetermined fit is overfitting, not evidence. This script
replaces it with a fit whose capacity is below the data and validated by 5-fold
cross-validation over concepts, and drops every number that cannot survive that control.

WHAT IS ACTUALLY MEASURED HERE

  A. Honest model-class residuals (5-fold CV over the 200 concepts):
     orthogonal vs unconstrained linear vs capacity-capped smooth.
     `CV` is over CONCEPTS, which is the right split because the question is whether the map
     generalises to held-out concepts -- not to held-out noise.

  B. Does the rigid residual PREDICT subject difficulty? This is the link between the geometry
     and the score, and it is what turns the theory into a falsifiable claim. If subjects with
     a larger rigid-model residual are systematically the weak subjects, then the model class
     is a live cause of the 31pp spread; if not, the rigidity story is decoration.

  C. THE GEOMETRY TEST, which decides the Gromov-Wasserstein proposal. GW/FGW aligns two
     domains using ONLY their intra-domain distance matrices, and only works if the concept
     metric is subject-invariant:
        corr(D_eeg_s, D_img)  -- does within-subject EEG geometry reflect concept geometry?
        corr(D_eeg_s, D_eeg_t) -- is the concept metric consistent across subjects?
     High values mean the intrinsic geometry is shared and structural alignment is viable.
     Low values mean the concept metric itself is subject-specific, in which case NO
     correspondence-free method can work and the theory's proposal is dead on arrival.
     This is deliberately a test that can fail.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FEAT = "outputs/probe/rigid_feats_sub{:02d}_seed{}.npz"


def _resid(P, Q):
    return float(np.linalg.norm(P - Q) / max(np.linalg.norm(Q), 1e-12))


def _zm(x):
    return x - x.mean(0, keepdims=True)


def _procrustes(A, B):
    """The rotation matrix `R` minimising `|A R - B|` (not the fitted product)."""
    u, _, vt = np.linalg.svd(A.T @ B)
    return u @ vt


def _ridge(A, B, lam):
    return np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ B)


def _rff(X, D, gamma, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(X.shape[1], D))
    b = rng.uniform(0, 2 * np.pi, size=D)
    return np.sqrt(2.0 / D) * np.cos(X @ w / max(np.sqrt(gamma), 1e-6) + b)


def cv_residual(A, B, kind, n_fold=5, rff_dim=32, gamma=1.0, lam=1e-2, seed=0):
    """Mean held-out residual |P-Q|/|Q| over folds; the map is fitted on the train folds only.

    Capacity is capped by construction: `rff_dim=32` gives `(64+32)*64 = 6144` parameters
    against a 160x64 training fold = 10240 constraints, so the fit is determined. The previous
    run's `rff_dim=1024` had 4x more parameters than constraints.
    """
    n = A.shape[0]
    idx = np.random.default_rng(seed).permutation(n)
    folds = np.array_split(idx, n_fold)
    out = []
    for f in folds:
        te = np.zeros(n, dtype=bool)
        te[f] = True
        Atr, Btr, Ate, Bte = A[~te], B[~te], A[te], B[te]
        if kind == "orth":
            P = _procrustes(Atr, Btr)
            Pte = Ate @ P
        elif kind == "linear":
            W = _ridge(Atr, Btr, 1e-6)
            Pte = Ate @ W
        else:
            Ftr = _rff(Atr, rff_dim, gamma, seed=seed)
            Fte = _rff(Ate, rff_dim, gamma, seed=seed)
            W = _ridge(np.hstack([Atr, Ftr]), Btr, lam)
            Pte = np.hstack([Ate, Fte]) @ W
        out.append(_resid(Pte, Bte))
    return float(np.mean(out))


def dist_corr(X, Y):
    """Correlation of upper-triangle pairwise distances between two clouds."""
    Dx = 1.0 - (X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)) @ \
        (X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)).T
    Dy = 1.0 - (Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-9)) @ \
        (Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-9)).T
    iu = np.triu_indices(Dx.shape[0], k=1)
    a, b = Dx[iu], Dy[iu]
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, nargs="*", default=list(range(1, 11)))
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--acc-summary", default="outputs/v8_s3r_summary.json",
                    help="for the geometry-vs-score link; any run_eval summary works")
    ap.add_argument("--acc-root", default="outputs/eval/v8_s3r",
                    help="per-fold reports, used to get per-subject accuracy")
    ap.add_argument("--out", default="outputs/probe/rigidity_cv.json")
    args = ap.parse_args()

    subs = [s for s in args.subjects
            if Path(FEAT.format(s, args.seed)).is_file()]
    if len(subs) < 3:
        raise SystemExit("need at least 3 subjects of cached features")

    data = {}
    for s in subs:
        z = np.load(FEAT.format(s, args.seed))
        data[s] = (_zm(z["z_e"].astype(np.float64)), _zm(z["z_i"].astype(np.float64)))

    # ------------------------------------------------------- A. honest model-class gap
    print("=" * 88)
    print("A. MODEL-CLASS RESIDUAL, 5-fold CV over concepts (capacity capped, no overfit)")
    print("-" * 88)
    print(f"{'subject':<9}{'orth':>9}{'linear':>9}{'smooth(32rff)':>16}"
          f"{'orth/lin':>10}{'resid 1d':>11}{'8d':>8}")
    rows = {}
    for s in subs:
        A, B = data[s]
        r_o = cv_residual(A, B, "orth")
        r_l = cv_residual(A, B, "linear")
        r_s = cv_residual(A, B, "smooth", rff_dim=32, gamma=1.0, lam=1.0)
        # residual spectrum after the in-sample orthogonal fit, for the low-rank question
        E = A @ _procrustes(A, B) - B
        e = np.linalg.svd(E, compute_uv=False) ** 2
        rows[s] = {"cv_orth": r_o, "cv_linear": r_l, "cv_smooth": r_s,
                   "orth_over_linear": r_o / r_l,
                   "resid_e1": float(e[0] / e.sum()), "resid_e8": float(e[:8].sum() / e.sum())}
        print(f"sub{s:02d}{r_o:>12.3f}{r_l:>9.3f}{r_s:>16.3f}"
              f"{r_o / r_l:>10.2f}{rows[s]['resid_e1']:>11.2f}{rows[s]['resid_e8']:>8.2f}")
    mo = np.mean([rows[s]["cv_orth"] for s in subs])
    ml = np.mean([rows[s]["cv_linear"] for s in subs])
    ms = np.mean([rows[s]["cv_smooth"] for s in subs])
    print("-" * 88)
    print(f"means: orth {mo:.3f}   linear {ml:.3f}   smooth {ms:.3f}")
    print(f"relaxing orth -> linear removes {(mo - ml) / mo * 100:.1f}% of the residual")
    print(f"relaxing linear -> smooth removes {(ml - ms) / ml * 100:.1f}% more")

    # ---------------------------------------------------- B. does geometry predict the score?
    print("\n" + "=" * 88)
    print("B. DOES THE RIGID RESIDUAL PREDICT SUBJECT ACCURACY?")
    print("-" * 88)
    acc = {}
    root = Path(args.acc_root)
    if root.is_dir():
        import glob
        for p in glob.glob(str(root / "sub*_seed*.json")):
            try:
                d = json.loads(Path(p).read_text())

                def find(o):
                    if isinstance(o, dict):
                        if isinstance(o.get("rows"), dict):
                            return o["rows"]
                        for v in o.values():
                            g = find(v)
                            if g is not None:
                                return g
                    return None
                r = find(d)
                if not r:
                    continue
                row = "+ T1(CSLS + recovery) + T2 reps"
                sub = int(Path(p).stem.split("_")[0][3:])
                acc.setdefault(sub, []).append(r[row]["top1"])
            except Exception:
                continue
    if acc:
        a = np.array([np.mean(acc[s]) for s in subs])
        for key in ("cv_orth", "cv_linear"):
            b = np.array([rows[s][key] for s in subs])
            print(f"  corr(subject acc, {key}) = {np.corrcoef(a, b)[0, 1]:+.3f}")
        b = np.array([rows[s]["orth_over_linear"] for s in subs])
        print(f"  corr(subject acc, orth/linear gap) = {np.corrcoef(a, b)[0, 1]:+.3f}")
        print(f"  subject acc range: {a.min():.1f} .. {a.max():.1f}  (n={len(a)})")
        for s, x in sorted(zip(subs, a), key=lambda t: t[1]):
            print(f"    sub{s:02d}  acc {x:5.1f}   cv_orth {rows[s]['cv_orth']:.3f}   "
                  f"cv_linear {rows[s]['cv_linear']:.3f}")
    else:
        print(f"  (no per-fold reports under {args.acc_root}; fit of geometry to score skipped)")

    # ------------------------------------------- C. the geometry test that decides GW
    print("\n" + "=" * 88)
    print("C. IS THE CONCEPT METRIC SUBJECT-INVARIANT?  (decides whether GW/FGW can work)")
    print("-" * 88)
    img = data[subs[0]][1]
    print("  corr(D_eeg_subject, D_image)   [within-subject EEG geometry vs concept geometry]")
    vs_img = {}
    for s in subs:
        vs_img[s] = dist_corr(data[s][0], img)
        print(f"    sub{s:02d}  {vs_img[s]:+.3f}")
    print("  corr(D_eeg_s, D_eeg_t)         [cross-subject consistency of the concept metric]")
    pair = []
    for i, s in enumerate(subs):
        for t in subs[i + 1:]:
            pair.append(dist_corr(data[s][0], data[t][0]))
    print(f"    mean over {len(pair)} pairs  {np.mean(pair):+.3f}   "
          f"(min {min(pair):+.3f}, max {max(pair):+.3f})")
    # Baseline: how much of this is just "both matrices come from 200 points in 64 dims"?
    rng = np.random.default_rng(0)
    rnd = [dist_corr(data[subs[0]][0], rng.normal(size=img.shape)) for _ in range(5)]
    print(f"    chance baseline (random 64-dim cloud)  {np.mean(rnd):+.3f}")
    print("-" * 88)
    print(f"  mean corr to image geometry = {np.mean(list(vs_img.values())):+.3f}")
    print(f"  mean cross-subject geometry agreement = {np.mean(pair):+.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"per_subject": {f"sub{s:02d}": rows[s] for s in subs},
         "d_image_geometry": {f"sub{s:02d}": vs_img[s] for s in subs},
         "cross_subject_geometry_mean": float(np.mean(pair)),
         "chance_geometry": float(np.mean(rnd))}, indent=2, default=str))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
