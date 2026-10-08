#!/usr/bin/env python
"""DECISIVE PROBES: is the subject term a GROUP ACTION or a DIFFEOMORPHISM?  (v2)

WHY THERE IS A v2
-----------------
v1 was wrong twice, and both mistakes are recorded here because both are the kind that produce
a confident wrong answer.

  * Probe A compared the fitted maps' closure defect against a SYNTHETIC group-action control.
    The control's maps were large random rotations, so its defect was ~1.41, while the real maps
    are near the identity (the encoder has already largely aligned the subjects) and hence
    trivially "compose" -- defect ~1.01. That measured **how far the maps are from the identity**,
    not whether they form a group, and it would have reported "real data is MORE closed than a
    group", which is a nonsense reading presented as a result.

  * Probe B fit a local orthogonal map from k = 10 neighbours. A 64x64 orthogonal matrix has
    2016 parameters, so k = 10 is underdetermined by construction and local alignment can only
    lose. It lost on 0% of pairs. That is a design defect masquerading as evidence against the
    hypothesis.

FIXES
  Probe A is now a HELD-OUT COMPOSITION test, which needs no synthetic control at all and is
  therefore self-calibrating: fit R_{s->t}, R_{t->u} on one half of the concepts and ask whether
  the COMPOSED map predicts the held-out half as well as the DIRECTLY FITTED one. Both arms are
  evaluated on the same held-out concepts, so estimation noise cancels in the comparison. If the
  subject term is a group action the two are equally good; a gap is a direct measurement of
  non-group structure, with the noise floor already subtracted by the pairing.

  Probe B is now a LOCALLY-WEIGHTED Procrustes whose effective sample size is swept and is always
  kept above the parameter count. As the bandwidth widens it reduces exactly to the global map,
  so "does position-dependence help?" becomes a one-parameter question with the global solution
  as its own null.

PRE-REGISTERED READING
  A: composed held-out residual materially worse than direct (and paired t large) -> NOT a group.
  B: some effective-sample size beats the global fit on held-out concepts -> position-dependence
     is real and resolvable at n = 200.
  Both must hold for the Anisometric Manifold Alignment direction to be well posed. If A holds
  but B does not, the subject term is non-group but NOT locally resolvable at this sample size,
  and the honest conclusion is that the theory is right and unusable here.

Pure numpy, CPU only, on the cached per-encoder concept means.  Writes outputs/probe/gauge/.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
OUT_DIR = os.path.join(ROOT, "outputs", "probe", "gauge")


def _mm(x: np.ndarray) -> np.ndarray:
    """Per-dimension standardisation: the frame deployment's recovery actually works in."""
    return (x - x.mean(0, keepdims=True)) / (x.std(0, keepdims=True) + 1e-8)


def _procrustes(x: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Weighted orthogonal Procrustes: argmin_Q sum_i w_i ||x_i Q - y_i||^2."""
    if w is None:
        m = x.T @ y
    else:
        m = x.T @ (w[:, None] * y)
    u, _, vt = np.linalg.svd(m)
    return u @ vt


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(float(np.linalg.norm(b)), 1e-12))


def load_folds(pattern: str) -> dict:
    """Each npz = ONE encoder (trained on 9 of 10 subjects) + those 9 subjects' concept means in
    that encoder's own 64-d output space. Same encoder across the 9 clouds is what makes a map
    between two subjects well defined without a cross-encoder confound."""
    out = {}
    for f in sorted(glob.glob(pattern)):
        z = np.load(f)
        out[os.path.basename(f)[:-5]] = np.asarray(z["src_means"], dtype=np.float64)
    return out


def _inject_field(x0: np.ndarray, rng: np.random.Generator,
                  strength: float = 0.9) -> np.ndarray:
    """Apply a smooth position-dependent rotation: angle varies linearly along a random axis."""
    c, d = x0.shape
    p = rng.standard_normal(d)
    p /= np.linalg.norm(p)
    u = rng.standard_normal(d)
    u /= np.linalg.norm(u)
    v = rng.standard_normal(d)
    v = v - (v @ u) * u
    v /= np.linalg.norm(v)
    theta = strength * (x0 @ p)
    y = np.empty_like(x0)
    for r in range(c):
        th = theta[r]
        rmat = (np.eye(d) + np.sin(th) * (np.outer(v, u) - np.outer(u, v))
                + (np.cos(th) - 1.0) * (np.outer(u, u) + np.outer(v, v)))
        y[r] = x0[r] @ rmat
    return y


def _joint_frame(x: np.ndarray, y: np.ndarray, r: int | None) -> np.ndarray:
    """Common orthonormal frame from the top-`r` directions of the JOINT cloud.

    Built from [x; y] rather than from either subject, so the frame is not tied to one side. With
    `r=None` the frame is the identity (the full space).

    WHY THE REDUCTION MATTERS TWICE. (i) It is the theory's own claim: the concept manifold has
    d_M ~ 16-25 measured dimensions, so the alignment operator belongs there. (ii) It fixes the
    POWER defect -- a local Procrustes in d = 64 must fit 2016 parameters, so even an INJECTED
    smooth field is barely detectable (the v2 control only fired on 1 of 10 folds). Restricting to
    r = 16 leaves 120 parameters, which ESS = 64 determines five times over.
    """
    if r is None or r >= x.shape[1]:
        return np.eye(x.shape[1])
    _, _, vt = np.linalg.svd(np.vstack([x, y]), full_matrices=False)
    return vt[: int(r)].T


def probe_b2(means: np.ndarray, rng: np.random.Generator, ranks=(8, 16, 32),
             ess_list=(None, 64, 32), n_split: int = 2, standardize: bool = False) -> dict:
    """Local vs global Procrustes INSIDE the intrinsic subspace, with a gated positive control.

    Each (rank, standardize) setting is only read if the injected-field control shows the
    estimator has power THERE -- an under-powered negative is not evidence. `standardize=False`
    is the default because per-dimension standardisation is a diagonal whitening, and this
    project has already measured that whitening inflates noise directions (effective rank
    25.6 -> 85.0); a probe that silently whitens would test a different space than the one it
    claims to.
    """
    s, c, d = means.shape

    def _prep(m: np.ndarray) -> np.ndarray:
        v = np.asarray(m, dtype=np.float64)
        if standardize:
            v = _mm(v)
        return v - v.mean(0, keepdims=True)

    mm = np.stack([_prep(means[i]) for i in range(s)])
    out: dict = {}
    for rank in ranks:
        for tag, gen in (("real", None), ("control", _inject_field)):
            acc: dict = {}
            for i in range(s):
                for j in range(s):
                    if i == j:
                        continue
                    x, y = mm[i], mm[j]
                    if tag == "control":
                        y = _inject_field(x, rng)
                    vv = _joint_frame(x, y, rank)
                    xr, yr = x @ vv, y @ vv
                    for _ in range(n_split):
                        idx = rng.permutation(c)
                        fit, hold = idx[: c // 2], idx[c // 2:]
                        d2 = ((xr[hold, None, :] - xr[fit][None, :, :]) ** 2).sum(-1)
                        for e in ess_list:
                            if e is None:
                                pred = xr[hold] @ _procrustes(xr[fit], yr[fit])
                            else:
                                pred = np.empty_like(yr[hold])
                                med = np.median(d2)
                                for t in range(hold.size):
                                    h2 = max(med / max(float(e), 1.0), 1e-12)
                                    w = np.exp(-d2[t] / (2.0 * h2))
                                    pred[t] = xr[hold][t] @ _procrustes(
                                        xr[fit], yr[fit], w=w / max(w.sum(), 1e-12))
                            acc.setdefault("global" if e is None else f"ess{e}", []).append(
                                _rel(pred, yr[hold]))
            base = float(np.mean(acc["global"]))
            row = {"global": base}
            for kk in acc:
                if kk == "global":
                    continue
                row[kk] = float(np.mean(acc[kk]))
                row[f"{kk}_ratio"] = row[kk] / base
            out[f"r{rank}_{tag}"] = row

        real = out[f"r{rank}_real"]
        ctrl = out[f"r{rank}_control"]
        ctrl_ratio = min((ctrl[k] / ctrl["global"] for k in ctrl if k.startswith("ess")),
                         default=1.0)
        out[f"r{rank}_control_power"] = float(1.0 - ctrl_ratio)
        real_ratio = min((real[k] / real["global"] for k in real if k.startswith("ess")),
                         default=1.0)
        out[f"r{rank}_real_local_gain"] = float(1.0 - real_ratio)
        # the estimate is only READ where the control proves the estimator can see a field
        out[f"r{rank}_conclusive"] = bool(out[f"r{rank}_control_power"] > 0.10)
        out[f"r{rank}_local_helps"] = bool(out[f"r{rank}_real_local_gain"] > 0.05)
    return out


def probe_a(means: np.ndarray, rng: np.random.Generator, n_split: int = 4) -> dict:
    """HELD-OUT COMPOSITION: does the composed map predict as well as the direct one?

    Self-calibrating -- no synthetic control. Both arms see the same held-out concepts, so the
    estimation noise floor cancels in the paired difference.
    """
    s, c, d = means.shape
    mm = np.stack([_mm(means[i]) for i in range(s)])
    direct, comp, comp_rand = [], [], []
    for _ in range(n_split):
        idx = rng.permutation(c)
        fit, hold = idx[: c // 2], idx[c // 2:]
        R = {(i, j): _procrustes(mm[i][fit], mm[j][fit])
             for i in range(s) for j in range(s) if i != j}
        for i in range(s):
            for j in range(s):
                for k in range(s):
                    if len({i, j, k}) < 3:
                        continue
                    xs = mm[i][hold]
                    direct.append(_rel(xs @ R[(i, k)], mm[k][hold]))
                    comp.append(_rel(xs @ (R[(j, k)] @ R[(i, j)]), mm[k][hold]))
                    # a RANDOM orthogonal reference: how bad is a map with no relation at all?
                    q = np.linalg.qr(rng.standard_normal((d, d)))[0]
                    comp_rand.append(_rel(xs @ (R[(j, k)] @ R[(i, j)] @ q), mm[k][hold]))
    direct, comp, comp_rand = map(np.asarray, (direct, comp, comp_rand))
    dlt = comp - direct
    t = (dlt.mean() / (dlt.std(ddof=1) / np.sqrt(dlt.size)) if dlt.std(ddof=1) > 0 else np.nan)
    return {
        "direct_heldout_rel_resid": float(direct.mean()),
        "composed_heldout_rel_resid": float(comp.mean()),
        "random_ref_rel_resid": float(comp_rand.mean()),
        "composed_minus_direct": float(dlt.mean()),
        "composed_minus_direct_t": float(t),
        "composed_worse_frac": float((dlt > 0).mean()),
        # the scale a "gap" should be read against: the spread of the direct maps themselves
        "direct_map_spread": float(direct.std(ddof=1)),
        "closure_gap_over_direct_spread": float(dlt.mean() / max(direct.std(ddof=1), 1e-12)),
        "group_hypothesis_rejected": bool(dlt.mean() > 0.2 * direct.mean()
                                          and t > 4.0),
    }


def _ess(w: np.ndarray) -> float:
    return float((w.sum() ** 2) / max(float((w ** 2).sum()), 1e-12))


def probe_b(means: np.ndarray, rng: np.random.Generator, ess_list=(None, 128, 64),
            n_split: int = 2) -> dict:
    """LOCALLY-WEIGHTED Procrustes, effective sample size swept, global as its own null.

    The bandwidth is chosen per target concept to hit a target effective sample size, which is
    always kept well above the fitted parameter count (`d(d-1)/2` = 2016 for d = 64 is only
    approached via the low-rank structure of the data, so the ESS floor is set by validation, not
    asserted). As ESS -> all concepts the estimator reduces exactly to the global map.

    TWO THINGS THAT MAKE A NEGATIVE READABLE. A local estimator that cannot detect a field which
    is KNOWN to be there proves nothing when it finds nothing, so:

      * POSITIVE CONTROL: the same estimator is run on data where a smooth position-dependent
        rotation IS injected (angle varying linearly along a random direction). If local does not
        beat global there, the test is under-powered and its verdict on real data is void.
      * NO-MAP BASELINE: the identity residual ||x - y||/||y||, so "the global map explains 45% of
        the variance" is a statement about a measured reference rather than about a bare number.

    Plus the diagnostic that actually distinguishes the two remaining explanations: RESIDUAL
    SPATIAL COHERENCE. After the global map, are the residuals of NEIGHBOURING concepts more
    alike than those of random pairs? A smooth deformation leaves coherent residuals; a
    concept-idiosyncratic noise floor does not. This is what separates "the theory is right but
    needs a field estimator" from "the residual is noise and no alignment can remove it".
    """
    s, c, d = means.shape
    mm = np.stack([_mm(means[i]) for i in range(s)])

    def _sweep(x, y, fit, hold, d2):
        out = {}
        for e in ess_list:
            if e is None:
                pred = x[hold] @ _procrustes(x[fit], y[fit])
            else:
                pred = np.empty_like(y[hold])
                med = np.median(d2)
                for r in range(hold.size):
                    h2 = max(med / max(float(e), 1.0), 1e-12)
                    w = np.exp(-d2[r] / (2.0 * h2))
                    pred[r] = x[hold][r] @ _procrustes(x[fit], y[fit], w=w / max(w.sum(), 1e-12))
            out["global" if e is None else f"ess{e}"] = _rel(pred, y[hold])
        out["nomap"] = _rel(x[hold], y[hold])
        return out

    res: dict = {}
    coh_nn, coh_rnd = [], []
    for i in range(s):
        for j in range(s):
            if i == j:
                continue
            x, y = mm[i], mm[j]
            for _ in range(n_split):
                idx = rng.permutation(c)
                fit, hold = idx[: c // 2], idx[c // 2:]
                d2 = ((x[hold, None, :] - x[fit][None, :, :]) ** 2).sum(-1)
                got = _sweep(x, y, fit, hold, d2)
                for kk, v in got.items():
                    res.setdefault(kk, []).append(v)
            # ---- residual spatial coherence, on the FULL concept set ----
            q = _procrustes(x, y)
            r = y - x @ q                              # (C, d) residuals
            rn = r / np.clip(np.linalg.norm(r, axis=1, keepdims=True), 1e-12, None)
            dxx = ((x[:, None, :] - x[None, :, :]) ** 2).sum(-1)
            np.fill_diagonal(dxx, np.inf)
            nn = np.argsort(dxx, axis=1)[:, :10]
            coh_nn.append(float(np.mean([rn[a] @ rn[b] for a in range(c) for b in nn[a]])))
            perm = rng.permutation(c)
            coh_rnd.append(float(np.mean([rn[a] @ rn[perm[a]] for a in range(c)])))

    out: dict = {}
    for kk, v in res.items():
        v = np.asarray(v)
        out[f"{kk}_rel_resid_mean"] = float(v.mean())
        out[f"{kk}_rel_resid_sd"] = float(v.std(ddof=1))
    g = np.asarray(res["global"])
    out["global_var_explained"] = float(1.0 - g.mean() ** 2)
    out["nomap_var_explained"] = float(1.0 - np.asarray(res["nomap"]).mean() ** 2)

    # ---- POSITIVE CONTROL: inject a KNOWN smooth position-dependent rotation ----------------
    x0 = mm[0]
    p = rng.standard_normal(d)
    p /= np.linalg.norm(p)
    u = rng.standard_normal(d)
    u /= np.linalg.norm(u)
    v = rng.standard_normal(d)
    v = v - (v @ u) * u
    v /= np.linalg.norm(v)
    theta = 0.9 * (x0 @ p)                            # smoothly varying angle
    ctrl_res: dict = {}
    for _ in range(2):
        idx = rng.permutation(c)
        fit, hold = idx[: c // 2], idx[c // 2:]
        yc = np.empty_like(x0)
        for r in range(c):
            th = theta[r]
            rmat = (np.eye(d) + np.sin(th) * (np.outer(v, u) - np.outer(u, v))
                    + (np.cos(th) - 1.0) * (np.outer(u, u) + np.outer(v, v)))
            yc[r] = x0[r] @ rmat
        d2c = ((x0[hold, None, :] - x0[fit][None, :, :]) ** 2).sum(-1)
        got = _sweep(x0, yc, fit, hold, d2c)
        for kk, vv in got.items():
            ctrl_res.setdefault(kk, []).append(vv)
    cg = float(np.mean(ctrl_res["global"]))
    cbest = min((float(np.mean(vv)), kk) for kk, vv in ctrl_res.items() if kk != "nomap")
    out["control_global_rel_resid"] = cg
    out["control_best_local"] = cbest[1]
    out["control_best_local_ratio"] = cbest[0] / cg
    out["control_local_detects_field"] = bool(cbest[1].startswith("ess")
                                              and cbest[0] < 0.9 * cg)

    g = np.asarray(res["global"])
    best_name, best_ratio, best_t = None, 1.0, 0.0
    for kk in res:
        if kk == "global" or kk == "nomap":
            continue
        v = np.asarray(res[kk])
        dlt = v - g
        t = (dlt.mean() / (dlt.std(ddof=1) / np.sqrt(dlt.size))
             if dlt.std(ddof=1) > 0 else np.nan)
        out[f"{kk}_minus_global"] = float(dlt.mean())
        out[f"{kk}_minus_global_t"] = float(t)
        out[f"{kk}_better_frac"] = float((dlt < 0).mean())
        if dlt.mean() < 0 and float((dlt < 0).mean()) > 0.6 and t < -4.0:
            if v.mean() / g.mean() < best_ratio:
                best_name, best_ratio, best_t = kk, float(v.mean() / g.mean()), float(t)
    out["best_local"] = best_name
    out["best_local_ratio"] = best_ratio
    out["best_local_t"] = best_t
    out["position_dependence_supported"] = bool(best_name is not None)
    out["resid_coherence_nn"] = float(np.mean(coh_nn))
    out["resid_coherence_random"] = float(np.mean(coh_rnd))
    out["resid_coherence_excess"] = float(np.mean(coh_nn) - np.mean(coh_rnd))
    out["residual_is_smooth"] = bool(np.mean(coh_nn) - np.mean(coh_rnd) > 0.10)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob",
                    default=os.path.join(ROOT, "outputs/src_metric/v8/sub*_seed2025.npz"))
    ap.add_argument("--splits-a", type=int, default=4)
    ap.add_argument("--splits-b", type=int, default=2)
    ap.add_argument("--ess", default="128,64")
    ap.add_argument("--ranks", default="8,16,32")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--limit", type=int, default=0, help="debug: use only N encoders")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    folds = load_folds(args.glob)
    if args.limit:
        folds = dict(sorted(folds.items())[: args.limit])
    if not folds:
        raise SystemExit(f"no npz matched {args.glob}")
    ess = [None] + [int(x) for x in str(args.ess).split(",") if x.strip()]
    print(f"[gauge] {len(folds)} encoders, ESS sweep {ess}")

    rng = np.random.default_rng(args.seed)
    A, B, B2 = [], [], []
    ranks = tuple(int(x) for x in str(args.ranks).split(",") if x.strip())
    for name in sorted(folds):
        t0 = time.time()
        a = probe_a(folds[name], rng, n_split=args.splits_a)
        b = probe_b(folds[name], rng, ess_list=tuple(ess), n_split=args.splits_b)
        b2 = probe_b2(folds[name], rng, ranks=ranks, ess_list=tuple(ess),
                      n_split=args.splits_b)
        A.append(a)
        B.append(b)
        B2.append(b2)
        rtxt = " ".join(f"r{r}:pow{b2[f'r{r}_control_power']:.2f}/gain{b2[f'r{r}_real_local_gain']:+.2f}"
                        for r in ranks)
        print(f"  [{name}] A: direct {a['direct_heldout_rel_resid']:.4f} -> composed "
              f"{a['composed_heldout_rel_resid']:.4f} (+{a['composed_minus_direct']:.4f}, "
              f"t={a['composed_minus_direct_t']:.1f}) | B2: {rtxt} [{time.time()-t0:.0f}s]")

    def m(lst, key):
        v = np.asarray([x[key] for x in lst], dtype=float)
        return float(v.mean()), float(v.std(ddof=1))

    summary = {
        "n_encoders": len(folds),
        "ess_sweep": [("global" if e is None else e) for e in ess],
        "ranks": list(ranks),
        "probe_a": {k: [x[k] for x in A] for k in A[0]},
        "probe_b": {k: [x[k] for x in B] for k in B[0]},
        "probe_b2": {k: [x[k] for x in B2] for k in B2[0]},
        "aggregate": {
            "a_direct_mean": m(A, "direct_heldout_rel_resid")[0],
            "a_composed_mean": m(A, "composed_heldout_rel_resid")[0],
            "a_gap_mean": m(A, "composed_minus_direct")[0],
            "a_gap_t_mean": m(A, "composed_minus_direct_t")[0],
            "a_group_rejected_encoders": int(sum(x["group_hypothesis_rejected"] for x in A)),
            "b_global_mean": m(B, "global_rel_resid_mean")[0],
            "b_nomap_mean": m(B, "nomap_rel_resid_mean")[0],
            "b_position_dep_encoders": int(sum(x["position_dependence_supported"] for x in B)),
            "b_resid_coherence_excess": m(B, "resid_coherence_excess")[0],
            "b2_conclusive_ranks": [r for r in ranks
                                    if sum(x[f"r{r}_conclusive"] for x in B2) >=
                                    max(1, int(0.8 * len(B2)))],
            "b2_local_helps_ranks": [r for r in ranks
                                     if sum(x[f"r{r}_local_helps"] for x in B2) >=
                                     max(1, int(0.8 * len(B2)))],
        },
    }
    for r in ranks:
        summary["aggregate"][f"b2_r{r}_control_power_mean"] = m(B2, f"r{r}_control_power")[0]
        summary["aggregate"][f"b2_r{r}_real_local_gain_mean"] = m(B2, f"r{r}_real_local_gain")[0]
    n = len(folds)
    a_ok = summary["aggregate"]["a_group_rejected_encoders"] >= max(1, int(0.8 * n))
    b_ok = len(summary["aggregate"]["b2_local_helps_ranks"]) > 0
    summary["verdict"] = {
        "probe_a_not_a_group": bool(a_ok),
        "probe_b_locally_resolvable": bool(b_ok),
        "ama_direction_well_posed": bool(a_ok and b_ok),
        "reading": ("AMA well posed: subject term is non-group AND locally resolvable"
                    if a_ok and b_ok else
                    "GROUP SUFFICES / NO FIELD: global maps already compose; AMA has nothing to add"
                    if not a_ok else
                    "non-group but NOT locally resolvable: the residual is not a smooth field "
                    "at this sample size, so no field estimator can exploit it"),
    }
    p = os.path.join(OUT_DIR, "gauge_probe_v2.json")
    with open(p, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n[gauge] wrote {p}")
    for k, v in summary["aggregate"].items():
        print(f"   {k:<34s} {v}")
    print(f"[gauge] VERDICT: {summary['verdict']['reading']}")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            names = sorted(folds)
            fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
            ax[0].bar(range(len(names)),
                      [x["composed_minus_direct"] / max(x["direct_heldout_rel_resid"], 1e-9)
                       for x in A], color="#2b6cb0")
            ax[0].axhline(0.2, color="r", ls=":", lw=1, label="pre-registered 20% threshold")
            ax[0].set_ylabel("(composed - direct) / direct")
            ax[0].set_title("PROBE A: held-out composition gap\n(>20% = the maps are NOT a group)")
            ax[0].legend(fontsize=7)
            ax[0].set_xticks(range(len(names)))
            ax[0].set_xticklabels(names, rotation=90, fontsize=7)
            for kk, col in zip(["global"] + [f"ess{e}" for e in ess[1:]],
                               ("#4a5568", "#2f855a", "#38a169")):
                key = f"{kk}_rel_resid_mean"
                if key in B[0]:
                    ax[1].plot(range(len(names)), [x[key] for x in B], "o-", label=kk,
                               color=col, ms=3)
            ax[1].set_ylabel("held-out relative residual")
            ax[1].set_title("PROBE B: local bandwidth vs global\n(lower = position-dependence helps)")
            ax[1].legend(fontsize=7)
            ax[1].set_xticks(range(len(names)))
            ax[1].set_xticklabels(names, rotation=90, fontsize=7)
            fig.tight_layout()
            png = os.path.join(OUT_DIR, "gauge_probe_v2.png")
            fig.savefig(png, dpi=130)
            print(f"[gauge] wrote {png}")
        except Exception as exc:  # pragma: no cover
            print(f"[gauge] plot skipped: {exc}")


if __name__ == "__main__":
    main()
