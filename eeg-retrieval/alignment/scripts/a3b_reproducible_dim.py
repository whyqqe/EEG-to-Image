#!/usr/bin/env python
"""
A3b / A6 -- REPRODUCIBLE DIMENSIONALITY AND CAPACITY-CORRECTED COVERAGE

Carries Claims 1 and 2 of docs/03_theory_and_method.md on THINGS-EEG2.

=============================================================================
WHAT THIS SCRIPT ESTABLISHED, IN ORDER (all measured, nothing assumed)
=============================================================================
Attempt 1 -- participation ratio of the noise-corrected covariance,

    r_PR = PR( psd( S_obs - S_noise / T ) ),   S_obs = Cov_i(bar x_i),

is unbiased (Prop. 1) and is what the field effectively reports.  MEASURED
FAILURE (sub-01, w1):

    D      100    200    400    800   1500
    r_PR    23     49     95    174    279
    r_PR/D  23%    25%    24%    22%    19%     <- tracks the budget

A genuine intrinsic dimension must saturate once D exceeds it.  Ours does not,
so it is not measured.  PR counts a smoothly decaying bulk with no spectral gap.

Attempt 2 -- held-out screening count, i.e. split the concepts in half, estimate
the signal subspace in each half, and count k where the top-k subspace overlap

    overlap(k) = mean_i cos^2(theta_i)

exceeds the quantile of E[overlap] = k/d for independent random k-frames.  This
is the same mechanism as RSCA, which GATE 1 validated.  MEASURED RESULT:

    D      150    300    400    700   1200
    r_screen 30     50     50     80    250      <- still tracks the budget

So the screened count ALSO scales with D.  The reason is now identified: the
whitened trial-mean covariance contains reproducible variance that is NOT
stimulus-locked (drift, non-stationarity, session effects).  Such components
reproduce across concept splits just as well as visual ones, and there is more
of them the more dimensions you keep.

=============================================================================
WHAT THIS SCRIPT THEREFORE MEASURES
=============================================================================
Three things, so that the claim is decided by data rather than by choice:

  (1) The FULL D-sweep of r_screen, r_PR and r_uncorr.  Reporting a single D
      would hide the budget dependence.  The sweep is the figure.

  (2) A STIMULUS-LOCKED CONTROL.  The same analysis on circularly time-shifted
      EEG (per concept, random offset; a1_controls' C3).  The shift destroys
      stimulus locking while preserving spectra, drift and dimensionality, so

          r_locked(D) = r_screen^{real}(D) - r_screen^{shift}(D)

      is the stimulus-locked reproducible dimension.  r_locked is the only
      version of Claim 1 that can survive, and it is reported with a bootstrap
      CI.  If r_locked still scales with D, Claim 1 must be abandoned.

  (3) CAPACITY-CORRECTED COVERAGE (Claim 2, Prop. 5a).  For a random m-dim
      subspace of R^n, E[tr(P_V P_M)] = r*m/n; with n=1654 and q=1024 a RANDOM
      model already covers ~62% of the reproducible subspace, so only the EXCESS
      over a shuffled-feature control is interpretable.

Conventions match the rest of the pipeline: EEG (n, T, ch, time), T = 10 repeat
groups, windows from lib_windows (the verified time axis).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_windows import EEG_ROOT, WINDOWS, slice_eeg  # noqa: E402

# dimensions evaluated in the sweep; the first entry is the primary
D_GRID = [400, 100, 150, 200, 300, 600, 900, 1400]
K_GRID = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 30, 40, 50, 65, 80, 100,
          130, 160, 200, 250, 320]

# ---------------------------------------------------------------------------
# linear algebra
# ---------------------------------------------------------------------------


def psd_project(S: np.ndarray) -> np.ndarray:
    """Projection onto the PSD cone (clip eigenvalues at 0).

    Prop. 1: this biases eigenvalues upward, so every dimensionality derived
    from it is an UPPER bound and the conclusions are conservative.
    """
    S = 0.5 * (S + S.T)
    w, V = np.linalg.eigh(S)
    return (V * np.maximum(w, 0.0)) @ V.T


def pr_from_eigs(w: np.ndarray) -> float:
    w = np.maximum(np.asarray(w, float), 0.0)
    return float(w.sum() ** 2 / max((w ** 2).sum(), 1e-300))


def pr(S: np.ndarray) -> float:
    return pr_from_eigs(np.linalg.eigvalsh(0.5 * (S + S.T)))


def inv_sqrt_psd(S: np.ndarray) -> np.ndarray:
    S = 0.5 * (S + S.T)
    w, V = np.linalg.eigh(S)
    return (V * (1.0 / np.sqrt(np.maximum(w, 1e-12)))) @ V.T


def pca_basis(Ec: np.ndarray, d: int) -> np.ndarray:
    """Top-d right singular vectors of Ec (n, p) via the n x n Gram matrix.

    Never forms a p x p matrix, so the 1 s window (p = 15750) is tractable.
    """
    G = Ec @ Ec.T
    w, U = np.linalg.eigh(G)
    order = np.argsort(w)[::-1][:d]
    w, U = w[order], U[:, order]
    s = np.sqrt(np.maximum(w, 0.0))
    keep = s > 1e-10
    if not np.all(keep):
        U, s = U[:, keep], s[keep]
    return (Ec.T @ U) / s


def cov(X: np.ndarray) -> np.ndarray:
    X = X - X.mean(0, keepdims=True)
    return (X.T @ X) / max(X.shape[0] - 1, 1)


def subspace_overlap(Qa: np.ndarray, Qb: np.ndarray) -> float:
    """mean cos^2 of principal angles.  The singular values of Qa^T Qb ARE the
    cosines, so no SVD is needed."""
    return float(np.sum((Qa.T @ Qb) ** 2) / Qa.shape[1])


_NULL_CACHE: Dict[Tuple[int, int], float] = {}


def null_q95(d: int, k: int, n_sim: int = 300, seed: int = 0) -> float:
    """95th percentile of overlap for INDEPENDENT random k-frames in R^d.

    E[overlap] = k/d exactly, so this turns the screen into a real test rather
    than the arbitrary margin 'ratio > 1'.
    """
    key = (d, k)
    if key in _NULL_CACHE:
        return _NULL_CACHE[key]
    n_sim = max(int(n_sim), 100)
    rng = np.random.default_rng(seed + 1000 * k)
    vals = np.empty(n_sim)
    for i in range(n_sim):
        A, _ = np.linalg.qr(rng.normal(size=(d, k)))
        B, _ = np.linalg.qr(rng.normal(size=(d, k)))
        vals[i] = np.sum((A.T @ B) ** 2) / k
    _NULL_CACHE[key] = float(np.quantile(vals, 0.95))
    return _NULL_CACHE[key]


# ---------------------------------------------------------------------------
# data loading (same conventions as a1_controls / a2_ceiling_effdim)
# ---------------------------------------------------------------------------


def load_eeg(sub: str, which: str = "train") -> np.ndarray:
    a = np.load(f"{EEG_ROOT}/preprocessed_eeg/{sub}/{which}.npy")
    if a.ndim == 5:                      # (n, T, sessions, ch, time)
        a = a.mean(2)
    return a.astype(np.float32)


def load_clip(name: str = "ViT-H-14", which: str = "train") -> np.ndarray:
    f = np.load(f"{EEG_ROOT}/image_feature/{name}/image_{which}.npy")
    if f.ndim == 3:
        f = f.mean(1)
    return f.astype(np.float32)


def circular_shift(E: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Per-concept circular time shift of the whole epoch.

    Applied on the 4-D array BEFORE slicing so the shift cannot accidentally
    leave the window aligned.  Destroys stimulus locking while preserving the
    spectrum, the drift and the dimensionality -- which is exactly what the
    stimulus-locked control needs.

    NOTE: this materialises a full-size copy of E (4 GB for a THINGS-EEG2 train
    split).  Prefer shifted_slice() when only one window is needed -- see below.
    """
    n, T, ch, nt = E.shape
    offs = rng.integers(1, nt, size=n)
    out = np.empty_like(E)
    for i in range(n):
        out[i] = np.roll(E[i], int(offs[i]), axis=-1)
    return out


def shifted_slice(E: np.ndarray, lo: int, hi: int, offs: np.ndarray) -> np.ndarray:
    """Sliced [lo, hi) of E after a per-concept circular shift, without the copy.

    Equivalent to slice_eeg(circular_shift(E, ...), lo, hi) but allocates only
    the window.  The wrapped index (arange(lo,hi) - off) % nt is exactly what
    np.roll followed by slicing would select, so the two agree sample for sample
    while avoiding a full-size 4 GB intermediate -- which matters because this
    runs once per shift replicate per window.
    """
    n, T, ch, nt = E.shape
    out = np.empty((n, T, ch, hi - lo), dtype=E.dtype)
    base = np.arange(lo, hi)
    for i in range(n):
        out[i] = E[i][:, :, (base - int(offs[i])) % nt]
    return out


def sample_shift_offsets(n: int, nt: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(1, nt, size=n)


# ---------------------------------------------------------------------------
# the estimator
# ---------------------------------------------------------------------------


def prep_projection(E: np.ndarray, d_max: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (Ep, Rp): trial means and residual trials in the top-d_max basis.

    Computed once at the largest D and sliced for the sweep, so the expensive
    p-dimensional products are paid for once instead of once per D.

    MEMORY: the obvious implementation promotes the whole (n, T, ch, nt) array to
    float64, which for a THINGS-EEG2 full-window slice is 8.3 GB on top of the
    4 GB float32 source -- and this runs once per window per shift replicate in
    parallel workers.  So the residuals are accumulated in row-chunks instead,
    and the per-concept mean is gathered by index rather than materialised with
    np.repeat.  Peak drops from ~19 GB to ~9 GB per worker.
    """
    n, T, p = E.shape
    Ebar = E.mean(1).astype(np.float64)                # (n, p)
    Ec = Ebar - Ebar.mean(0, keepdims=True)
    Wp = pca_basis(Ec, int(min(d_max, n - 1, p)))      # (p, d)
    Ep = Ec @ Wp

    Ef = E.reshape(n * T, p)                           # view, float32, no copy
    rows = np.arange(n * T) // T                       # concept index per trial
    d = Wp.shape[1]
    Rp = np.empty((n * T, d), np.float64)
    step = max(1, int(2.0e7 // max(p, 1)))             # ~160 MB float64 chunk
    for a in range(0, n * T, step):
        b = min(a + step, n * T)
        blk = Ef[a:b].astype(np.float64)
        blk -= Ebar[rows[a:b]]
        Rp[a:b] = blk @ Wp
    return Ep, Rp


def analyze_at_D(
    Ep_full: np.ndarray,
    Rp_full: np.ndarray,
    T: int,
    D: int,
    *,
    shrink: float = 0.05,
    n_rep: int = 6,
    n_sim_null: int = 300,
    n_boot: int = 0,
    n_shift: int = 0,
    k_grid: Optional[List[int]] = None,
    seed: int = 0,
) -> Dict:
    """All dimensionality statistics at one dimension budget D.

    Steps (each maps to a proposition in docs/03_theory_and_method.md):
      1. take the first D principal directions of the trial means        [P1]
      2. residual covariance from within-concept trial scatter           [P1]
      3. shrink + whiten so the noise in whitened coordinates is ~I      [P3]
      4. S_sig = Cov_i(whitened trial mean) - S_noise/T  (unbiased W)    [P1]
      5. r_uncorr = PR(Cov_i(whitened mean))   <- the "full rank" value  [P2]
         r_PR     = PR(psd(S_sig))             <- budget-dependent       [P2]
         r_screen = screening count            <- still budget-dependent
    """
    rng = np.random.default_rng(seed)
    n = Ep_full.shape[0]
    d = int(min(D, Ep_full.shape[1]))
    Ep = Ep_full[:, :d]
    Rp = Rp_full[:, :d]

    tr_mean = float(np.trace((Rp.T @ Rp) / max(Rp.shape[0] - 1, 1)) / d)
    S_res = (Rp.T @ Rp) / max(Rp.shape[0] - 1, 1)
    L = inv_sqrt_psd((1.0 - shrink) * S_res + shrink * tr_mean * np.eye(d))

    Ep_w = Ep @ L
    S_obs = cov(Ep_w)
    S_noise = L.T @ S_res @ L
    S_sig = psd_project(S_obs - S_noise / T)

    w_sig, V_sig = np.linalg.eigh(S_sig)
    order = np.argsort(w_sig)[::-1]
    w_sig, V_sig = w_sig[order], V_sig[:, order]

    # --- screening curve ---------------------------------------------------
    # never test k = d: the top-d space of a d-dimensional space is trivially
    # everything, so overlap = 1 and the statistic is degenerate there
    kg = [k for k in (k_grid or K_GRID) if k <= 0.75 * d]

    def _sub(idx: np.ndarray) -> np.ndarray:
        S = psd_project(cov(Ep_w[idx]) - S_noise / T)
        w, V = np.linalg.eigh(S)
        return V[:, np.argsort(w)[::-1]]

    ov = np.zeros((n_rep, len(kg)))
    for r_i in range(n_rep):
        perm = rng.permutation(n)
        h = n // 2
        QA = np.linalg.qr(_sub(perm[:h])[:, :kg[-1]])[0]
        QB = np.linalg.qr(_sub(perm[h:])[:, :kg[-1]])[0]
        for j, k in enumerate(kg):
            ov[r_i, j] = subspace_overlap(QA[:, :k], QB[:, :k])
    ov_mean = ov.mean(0)
    thr = np.array([null_q95(d, k, n_sim=n_sim_null, seed=seed) for k in kg])
    passing = [k for j, k in enumerate(kg) if ov_mean[j] > thr[j]]
    r_screen = int(max(passing)) if passing else 0

    # --- optional bootstrap for the PR-type quantities ---------------------
    boot_rpr, boot_unc = [], []
    if n_boot > 0:
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            Sb = cov(Ep_w[idx])
            boot_rpr.append(pr(psd_project(Sb - S_noise / T)))
            boot_unc.append(pr(Sb))

    def _ci(a, q):
        return float(np.quantile(a, q)) if len(a) else float("nan")

    return {
        "D": int(d),
        "r_screen": r_screen,
        "r_PR": pr(S_sig),
        "r_uncorr": pr(S_obs),
        "r_PR_over_d": pr(S_sig) / d,
        "r_uncorr_over_d": pr(S_obs) / d,
        "r_screen_over_d": r_screen / d,
        "r_PR_ci": [_ci(boot_rpr, 0.025), _ci(boot_rpr, 0.975)],
        "r_uncorr_ci": [_ci(boot_unc, 0.025), _ci(boot_unc, 0.975)],
        "signal_var_frac": float(S_sig.trace() / max(S_obs.trace(), 1e-300)),
        "k_grid": kg,
        "overlap": [float(x) for x in ov_mean],
        "null_q95": [float(x) for x in thr],
        "overlap_ratio": [float(ov_mean[j] / max(thr[j], 1e-12))
                          for j in range(len(kg))],
        "top_eig_sig": [float(x) for x in w_sig[:8]],
        "_Ep_w": Ep_w, "_V_sig": V_sig,
    }


def coverage(Ep_w: np.ndarray, V_hat: np.ndarray, C: np.ndarray,
             *, n_perm: int = 200, seed: int = 0) -> Dict:
    """Capacity-corrected coverage of the reproducible subspace (Prop. 5a).

    The model can produce any concept-space pattern in range(C) subset R^n; the
    reproducible subspace maps to span(Ep_w @ V_hat) subset R^n.  Hence
    cov = tr(P_repro P_model), and a random m-dim subspace has E[cov] = r*m/n.
    """
    rng = np.random.default_rng(seed)
    n, r = Ep_w.shape[0], V_hat.shape[1]
    if r == 0:
        return {"r": 0}
    Q_repro, _ = np.linalg.qr(Ep_w @ V_hat)
    Uc, sc, _ = np.linalg.svd(C, full_matrices=False)
    m = int((sc > sc[0] * 1e-8).sum())
    cov_obs = float(np.sum((Uc[:, :m].T @ Q_repro) ** 2))
    null = np.empty(n_perm)
    for b in range(n_perm):
        Up, sp, _ = np.linalg.svd(C[rng.permutation(n)], full_matrices=False)
        mp = int((sp > sp[0] * 1e-8).sum())
        null[b] = float(np.sum((Up[:, :mp].T @ Q_repro) ** 2))
    nm = float(null.mean())
    return {
        "r": int(r), "m_model": int(m),
        "cov": cov_obs, "cov_frac": cov_obs / max(r, 1),
        "null_mean": nm, "null_theory": float(r * m / n),
        "excess": cov_obs - nm,
        "excess_frac": (cov_obs - nm) / max(r, 1),
        "p_perm": float((null >= cov_obs).mean()),
        "uncovered_frac_capped": float(max(0.0, 1.0 - (cov_obs - nm) / max(r, 1))),
    }


# ---------------------------------------------------------------------------
# per-subject worker (top-level so ProcessPoolExecutor can pickle it)
# ---------------------------------------------------------------------------


def slice_window_feats(win: np.ndarray) -> np.ndarray:
    """(n, T, ch, nt) window -> (n, T, ch*nt), exactly as lib_windows.slice_eeg.

    Kept here so that a window produced by shifted_slice() gets the identical
    per-channel baseline removal and flattening as one produced by slice_eeg().
    The two paths must agree sample for sample or the control is not a control;
    test_shift_equivalence() asserts this.
    """
    W = win.astype(np.float32)
    W = W - W.mean(-1, keepdims=True)
    n, T, c, t = W.shape
    return W.reshape(n, T, c * t)


def _process_subject(kw: Dict) -> Tuple[str, Dict, List[str]]:
    """All windows for one subject.  Returns (sub, sub_out, log_lines).

    Everything here is independent across subjects, and a subject costs ~4 GB of
    source array plus a float64 working copy, so this is the natural unit of
    parallelism.  Output is returned as strings rather than printed so that the
    parent can emit it in a deterministic order.
    """
    sub = kw["sub"]
    whichs: List[str] = list(kw["which"])
    windows: List[str] = list(kw["windows"])
    d_grid: List[int] = list(kw["d_grid"])
    shrink: float = kw["shrink"]
    n_rep: int = kw["n_rep"]
    n_sim_null: int = kw["n_sim_null"]
    n_boot: int = kw["n_boot"]
    n_perm: int = kw["n_perm"]
    n_shift: int = kw["n_shift"]
    clips: List[str] = list(kw["clips"])
    seed: int = kw["seed"]
    win_map: Dict[str, Tuple[int, int]] = kw["win_map"]
    d_max: int = kw["d_max"]
    primary_D = kw["primary_D"]

    logs: List[str] = []
    sub_out: Dict = {"windows": {}}

    for which in whichs:
        try:
            E_all = load_eeg(sub, which)
        except FileNotFoundError:
            logs.append(f"  {sub}: no {which}.npy, skipped")
            continue
        n_eeg = E_all.shape[0]
        cons: Dict[str, np.ndarray] = {}
        if not kw["no_coverage"] and which == "train":
            for cn in clips:
                cons[cn] = load_clip(cn, which)[:n_eeg]

        for wname in windows:
            lo, hi = win_map[wname]
            key = f"{which}|{wname}"
            E = slice_eeg(E_all, lo, hi)
            T = E.shape[1]

            Ep, Rp = prep_projection(E, d_max)

            sweep: Dict = {}
            for D in sorted(d_grid, reverse=True):
                is_primary = (D == primary_D)
                sweep[D] = analyze_at_D(
                    Ep, Rp, T, D, shrink=shrink, n_rep=n_rep,
                    n_sim_null=n_sim_null,
                    n_boot=n_boot if is_primary else 0,
                    k_grid=K_GRID, seed=seed,
                )

            # ---- stimulus-locked control ----------------------------------
            # Circularly time-shifted EEG destroys stimulus locking but keeps the
            # spectrum, the drift and the dimensionality, so real - shift is the
            # only part of the count that can be attributed to the stimulus.
            if not kw["no_control"] and n_shift > 0:
                rng = np.random.default_rng(seed + 7)
                sh: List[Dict[int, int]] = []
                for _ in range(n_shift):
                    offs = sample_shift_offsets(E_all.shape[0], E_all.shape[-1],
                                                rng)
                    Es = slice_window_feats(shifted_slice(E_all, lo, hi, offs))
                    Eps, Rps = prep_projection(Es, d_max)
                    sh.append({
                        D: analyze_at_D(
                            Eps, Rps, T, D, shrink=shrink, n_rep=n_rep,
                            n_sim_null=n_sim_null, n_boot=0, k_grid=K_GRID,
                            seed=seed,
                        )["r_screen"]
                        for D in d_grid
                    })
                locked = {}
                for D in d_grid:
                    real = sweep[D]["r_screen"]
                    ctrl = [s[D] for s in sh]
                    locked[str(D)] = {
                        "real": int(real),
                        "shift_mean": float(np.mean(ctrl)),
                        "shift_all": ctrl,
                        "locked": int(real - np.mean(ctrl)),
                        "locked_frac_of_real": float(
                            (real - np.mean(ctrl)) / max(real, 1)),
                    }
                sweep["stimulus_locked_control"] = locked

            # ---- coverage (Claim 2) ---------------------------------------
            prim = sweep[primary_D]
            Ep_w = prim.pop("_Ep_w")
            V_sig = prim.pop("_V_sig")
            for D, rec in sweep.items():
                if isinstance(rec, dict) and "_Ep_w" in rec:
                    rec.pop("_Ep_w")
                    rec.pop("_V_sig")

            if cons and prim["r_screen"] > 0:
                Vhat = V_sig[:, :prim["r_screen"]]
                prim["coverage"] = {
                    cn: coverage(Ep_w, Vhat, cc, n_perm=n_perm, seed=seed)
                    for cn, cc in cons.items()
                }

            sub_out["windows"][key] = {"d_sweep": {str(k): v
                                                   for k, v in sweep.items()}}

            # ---- log ------------------------------------------------------
            logs.append(f"\n  {sub} {key}   p={E.shape[-1]} T={T}")
            logs.append(f"    {'D':>5} {'r_screen':>9} {'r_PR':>8} "
                        f"{'r_uncorr':>9} {'r_PR/D':>7} {'r_unc/D':>8}")
            for D in sorted(d_grid):
                s = sweep[D]
                logs.append(f"    {D:5d} {s['r_screen']:9d} {s['r_PR']:8.1f} "
                            f"{s['r_uncorr']:9.1f} {s['r_PR_over_d']:7.1%} "
                            f"{s['r_uncorr_over_d']:8.1%}")
            if "stimulus_locked_control" in sweep:
                logs.append("    stimulus-locked control (real - circular shift):")
                for D in sorted(d_grid):
                    L = sweep["stimulus_locked_control"][str(D)]
                    logs.append(f"      D={D:5d} real={L['real']:4d} "
                                f"shift={L['shift_mean']:6.1f} "
                                f"locked={L['locked']:4d} "
                                f"({L['locked_frac_of_real']:6.1%} of real)")
            for cn, cv in prim.get("coverage", {}).items():
                logs.append(f"    {cn}: cov={cv['cov']:6.2f} r={cv['r']:3d} "
                            f"null={cv['null_mean']:6.2f} "
                            f"(theory {cv['null_theory']:6.2f}) "
                            f"excess={cv['excess']:6.2f} "
                            f"uncovered={cv['uncovered_frac_capped']:6.1%} "
                            f"p={cv['p_perm']:.4f}")
    return sub, sub_out, logs


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",
                    default="/project/peilab/why/eeg-retrieval/alignment/outputs/a3b")
    ap.add_argument("--subs", nargs="+",
                    default=[f"sub-{i:02d}" for i in range(1, 11)])
    ap.add_argument("--windows", nargs="+", default=[w[0] for w in WINDOWS])
    ap.add_argument("--which", nargs="+", default=["train"])
    ap.add_argument("--clips", nargs="+", default=["ViT-H-14", "RN50"])
    ap.add_argument("--d-grid", nargs="+", type=int, default=D_GRID)
    ap.add_argument("--shrink", type=float, default=0.05)
    ap.add_argument("--n-rep", type=int, default=8)
    ap.add_argument("--n-sim-null", type=int, default=400)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--n-shift", type=int, default=3,
                    help="number of circular-shift control replicates")
    ap.add_argument("--no-control", action="store_true")
    ap.add_argument("--no-coverage", action="store_true")
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel workers over subjects. Subjects are "
                         "independent; each needs ~14 GB peak (4 GB source "
                         "array + a float64 working copy), so bound this by "
                         "memory as well as by cores.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    win_map = {w[0]: (w[1], w[2]) for w in WINDOWS}
    d_max = max(args.d_grid).__int__()

    print("A3b/A6 -- reproducible dimensionality and capacity-corrected coverage")
    print(f"  subjects {len(args.subs)}   windows {len(args.windows)}   "
          f"which {args.which}")
    print(f"  D sweep {sorted(args.d_grid)} (primary {args.d_grid[0]})")
    print(f"  n_rep={args.n_rep} n_sim_null={args.n_sim_null} "
          f"n_boot={args.n_boot} n_shift={args.n_shift}")
    print("  KEY: r_screen must SATURATE in D to be an intrinsic dimension;")
    print("       r_locked = real - circular-shift is the stimulus-locked part")
    print("=" * 120, flush=True)

    out: Dict = {"config": vars(args), "subjects": {}}
    t0 = time.time()

    # ---- parallel over subjects -------------------------------------------
    # The subjects are fully independent and a `normal` node has 224 cores and
    # ~2 TB, so the wall clock is set by the slowest subject rather than by the
    # sum.  Each worker holds ~4 GB of source array plus a float64 working copy,
    # so --jobs is bounded by memory as much as by cores.
    #
    # Pre-populate the random-frame null table BEFORE forking: the children
    # inherit it through the module-level cache, so the ~20 min of QR nulls is
    # paid once instead of once per worker.
    t_null = time.time()
    for D in sorted(args.d_grid):
        dd = int(min(D, d_max, 1654 - 1))
        kg = [k for k in K_GRID if k <= 0.75 * dd]
        if not kg:
            continue
        for k in kg:
            null_q95(dd, k, n_sim=args.n_sim_null, seed=args.seed)
    print(f"  null tables pre-computed in {time.time() - t_null:.1f}s "
          f"({len(_NULL_CACHE)} (d,k) entries)", flush=True)

    tasks = [dict(sub=s, which=args.which, windows=args.windows,
                  d_grid=args.d_grid, shrink=args.shrink, n_rep=args.n_rep,
                  n_sim_null=args.n_sim_null, n_boot=args.n_boot,
                  n_perm=args.n_perm, n_shift=args.n_shift,
                  no_control=args.no_control, no_coverage=args.no_coverage,
                  clips=args.clips, seed=args.seed, win_map=win_map,
                  d_max=d_max, primary_D=args.d_grid[0])
             for s in args.subs]

    jobs = max(1, min(args.jobs, len(tasks)))
    print(f"  running {len(tasks)} subjects with {jobs} worker(s)", flush=True)
    t_par = time.time()
    if jobs == 1:
        results = [_process_subject(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            results = list(ex.map(_process_subject, tasks))
    print(f"  subject loop done in {time.time() - t_par:.1f}s", flush=True)

    for sub, sub_out, logs in results:
        out["subjects"][sub] = sub_out
        print("\n" + "-" * 120, flush=True)
        for line in logs:
            print(line, flush=True)

    # ---- group summary across subjects -----------------------------------
    print("\n" + "=" * 120)
    print("GROUP SUMMARY  (mean +- sd across subjects)")
    keys = sorted({k for s in out["subjects"].values() for k in s["windows"]})
    summary = {}
    for k in keys:
        recs = [s["windows"][k]["d_sweep"] for s in out["subjects"].values()
                if k in s["windows"]]
        if not recs:
            continue
        entry: Dict = {"n_sub": len(recs), "D_grid": sorted(args.d_grid)}
        for tag in ("r_screen", "r_PR", "r_uncorr"):
            entry[tag] = {
                str(D): [float(np.mean([r[str(D)][tag] for r in recs])),
                         float(np.std([r[str(D)][tag] for r in recs]))]
                for D in args.d_grid
            }
        if "stimulus_locked_control" in recs[0]:
            entry["locked"] = {
                str(D): [float(np.mean([r["stimulus_locked_control"][str(D)]["locked"]
                                        for r in recs])),
                         float(np.std([r["stimulus_locked_control"][str(D)]["locked"]
                                       for r in recs]))]
                for D in args.d_grid
            }
        for cn in args.clips:
            ex = [r[str(args.d_grid[0])]["coverage"][cn]["excess"] for r in recs
                  if "coverage" in r[str(args.d_grid[0])]
                  and cn in r[str(args.d_grid[0])]["coverage"]]
            un = [r[str(args.d_grid[0])]["coverage"][cn]["uncovered_frac_capped"]
                  for r in recs if "coverage" in r[str(args.d_grid[0])]
                  and cn in r[str(args.d_grid[0])]["coverage"]]
            if ex:
                entry[f"{cn}_excess"] = float(np.mean(ex))
                entry[f"{cn}_uncovered"] = float(np.mean(un))
        summary[k] = entry

        print(f"\n  {k}   (n={entry['n_sub']})")
        for D in sorted(args.d_grid):
            line = (f"    D={D:5d}  r_screen={entry['r_screen'][str(D)][0]:7.1f}"
                    f"+-{entry['r_screen'][str(D)][1]:4.1f}"
                    f"   r_PR={entry['r_PR'][str(D)][0]:7.1f}"
                    f"   r_uncorr={entry['r_uncorr'][str(D)][0]:7.1f}")
            if "locked" in entry:
                line += (f"   locked={entry['locked'][str(D)][0]:7.1f}"
                         f"+-{entry['locked'][str(D)][1]:4.1f}")
            print(line)
        for cn in args.clips:
            if f"{cn}_excess" in entry:
                print(f"    {cn}: excess={entry[f'{cn}_excess']:6.2f} "
                      f"uncovered={entry[f'{cn}_uncovered']:6.1%}")
    out["summary"] = summary

    # ---- verdicts --------------------------------------------------------
    print("\n" + "=" * 120)
    dg = sorted(args.d_grid)
    def _slope(tag):
        """ratio of the largest-D value to the smallest-D value for a statistic"""
        v = [np.mean([s["windows"][k]["d_sweep"][str(D)][tag]
                      for s in out["subjects"].values()
                      for k in s["windows"]]) for D in dg]
        return float(v[-1] / max(v[0], 1e-9))
    try:
        slope_screen = _slope("r_screen")
        slope_pr = _slope("r_PR")
    except Exception:
        slope_screen = slope_pr = float("nan")
    unc_all = [v[f"{args.clips[0]}_uncovered"] for v in summary.values()
               if f"{args.clips[0]}_uncovered" in v]
    verdict = {
        "D_grid": dg,
        "slope_r_screen_max_over_min": slope_screen,
        "slope_r_PR_max_over_min": slope_pr,
        "CLAIM1_identifiable":
            "YES" if slope_screen < 2.0 else
            "NO_r_screen_tracks_budget_like_r_PR",
        "CLAIM2_uncovered_frac_mean": float(np.mean(unc_all)) if unc_all else None,
        "CLAIM2_verdict":
            ("SUPPORTED" if unc_all and np.mean(unc_all) > 0.5 else
             "PARTIAL" if unc_all and np.mean(unc_all) > 0.2 else
             "NOT_SUPPORTED" if unc_all else "NO_DATA"),
    }
    out["verdict"] = verdict
    for k, v in verdict.items():
        print(f"  {k}: {v}")

    path = os.path.join(args.out, "a3b_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("\n" + "=" * 120)
    print(f"A3b/A6 DONE.  wrote {path}   elapsed {time.time() - t0:.1f}s")
    print(f"VERDICT_LINE claim1_identifiable={verdict['CLAIM1_identifiable']} "
          f"slope_screen={slope_screen:.2f} slope_pr={slope_pr:.2f} "
          f"claim2_uncovered={verdict['CLAIM2_uncovered_frac_mean']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
