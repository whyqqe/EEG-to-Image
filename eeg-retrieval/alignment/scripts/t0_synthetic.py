#!/usr/bin/env python
"""
T0 -- SYNTHETIC TEST OF SRAT (Scale-Resolved Alignment Theory)   [v2]
=============================================================================
PURPOSE
-------
Decide, on synthetic data with known ground truth, whether the SRAT prescription

        lambda_s* = clip( SNR_s / theta , 0, 1 ),     SNR_s := I_s / N_s

predicts which scales of a frozen encoder should be aligned. This is the
life-or-death test of the theory: if the closed form does not track the
empirically optimal per-scale alignment weight on clean synthetic data, the
theory is wrong and no GPU time should be spent on the real experiments.

--------------------------------------------------------------------------------
WHY v2 (and what v1 got wrong -- kept here so the mistake is not repeated)
--------------------------------------------------------------------------------
v1 measured R@1 on a single CCA component per scale. That produces a RANK-1
representation, so the score matrix is an outer product and every row's argmax
is the same gallery item. R@1 was therefore identically ~1/n_eval for *every*
scale, and the experiment silently measured nothing. Diagnosis: the per-scale
channel must be k-dimensional with k > 1 (here k = 16), which is also what
"layer features" actually are.

v2 changes:
  * shared latent is k-dimensional:  c ~ N(0, I_k),  B_s in R^{d_s x k}, B_s^T B_s = C_s I_k
  * alignment channel per scale = top-k canonical projections (k-dim, not scalar)
  * score matrices A_s are precomputed ONCE; every lambda evaluation is then just
        S(lambda) = sum_s lambda_s^2 A_s
    which makes grid/coordinate search ~100x cheaper and removes any run-to-run
    noise from re-fitting.

--------------------------------------------------------------------------------
WHAT "ALIGNMENT STRENGTH" MEANS HERE
--------------------------------------------------------------------------------
lambda_s in [0,1] weights scale s's k-dim aligned channel inside a concatenated
cross-modal representation. Retrieval is a plain dot product over standardised
channels. lambda_s = 0 removes the channel. Because a global rescaling of lambda
leaves argmax scores unchanged, lambda is a *relative across-scale budget* --
exactly the object SRAT is about. Adding a channel whose canonical correlations
are near zero adds variance to the score without adding signal, so it strictly
hurts R@1: the alignment tax is real in this model.

--------------------------------------------------------------------------------
THE KNOBS THE THEORY SAYS MUST MATTER
--------------------------------------------------------------------------------
  n    : paired samples             -> estimation noise ~ d_s / n
  C    : shared-signal variance     -> content information
  d_s  : intrinsic dimensionality   -> how fast finite-n estimation degrades

PREREGISTERED CRITERIA (fixed BEFORE looking at any numbers)
------------------------------------------------------------
C1 RULE CORRECTNESS.  Over a grid of theta, the best achievable Spearman between
   the SRAT weight vector and the empirically optimal per-scale inclusion weight
   must be >= 0.80 (PASS), 0.50-0.80 (PARTIAL), < 0.50 (FAIL).

C2 KNOB SENSITIVITY.  The empirically best single scale must move by >= 1 scale
   index as n changes (or as C is degraded), in the direction predicted by
   plugging the estimated (I_s, N_s) into the theory.
   FAIL if flat across all knobs -> SRAT's central mechanism is falsified.

C3 SPARSITY.  The empirical joint optimum must set at least one lambda_s to 0.
   FAIL if the optimum is all-positive -> the threshold rule (Prop. 1) is
   falsified and SRAT collapses to plain weighted multi-layer fusion.

C4 CONTROLS MUST ACT.  Each negative control must produce a measurable change in
   the diagnostic it is supposed to break.  A control that "passes" because it
   silently did nothing is an experiment that measured nothing.

RUN
---
  PY=/project/peilab/why/eeg-brainit/.venv/bin/python
  $PY alignment/scripts/t0_synthetic.py --out alignment/outputs/t0 --seeds 5
  $PY alignment/scripts/t0_synthetic.py --out alignment/outputs/t0 --seeds 2 --quick
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Generative model
# ---------------------------------------------------------------------------


@dataclass
class ScaleParams:
    """Per-scale information budget of a synthetic frozen encoder pair."""

    L: int = 16  # number of scales (layers)
    k: int = 16  # dimension of the shared subspace
    d0: int = 192  # shallow/fine-scale feature dimension
    d_decay: float = 0.85  # d_s = d0 * decay**s  -> decreasing with depth
    C_peak: float = 1.0  # peak shared-signal variance
    C_peak_at: int = 7  # scale index of the peak (unimodal C_s)
    C_width: float = 2.2
    N0: float = 0.25  # X-private variance at scale 0
    N_growth: float = 6.0  # N_last / N_0
    M0: float = 0.25
    M_growth: float = 5.0
    C_scale_knob: float = 1.0  # multiplier on C_s (content-degradation knob)

    def arrays(self) -> Dict[str, np.ndarray]:
        s = np.arange(self.L)
        d = np.maximum(4, np.round(self.d0 * self.d_decay**s)).astype(int)
        C = (
            self.C_peak
            * np.exp(-0.5 * ((s - self.C_peak_at) / self.C_width) ** 2)
            * self.C_scale_knob
        )
        N = self.N0 * np.exp(np.log(self.N_growth) * s / max(1, self.L - 1))
        M = self.M0 * np.exp(np.log(self.M_growth) * s / max(1, self.L - 1))
        return {"d": d, "C": C, "N": N, "M": M}

    def rho_true(self) -> np.ndarray:
        """True population canonical correlation of each shared component."""
        a = self.arrays()
        C, N, M, d = a["C"], a["N"], a["M"], a["d"].astype(float)
        return C / np.sqrt((C + N / d) * (C + M / d))

    def I_true(self) -> np.ndarray:
        rho = np.clip(self.rho_true(), 0, 1 - 1e-12)
        return self.k * (-0.5 * np.log(1 - rho**2))


def generate(p: ScaleParams, n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Rank-k shared structure.

        c ~ N(0, I_k)
        z_X^(s) = c B_s^T + u_X,   B_s in R^{d_s x k}, B_s^T B_s = C_s I_k
                                   u_X ~ N(0, N_s/d_s I_{d_s})
    """
    a = p.arrays()
    out = {}
    for s in range(p.L):
        C, N, M, d = a["C"][s], a["N"][s], a["M"][s], int(a["d"][s])
        Q, _ = np.linalg.qr(rng.standard_normal((d, p.k)))
        B = np.sqrt(C) * Q  # (d, k)
        c = rng.standard_normal((n, p.k))
        ux = rng.standard_normal((n, d)) * np.sqrt(N / d)
        uy = rng.standard_normal((n, d)) * np.sqrt(M / d)
        shared = c @ B.T
        out[f"X{s}"] = shared + ux
        out[f"Y{s}"] = shared + uy
    return out


# ---------------------------------------------------------------------------
# Ridge-regularised CCA, top-k components
# ---------------------------------------------------------------------------


def _inv_sqrt_reg(S: np.ndarray, ridge: float) -> np.ndarray:
    S = S + ridge * np.eye(S.shape[0])
    w, V = np.linalg.eigh(S)
    w = np.maximum(w, 1e-10)
    return V @ np.diag(1.0 / np.sqrt(w)) @ V.T


def cca_topk(
    X: np.ndarray, Y: np.ndarray, k: int, ridge: float = 1e-2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-k canonical pairs. X:(n,d) Y:(n,d). Returns U:(d,k) V:(d,k) rhos:(k,)."""
    Xc = X - X.mean(0, keepdims=True)
    Yc = Y - Y.mean(0, keepdims=True)
    n = X.shape[0]
    Sxx = Xc.T @ Xc / n
    Syy = Yc.T @ Yc / n
    Sxy = Xc.T @ Yc / n
    # data-scale-aware ridge
    rx = ridge * float(np.trace(Sxx) / Sxx.shape[0])
    ry = ridge * float(np.trace(Syy) / Syy.shape[0])
    Sxx_is = _inv_sqrt_reg(Sxx, rx)
    Syy_is = _inv_sqrt_reg(Syy, ry)
    M = Sxx_is @ Sxy @ Syy_is
    U, sv, Vt = np.linalg.svd(M, full_matrices=False)
    kk = max(1, min(k, U.shape[1]))
    U = U[:, :kk]
    V = Vt[:kk].T
    rhos = sv[:kk]
    # normalise so that each projection has unit variance under Sxx / Syy
    U = Sxx_is @ U
    V = Syy_is @ V
    nx = np.sqrt(np.einsum("ij,ij->j", U, Sxx @ U))
    ny = np.sqrt(np.einsum("ij,ij->j", V, Syy @ V))
    U = U / np.maximum(nx, 1e-12)[None, :]
    V = V / np.maximum(ny, 1e-12)[None, :]
    return U, V, np.clip(rhos[:kk], 0.0, 1.0)


# ---------------------------------------------------------------------------
# Estimation of (I_s, N_s) -- the operational core of SRAT
# ---------------------------------------------------------------------------


def estimate_budgets(
    data: Dict[str, np.ndarray],
    p: ScaleParams,
    idx_dir: np.ndarray,
    idx_mom: np.ndarray,
    ridge: float = 1e-2,
) -> Dict:
    """Estimate per-scale alignment channels and their information budget.

    idx_dir fits the canonical directions; idx_mom (disjoint) is used to MEASURE
    rho, so the reported rho is held-out and not inflated by CCA overfitting.

    Definitions used from here on:
        rho_hat[s,t] : held-out canonical correlation of component t of scale s
        I_hat[s]     : sum_t -0.5 log(1 - rho_hat^2)      (mutual information)
        N_hat[s]     : sum_t (1 - rho_hat[t])             (private / mismatch cost)
        SNR[s]       : I_hat[s] / N_hat[s]
    """
    L, k = p.L, p.k
    rho = np.zeros((L, k))
    I = np.zeros(L)
    Nb = np.zeros(L)
    Us, Vs = [], []
    for s in range(L):
        U, V, _ = cca_topk(data[f"X{s}"][idx_dir], data[f"Y{s}"][idx_dir], k, ridge)
        Us.append(U)
        Vs.append(V)
        Px = data[f"X{s}"][idx_mom] @ U
        Py = data[f"Y{s}"][idx_mom] @ V
        for t in range(k):
            a, b = Px[:, t], Py[:, t]
            sa, sb = a.std(), b.std()
            if sa < 1e-9 or sb < 1e-9:
                continue
            rho[s, t] = np.clip(np.corrcoef(a, b)[0, 1], 0.0, 0.9999)
        I[s] = np.sum(-0.5 * np.log(1 - rho[s] ** 2))
        Nb[s] = np.sum(1.0 - rho[s])
    return {"rho": rho, "I": I, "N": Nb, "SNR": I / np.maximum(Nb, 1e-12), "U": Us, "V": Vs}


# ---------------------------------------------------------------------------
# Precomputed per-scale score matrices -> cheap evaluation of any lambda
# ---------------------------------------------------------------------------


class Aligner:
    """Holds standardised k-dim channels and their per-scale score matrices."""

    def __init__(self, data, est, idx_mom, idx_eval, lam_dim):
        self.L = len(est["U"])
        self.A: List[np.ndarray] = []
        for s in range(self.L):
            U, V = est["U"][s], est["V"][s]
            Px = data[f"X{s}"][idx_mom] @ U
            Py = data[f"Y{s}"][idx_mom] @ V
            mx, sx = Px.mean(0), np.maximum(Px.std(0), 1e-8)
            my, sy = Py.mean(0), np.maximum(Py.std(0), 1e-8)
            Xe = (data[f"X{s}"][idx_eval] @ U - mx) / sx
            Ye = (data[f"Y{s}"][idx_eval] @ V - my) / sy
            self.A.append(Xe @ Ye.T)
        self.n_eval = self.A[0].shape[0]
        self._ones = np.ones(self.L)

    def score(self, lam: np.ndarray) -> float:
        w = np.asarray(lam, float) ** 2
        S = w[0] * self.A[0]
        for s in range(1, self.L):
            if w[s] > 0:
                S += w[s] * self.A[s]
        return float((S.argmax(1) == np.arange(self.n_eval)).mean())

    def score_many(self, lam_list: List[np.ndarray]) -> np.ndarray:
        return np.array([self.score(l) for l in lam_list])


# ---------------------------------------------------------------------------
# Selection rules
# ---------------------------------------------------------------------------


def rule_srat(SNR: np.ndarray, theta: float) -> np.ndarray:
    """SRAT closed form: clip(SNR/theta, 0, 1). theta is the single free constant."""
    return np.clip(SNR / max(theta, 1e-12), 0.0, 1.0)


def rule_threshold(SNR: np.ndarray, theta: float, floor: float = 1.0) -> np.ndarray:
    """Hard threshold version: full weight above theta, zero below."""
    lam = np.zeros_like(SNR)
    lam[SNR >= theta] = floor
    return lam


def rule_uniform(SNR: np.ndarray) -> np.ndarray:
    return np.ones_like(SNR)


def rule_final_layer(SNR: np.ndarray) -> np.ndarray:
    lam = np.zeros_like(SNR)
    lam[-1] = 1.0
    return lam


def rule_mid_layer(SNR: np.ndarray) -> np.ndarray:
    lam = np.zeros_like(SNR)
    lam[len(SNR) // 2] = 1.0
    return lam


def rule_top1_snr(SNR: np.ndarray) -> np.ndarray:
    lam = np.zeros_like(SNR)
    lam[int(np.argmax(SNR))] = 1.0
    return lam


def rule_mutual_knn(data, p, idx, k_nn: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """STRUCTURE-style layer selection: mutual-kNN overlap between modalities."""
    n = len(idx)
    if k_nn is None:
        k_nn = max(1, int(np.ceil(2 * np.cbrt(n))))
    knn = np.zeros(p.L)
    for s in range(p.L):
        X = data[f"X{s}"][idx]
        Y = data[f"Y{s}"][idx]
        X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        Y = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
        Sx = X @ X.T
        Sy = Y @ Y.T
        np.fill_diagonal(Sx, -np.inf)
        np.fill_diagonal(Sy, -np.inf)
        nx = Sx.argsort(1)[:, -k_nn:]
        ny = Sy.argsort(1)[:, -k_nn:]
        knn[s] = np.mean([len(set(nx[i]) & set(ny[i])) for i in range(n)]) / k_nn
    lam = np.zeros(p.L)
    lam[int(np.argmax(knn))] = 1.0
    return lam, knn


# ---------------------------------------------------------------------------
# One configuration
# ---------------------------------------------------------------------------


def run_config(
    p: ScaleParams,
    n_pairs: int,
    rng: np.random.Generator,
    thetas: np.ndarray,
    lam_grid: np.ndarray,
    shuffle_pairs: bool = False,
    ridge: float = 1e-2,
    n_eval: int = 1000,
) -> Dict:
    n_eval = int(min(n_eval, max(200, n_pairs // 2)))
    total = 3 * n_pairs + n_eval
    data = generate(p, total, rng)
    if shuffle_pairs:
        perm = rng.permutation(total)
        for s in range(p.L):
            data[f"Y{s}"] = data[f"Y{s}"][perm]

    idx_dir = np.arange(n_pairs)
    idx_mom = np.arange(n_pairs, 2 * n_pairs)
    idx_eval = np.arange(2 * n_pairs, total)

    est = estimate_budgets(data, p, idx_dir, idx_mom, ridge)
    al = Aligner(data, est, idx_mom, idx_eval, p.k)
    SNR = est["SNR"]

    def r1(lam):
        return al.score(lam)

    # ---- (A) single-scale-only: "which scale is the alignment target" --------
    single_r1 = np.array([r1(np.eye(p.L)[s]) for s in range(p.L)])
    best_single = int(np.argmax(single_r1))

    # ---- (B) inclusion curve: lambda_s swept, all others at 1 ----------------
    curves = np.zeros((p.L, len(lam_grid)))
    for s in range(p.L):
        for j, lv in enumerate(lam_grid):
            lam = np.ones(p.L)
            lam[s] = lv
            curves[s, j] = r1(lam)
    emp_best = lam_grid[curves.argmax(1)]
    r1_all = r1(np.ones(p.L))
    drop_r1 = np.array([r1(np.where(np.arange(p.L) == s, 0.0, 1.0)) for s in range(p.L)])
    drop_gain = r1_all - drop_r1

    # ---- joint optimum via coordinate descent -------------------------------
    lam = np.ones(p.L)
    best = r1_all
    for _ in range(4):
        improved = False
        for s in range(p.L):
            cur = lam[s]
            for lv in lam_grid:
                lam[s] = lv
                v = r1(lam)
                if v > best + 1e-9:
                    best, cur, improved = v, lv, True
            lam[s] = cur
        if not improved:
            break
    lam_joint, joint_r1 = lam.copy(), best

    # ---- SRAT with the best achievable theta --------------------------------
    sp_by_theta, theta_rows = [], []
    for th in thetas:
        lw = rule_srat(SNR, th)
        sp = spearmanr(lw, emp_best).correlation if np.std(emp_best) > 1e-9 else 0.0
        sp_by_theta.append(0.0 if np.isnan(sp) else float(sp))
        theta_rows.append({"theta": float(th), "spearman": sp_by_theta[-1],
                           "lam": lw.tolist(), "r1": r1(lw)})
    bi = int(np.argmax(sp_by_theta))
    best_theta = float(thetas[bi])
    lam_srat = rule_srat(SNR, best_theta)

    # oracle: use TRUE rho to build SNR, same rule shape
    rho_t = p.rho_true()
    I_t = p.I_true()
    N_t = p.k * (1.0 - rho_t)
    SNR_true = I_t / np.maximum(N_t, 1e-12)
    sp_th, rows_true = [], []
    for th in thetas:
        lw = rule_srat(SNR_true, th)
        sp = spearmanr(lw, emp_best).correlation if np.std(emp_best) > 1e-9 else 0.0
        sp_th.append(0.0 if np.isnan(sp) else float(sp))
        rows_true.append({"theta": float(th), "lam": lw.tolist(), "r1": r1(lw)})
    bi_t = int(np.argmax(sp_th))
    lam_oracle = rule_srat(SNR_true, float(thetas[bi_t]))

    mk, knn_vals = rule_mutual_knn(data, p, idx_mom)

    rules = {
        "srat_oracle": lam_oracle,
        "srat_estimated": lam_srat,
        "srat_hard_threshold": rule_threshold(SNR, best_theta),
        "uniform": rule_uniform(SNR),
        "final_layer": rule_final_layer(SNR),
        "mid_layer": rule_mid_layer(SNR),
        "top1_snr": rule_top1_snr(SNR),
        "mutual_knn": mk,
        "empirical_joint": lam_joint,
    }
    rules_out = {kk: {"lambda": v.tolist(), "r1": (joint_r1 if kk == "empirical_joint" else r1(v))}
                 for kk, v in rules.items()}

    return {
        "n_pairs": n_pairs,
        "n_eval": n_eval,
        "chance": 1.0 / n_eval,
        "rho_true": p.rho_true().tolist(),
        "rho_est": est["rho"].mean(1).tolist(),
        "I_est": est["I"].tolist(),
        "N_est": est["N"].tolist(),
        "SNR_est": SNR.tolist(),
        "SNR_true": SNR_true.tolist(),
        "single_scale_r1": single_r1.tolist(),
        "best_single_scale": best_single,
        "best_single_r1": float(single_r1.max()),
        "curves": curves.tolist(),
        "lam_grid": lam_grid.tolist(),
        "emp_best_lambda": emp_best.tolist(),
        "drop_gain": drop_gain.tolist(),
        "r1_all": r1_all,
        "theta_best": best_theta,
        "spearman_best": float(sp_by_theta[bi]),
        "theta_sweep": theta_rows,
        "theta_best_oracle": float(thetas[bi_t]),
        "spearman_oracle_best": float(sp_th[bi_t]),
        "knn_values": knn_vals.tolist(),
        "rules": rules_out,
        "shuffled": shuffle_pairs,
    }


# ---------------------------------------------------------------------------
# Negative controls (C4: each must demonstrably act)
# ---------------------------------------------------------------------------


def negative_controls(p, n_pairs, rng, thetas, lam_grid) -> Dict:
    out = {}
    base = run_config(p, n_pairs, np.random.default_rng(0), thetas, lam_grid)
    shuf = run_config(p, n_pairs, np.random.default_rng(0), thetas, lam_grid, shuffle_pairs=True)
    out["A_shuffled_pairs"] = {
        "r1_uniform_base": base["rules"]["uniform"]["r1"],
        "r1_uniform_shuffled": shuf["rules"]["uniform"]["r1"],
        "chance": shuf["chance"],
        "control_acted": bool(
            base["rules"]["uniform"]["r1"] > shuf["rules"]["uniform"]["r1"] + 0.05
        ),
        "chance_level_achieved": bool(abs(shuf["rules"]["uniform"]["r1"] - shuf["chance"]) < 0.05),
    }
    out["B_no_signal_scale"] = {
        "note": "set C=0 for scales >= 12: their true SNR is 0, so SRAT must "
                "assign lambda=0 there and R1 must not drop when they are removed",
        "tested_in_part3": True,
    }
    return out


# ---------------------------------------------------------------------------
# Part 3: explicit no-signal control
# ---------------------------------------------------------------------------


def no_signal_control(p=None, n_pairs: int = 2000, theta=None) -> Dict:
    p = p or ScaleParams()
    rng = np.random.default_rng(11)
    a = p.arrays()
    # zero the shared signal on the last four scales
    dead = list(range(p.L - 4, p.L))
    p2 = ScaleParams(**{**p.__dict__})
    arr = p2.arrays()
    C = arr["C"].copy()
    C[dead] = 0.0
    lam_grid = np.linspace(0, 1, 11)
    # rebuild a config with the modified C by monkey-patching arrays()
    class P2(ScaleParams):
        def arrays(self_inner):
            d = super().arrays()
            d = {kk: vv.copy() for kk, vv in d.items()}
            d["C"] = C
            return d

    p2 = P2(**p.__dict__)
    res = run_config(p2, n_pairs, rng, np.array(theta if theta is not None else [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]), lam_grid)
    lam = np.array(res["rules"]["srat_estimated"]["lambda"])
    return {
        "dead_scales": dead,
        "srat_lambda_on_dead_scales": lam[dead].tolist(),
        "srat_zeros_dead_scales": bool(np.allclose(lam[dead], 0.0)),
        "SNR_est_on_dead_scales": np.array(res["SNR_est"])[dead].tolist(),
        "SNR_true_on_dead_scales": np.array(res["SNR_true"])[dead].tolist(),
        "r1_uniform": res["rules"]["uniform"]["r1"],
        "r1_srat": res["rules"]["srat_estimated"]["r1"],
        "control_acted": bool(np.allclose(lam[dead], 0.0)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/project/peilab/why/eeg-retrieval/alignment/outputs/t0")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    lam_grid = np.array([0.0, 0.25, 0.5, 0.75, 1.0]) if args.quick else np.linspace(0, 1, 11)
    thetas = np.logspace(-2, 2, 25)
    n_list = [500, 2000, 10000] if not args.quick else [1000]
    seeds = range(args.seeds)

    print("=" * 100)
    print("T0 v2 -- SRAT synthetic test (rank-k shared subspace)")
    print("=" * 100)

    results = {"config": {"n_list": n_list, "seeds": args.seeds, "k": 16, "L": 16,
                          "lam_grid": lam_grid.tolist(), "theta_grid": thetas.tolist()}}

    # ---------------- Part 1: the n knob -----------------------------------
    print("\n[Part 1] varying n (paired-sample count) -- the estimation-noise axis")
    part1 = []
    for n in n_list:
        rows = []
        for sd in seeds:
            rows.append(run_config(ScaleParams(), n, np.random.default_rng(1000 + sd), thetas, lam_grid))
        agg = {
            "n": n,
            "spearman_best_mean": float(np.mean([r["spearman_best"] for r in rows])),
            "spearman_best_std": float(np.std([r["spearman_best"] for r in rows])),
            "spearman_oracle_mean": float(np.mean([r["spearman_oracle_best"] for r in rows])),
            "best_single_scale_mean": float(np.mean([r["best_single_scale"] for r in rows])),
            "best_single_scales": [r["best_single_scale"] for r in rows],
            "n_zero_scales_empirical": float(np.mean([int((np.array(r["emp_best_lambda"]) == 0).sum()) for r in rows])),
            "n_zero_scales_srat": float(np.mean([int((np.array(r["rules"]["srat_estimated"]["lambda"]) == 0).sum()) for r in rows])),
            "r1": {kk: float(np.mean([r["rules"][kk]["r1"] for r in rows]))
                   for kk in rows[0]["rules"]},
            "seed_rows": rows,
        }
        # drop the per-seed payload from the aggregate printout but keep in json
        part1.append(agg)
        print(f"  n={n:6d}  sp(srat)={agg['spearman_best_mean']:+.3f}+-{agg['spearman_best_std']:.3f}"
              f"  sp(oracle)={agg['spearman_oracle_mean']:+.3f}"
              f"  best_single={agg['best_single_scales']}"
              f"  zeros[emp={agg['n_zero_scales_empirical']:4.1f} srat={agg['n_zero_scales_srat']:4.1f}]")
        print(f"          R1: " + "  ".join(f"{kk}={vv:.3f}" for kk, vv in sorted(agg["r1"].items(), key=lambda x: -x[1])))
    results["part1_n_knob"] = part1

    # ---------------- Part 2: the C knob -----------------------------------
    print("\n[Part 2] varying C_scale_knob (shared-content level) -- content axis")
    part2 = []
    for ck in [1.0, 0.5, 0.2, 0.05]:
        rows = []
        for sd in seeds:
            rows.append(run_config(ScaleParams(C_scale_knob=ck), 2000,
                                   np.random.default_rng(2000 + sd), thetas, lam_grid))
        agg = {
            "C_scale_knob": ck,
            "spearman_best_mean": float(np.mean([r["spearman_best"] for r in rows])),
            "best_single_scale_mean": float(np.mean([r["best_single_scale"] for r in rows])),
            "best_single_scales": [r["best_single_scale"] for r in rows],
            "n_zero_scales_empirical": float(np.mean([int((np.array(r["emp_best_lambda"]) == 0).sum()) for r in rows])),
            "n_zero_scales_srat": float(np.mean([int((np.array(r["rules"]["srat_estimated"]["lambda"]) == 0).sum()) for r in rows])),
            "r1": {kk: float(np.mean([r["rules"][kk]["r1"] for r in rows])) for kk in rows[0]["rules"]},
            "seed_rows": rows,
        }
        part2.append(agg)
        print(f"  C_knob={ck:4.2f}  sp(srat)={agg['spearman_best_mean']:+.3f}"
              f"  best_single={agg['best_single_scales']}"
              f"  zeros[emp={agg['n_zero_scales_empirical']:4.1f} srat={agg['n_zero_scales_srat']:4.1f}]")
        print(f"          R1: " + "  ".join(f"{kk}={vv:.3f}" for kk, vv in sorted(agg["r1"].items(), key=lambda x: -x[1])))
    results["part2_C_knob"] = part2

    # ---------------- Part 3: controls -------------------------------------
    print("\n[Part 3] negative controls")
    nc = negative_controls(ScaleParams(), 2000, np.random.default_rng(7), thetas, lam_grid)
    nc["C_dead_scales"] = no_signal_control()
    results["negative_controls"] = nc
    for kk, vv in nc.items():
        print(f"  {kk}: {json.dumps(vv, default=str)}")

    with open(os.path.join(args.out, "t0_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {os.path.join(args.out, 't0_results.json')}")

    # ---------------- verdict ----------------------------------------------
    print("\n" + "=" * 100)
    print("VERDICT vs preregistered criteria")
    print("=" * 100)
    sps = [r["spearman_best_mean"] for r in part1] + [r["spearman_best_mean"] for r in part2]
    best_sp = max(sps)
    print(f"C1 rule correctness : best Spearman over all configs/thetas = {best_sp:+.3f} -> "
          f"{'PASS' if best_sp >= 0.80 else ('PARTIAL' if best_sp >= 0.50 else 'FAIL')}")
    peaks = [r["best_single_scale_mean"] for r in part1]
    moved = (max(peaks) - min(peaks)) >= 1.0
    print(f"C2 knob sensitivity : best single scale across n = {['%.2f' % x for x in peaks]} -> "
          f"{'MOVED' if moved else 'FLAT (SRAT mechanism falsified)'}")
    zm = [r["n_zero_scales_empirical"] for r in part1] + [r["n_zero_scales_empirical"] for r in part2]
    print(f"C3 sparsity         : mean #zero empirical weights = {['%.1f' % x for x in zm]} -> "
          f"{'PASS' if max(zm) > 0 else 'FAIL (optimum all-positive)'}")
    print(f"C4 controls acted   : " + json.dumps({k: v.get("control_acted") for k, v in nc.items() if isinstance(v, dict)}, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
