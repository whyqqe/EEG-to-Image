#!/usr/bin/env python
"""Diagnostic dump for T0 -- understand the structure before scaling up."""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t0_synthetic import (  # noqa: E402
    ScaleParams, generate, estimate_budgets, prepare_scale_stats,
    build_reps, retrieval_r1, mi_from_budgets, cca_top1,
)

np.set_printoptions(precision=4, suppress=True, linewidth=200)

p = ScaleParams()
a = p.arrays()
L = p.L
lam_grid = np.linspace(0, 1, 11)

N_PAIRS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
CKN = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
p = ScaleParams(C_scale_knob=CKN)

n_eval = min(1000, max(200, N_PAIRS // 2))
total = N_PAIRS + N_PAIRS + n_eval
rng = np.random.default_rng(SEED)
data = generate(p, total, rng)
idx_dir = np.arange(N_PAIRS)
idx_mom = np.arange(N_PAIRS, 2 * N_PAIRS)
idx_eval = np.arange(2 * N_PAIRS, total)

est = estimate_budgets(data, idx_dir, idx_mom, p)
stats = prepare_scale_stats(data, idx_mom, est["dirs"])

C, N, M, rho = est["C"], est["N"], est["M"], est["rho"]
I = mi_from_budgets(C, N, M)

# true population canonical correlations
C_t, N_t, M_t, d_t = a["C"], a["N"], a["M"], a["d"].astype(float)
rho_true = C_t / np.sqrt((C_t + N_t / d_t) * (C_t + M_t / d_t))

# ---- (A) single-scale-only retrieval: "which scale is the best alignment target"
single_r1 = np.zeros(L)
for s in range(L):
    lam = np.zeros(L); lam[s] = 1.0
    AX, AY = build_reps(data, idx_eval, est["dirs"], stats, lam)
    single_r1[s] = retrieval_r1(AX, AY)

# ---- (B) per-scale benefit curve given others ON: "should this scale be included"
curves = np.zeros((L, len(lam_grid)))
for s in range(L):
    for j, lv in enumerate(lam_grid):
        lam = np.ones(L); lam[s] = lv
        AX, AY = build_reps(data, idx_eval, est["dirs"], stats, lam)
        curves[s, j] = retrieval_r1(AX, AY)
emp_best = lam_grid[curves.argmax(1)]

# ---- (C) leave-one-out: how much does dropping scale s hurt?
loo = np.zeros(L)
for s in range(L):
    lam = np.ones(L); lam[s] = 0.0
    AX, AY = build_reps(data, idx_eval, est["dirs"], stats, lam)
    loo[s] = retrieval_r1(AX, AY)
r1_all = retrieval_r1(*build_reps(data, idx_eval, est["dirs"], stats, np.ones(L)))

print("=" * 150)
print(f"n_pairs={N_PAIRS} seed={SEED} C_knob={CKN}  n_eval={n_eval}  chance={1/n_eval:.4f}")
print(f"R1 (all scales on) = {r1_all:.4f}")
print("=" * 150)
hdr = f"{'s':>3} {'d':>4} {'C_true':>9} {'N_true':>8} {'M_true':>8} {'rho_true':>9} {'rho_est':>8} {'I_est':>8} {'I/N':>8} | {'single':>7} {'best_lam':>8} {'loo':>7}"
print(hdr); print("-" * len(hdr))
for s in range(L):
    print(f"{s:>3} {int(d_t[s]):>4} {C_t[s]:>9.5f} {N_t[s]:>8.4f} {M_t[s]:>8.4f} "
          f"{rho_true[s]:>9.4f} {rho[s]:>8.4f} {I[s]:>8.4f} {I[s]/max(N[s],1e-9):>8.4f} | "
          f"{single_r1[s]:>7.4f} {emp_best[s]:>8.2f} {loo[s]:>7.4f}")

print()
print(f"{'BEST single-scale target':<26}: s={int(np.argmax(single_r1))} (R1={single_r1.max():.4f})")
print(f"{'WORST (most droppable)':<26}: s={int(np.argmin(loo))} (loo R1={loo.min():.4f}, drop gain={r1_all-loo.min():+.4f})")
print(f"{'theory I/N argmax':<26}: s={int(np.argmax(I/np.maximum(N,1e-9)))}")

print()
print("theory lambda* for several c:")
for c in [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
    lam = np.clip(I / (c * np.maximum(N, 1e-12)), 0, 1)
    print(f"  c={c:5.2f}  zeros={int((lam==0).sum()):2d}  lam={np.round(lam,2)}")

# correlation between theory and empirical
from scipy.stats import spearmanr
for c in [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]:
    lam = np.clip(I / (c * np.maximum(N, 1e-12)), 0, 1)
    sp = spearmanr(lam, emp_best).correlation
    sp2 = spearmanr(lam, single_r1).correlation
    print(f"  c={c:5.2f}  spearman(lam*, emp_best_lam)={sp:+.3f}   spearman(lam*, single_r1)={sp2:+.3f}")
