#!/usr/bin/env python
"""
Verify the noise-ceiling CONVENTION.  This resolves a contradiction in the A2 output.

THE CONTRADICTION
    A2 reported RN50 RSA = 0.0474 against a full-window ceiling of 0.0209, i.e.
    226% of the ceiling.  A model cannot exceed the noise ceiling, so either the
    ceiling formula is wrong or the two quantities are not comparable.

WHY sqrt(reliability) IS THE CEILING
    Classical test theory: R_obs is a noisy measurement of a true RDM R whose
    reliability is rho (= split-half correlation, Spearman-Brown corrected to full
    length).  A MODEL RDM that perfectly reproduced R would still only correlate
    with R_obs by
        corr(R_obs, R_model) = corr(R_obs, R) = sqrt(rho)
    because the noise in R_obs is independent of the model.  So the ceiling is
    sqrt(rho), NOT rho.  Reporting rho understates the ceiling and lets RSA scores
    appear to exceed 100%.

WHAT THIS SCRIPT DOES
    Simulates RDMs with a KNOWN true RDM and known noise, then measures
        A = corr(R_obs, R_true)          the actual attainable ceiling
        rho = split-half reliability     (SB corrected)
    and checks A vs sqrt(rho) and A vs rho.  This is a direct empirical answer to
    which convention is right, rather than an appeal to authority.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_rsca import rdm, spearman_brown, upper_tri  # noqa: E402


def main() -> int:
    rng = np.random.default_rng(0)
    print("=" * 96)
    print("Noise-ceiling convention check")
    print("=" * 96)

    # Concept-level feature space: n concepts, d dims. The true RDM comes from the
    # noiseless features; trials add independent per-trial noise.
    n, d, T = 300, 60, 10          # T must be even for split-half
    n_rep = 40

    print(f"  n_concepts={n}  d={d}  T_trials={T}  reps={n_rep}")
    print(f"\n  {'noise_sd':>9} {'A=corr(obs,true)':>17} {'rho(SB)':>9} {'sqrt(rho)':>10} "
          f"{'A/rho':>7} {'A/sqrt(rho)':>12}")

    for noise_sd in [0.0, 0.5, 1.0, 2.0, 4.0]:
        A_list, rho_list = [], []
        for _ in range(n_rep):
            F_true = rng.standard_normal((n, d))
            R_true = rdm(F_true)

            # trials = true features + noise; the "signal" is the concept mean
            trials = F_true[:, None, :] + noise_sd * rng.standard_normal((n, T, d))
            F_obs = trials.mean(1)
            R_obs = rdm(F_obs)
            # split-half
            h1 = trials[:, : T // 2].mean(1)
            h2 = trials[:, T // 2:].mean(1)
            r12 = spearmanr(upper_tri(rdm(h1)), upper_tri(rdm(h2))).correlation
            rho = spearman_brown(r12)

            A = spearmanr(upper_tri(R_obs), upper_tri(R_true)).correlation
            A_list.append(A)
            rho_list.append(rho)

        A_m, rho_m = float(np.mean(A_list)), float(np.mean(rho_list))
        print(f"  {noise_sd:>9.1f} {A_m:>17.4f} {rho_m:>9.4f} {np.sqrt(max(rho_m,0)):>10.4f} "
              f"{A_m/max(rho_m,1e-9):>7.2f} {A_m/max(np.sqrt(max(rho_m,0)),1e-9):>12.2f}")

    print("""
  INTERPRETATION
    A/rho should be ~1 if rho were the ceiling; A/sqrt(rho) should be ~1 if
    sqrt(rho) were.  Read the last two columns above.""")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    sys.exit(main())
