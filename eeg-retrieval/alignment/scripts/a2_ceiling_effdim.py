#!/usr/bin/env python
"""
A2/A3 -- NOISE CEILING AND EFFECTIVE DIMENSIONALITY

These two numbers bound EVERYTHING downstream and cannot be worked around by a
better model (docs Prop. 1 and Prop. 5), so they must be measured before any
main analysis.

A2  NOISE CEILING (Prop. 5)
    Split-half RDM reliability with Spearman-Brown correction.  This is the
    maximum RSA score any image-feature model can reach against this data --
    every reported RSA must be expressed as a fraction of it.
    Also inverts the model to give the neural SNR ratio N/S = (T/2)(1-rel)/rel,
    which makes the ceiling a *parameter* of the generative model rather than an
    external nuisance (docs 5.2).

A3  EFFECTIVE DIMENSIONALITY (Prop. 1)
    Participation ratio and d_eff@90/99 of the 63-channel field.  Volume
    conduction makes the scalp field a smooth low-rank mixture, so the number of
    recoverable latent directions is bounded by this, not by 63.
    This caps min(r_E^eff, r_C^eff) in the rank bound.

Both are computed per subject, per time window, and per encoder for the ceiling.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_rsca import (  # noqa: E402
    effective_rank_frac, participation_ratio, rdm, spearman_brown, upper_tri,
)

EEG_ROOT = "/project/peilab/why/NeuroBridge/data/things_eeg"
BASELINE = 50

# Windows come from lib_windows, the single source of truth for the verified
# time axis.  Do NOT hard-code indices here again: info.json's `times` field does
# not describe the stored arrays, and re-deriving windows from it shifts
# everything by 200 ms.
from lib_windows import WINDOWS  # noqa: E402


def load_eeg(sub: str, which: str = "train") -> np.ndarray:
    a = np.load(f"{EEG_ROOT}/preprocessed_eeg/{sub}/{which}.npy")
    if a.ndim == 5:
        a = a.mean(2)
    return a.astype(np.float32)


def load_clip(name: str, which: str = "train") -> np.ndarray:
    f = np.load(f"{EEG_ROOT}/image_feature/{name}/image_{which}.npy")
    if f.ndim == 3:
        f = f.mean(1)
    return f.astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/project/peilab/why/eeg-retrieval/alignment/outputs/a2")
    ap.add_argument("--subs", nargs="+", default=[f"sub-{i:02d}" for i in range(1, 11)])
    ap.add_argument("--clips", nargs="+", default=["RN50", "ViT-H-14"])
    ap.add_argument("--n-perm", type=int, default=500)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.quick:
        args.subs = args.subs[:2]
        args.n_perm = 50

    print("=" * 100)
    print("A2/A3 -- noise ceiling and effective dimensionality")
    print("=" * 100)

    t0 = time.time()
    out = {"config": vars(args), "subjects": {}}

    for sub in args.subs:
        E = load_eeg(sub)
        n, T, ch, tp = E.shape
        # no pre-stimulus baseline exists in the stored arrays; per-window channel
        # mean removal happens in slice_eeg via lib_windows
        h1 = E[:, : T // 2].mean(1)
        h2 = E[:, T // 2:].mean(1)
        full = E.mean(1)

        sub_out = {"shape": [n, T, ch, tp], "a2_ceiling": {}, "a3_effdim": {}}

        for name, lo, hi in WINDOWS:
            R1 = rdm(h1[:, :, lo:hi].reshape(n, -1))
            R2 = rdm(h2[:, :, lo:hi].reshape(n, -1))
            r = float(spearmanr(upper_tri(R1), upper_tri(R2)).correlation)
            sb = spearman_brown(r)
            # ------------------------------------------------------------------
            # TWO DIFFERENT QUANTITIES, DO NOT CONFLATE THEM.
            #   `reliability` = SB-corrected split-half correlation = how well the
            #      observed RDM measures itself.
            #   `ceiling_for_rsa` = sqrt(reliability) = the maximum correlation any
            #      model RDM can attain against the observed RDM, because a model
            #      that perfectly reproduced the TRUE RDM would still correlate
            #      with the noisy observed RDM only by sqrt(reliability).
            # Using `reliability` as the ceiling lets RSA exceed 100% (A2 originally
            # reported 226%), which is impossible.  Verified by simulation in
            # scripts/verify_ceiling_convention.py: corr(obs, true)/sqrt(reliability)
            # = 1.00-1.15 while corr(obs, true)/reliability reaches 3.59.
            # ------------------------------------------------------------------
            ceil = float(np.sqrt(max(sb, 0.0))) if np.isfinite(sb) else float("nan")
            # invert the model for the neural SNR (docs 5.2)
            ns_over_s = (T / 2) * (1 - r) / max(r, 1e-9) if r > 0 else float("inf")
            sub_out["a2_ceiling"][name] = {
                "rho_half": r,
                "reliability_spearman_brown": sb,
                "ceiling_for_rsa": ceil,
                "reproducible_var_frac": sb ** 2 if np.isfinite(sb) else None,
                "N_over_S": ns_over_s,
            }
        # per encoder RSA vs ceiling
        for cname in args.clips:
            C = load_clip(cname)
            if C.shape[0] != n:
                C = C[:n]
            for name, lo, hi in [("w1_100_200ms", 25, 50),
                                 ("full_000_1000ms", 0, 250)]:
                Re = upper_tri(rdm(full[:, :, lo:hi].reshape(n, -1)))
                Rc = upper_tri(rdm(C))
                rho = float(spearmanr(Re, Rc).correlation)
                ceil = sub_out["a2_ceiling"][name]["ceiling_for_rsa"] or 1e-9
                # permutation null for the RSA (docs C4: never report a bare rho)
                rng = np.random.default_rng(0)
                null = np.array([
                    spearmanr(Re, upper_tri(rdm(C[rng.permutation(n)]))).correlation
                    for _ in range(args.n_perm)
                ])
                sub_out.setdefault("rsa", {})[f"{cname}|{name}"] = {
                    "rho": rho, "ceiling": ceil,
                    "frac_of_ceiling": rho / max(ceil, 1e-9),
                    "null_95": float(np.quantile(null, 0.95)),
                    "null_99": float(np.quantile(null, 0.99)),
                    "p_perm": float((null >= rho).mean()),
                }

        for name, lo, hi in WINDOWS:
            Xc = full[:, :, lo:hi].transpose(1, 0, 2).reshape(ch, -1)
            Xc = Xc - Xc.mean(1, keepdims=True)
            S = Xc @ Xc.T / Xc.shape[1]
            sub_out["a3_effdim"][name] = {
                "participation_ratio": participation_ratio(S),
                "d_eff_90": effective_rank_frac(S, 0.90),
                "d_eff_99": effective_rank_frac(S, 0.99),
            }

        # encoder effective ranks (caps r_C^eff in Prop. 1)
        for cname in args.clips:
            C = load_clip(cname)
            Cc = C - C.mean(0, keepdims=True)
            Sc = Cc.T @ Cc / max(Cc.shape[0] - 1, 1)
            sub_out.setdefault("a3_effdim_encoder", {})[cname] = {
                "participation_ratio": participation_ratio(Sc),
                "d_eff_90": effective_rank_frac(Sc, 0.90),
            }

        out["subjects"][sub] = sub_out
        c = sub_out["a2_ceiling"]
        d = sub_out["a3_effdim"]
        print(f"\n{sub}: ceiling_for_rsa[w1={c['w1_100_200ms']['ceiling_for_rsa']:.3f} "
              f"full={c['full_000_1000ms']['ceiling_for_rsa']:.3f}] "
              f"(reliability w1={c['w1_100_200ms']['reliability_spearman_brown']:.3f})")
        print(f"      effdim[full: pr={d['full_000_1000ms']['participation_ratio']:.1f} "
              f"d90={d['full_000_1000ms']['d_eff_90']}]")
        for tag, v in sub_out.get("rsa", {}).items():
            print(f"      RSA {tag:34s} rho={v['rho']:+.4f} "
                  f"({100*v['frac_of_ceiling']:5.1f}% of ceiling) p={v['p_perm']:.3f}",
                  flush=True)

    # ---- aggregate --------------------------------------------------------
    # the ceiling that bounds RSA is sqrt(reliability); see the note in the loop
    ceil_full = [out["subjects"][s]["a2_ceiling"]["full_000_1000ms"]["ceiling_for_rsa"]
                 for s in out["subjects"]]
    ceil_early = [out["subjects"][s]["a2_ceiling"]["w1_100_200ms"]["ceiling_for_rsa"]
                  for s in out["subjects"]]
    ceil_early_full = [out["subjects"][s]["a2_ceiling"]["w1_100_200ms"]["ceiling_for_rsa"]
                       for s in out["subjects"]]
    pr = [out["subjects"][s]["a3_effdim"]["full_000_1000ms"]["participation_ratio"]
          for s in out["subjects"]]
    out["summary"] = {
        "ceiling_full_mean": float(np.nanmean(ceil_full)),
        "ceiling_full_range": [float(np.nanmin(ceil_full)), float(np.nanmax(ceil_full))],
        "ceiling_early_evoked_mean": float(np.nanmean(ceil_early)),
        "participation_ratio_mean": float(np.mean(pr)),
        "participation_ratio_range": [float(np.min(pr)), float(np.max(pr))],
        "branch": None,
    }
    # docs E5: decide the paper branch.
    # IMPORTANT: branch on the window where the response actually lives (early
    # evoked), not on the whole epoch.  A whole-epoch ceiling is diluted by the
    # 300-1000 ms stretch that carries almost no reliable visual signal, so
    # branching on it would engineer a pessimistic conclusion.
    cf = out["summary"]["ceiling_early_evoked_mean"] or out["summary"]["ceiling_full_mean"]
    out["summary"]["branch"] = (
        "MAIN_RESULT_low_ceiling" if cf < 0.10 else
        "MAIN_RESULT_high_ceiling" if cf >= 0.20 else "BORDERLINE"
    )
    out["summary"]["branch_basis"] = (
        f"ceiling_for_rsa with the signal-carrying window = {cf:.4f}")
    out["summary"]["elapsed_s"] = round(time.time() - t0, 1)
    out["summary"]["n_perm"] = args.n_perm

    with open(os.path.join(args.out, "a2_results.json"), "w") as f:
        json.dump(out, f, indent=2)

    s = out["summary"]
    print("\n" + "=" * 100)
    print(f"SUMMARY  ceiling(post-window) = {s['ceiling_full_mean']:.4f} "
          f"range {s['ceiling_full_range']}")
    print(f"         ceiling_for_rsa(early evoked w1) = {s['ceiling_early_evoked_mean']:.4f}"
          f"   <- where the visual response actually is")
    print(f"         NOTE: ceilings are sqrt(reliability); the branch decision below")
    print(f"         uses the window where the signal lives, not the whole epoch.")
    print(f"         EEG participation ratio = {s['participation_ratio_mean']:.1f} "
          f"range {s['participation_ratio_range']}   <- caps r_E^eff")
    print(f"         BRANCH DECISION: {s['branch']}")
    if s["ceiling_full_mean"] < 0.10:
        print("         -> docs E5 low branch: the publishable claim becomes the")
        print("            upper bound on EEG neural visibility, not the NV profile.")
    print(f"  wrote {os.path.join(args.out, 'a2_results.json')}  ({s['elapsed_s']}s)")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
