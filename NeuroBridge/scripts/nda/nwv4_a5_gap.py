#!/usr/bin/env python
"""Why A5, and what a structure condition must contain to reach it.

A5 is the arm we want to deploy:

    cc3 layout, no ControlNet, empty prompt, turbo 15 steps CFG 0, band anchor sigma 3.0
    conditions: our semantic bank + GROUND-TRUTH depth-CLIP and edge-CLIP rows

    pixcorr 0.1980  ssim 0.3756  incep 0.8400  clip 0.9123  alex2 0.8248  alex5 0.9245
    swav 0.5205  fid 160.96

Its only non-deployable ingredient is those GT structure rows.  The deployable twin E2
(same operator, same semantic bank, variance-restored linear EEG->structure instead of
GT) reaches

    pixcorr 0.1940  ssim 0.3677  incep 0.7363  clip 0.8215  alex2 0.8063  alex5 0.8874
    swav 0.5752  fid 171.81

so the gap this script is about is incep +0.104, clip +0.091, swav +0.055, alex5 +0.037,
alex2 +0.019, pixcorr +0.004, ssim +0.008 -- i.e. almost entirely the two semantic axes,
and the structure branch is the only thing that differs.

WHAT WE ALREADY KNOW (nw5b / nw5c)
----------------------------------
  1. The branch needs TRIAL-ALIGNED information, not a prior.  A constant centroid (C1),
     misaligned GT rows (C3), and the honest linear map (C2) all landed at or below the
     semantic-only floor.
  2. Our structure banks are SHRUNK: mean pairwise cosine across the 200 trials is
     0.955 (ridge) / 0.874 / 0.938 (UCK) against 0.566 / 0.530 for the GT rows.  Least
     squares reduces squared error by predicting the mean, so the trial-specific
     component got crushed.
  3. Rescaling only the deviation from the bank mean (identical weights, identical EEG)
     lifted top-1 retrieval against the GT bank from 3x/11x to 26x/35x chance.  Yet the
     generated arms moved by incep +0.0061 / clip +0.0093, against +0.128 / +0.085 for
     the GT rows.  So retrieval improved 8.7x and the gain improved from 0.0% to 4.8% of
     what GT delivers.  Something else is binding.

WHAT THIS SCRIPT MEASURES
-------------------------
Every condition bank is split into the part shared by all 200 trials and the part that
varies between them:

    bank_i = m + d_i          m = mean row, d_i = trial-specific deviation

and for each we report

    %E(mean)   share of the row energy sitting in m        (the shrinkage axis)
    cos(m)     alignment of m with the GT bank's mean       (is the prior in the right place)
    dev_corr   mean_i cos(d_i, d_GT_i)                      (is the deviation the RIGHT one)
    dev_top1   does the deviation identify the trial among all 200 GT deviations
                                                            (the same test as top1, but with
                                                             the mean projected out)

dev_top1 is the number that matters.  Full-row top1 can be earned by the mean alone if a
bank is concentrated enough; dev_top1 cannot, because the mean has been removed from both
sides.  If a bank scores high on dev_top1, its trial information is real and the fix is
to stop burying it under the mean.  If it scores near chance, the trial information was
never there.

The semantic bank is measured the same way and acts as the CALIBRATION: it is the one
branch we know works (it is what every arm in the table is built on), so its dev_top1 is
the level a structure head has to be compared against, not 1.0.

Part B then asks the constructive question: if the head is trained on the DEVIATION
instead of on the full row, how much of the gap closes?  That is a ridge fit on
z -> (GT row - GT train centroid), evaluated both ways, with alpha chosen on the
held-out validation concepts exactly as before.  Nothing is trained on test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

NB_ROOT = Path("/project/peilab/why/NeuroBridge")


def l2n(x: np.ndarray) -> np.ndarray:
    return (x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)).astype(np.float32)


def split_mean_dev(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (unit mean direction, unit-normalised deviations)."""
    x = l2n(x)
    m = x.mean(0, keepdims=True)
    return l2n(m), l2n(x - m)


def energy_split(x: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Row energy split into the shared part and the trial-specific part.

    Rows are unit-norm, the deviations sum to zero by construction, so
    |m|^2 + mean_i|d_i|^2 = 1 and the two shares are directly comparable.
    Returns (share_in_mean, share_in_dev, unit mean direction, unit deviations).
    """
    x = l2n(x)
    m = x.mean(0, keepdims=True)
    d = x - m
    e_mean = float((m ** 2).sum())
    e_dev = float((d ** 2).sum(1).mean())
    return e_mean, e_dev, l2n(m), l2n(d)


def describe(B: np.ndarray, m_g: np.ndarray, d_g: np.ndarray) -> dict:
    """m_g / d_g are the GT bank's mean direction and unit deviations (already l2n)."""
    e_mean, e_dev, m_b, d_b = energy_split(B)
    Bn = l2n(B)
    cos_mean = float((m_b * m_g).sum())
    dev_corr = float((d_b * d_g).sum(1).mean())
    S = l2n(d_b) @ l2n(d_g).T
    dev_top1 = float((S.argmax(1) == np.arange(len(d_g))).mean())
    return {"rowcos": float((Bn @ Bn.T).sum() - np.trace(Bn @ Bn.T)) / (len(B) * (len(B) - 1)),
            "energy_mean": e_mean, "energy_dev": e_dev,
            "cos_mean": cos_mean, "dev_corr": dev_corr, "dev_top1": dev_top1}


def full_row_top1(B: np.ndarray, G: np.ndarray) -> float:
    return float(((l2n(B) @ l2n(G).T).argmax(1) == np.arange(len(G))).mean())


def ridge(ztr, ytr, zte, alphas, fit_i, val_i):
    """Alpha selected on val only; returns (alpha, weights, val score)."""
    zf_all = np.concatenate([ztr, np.ones((len(ztr), 1), np.float32)], 1)
    best = None
    for a in alphas:
        zf = zf_all[fit_i]
        g = zf.T @ zf
        if a > 0:
            pen = np.eye(g.shape[0], dtype=np.float32) * a
            pen[-1, -1] = 0.0
            g = g + pen
        w = np.linalg.solve(g, zf.T @ ytr[fit_i])
        pred_val = np.concatenate([ztr[val_i], np.ones((len(val_i), 1), np.float32)], 1) @ w
        sc = float(((l2n(pred_val) @ l2n(ytr[val_i]).T).argmax(1)
                    == np.arange(len(val_i))).mean())
        if best is None or sc > best[0]:
            best = (sc, a, w)
    sc, a, w = best
    pred = np.concatenate([zte, np.ones((len(zte), 1), np.float32)], 1) @ w
    return a, sc, pred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, default=8)
    ap.add_argument("--z-root", type=str, default=str(NB_ROOT / "outputs/ocf/intra_z"))
    ap.add_argument("--cond-cache", type=str, default=str(NB_ROOT / "outputs/gem/cond_cache"))
    ap.add_argument("--split-json", type=str, default=str(NB_ROOT / "outputs/leakfree/split.json"))
    ap.add_argument("--conds", type=str, default=str(NB_ROOT / "outputs/nw5_s08/conds"))
    ap.add_argument("--stag", type=str, default="sub-08")
    ap.add_argument("--sem", type=str,
                    default=str(NB_ROOT / "outputs/nw4_10s/arms/a_hi/conds/sub-08/cal_test.npy"))
    ap.add_argument("--alphas", type=str, default="0,0.01,0.1,1,10,100,1000")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    sid, stag = args.subject, args.stag
    cc, conds = Path(args.cond_cache), Path(args.conds)
    alphas = [float(x) for x in args.alphas.split(",")]

    sp = json.loads(Path(args.split_json).read_text())
    fit_i = np.asarray(sp["fit_rows"], dtype=int)
    val_i = np.asarray(sp.get("val_b_rows") or sp["val_a_rows"], dtype=int)

    zdir = Path(args.z_root) / f"sub-{sid:02d}"
    ztr = l2n(np.load(zdir / "shared_r_train.npy").astype(np.float32))
    zte = l2n(np.load(zdir / "shared_r_test.npy").astype(np.float32))
    print(f"[gap] sub-{sid}  EEG train {ztr.shape} test {zte.shape}  "
          f"fit={len(fit_i)} val={len(val_i)}")

    # ---------------- Part A: where is each bank's information? ----------------
    print()
    print("=" * 118)
    print("PART A  every bank split into the part shared by all 200 trials and the part that "
          "varies between them")
    print("=" * 118)
    print("  dev_top1 is the decisive column: it removes the shared part from both sides, so")
    print("  unlike full-row top1 it cannot be earned by a concentrated bank's mean.")
    print()

    out: dict = {"subject": sid, "partA": {}, "partB": {}}
    target_specs = [("depth", "clip_depth1024"), ("edge", "clip_edge1024"),
                    ("image (semantic)", "clip_img1024")]

    for tname, fn in target_specs:
        Gtr = l2n(np.load(cc / f"{fn}_train.npy").astype(np.float32))
        Gte = l2n(np.load(cc / f"{fn}_test.npy").astype(np.float32))
        m_g, d_g = split_mean_dev(Gte)
        gt_e_mean, gt_e_dev, _, _ = energy_split(Gte)
        print(f"--- target: {tname}  (GT rows: {gt_e_mean:.3f} of row energy in the shared "
              f"part, {gt_e_dev:.3f} trial-specific) ---")

        banks: dict[str, np.ndarray] = {}
        if tname == "image (semantic)":
            banks["a_hi SEM (shipped)"] = np.load(args.sem).astype(np.float32)
        else:
            lab = tname
            for name, p in (
                ("GT rows (A5)",
                 None),
                ("ridge raw (C2, shrunk)", conds / f"ridge_{lab}1024_{stag}_test.npy"),
                ("ridge varest (E2)", conds / f"varestridge_{lab}1024_{stag}_test.npy"),
                ("UCK raw (A8, shrunk)", conds / f"eeg_{lab}1024_{stag}_test.npy"),
                ("UCK varest (E3)", conds / f"varest_uck_{lab}1024_{stag}_test.npy"),
                ("constant centroid (C1)", conds / f"const_{lab}1024_{stag}_test.npy"),
                ("GT rows shuffled (C3)", conds / f"shuf_{lab}1024_{stag}_test.npy"),
            ):
                if name == "GT rows (A5)":
                    banks[name] = Gte
                elif Path(p).is_file():
                    banks[name] = np.load(p).astype(np.float32)

        print(f"    {'bank':<26}{'%E(mean)':>10}{'%E(dev)':>9}{'cos(m)':>9}"
              f"{'dev_corr':>10}{'dev_top1':>10}{'xchance':>9}{'row_top1':>10}")
        print("    " + "-" * 91)
        for name, B in banks.items():
            d = describe(B, m_g, d_g)
            rt1 = full_row_top1(B, Gte)
            d["row_top1"] = rt1
            print(f"    {name:<26}{d['energy_mean']:>10.3f}{d['energy_dev']:>9.3f}"
                  f"{d['cos_mean']:>9.4f}{d['dev_corr']:>10.4f}{d['dev_top1']:>10.4f}"
                  f"{d['dev_top1']*len(Gte):>9.1f}{rt1:>10.4f}")
            out["partA"].setdefault(tname, {})[name] = d
        print()

    # ---------------- Part B: train on the deviation instead of the row ------------
    print("=" * 118)
    print("PART B  if the head is trained on the DEVIATION instead of the full row, how much "
          "of the gap closes?")
    print("=" * 118)
    print("  Full-row training spends its capacity on the mean, which the shrink in Part A shows")
    print("  up as %E(mean) ~ 0.95.  Deviation training removes the mean from the target first.")
    print("  Both fit on the same leak-free `fit` rows, alpha chosen on the same held-out val.")
    print()
    print(f"  {'target':<8}{'mode':<12}{'alpha':>8}{'val_top1':>10}{'test_dev_top1':>15}"
          f"{'test_dev_corr':>15}{'row_top1':>10}")
    print("  " + "-" * 78)
    for lab in ("depth", "edge"):
        Gtr = np.load(cc / f"clip_{lab}1024_train.npy").astype(np.float32)
        Gte = l2n(np.load(cc / f"clip_{lab}1024_test.npy").astype(np.float32))
        m_g_tr = Gtr.mean(0, keepdims=True)          # deployable: train rows only
        Dtr = Gtr - m_g_tr
        m_gt, d_g = split_mean_dev(Gte)

        for mode, Y in (("full row", Gtr), ("deviation", Dtr)):
            a, vsc, pred = ridge(ztr, Y, zte, alphas, fit_i, val_i)
            if mode == "deviation":
                cond = l2n(m_g_tr + pred)             # put the train centroid back
            else:
                cond = l2n(pred)
            dd = describe(cond, m_gt, d_g)
            rt1 = full_row_top1(cond, Gte)
            print(f"  {lab:<8}{mode:<12}{a:>8.2f}{vsc:>10.4f}{dd['dev_top1']:>15.4f}"
                  f"{dd['dev_corr']:>15.4f}{rt1:>10.4f}")
            out["partB"].setdefault(lab, {})[mode] = {
                "alpha": a, "val_top1": vsc, "test_dev_top1": dd["dev_top1"],
                "test_dev_corr": dd["dev_corr"], "row_top1": rt1,
                "energy_mean": dd["energy_mean"], "cos_mean": dd["cos_mean"]}
        print()

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
