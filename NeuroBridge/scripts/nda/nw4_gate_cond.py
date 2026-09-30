#!/usr/bin/env python
"""The condition gate, on the axis that actually predicts generation.

WHY THIS REPLACED THE vs_cl GATE
--------------------------------
The previous gate used

    vs_cl = mean_i cos(cond_i, true_clip_i) - mean_i cos(constant, true_clip_i)

fitted from seven (condition, generation) pairs.  The full nw4 run produced an eighth
that broke it:

    condition      vs_cl      inception
    hybrid ip_uck  +0.0367    0.7280
    nw4 img        +0.0451    0.6228     <- better vs_cl, worse image

Worse, tracing what actually reached the generator showed the compared quantity was
not even the one used: hybrid generates from the CALIBRATED bank
(`outputs/hybrid_s08/conds/uck.npy`, vs_cl -0.0550), not the un-calibrated
`full/conds/ip_uck_test.npy` (+0.0367) the fit had used.  The seven-point fit was
therefore measured on the wrong column and never predicted the 0.7280 row.

`nw4_diag_rsa.py` scored every condition with a known generation score on both axes:

    corr(RSA,  inception) = +0.967      <- the gate is this one
    corr(vs_cl, inception) = +0.687

where RSA is the second-order statistic vs_cl ignores -- whether the SIMILARITY
STRUCTURE between trials is preserved:

    RSA = corr( cos(cond_i, cond_j),  cos(T_i, T_j) )   over all i < j

with T the true trial-level CLIP image embeddings.  A constant condition scores 0, an
oracle scores 1, and it does not care how the rows are embedded as long as their
relative geometry matches what the adapter must render.

WHY THE DEFECT WAS CONCENTRATION, NOT DIRECTION
-----------------------------------------------
    condition              RSA     rowcos    top1    inception
    real CLIP bank        1.0000   0.6275    --        --
    nw4 img, raw          0.1362   0.9380   0.035      0.6228
    nw4 img + gem_calib   0.2085   0.6123   0.215        --
    hybrid ip_uck calib   0.2120   0.6119   0.160      0.7280

`rowcos` 0.938 against the real bank's own 0.6275 means every row points essentially the
same way: the bank is one direction plus noise, so trial identity is unrecoverable and
the adapter renders one averaged appearance.  `gem_calib.py` quantile-matches each
row's concentration onto the TRAIN bank's own distribution to undo exactly this, and it
restores both RSA (0.136 -> 0.208) and retrieval (0.035 -> 0.215).  hybrid_s08 applied
it; this pipeline did not.

This also retires A2 as the fix: the closed-form manifold projection acted on each
row's DIRECTION by mixing top-K neighbours, when the defect was the CONCENTRATION of
the bank as a whole, so it could not have repaired this however it was tuned.

The gate therefore:
  1. reports RSA first and gates on it,
  2. reports `rowcos` against the real bank's `rowcos` -- a near-free sanity check that
     catches the failure mode directly, since a bank far more concentrated than the real
     one cannot carry trial structure no matter what RSA says,
  3. keeps vs_cl and retrieval only as diagnostics, never as the verdict.

Usage:
  python scripts/nda/nw4_gate_cond.py --conds outputs/nw4/sub-08/s2/conds_cal \
      --out outputs/nw4/sub-08/cond_gate.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

NB_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NB_ROOT / "scripts" / "nda"))

from nw4_s2_project import l2n, row2mean, retrieval  # noqa: E402

DEFAULT_BANK = "outputs/gem/cond_cache/clip_img1024_test.npy"
DEFAULT_REF = "outputs/gem/cond_cache/clip_img1024_train.npy"

# RSA gate.  Calibrated points: 0.048 -> 0.5236 (nw3), 0.136 -> 0.6228 (our raw),
# 0.192 -> 0.7053, 0.212 -> 0.7280 (hybrid).  A least-squares fit through those gives
# inception ~= 0.464 + 1.25 * RSA, so 0.190 lands at ~0.70 -- the threshold below which
# generation cannot be argued to match the honest reference, and above which it can.
RSA_GATE = 0.190
# The bank's own concentration.  Anything much tighter cannot carry trial structure.
ROWCOS_SLACK = 0.15


def rsa(z: np.ndarray, t: np.ndarray) -> float:
    cz, ct = l2n(z) @ l2n(z).T, l2n(t) @ l2n(t).T
    iu = np.triu_indices(len(cz), k=1)
    a, b = cz[iu], ct[iu]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conds", type=str, default="",
                    help="directory of *_test.npy condition banks emitted by S2")
    ap.add_argument("--sweep", type=str, default="",
                    help="directory of trained S1 runs (each <run>/conds/*_test.npy)")
    ap.add_argument("--bank", type=str, default=str(NB_ROOT / DEFAULT_BANK))
    ap.add_argument("--bank-train", type=str, default=str(NB_ROOT / DEFAULT_REF))
    ap.add_argument("--gate", type=float, default=RSA_GATE)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    if not args.conds and not args.sweep:
        raise SystemExit("[gate] pass --conds or --sweep")

    B_te = l2n(np.load(args.bank).astype(np.float32))
    B_tr = l2n(np.load(args.bank_train).astype(np.float32))
    c_raw = l2n(B_tr.mean(0, keepdims=True))[0]
    floor = float((np.tile(c_raw, (len(B_te), 1)) * B_te).sum(1).mean())
    bank_rowcos = row2mean(B_te)

    # a control with a KNOWN score, so the gate reports a calibration every run
    ctrl_p = NB_ROOT / "outputs/hybrid_s08/conds/uck.npy"
    ctrl_rsa = None
    if ctrl_p.is_file():
        cz = np.load(ctrl_p).astype(np.float32)
        if cz.shape[1] == B_te.shape[1]:
            ctrl_rsa = rsa(l2n(cz), B_te)

    print(f"real bank rowcos = {bank_rowcos:.4f}   constant floor vs_true = {floor:.4f}")
    print(f"RSA gate >= {args.gate:.4f}" +
          (f"   control: hybrid ip_uck (calibrated) RSA {ctrl_rsa:.4f} -> 0.7280 inception"
           if ctrl_rsa is not None else ""))

    files: list[tuple[str, Path]] = []
    if args.sweep:
        for run in sorted(Path(args.sweep).iterdir()):
            for f in sorted((run / "conds").glob("*_test.npy")) if (run / "conds").is_dir() else []:
                files.append((f"{run.name}/{f.stem}", f))
    else:
        for f in sorted(Path(args.conds).glob("*_test.npy")):
            files.append((f.stem, f))
    if not files:
        raise SystemExit(f"[gate] no *_test.npy found (looked in {args.conds or args.sweep})")

    print()
    print(f"{'condition':<22}{'RSA':>9}{'vs_cl':>9}{'rowcos':>9}{'dev':>8}"
          f"{'top1':>7}   verdict")
    print("-" * 82)
    rows = []
    for name, f in files:
        z = np.load(f).astype(np.float32)
        if z.shape[1] != B_te.shape[1]:
            print(f"{name:<22}  dim {z.shape[1]} != {B_te.shape[1]} (not clip_img space)")
            continue
        z = l2n(z)
        r = rsa(z, B_te)
        rc = row2mean(z)
        vt = float((z * B_te).sum(1).mean()) - floor
        top1 = retrieval(z, B_te)["top1"]
        too_tight = rc > bank_rowcos + ROWCOS_SLACK
        ok = (r >= args.gate) and not too_tight
        why = "PASS" if ok else (
            "over-concentrated (run gem_calib)" if too_tight else
            f"RSA below gate (need +{args.gate - r:.4f})")
        try:
            rel = str(f.resolve().relative_to(NB_ROOT))
        except ValueError:
            rel = str(f)
        rows.append({"name": name, "path": rel, "rsa": round(r, 4),
                     "vs_cl": round(vt, 4), "rowcos": round(rc, 4),
                     "rowcos_dev_from_bank": round(rc - bank_rowcos, 4),
                     "top1": top1, "pass": bool(ok), "reason": why,
                     "predicted_inception": round(0.464 + 1.25 * r, 4)})
        print(f"{name:<22}{r:>9.4f}{vt:>+9.4f}{rc:>9.4f}{rc - bank_rowcos:>+8.4f}"
              f"{top1:>7.3f}   {why}")

    passing = [r for r in rows if r["pass"]]
    best = max(rows, key=lambda r: r["rsa"]) if rows else None
    print()
    verdict: list[str] = []
    if passing:
        b = max(passing, key=lambda r: r["rsa"])
        verdict.append(f"{len(passing)} condition(s) clear the RSA gate; best is {b['name']} "
                       f"RSA {b['rsa']:.4f} -> predicted inception {b['predicted_inception']:.4f}")
        if ctrl_rsa is not None:
            verdict.append(
                f"vs the hybrid control (RSA {ctrl_rsa:.4f}, measured 0.7280): "
                f"{'ABOVE' if b['rsa'] > ctrl_rsa else 'below'} by "
                f"{b['rsa'] - ctrl_rsa:+.4f} RSA")
    else:
        verdict.append(f"NO condition clears RSA >= {args.gate:.4f}; generation would "
                       f"test the adapter's tolerance, not the architecture")
        if best:
            verdict.append(f"best RSA is {best['rsa']:.4f} ({best['name']}), "
                           f"{args.gate - best['rsa']:+.4f} short")
            tight = [r for r in rows if r["rowcos"] > bank_rowcos + ROWCOS_SLACK]
            if tight:
                verdict.append(f"{len(tight)}/{len(rows)} banks are over-concentrated "
                               f"(rowcos > {bank_rowcos + ROWCOS_SLACK:.3f} vs the real "
                               f"bank's {bank_rowcos:.4f}) -- run gem_calib.py")
    for v in verdict:
        print(f"[gate] {v}")

    rep = {"stage": "nw4_gate_cond", "axis": "rsa", "bank_rowcos": round(bank_rowcos, 4),
           "floor": floor, "gate": args.gate, "control_rsa": ctrl_rsa,
           "rows": rows, "any_pass": bool(passing), "verdict": verdict,
           "best_rsa": best["rsa"] if best else None}
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=2), encoding="utf-8")
        print(f"[gate] wrote {args.out}")
    # a gate that cannot be cleared must fail loudly so the caller can stop
    sys.exit(0 if passing else 1)


if __name__ == "__main__":
    main()
