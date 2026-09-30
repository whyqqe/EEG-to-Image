#!/usr/bin/env python
"""Does the condition preserve the RELATIVE structure between trials?

WHY.  `vs_cl` was established as the gate from seven (condition, generation score)
pairs, but the full nw4 run produced an eighth that breaks it:

    condition      vs_cl     inception
    hybrid ip_uck  +0.0367   0.7280
    nw4 img        +0.0451   0.6228    <- better vs_cl, worse image

So vs_cl is necessary but not sufficient.  It measures only the AVERAGE direction of
the rows: mean_i cos(z_i, T_i), minus the same for a constant.  A condition can score
well there while getting every trial's *relative* placement wrong -- e.g. every row
points vaguely at the target region but the rows are not ordered or clustered the way
the true targets are.

The missing quantity is second-order: does the condition reproduce the similarity
STRUCTURE of the targets?  With C_cond[i,j] = cos(z_i, z_j) and C_true[i,j] =
cos(T_i, T_j), the RSA statistic corr(C_cond, C_true) over i<j is 0 for a constant
condition and 1 for a perfect one, regardless of how the rows are embedded.

This script scores every condition we have (including all the hybrid ones, whose
generation scores are known) on both axes, so the two can be compared directly.

Usage:
  python scripts/nda/nw4_diag_rsa.py --out outputs/nw4/sub-08/diag_rsa.json
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


def rsa(z: np.ndarray, t: np.ndarray) -> float:
    """Pearson correlation of the two off-diagonal similarity matrices."""
    cz = l2n(z) @ l2n(z).T
    ct = l2n(t) @ l2n(t).T
    iu = np.triu_indices(len(cz), k=1)
    a, b = cz[iu], ct[iu]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--bank", type=str,
                    default=str(NB_ROOT / "outputs/gem/cond_cache/clip_img1024_test.npy"))
    ap.add_argument("--bank-train", type=str,
                    default=str(NB_ROOT / "outputs/gem/cond_cache/clip_img1024_train.npy"))
    args = ap.parse_args()

    B_te = l2n(np.load(args.bank).astype(np.float32))
    B_tr = l2n(np.load(args.bank_train).astype(np.float32))
    c_raw = l2n(B_tr.mean(0, keepdims=True))[0]
    floor = float((np.tile(c_raw, (len(B_te), 1)) * B_te).sum(1).mean())
    print(f"constant floor vs_true = {floor:.4f}   (RSA of a constant = 0 by definition)")
    print()

    # label -> (path, known generation inception or None)
    cands = [
        ("CONSTANT (train mean)", None, None),
        ("ORACLE true clip_img", "__oracle__", None),
        ("hybrid ip_uck", "outputs/hybrid_s08/conds/uck.npy", 0.7280),
        ("hybrid ip_blend", "outputs/hybrid_s08/conds/blend.npy", 0.7053),
        ("hybrid ip_hard", "outputs/hybrid_s08/conds/hard.npy", 0.6324),
        ("hybrid ip_short", "outputs/hybrid_s08/conds/short.npy", 0.7048),
        ("hybrid ip_nat", "outputs/hybrid_s08/conds/nat.npy", 0.6922),
        ("nw3 z_fused", "outputs/nw3/sub-08/s3/conds/z_fused_test.npy", 0.5236),
        ("nw4 s2 img", "outputs/nw4/sub-08/s2/conds/img_test.npy", 0.6228),
        ("nw4 s2 fused", "outputs/nw4/sub-08/s2/conds/fused_test.npy", 0.5986),
        ("nw4 s2 attr", "outputs/nw4/sub-08/s2/conds/attr_test.npy", None),
        ("nw4 s2 depth", "outputs/nw4/sub-08/s2/conds/depth_test.npy", None),
        ("nw4 s2 edge", "outputs/nw4/sub-08/s2/conds/edge_test.npy", None),
    ]

    rows: list[dict] = []
    print(f"{'condition':<24}{'vs_cl':>9}{'RSA':>9}{'rowcos':>9}{'top1':>8}"
          f"{'  known incep':>14}")
    for name, p, incep in cands:
        if p is None:
            z = np.tile(c_raw, (len(B_te), 1))
        elif p == "__oracle__":
            z = B_te
        else:
            fp = Path(p)
            if not fp.is_file():
                print(f"{name:<24}  (missing: {p})")
                continue
            z = np.load(fp).astype(np.float32)
            if z.shape[1] != B_te.shape[1]:
                print(f"{name:<24}  (dim {z.shape[1]} != clip_img space)")
                continue
        z = l2n(z)
        vt = float((z * B_te).sum(1).mean())
        r = rsa(z, B_te)
        rt = retrieval(z, B_te)
        rows.append({"name": name, "vs_cl": round(vt - floor, 4), "rsa": round(r, 4),
                     "rowcos": round(row2mean(z), 4), "top1": rt["top1"],
                     "known_incep": incep, "path": p})
        k = f"{incep:.4f}" if incep is not None else "--"
        print(f"{name:<24}{vt - floor:>+9.4f}{r:>9.4f}{row2mean(z):>9.4f}"
              f"{rt['top1']:>8.3f}{k:>14}")

    # which axis predicts the known generation scores?
    print()
    print("=" * 84)
    print("WHICH AXIS PREDICTS GENERATION?  (over the conditions with known inception)")
    print("=" * 84)
    known = [r for r in rows if r["known_incep"] is not None]
    verdict = []
    if len(known) >= 3:
        for axis in ("vs_cl", "rsa"):
            a = np.array([r[axis] for r in known])
            b = np.array([r["known_incep"] for r in known])
            if a.std() < 1e-12:
                verdict.append(f"{axis:<6} constant across samples")
                continue
            c = float(np.corrcoef(a, b)[0, 1])
            verdict.append(f"corr({axis}, inception) = {c:+.3f} over {len(known)} conditions")
        for v in verdict:
            print(f"  {v}")
        # the specific pair that broke vs_cl
        hu = next((r for r in known if r["name"] == "hybrid ip_uck"), None)
        n4 = next((r for r in known if r["name"] == "nw4 s2 img"), None)
        if hu and n4:
            print()
            print(f"  the pair that broke the vs_cl gate:")
            print(f"    {hu['name']:<18} vs_cl {hu['vs_cl']:+.4f}  RSA {hu['rsa']:.4f}  "
                  f"incep {hu['known_incep']:.4f}")
            print(f"    {n4['name']:<18} vs_cl {n4['vs_cl']:+.4f}  RSA {n4['rsa']:.4f}  "
                  f"incep {n4['known_incep']:.4f}")
            dv = n4["vs_cl"] - hu["vs_cl"]
            dr = n4["rsa"] - hu["rsa"]
            print(f"    delta            vs_cl {dv:+.4f}  RSA {dr:+.4f}")
            if dv > 0 and dr < 0:
                print("    -> our condition is BETTER aligned on average but has WORSE")
                print("       relative structure.  RSA is the axis that explains the")
                print("       worse images, so RSA must join vs_cl in the gate.")
    for v in verdict:
        print(f"  - {v}")
    for r in rows:
        del r["path"]

    rep = {"stage": "nw4_diag_rsa", "floor": floor, "rows": rows, "verdict": verdict}
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=2), encoding="utf-8")
        print(f"\n[diag] wrote {args.out}")


if __name__ == "__main__":
    main()
