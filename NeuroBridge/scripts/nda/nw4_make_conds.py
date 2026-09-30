#!/usr/bin/env python
"""Build the generation condition bank from an S1 head, in the variants the 10-subject
run compares.

WHY THESE VARIANTS, AND WHY "SNAP TO REAL ROWS"
-----------------------------------------------
The oracle test (real CLIP test embeddings fed straight to the generator) returns
inception 0.7946 on the same chain that our best EEG condition reaches 0.6757.  Since
the only difference is that the oracle's rows ARE real image embeddings, the question is
what property of ours is missing.  Measuring the candidates:

  condition            rowcos   RSA     vs_true  nn_true  inception
  real CLIP bank       0.6275   1.0000  1.0000   1.0000    0.7946 (oracle)
  hybrid ip_uck calib  0.6119   0.2120  0.5651   0.6745    0.6952
  spa0 calib           0.6123   0.2414  0.5711   0.6389    0.6757
  spa5 calib           0.6124   0.2884  0.5712   0.6363    0.6744
  nw4 raw img          0.9380   0.1362  0.6653   0.7693    0.6228

Two things fall out.

First, `vs_true` is not a quality axis and never was: the raw bank has the HIGHEST
vs_true of any row here (0.6653, above the constant's 0.6201) and the WORST image
(0.6228), because 0.938 rowcos means every row sits in the same central region near
every target.  Concentration inflates it.  Calibration drops rowcos to the bank's own
0.6123 and the image improves by +0.053.  That is why the old vs_cl gate was measuring
an artefact.

Second, and this is what motivates the variants, RSA stops working above ~0.21:
spa0 and spa5 differ by +0.047 RSA and by -0.0013 inception, while hybrid beats both
with LESS RSA.  So `spa5`/`spa20` are not worth GPU hours on their own.  What hybrid has
that ours lacks shows up in `nn_true` (0.6745 vs 0.6389): its rows each land closer to
some real target.  Snapping to real bank rows tests that directly, and it also has the
highest RSA measured (0.2973), i.e. it dominates spa0 on BOTH axes:

  ret@1 from spa0 head   RSA 0.2973   nn_true 0.6634

This is what A2 (the closed-form manifold projection) was reaching for and missed: A2
mixed the top-K neighbour DIRECTIONS when the defect was the bank's CONCENTRATION, so it
could not help.  With concentration already repaired by `gem_calib`, snapping to the
nearest real row is the on-manifold operation A2 wanted to be.  Predict-then-retrieve is
standard practice, needs no test targets (only the TRAIN bank), and is label-free.

Variants emitted:
  direct        the S1 head as-is
  cal           direct + concentration calibration
  ret1          snap each row to its nearest TRAIN bank row (on-manifold, real vectors)
  ret1_cal      ret1 + calibration
  ret3soft      softmax-weighted top-3 real rows (tests a softer snap)
  ret3soft_cal  ret3soft + calibration

Usage:
  python scripts/nda/nw4_make_conds.py --head <s1out>/conds/z_img_test.npy \
      --out-dir <dir> --variants cal,ret1_cal,ret1,ret3soft_cal --json <report>
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
from ocf_train import calibrate_quantile  # noqa: E402

DEFAULT_REF = "outputs/gem/cond_cache/clip_img1024_train.npy"
DEFAULT_BANK = "outputs/gem/cond_cache/clip_img1024_test.npy"


def rsa(z: np.ndarray, t: np.ndarray) -> float:
    z, t = l2n(z), l2n(t)
    cz, ct = z @ z.T, t @ t.T
    iu = np.triu_indices(len(cz), 1)
    a, b = cz[iu], ct[iu]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", type=str, required=True,
                    help="an S1 head's conds/z_img_test.npy")
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--json", type=str, default="")
    ap.add_argument("--ref", type=str, default=str(NB_ROOT / DEFAULT_REF))
    ap.add_argument("--bank", type=str, default=str(NB_ROOT / DEFAULT_BANK))
    ap.add_argument("--variants", type=str,
                    default="direct,cal,ret1,ret1_cal,ret3soft,ret3soft_cal")
    ap.add_argument("--tau", type=float, default=0.07, help="softmax temperature for ret3")
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    z0 = l2n(np.load(args.head).astype(np.float32))
    R = l2n(np.load(args.ref).astype(np.float32))     # TRAIN bank (no test targets used)
    B = l2n(np.load(args.bank).astype(np.float32))    # TEST bank, for reporting only
    bank_rowcos = row2mean(B)
    c_raw = l2n(R.mean(0, keepdims=True))[0]
    floor = float((np.tile(c_raw, (len(B), 1)) * B).sum(1).mean())

    # retrieval index is against the TRAIN bank only -- never the test targets
    sim = z0 @ R.T
    idx1 = np.argsort(-sim, 1)[:, 0]

    want = [v.strip() for v in args.variants.split(",") if v.strip()]
    made: dict[str, Path] = {}

    def emit(name: str, z: np.ndarray, do_cal: bool) -> None:
        zz = l2n(np.asarray(z, dtype=np.float32))
        if do_cal:
            zz = l2n(np.asarray(calibrate_quantile(zz, R, two_sided=True)[0], dtype=np.float32))
        p = out / f"{name}_test.npy"
        np.save(p, zz.astype(np.float32))
        made[name] = p

    if "direct" in want:
        emit("direct", z0, False)
    if "cal" in want:
        emit("cal", z0, True)
    if "ret1" in want:
        emit("ret1", R[idx1], False)
    if "ret1_cal" in want:
        emit("ret1_cal", R[idx1], True)
    if "ret3soft" in want or "ret3soft_cal" in want:
        k = 3
        idxk = np.argsort(-sim, 1)[:, :k]
        sk = sim[np.arange(len(z0))[:, None], idxk]
        w = np.exp((sk - sk.max(1, keepdims=True)) / args.tau)
        w = w / w.sum(1, keepdims=True)
        soft = (R[idxk] * w[:, :, None]).sum(1)
        if "ret3soft" in want:
            emit("ret3soft", soft, False)
        if "ret3soft_cal" in want:
            emit("ret3soft_cal", soft, True)

    rows = []
    print(f"[conds] {'variant':<14}{'RSA':>9}{'rowcos':>9}{'dev':>8}{'vs_true':>9}{'top1':>7}")
    print(f"[conds] {'real bank':<14}{1.0:>9.4f}{bank_rowcos:>9.4f}{0.0:>+8.4f}{1.0:>9.4f}{'--':>7}")
    for name, p in made.items():
        z = l2n(np.load(p).astype(np.float32))
        r = rsa(z, B)
        rc = row2mean(z)
        vt = float((z * B).sum(1).mean())
        t1 = retrieval(z, B)["top1"]
        try:
            rel = str(p.resolve().relative_to(NB_ROOT))
        except ValueError:
            rel = str(p)
        rows.append({"variant": name, "path": rel,
                     "rsa": round(r, 4), "rowcos": round(rc, 4),
                     "rowcos_dev": round(rc - bank_rowcos, 4),
                     "vs_true": round(vt, 4), "vs_cl": round(vt - floor, 4),
                     "top1": t1})
        print(f"[conds] {name:<14}{r:>9.4f}{rc:>9.4f}{rc - bank_rowcos:>+8.4f}{vt:>9.4f}{t1:>7.3f}")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "stage": "nw4_make_conds", "head": args.head, "tag": args.tag,
            "bank_rowcos": round(bank_rowcos, 4), "floor": floor, "rows": rows,
        }, indent=2), encoding="utf-8")
        print(f"[conds] wrote {args.json}")


if __name__ == "__main__":
    main()
