#!/usr/bin/env python
"""The metric that actually predicts generation quality: `vs_true`.

`nw4_diag_collapse.py` showed the S1 head is a strong *retriever* (200-way top1
0.35).  That is NOT the number that matters downstream.  An audit of the older
`hybrid_s08` export report shows the generator responds to the cosine against the
trial's TRUE CLIP embedding, and that this ranking is *opposite* to the
discriminative ranking:

    variant      rowcos(lower=less collapsed)   vs_true   generated inception
    hard         0.508 (best)                   0.529 (worst)   0.632 (worst)
    uck          0.753                           0.657           0.728 (best)
    blend        0.830                           0.665 (best)    0.705
    centreline   --  (a CONSTANT vector)         0.628

A constant vector scores vs_true 0.628 because CLIP image embeddings share a large
common component.  So a condition can be "more discriminative" and yet land
*closer to a constant* in the only metric the adapter can read.

Consequence for nw4: before spending GPU hours on 11 generation arms we must know
whether our S1 heads clear the constant-vector bar at all.  If vs_true <= 0.628 the
adapter sees no better than a blank condition, and no amount of A3/A4 injection
engineering can help.

This script prints that decision table for every candidate condition we can build
from the existing artifacts, with the constant vector and the GT bank as bars.

Usage:
  python scripts/nda/nw4_diag_vstrue.py --s1 outputs/nw4/sub-08/s1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

NB_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NB_ROOT / "scripts" / "nda"))

from nw4_s1_train import FactorizedEncoder, VITH_LEVELS, ATTR_FIELDS  # noqa: E402,F401
from nw4_s2_project import (l2n, row2mean, offdiag, erank, retrieval,  # noqa: E402
                            manifold_project, fit_ridge, concept_means)
import leakfree as LF  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1", type=str, required=True)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--test-subject", type=int, default=8)
    ap.add_argument("--z-root", type=str, default=str(NB_ROOT / "outputs/ocf/intra_z"))
    ap.add_argument("--cond-cache", type=str, default=str(NB_ROOT / "outputs/gem/cond_cache"))
    ap.add_argument("--g2-targets", type=str, default=str(NB_ROOT / "outputs/g2/targets"))
    ap.add_argument("--split-json", type=str, default=str(NB_ROOT / "outputs/leakfree/split.json"))
    ap.add_argument("--nw3-s3", type=str,
                    default=str(NB_ROOT / "outputs/nw3/sub-08/s3/conds"))
    ap.add_argument("--proj-tau", type=float, default=0.07)
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    sid = f"{args.test_subject:02d}"
    s1d = Path(args.s1)
    cc, g2 = Path(args.cond_cache), Path(args.g2_targets)

    ztr = l2n(np.load(Path(args.z_root) / f"sub-{sid}" / "shared_r_train.npy").astype(np.float32))
    zte = l2n(np.load(Path(args.z_root) / f"sub-{sid}" / "shared_r_test.npy").astype(np.float32))
    n_flat = len(ztr)
    reps = n_flat // 1654
    cid_tr = np.arange(n_flat, dtype=np.int64) // reps

    B_img_tr = l2n(np.load(cc / "clip_img1024_train.npy").astype(np.float32))
    B_img_te = l2n(np.load(cc / "clip_img1024_test.npy").astype(np.float32))

    sd = torch.load(s1d / "best.pth", map_location=dev, weights_only=False)
    dims = {k: sd["state_dict"][k].shape[0] for k in
            ("head_img.weight", "head_vith.weight", "head_attr.weight")}
    model = FactorizedEncoder(z_dim=ztr.shape[1], dim_img=dims["head_img.weight"],
                              dim_vith=dims["head_vith.weight"],
                              dim_attr=dims["head_attr.weight"]).to(dev)
    model.load_state_dict(sd["state_dict"])
    model.eval()
    with torch.no_grad():
        o_te = model(torch.from_numpy(zte).to(dev))
        o_tr = model(torch.from_numpy(ztr).to(dev))
    z_img_te = l2n(o_te["z_img"].float().cpu().numpy())
    z_img_tr = l2n(o_tr["z_img"].float().cpu().numpy())

    split = LF.load(args.split_json)
    fit_i = LF.rows_for(split, "fit", n_flat)
    val_i = LF.rows_for(split, "val_b", n_flat)

    nw3_s3 = Path(args.nw3_s3)
    have_nw3 = (nw3_s3 / "z_fused_test.npy").is_file()

    # ---------------------------------------------------------------- the bars
    print("=" * 100)
    print("BARS (a condition only earns its keep if it beats the CONSTANT vector)")
    print("=" * 100)
    rows: list[dict] = []

    def add(tag: str, z: np.ndarray, target: np.ndarray = B_img_te,
            note: str = "") -> dict:
        z = l2n(np.asarray(z, dtype=np.float32))
        t = l2n(np.asarray(target, dtype=np.float32))
        vs_true = float((z * t).sum(1).mean())          # mean cosine to the TRUE target
        best = float((z * t).sum(1).max())
        r = {"tag": tag, "vs_true": round(vs_true, 4), "vs_true_best": round(best, 4),
             "rowcos": round(row2mean(z), 4), "offdiag": round(offdiag(z), 4),
             "erank": round(erank(z), 1), "dim": int(z.shape[1]),
             "retrieval_vs_gt_test": retrieval(z, B_img_te), "note": note}
        rows.append(r)
        print(f"  {tag:<38} vs_true={vs_true:.4f} (best {best:.4f})  "
              f"rowcos={r['rowcos']:.4f} offdiag={r['offdiag']:.4f} erank={r['erank']:5.1f}"
              f"  top1={r['retrieval_vs_gt_test']['top1']:.3f}  {note}")
        return r

    add("__constant(uniform)", np.ones((len(B_img_te), 1024), dtype=np.float32) / np.sqrt(1024),
        note="FLOOR: the adapter sees nothing trial-specific")
    add("__bank_mean(constant)", B_img_tr.mean(0, keepdims=True).repeat(len(B_img_te), 0),
        note="FLOOR: the real CLIP centroid")
    print()

    print("=" * 100)
    print("CANDIDATES")
    print("=" * 100)
    add("S1 head_img RAW (l2n only)", z_img_te, note="our healthy retriever")
    for k in (1, 2, 8):
        for g in (0.0, 0.5):
            zp, _ = manifold_project(z_img_te, B_img_tr, k, args.proj_tau, g)
            add(f"manifold_proj k={k} g={g}", zp)
    if have_nw3:
        zf = l2n(np.load(nw3_s3 / "z_fused_test.npy").astype(np.float32))
        add("nw3 z_fused (what nw3 generated)", zf,
            note="nw3 best: inception 0.5236 / clip 0.6164")
        zw = l2n(np.load(nw3_s3 / "z_img_test.npy").astype(np.float32))
        add("nw3 z_img (S2 projected)", zw)
        zs = l2n(np.load(nw3_s3.parent.parent / "s1/conds/z_sem_test.npy").astype(np.float32))
        add("nw3 z_sem (S1 head)", zs, note="nw3 never generated from this")

    # ORACLE ceiling: feed the true test clip_img embedding itself
    add("ORACLE gt clip_img (ceiling)", B_img_te,
        note="the condition that generated ack_s08 uck_oracle=0.8835 inception")

    # ---------------------------------------------------------------- verdict
    print()
    print("=" * 100)
    print("VERDICT")
    print("=" * 100)
    const_bar = [r for r in rows if r["tag"].startswith("__")][0]["vs_true"]
    oracle = [r for r in rows if r["tag"].startswith("ORACLE")][0]["vs_true"]
    verdict = []
    for r in rows:
        if r["tag"].startswith("__") or r["tag"].startswith("ORACLE"):
            continue
        d = r["vs_true"] - const_bar
        frac = (r["vs_true"] - const_bar) / max(oracle - const_bar, 1e-9)
        r["above_constant"] = round(d, 4)
        r["frac_of_headroom"] = round(frac, 4)
        if d <= 0.002:
            verdict.append(f"FAIL  {r['tag']}: vs_true {r['vs_true']:.4f} <= constant "
                           f"{const_bar:.4f} -- adapter sees no better than a blank "
                           f"condition; do NOT spend generation budget on it")
        else:
            verdict.append(f"OK    {r['tag']}: +{d:.4f} over constant "
                           f"({100 * frac:.1f}% of the oracle headroom "
                           f"{oracle - const_bar:.4f})")
    for v in verdict:
        print(f"  {v}")

    rep = {"stage": "nw4_diag_vstrue", "subject": sid,
           "bars": {"constant": const_bar, "oracle": oracle},
           "candidates": rows, "verdict": verdict}
    js = json.dumps(rep, indent=2)
    if args.out:
        Path(args.out).write_text(js, encoding="utf-8")
        print(f"\n[diag] wrote {args.out}")


if __name__ == "__main__":
    main()
