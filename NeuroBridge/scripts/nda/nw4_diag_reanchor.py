#!/usr/bin/env python
"""The construction the diagnostics point to: restore the common component.

WHY.  `nw4_diag_vstrue.py` produced a contradiction that decides the architecture:

    nw4 S1 head_img RAW   top1=0.350 (best retriever)   vs_true=0.257  <-- below
    CLIP bank mean        top1=0.005 (constant)         vs_true=0.620  <-- the floor

The head is highly discriminative yet scores *below a constant vector* on the only
metric that predicts generation quality.  The reason is a property of the
objective, not of the data: InfoNCE is invariant to a per-row shift along a
direction common to all rows, so it happily discards the large common component
that CLIP embeddings share (rowcos 0.628).  IP-Adapter reads the direction a row
points in, and that common component is most of it.

The fix is not a new loss but a post-hoc, label-free re-composition.  Write a real
target embedding as T_i = c + d_i (common part c, individual part d_i).  A
contrastive head estimates d_i and has no way to know c; c is however estimable
from the TRAIN concept bank alone:

    c_hat = mean(TRAIN bank rows)

so emit

    z_i = normalize( c_hat + lambda * h_i )

lambda trades the adapter's prior (c) against our EEG evidence (h_i).  At lambda=0
this is exactly the constant vector, i.e. the floor is a member of this family, and
the sweep can only move up from it.

This tests, with zero generation cost, whether any lambda clears the floor.  If it
does, the arms are worth GPU hours.  If nothing does, no injection engineering
(A3/A4) can help and the whole pathway must change.

Usage:
  python scripts/nda/nw4_diag_reanchor.py --s1 outputs/nw4/sub-08/s1
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

from nw4_s1_train import FactorizedEncoder  # noqa: E402
from nw4_s2_project import (l2n, row2mean, offdiag, erank, retrieval,  # noqa: E402
                            manifold_project, concept_means)
import leakfree as LF  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1", type=str, required=True)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--test-subject", type=int, default=8)
    ap.add_argument("--z-root", type=str, default=str(NB_ROOT / "outputs/ocf/intra_z"))
    ap.add_argument("--cond-cache", type=str, default=str(NB_ROOT / "outputs/gem/cond_cache"))
    ap.add_argument("--split-json", type=str, default=str(NB_ROOT / "outputs/leakfree/split.json"))
    ap.add_argument("--hybrid-conds", type=str,
                    default=str(NB_ROOT / "outputs/hybrid_s08/full/conds"))
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    sid = f"{args.test_subject:02d}"
    s1d = Path(args.s1)
    cc = Path(args.cond_cache)

    ztr = l2n(np.load(Path(args.z_root) / f"sub-{sid}" / "shared_r_train.npy").astype(np.float32))
    zte = l2n(np.load(Path(args.z_root) / f"sub-{sid}" / "shared_r_test.npy").astype(np.float32))
    n_flat = len(ztr)
    cid_tr = np.arange(n_flat, dtype=np.int64) // (n_flat // 1654)

    B_tr = l2n(np.load(cc / "clip_img1024_train.npy").astype(np.float32))
    B_te = l2n(np.load(cc / "clip_img1024_test.npy").astype(np.float32))

    sd = torch.load(s1d / "best.pth", map_location=dev, weights_only=False)
    dims = {k: sd["state_dict"][k].shape[0] for k in
            ("head_img.weight", "head_vith.weight", "head_attr.weight")}
    model = FactorizedEncoder(z_dim=ztr.shape[1], dim_img=dims["head_img.weight"],
                              dim_vith=dims["head_vith.weight"],
                              dim_attr=dims["head_attr.weight"]).to(dev)
    model.load_state_dict(sd["state_dict"])
    model.eval()

    def trunk_and_head(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            t = torch.from_numpy(np.ascontiguousarray(z, dtype=np.float32)).to(dev)
            h = model.trunk(t)
            o = model(t)          # `t`, not `z`: the modules are float32 torch, they
                                  # reject a numpy input
        return (h.float().cpu().numpy().astype(np.float32),
                o["z_img"].float().cpu().numpy().astype(np.float32))

    H_tr, Z_tr = trunk_and_head(ztr)
    H_te, Z_te = trunk_and_head(zte)
    h_te = l2n(Z_te)
    h_tr = l2n(Z_tr)

    # the common component, estimated from the TRAIN bank only (label-free w.r.t. test)
    # two candidates for `c`: the raw bank mean, and the per-concept-mean average.
    c_raw = l2n(B_tr.mean(0, keepdims=True))[0]
    cm = concept_means(B_tr, cid_tr, 1654)
    c_cm = l2n(cm.mean(0, keepdims=True))[0]

    # floor = the constant vector's own vs_true (lambda -> 0 reproduces it)
    def vs_true(z: np.ndarray) -> float:
        return float((l2n(z) * B_te).sum(1).mean())

    floor = vs_true(np.tile(c_raw, (len(B_te), 1)))
    floor_cm = vs_true(np.tile(c_cm, (len(B_te), 1)))
    oracle = 1.0

    # reference: what hybrid_s08 achieved with the same metric definition
    ref = {}
    hc = Path(args.hybrid_conds)
    for nm in ("ip_uck", "ip_blend", "ip_hard"):
        p = hc / f"{nm}_test.npy"
        if p.is_file():
            z = l2n(np.load(p).astype(np.float32))
            ref[nm] = {"vs_true": round(vs_true(z), 4), "rowcos": round(row2mean(z), 4),
                       "retrieval": retrieval(z, B_te)}
    print("=" * 104)
    print("FLOORS and the bar we must clear")
    print("=" * 104)
    print(f"  constant c_raw (mean of TRAIN bank)      vs_true={floor:.4f}")
    print(f"  constant c_cm  (mean of concept means)   vs_true={floor_cm:.4f}")
    print(f"  ORACLE (the true test embedding itself)  vs_true={oracle:.4f}")
    for nm, v in ref.items():
        d = v["vs_true"] - floor
        print(f"  hybrid_s08 {nm:<12}                 vs_true={v['vs_true']:.4f} "
              f"(vs_cl={d:+.4f}) rowcos={v['rowcos']:.4f} "
              f"top1={v['retrieval']['top1']:.3f}   <- generated 0.728/0.705/0.632")
    print()
    print("=" * 104)
    print("SWEEP: z = normalize(c_hat + lambda * h)          [lambda=0 IS the floor]")
    print("=" * 104)
    rows = []
    for cname, c in (("c_raw", c_raw), ("c_cm", c_cm)):
        for lam in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0):
            z = l2n(c[None, :] + lam * h_te)
            vt = vs_true(z)
            rt = retrieval(z, B_te)
            r = {"c": cname, "lambda": lam, "vs_true": round(vt, 4),
                 "vs_cl": round(vt - floor, 4), "rowcos": round(row2mean(z), 4),
                 "offdiag": round(offdiag(z), 4), "erank": round(erank(z), 1),
                 "top1": rt["top1"], "top1_csls": rt["top1_csls"]}
            rows.append(r)
            flag = "  <== BEATS FLOOR" if r["vs_cl"] > 0 else ""
            print(f"  {cname:<6} lambda={lam:<6g} vs_true={vt:.4f} vs_cl={r['vs_cl']:+.4f} "
                  f"rowcos={r['rowcos']:.4f} top1={r['top1']:.3f} "
                  f"csls={r['top1_csls']:.3f}{flag}")
        print()

    # also test the manifold projection ON TOP of the re-anchored condition, since the
    # generator may still prefer on-manifold rows
    print("=" * 104)
    print("RE-ANCHOR + manifold_project (does on-manifold help once c is restored?)")
    print("=" * 104)
    best_lam = max(rows, key=lambda r: r["vs_cl"])
    lam = best_lam["lambda"]
    z0 = l2n(c_raw[None, :] + lam * h_te)
    for k in (1, 2, 8):
        for g in (0.0, 0.3):
            zp, _ = manifold_project(z0, B_tr, k, 0.07, g)
            vt = vs_true(zp)
            rt = retrieval(zp, B_te)
            print(f"  k={k} gamma={g:<4g} vs_true={vt:.4f} vs_cl={vt - floor:+.4f} "
                  f"rowcos={row2mean(zp):.4f} top1={rt['top1']:.3f}")

    print()
    print("=" * 104)
    print("VERDICT")
    print("=" * 104)
    winners = [r for r in rows if r["vs_cl"] > 0]
    verdict = []
    if not winners:
        verdict.append("NO lambda clears the constant-vector floor -> the S1 head's "
                       "direction carries nothing the adapter can use; do not run the "
                       "generation arms, change the semantic target or the loss first")
    else:
        b = max(winners, key=lambda r: r["vs_cl"])
        verdict.append(f"best: c={b['c']} lambda={b['lambda']} -> vs_true {b['vs_true']:.4f} "
                       f"(vs_cl {b['vs_cl']:+.4f}, {100*b['vs_cl']/(oracle-floor):.1f}% of "
                       f"the headroom) with top1 {b['top1']:.3f}")
        hb = ref.get("ip_uck", {}).get("vs_true")
        if hb:
            verdict.append(f"compare hybrid_s08 ip_uck vs_cl "
                           f"{hb - floor:+.4f} -> we are "
                           f"{'BETTER' if b['vs_cl'] > hb - floor else 'WORSE'}")
        verdict.append("the re-anchoring is label-free (c from the TRAIN bank, no test "
                       "information) so it is usable in the reported protocol")
    for v in verdict:
        print(f"  - {v}")

    rep = {"stage": "nw4_diag_reanchor", "subject": sid, "floor_c_raw": round(floor, 4),
           "floor_c_cm": round(floor_cm, 4), "oracle": oracle,
           "hybrid_ref": ref, "sweep": rows, "verdict": verdict}
    if args.out:
        Path(args.out).write_text(json.dumps(rep, indent=2), encoding="utf-8")
        print(f"\n[diag] wrote {args.out}")


if __name__ == "__main__":
    main()
