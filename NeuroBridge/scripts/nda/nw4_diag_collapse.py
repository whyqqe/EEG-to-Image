#!/usr/bin/env python
"""Trace WHERE a conditioning bank collapses.

The first nw4 S2 run emitted conditions that were *worse* than nw3's:

    img   offdiag raw->proj 0.9065->0.8914   (nw3 S3 it had to beat: 0.6994)
    attr  offdiag raw->proj 0.9285->0.8299
    fused                     0.9168

`row2mean` ~0.94 means every row is 94% aligned with the average row, i.e. the
bank carries almost no per-trial variation.  That can be produced at three
different places, and they need completely different fixes:

  (1) S1's head      -- the trained NCE head may already be a constant.
                        FIX: none needed in S2; the encoder is at fault.
  (2) the EEG->bank ridge `hte @ W`
                        FIX: ridge minimises MSE, so under a weak per-trial
                        signal its optimum IS "predict the conditional mean".
                        Replace it with the contrastive head instead.
  (3) manifold_project -- the top-K convex combination may be pulling every row
                        onto the same few bank rows (k too small, gamma too
                        large).  FIX: retune k/gamma.

Guessing between (1)(2)(3) is what produced the bad run.  This script measures
each stage on the same rows, then prints the verdict.

Usage:
  python scripts/nda/nw4_diag_collapse.py --s1 outputs/nw4/sub-08/s1 \
      --out outputs/nw4/sub-08/diag_collapse.json
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

from nw4_s1_train import FactorizedEncoder, VITH_LEVELS, ATTR_FIELDS  # noqa: E402
from nw4_s2_project import (l2n, row2mean, offdiag, erank, retrieval,  # noqa: E402
                            manifold_project, fit_ridge, concept_means)
import leakfree as LF  # noqa: E402


def line(tag: str, z: np.ndarray, gal_tr: np.ndarray | None = None) -> dict:
    """Collapse fingerprint of one conditioning bank."""
    z = l2n(np.asarray(z, dtype=np.float32))
    d = {"dim": int(z.shape[1]), "row2mean": round(row2mean(z), 4),
         "offdiag": round(offdiag(z), 4), "erank": round(erank(z), 1),
         "std_across_rows": round(float(z.std(0).mean()), 5)}
    tag_w = f"{tag:<34}"
    print(f"  {tag_w} r2m={d['row2mean']:.4f} offdiag={d['offdiag']:.4f} "
          f"erank={d['erank']:5.1f} rowstd={d['std_across_rows']:.5f}"
          + (f"  n_gal={gal_tr.shape[0]}" if gal_tr is not None else ""))
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1", type=str, required=True)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--test-subject", type=int, default=8)
    ap.add_argument("--z-root", type=str, default=str(NB_ROOT / "outputs/ocf/intra_z"))
    ap.add_argument("--cond-cache", type=str, default=str(NB_ROOT / "outputs/gem/cond_cache"))
    ap.add_argument("--g2-targets", type=str, default=str(NB_ROOT / "outputs/g2/targets"))
    ap.add_argument("--split-json", type=str, default=str(NB_ROOT / "outputs/leakfree/split.json"))
    ap.add_argument("--alphas", type=str, default="0.1,1,10,100,1000")
    ap.add_argument("--proj-tau", type=float, default=0.07)
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    sid = f"{args.test_subject:02d}"
    s1d = Path(args.s1)
    cc, g2 = Path(args.cond_cache), Path(args.g2_targets)
    alphas = [float(x) for x in args.alphas.split(",")]

    ztr = l2n(np.load(Path(args.z_root) / f"sub-{sid}" / "shared_r_train.npy").astype(np.float32))
    zte = l2n(np.load(Path(args.z_root) / f"sub-{sid}" / "shared_r_test.npy").astype(np.float32))
    n_flat = len(ztr)
    reps = n_flat // 1654
    cid_tr = np.arange(n_flat, dtype=np.int64) // reps
    print(f"[diag] {n_flat} train rows = 1654 concepts x {reps} reps; test {zte.shape}")

    sd = torch.load(s1d / "best.pth", map_location=dev, weights_only=False)
    dims = {k: sd["state_dict"][k].shape[0] for k in
            ("head_img.weight", "head_vith.weight", "head_attr.weight")}
    model = FactorizedEncoder(z_dim=ztr.shape[1], dim_img=dims["head_img.weight"],
                              dim_vith=dims["head_vith.weight"],
                              dim_attr=dims["head_attr.weight"]).to(dev)
    model.load_state_dict(sd["state_dict"])
    model.eval()

    def run(z: np.ndarray) -> dict:
        with torch.no_grad():
            o = model(torch.from_numpy(np.ascontiguousarray(z, dtype=np.float32)).to(dev))
        return {k: o[k].float().cpu().numpy().astype(np.float32)
                for k in ("z_img", "z_vith", "z_attr")}

    htr = np.empty((0,), dtype=np.float32)

    def trunk(z: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            h = model.trunk(torch.from_numpy(np.ascontiguousarray(z, dtype=np.float32)).to(dev))
        return h.float().cpu().numpy().astype(np.float32)

    O_tr, O_te = run(ztr), run(zte)
    H_tr, H_te = trunk(ztr), trunk(zte)

    # reference banks, in the anchor (clip_img) space
    B_img_tr = l2n(np.load(cc / "clip_img1024_train.npy").astype(np.float32))
    B_img_te = l2n(np.load(cc / "clip_img1024_test.npy").astype(np.float32))
    B_attr_tr = l2n(np.concatenate(
        [np.load(g2 / f"sem_{f}_train.npy").astype(np.float32) for f in ATTR_FIELDS], 1))
    B_attr_te = l2n(np.concatenate(
        [np.load(g2 / f"sem_{f}_test.npy").astype(np.float32) for f in ATTR_FIELDS], 1))

    split = LF.load(args.split_json)
    fit_i = LF.rows_for(split, "fit", n_flat)
    val_i = LF.rows_for(split, "val_b", n_flat)

    rep: dict = {"stage": "nw4_diag_collapse", "subject": sid, "stages": {}}

    print("\n[REFERENCE] the real manifolds (what a healthy bank looks like)")
    rep["stages"]["ref_clip_img_test_gt"] = line("GT clip_img test bank (200)", B_img_te)
    rep["stages"]["ref_clip_img_train_concept_mean"] = line(
        "GT clip_img train CONCEPT MEANS",
        concept_means(B_img_tr, cid_tr, 1654))
    rep["stages"]["ref_attr_test_gt"] = line("GT attr test bank (200)", B_attr_te)

    print("\n[1] S1 HEAD OUTPUT  (if this is collapsed, S2 cannot fix it)")
    rep["stages"]["s1_head_z_img_test"] = line("S1 head_img(z_test)", O_te["z_img"])
    rep["stages"]["s1_head_z_img_train"] = line("S1 head_img(z_train)", O_tr["z_img"])
    rep["stages"]["s1_head_z_vith_test"] = line("S1 head_vith(z_test)", O_te["z_vith"])
    rep["stages"]["s1_head_z_attr_test"] = line("S1 head_attr(z_test)", O_te["z_attr"])
    rep["stages"]["s1_head_z_img_test_own"] = {
        "test_vs_gt_test": retrieval(O_te["z_img"], B_img_te),
        "train_vs_gt_train": retrieval(O_tr["z_img"], B_img_tr),
    }
    print(f"    head_img retrieval: test_vs_gt_test "
          f"{rep['stages']['s1_head_z_img_test_own']['test_vs_gt_test']}")
    print(f"    head_img retrieval: train_vs_gt_train "
          f"{rep['stages']['s1_head_z_img_test_own']['train_vs_gt_train']}")

    print("\n[2] EEG->BANK RIDGE  (least squares -> conditional mean)")
    ridges = {}
    for tag, B, B_te in (("img", B_img_tr, B_img_te), ("attr", B_attr_tr, B_attr_te)):
        r = fit_ridge(H_tr[fit_i], B[fit_i], alphas)
        ridges[tag] = r["W"]
        rep["stages"][f"ridge_{tag}_u_test"] = line(f"ridge img<-EEG  u_test ({tag})",
                                                    H_te @ r["W"])
        rep["stages"][f"ridge_{tag}_u_train"] = line(f"ridge img<-EEG  u_train ({tag})",
                                                     H_tr @ r["W"])
        rep["stages"][f"ridge_{tag}_retrieval"] = {
            "alpha": r["alpha"],
            "test_vs_bank": retrieval(l2n(H_te @ r["W"]), B_te),
        }
        print(f"    alpha={r['alpha']:<7g} test_vs_{tag}_bank "
              f"{rep['stages'][f'ridge_{tag}_retrieval']['test_vs_bank']}")

    print("\n[3] IMAGE-SPACE bank->anchor map  (attrs -> clip_img, no EEG involved)")
    P = fit_ridge(B_attr_tr[fit_i], B_img_tr[fit_i], alphas)["W"]
    rep["stages"]["attr_bank_mapped_test"] = line("attr bank @ P (image-space map)",
                                                 B_attr_te @ P)
    rep["stages"]["attr_bank_mapped_train"] = line("attr train @ P (image-space map)",
                                                  B_attr_tr @ P)
    rep["stages"]["attr_bank_mapped_retrieval"] = retrieval(l2n(B_attr_te @ P), B_img_te)
    print(f"    retrieval vs GT clip_img test bank: "
          f"{rep['stages']['attr_bank_mapped_retrieval']}")

    print("\n[4] manifold_project  (top-K convex combination, k/gamma sweep)")
    u_te_img = l2n(H_te @ ridges["img"])
    for k in (1, 2, 4, 8, 16):
        for g in (0.0, 0.2, 0.4, 0.8):
            zp, _ = manifold_project(u_te_img, B_img_tr, k, 0.07, g)
            d = line(f"proj k={k} gamma={g}", zp)
            rep["stages"][f"proj_k{k}_g{g}"] = d
    # what the GT bank looks like when pushed through the same machinery, i.e. the
    # ceiling for any top-K projection of a *perfect* query
    zp_gt, _ = manifold_project(B_img_te, B_img_tr, 2, 0.07, 0.2)
    rep["stages"]["proj_oracle_gt_query"] = line("proj of GT query (oracle ceiling)", zp_gt)

    # ---- [5] THE decisive sweep: project the GOOD query (S1's head output) ----
    # Stage [4] fed a collapsed ridge output, so it only showed that collapsing
    # survives projection.  The question that actually sets the S2 design is
    # whether projecting the *healthy* head output costs its discriminability.
    # The head is NCE-aligned with clip_img, so cosine against the bank is already
    # meaningful (stage [1] retrieval is exactly that comparison).
    print("\n[5] manifold_project on the S1 HEAD query (does the manifold cost us?)")
    q_head = l2n(O_te["z_img"])
    rep["stages"]["head_query_raw_retrieval"] = retrieval(q_head, B_img_te)
    print(f"    raw head query            "
          f"{'':<3}r2m={row2mean(q_head):.4f} offdiag={offdiag(q_head):.4f} "
          f"-> test_vs_gt_test {rep['stages']['head_query_raw_retrieval']}")
    for k in (1, 2, 4, 8):
        for g in (0.0, 0.2, 0.5):
            zp, _ = manifold_project(q_head, B_img_tr, k, args.proj_tau, g)
            d = line(f"head-proj k={k} gamma={g}", zp)
            d["retrieval_vs_gt_test"] = retrieval(zp, B_img_te)
            d["on_manifold_gap"] = round(
                float(np.linalg.norm(zp - l2n(B_img_tr[
                    (l2n(zp) @ l2n(B_img_tr).T).argmax(1)]), axis=1).mean()), 5)
            rep["stages"][f"headproj_k{k}_g{g}"] = d
            print(f"        -> top1={d['retrieval_vs_gt_test']['top1']:.3f} "
                  f"csls={d['retrieval_vs_gt_test']['top1_csls']:.3f} "
                  f"gap_to_nn_bankrow={d['on_manifold_gap']:.4f}")
    # and the plain L2-normalised head output (the "no projection at all" arm)
    print(f"    raw  head (l2n only)      "
          f"{'':<3}r2m={row2mean(q_head):.4f} offdiag={offdiag(q_head):.4f} "
          f"-> top1={rep['stages']['head_query_raw_retrieval']['top1']:.3f}")

    print("\n[VERDICT]")
    s1_r2m = rep["stages"]["s1_head_z_img_test"]["row2mean"]
    ridge_r2m = rep["stages"]["ridge_img_u_test"]["row2mean"]
    verdict = []
    if s1_r2m > 0.95:
        verdict.append("S1 head is itself near-constant -> retrain the encoder (A0/A1), "
                       "S2 is not the problem")
    else:
        verdict.append(f"S1 head is healthy (r2m={s1_r2m:.3f})")
    if ridge_r2m - s1_r2m > 0.05:
        verdict.append(f"the ridge ADDS collapse ({s1_r2m:.3f} -> {ridge_r2m:.3f}) -> "
                       "drop the EEG->bank ridge and use the contrastive head output")
    else:
        verdict.append(f"the ridge does not add collapse ({s1_r2m:.3f} -> {ridge_r2m:.3f})")
    raw_top1 = rep["stages"]["head_query_raw_retrieval"]["top1"]
    proj_best = max(
        (d for k, d in rep["stages"].items() if k.startswith("headproj_")),
        key=lambda d: d["retrieval_vs_gt_test"]["top1_csls"])
    proj_top1 = proj_best["retrieval_vs_gt_test"]["top1"]
    verdict.append(f"projecting the head query: top1 {raw_top1:.3f} -> {proj_top1:.3f} "
                   f"(r2m {row2mean(q_head):.3f} -> {proj_best['row2mean']:.3f}, "
                   f"gap {proj_best['on_manifold_gap']:.4f})")
    if proj_top1 >= raw_top1 - 0.03:
        verdict.append("the manifold projection is ~free on the head query -> keep it "
                       "for IP-Adapter compatibility")
    else:
        verdict.append("the manifold projection COSTS discriminability -> emit BOTH the "
                       "projected and the raw head condition as separate arms")
    for v in verdict:
        print(f"  - {v}")
    rep["verdict"] = verdict

    js = json.dumps(rep, indent=2)
    if args.out:
        Path(args.out).write_text(js, encoding="utf-8")
        print(f"\n[diag] wrote {args.out}")
    else:
        print(js)


if __name__ == "__main__":
    main()
