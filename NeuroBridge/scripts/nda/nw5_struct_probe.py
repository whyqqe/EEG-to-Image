#!/usr/bin/env python
"""Is a linear EEG -> structure(CLIP) map good enough to be worth a trained head?

WHY THIS EXISTS
---------------
The NW5 sub-08 run isolated a single bottleneck.  Same operator, same subject, same
conditions -- only the *structure* branch changes:

    A1  semantic only                       incep 0.7116  clip 0.8245
    A2  + GT depth + GT edge (feature sp.)  incep 0.8394  clip 0.9094   (+0.113/+0.085)
    A8  + our UCK depth  -> CLIP            incep 0.7077  clip 0.8244   (+0.000!)
    A3b same mass, uniform layout           incep 0.7280  clip 0.8265
    A9  full GT oracle                      incep 0.9831  clip 0.9920

So real structure is worth +0.11 Inception, our current EEG-derived structure is worth
exactly nothing, and the whole remaining gap (0.824 vs 0.909) lives in the EEG->structure
map.  Before spending a training run on a new head, this script asks the two questions
that decide whether such a head can work at all -- both answerable with a closed-form
ridge, no training:

  Q1 ABSOLUTE -- can EEG predict depth/edge CLIP above what we already achieve?
     Our shipped path (UCK depth head -> image -> OpenCLIP) reaches cos-to-own-true
     0.4757 for depth.  If a linear probe cannot beat that, a bigger head is not the fix.

  Q2 INCREMENTAL -- does the predictable part carry anything the semantic head lacks?
     Depth is largely determined by the CONCEPT (every chair's depth map looks chair-like).
     If EEG -> depth only recovers the concept mean, then it is redundant with the
     semantic head and explains the A8 null result exactly.  The discriminating quantity
     is therefore the WITHIN-CONCEPT residual, not the raw row.

Split hygiene mirrors the rest of the project: ridge coefficients are fit on `fit_rows`
only, alpha is chosen on `val_b_rows` only, and the 200 test rows are read once for the
reported numbers.  No test row touches training.

Usage:
  python scripts/nda/nw5_struct_probe.py --subject 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

NB_ROOT = Path(__file__).resolve().parents[2]


def l2n(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)


def concept_ids(n_concepts: int, reps: int, n_rows: int) -> np.ndarray:
    """Row -> concept id. Train is laid out concept-major, `reps` rows each."""
    cid = np.repeat(np.arange(n_concepts), reps)
    if len(cid) < n_rows:
        cid = np.concatenate([cid, np.full(n_rows - len(cid), n_concepts - 1)])
    return cid[:n_rows]


def ridge_fit(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Closed-form ridge with an intercept; `z` is expected L2-normalised."""
    zc = np.concatenate([z, np.ones((len(z), 1), dtype=z.dtype)], axis=1)
    g = zc.T @ zc
    return np.linalg.solve(g, zc.T @ y)


def ridge_apply(w: np.ndarray, z: np.ndarray) -> np.ndarray:
    zc = np.concatenate([z, np.ones((len(z), 1), dtype=z.dtype)], axis=1)
    return zc @ w


def evaluate(pred: np.ndarray, tgt: np.ndarray, cid_te: np.ndarray | None = None,
             cid_tr: np.ndarray | None = None, tgt_tr: np.ndarray | None = None) -> dict:
    """Fidelity + identifiability of a predicted bank against its own true bank."""
    p, t = l2n(pred.astype(np.float64)), l2n(tgt.astype(np.float64))
    sim = p @ t.T
    n = len(p)
    diag = float(np.diag(sim).mean())
    order = np.argsort(-sim, axis=1)
    top1 = float((order[:, 0] == np.arange(n)).mean())
    top5 = float(np.mean([np.isin(i, order[i, :5]) for i in range(n)]))
    rank = float(np.mean([np.where(order[i] == i)[0][0] + 1 for i in range(n)]))
    iu = np.triu_indices(n, 1)
    out = {"diag_cos": round(diag, 4), "top1": round(top1, 4), "top5": round(top5, 4),
           "mean_rank": round(rank, 2),
           "rsa": round(float(np.corrcoef((p @ p.T)[iu], (t @ t.T)[iu])[0, 1]), 4)}
    return out


def within_concept_top1(pred: np.ndarray, tgt: np.ndarray, concepts: np.ndarray) -> dict:
    """Is a prediction IMAGE-specific, or does it only know which concept it is?

    Each held-out concept appears `reps` times, and a predictor that knows only the
    concept would emit the same vector for all its repetitions -- so its own rep is
    indistinguishable from the other nine.  Scoring identification WITHIN each concept
    (candidates are that concept's own true targets, chance = 1/reps) therefore ranks
    concept-priors at the bottom and genuinely image-specific predictions near the top.
    This is the quantity that decides Q2: the semantic head's target is constant across
    a concept's repetitions by construction, so any structure branch that is also
    concept-constant adds nothing, which is exactly what A8's null result would look
    like from the inside.
    """
    p, t = l2n(pred.astype(np.float64)), l2n(tgt.astype(np.float64))
    hits, tot, margins = 0, 0, []
    for c in np.unique(concepts):
        idx = np.where(concepts == c)[0]
        if len(idx) < 2:
            continue
        sim = p[idx] @ t[idx].T                      # rows: reps, cols: true reps
        order = np.argsort(-sim, axis=1)
        hits += int((order[:, 0] == np.arange(len(idx))).sum())
        tot += len(idx)
        # how much the true rep beats the average other rep of the SAME concept
        own = np.diag(sim)
        other = (sim.sum(1) - own) / (len(idx) - 1)
        margins.append(float((own - other).mean()))
    return {"within_top1": round(hits / max(tot, 1), 4),
            "chance": round(1.0 / max(int(np.bincount(concepts.astype(int)).max()), 2), 4),
            "own_minus_other_cos": round(float(np.mean(margins)), 4),
            "n_rows": tot}


def constant_baseline(ytr: np.ndarray, yte: np.ndarray) -> dict:
    """What a CONSTANT condition already scores on this bank.

    This is not optional.  The depth-CLIP bank is heavily concentrated (mean pairwise
    cosine 0.57), so its train mean already sits at cosine ~0.68 to every test row --
    which means raw `diag_cos` flatters any predictor, including a useless one.  The
    quantity that carries information is the EXCESS over this baseline, together with
    top1 (chance = 1/n) and RSA.  Learned the hard way once already: `vs_cl` had to be
    redefined against a constant for the same reason.
    """
    c = l2n(ytr.mean(0, keepdims=True))
    sim = l2n(yte.astype(np.float64)) @ c.T
    return {"diag_cos": round(float(sim.mean()), 4),
            "diag_cos_excess": 0.0,
            "top1": 0.0, "rsa": None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, default=8)
    ap.add_argument("--z-root", type=str, default=str(NB_ROOT / "outputs/ocf/intra_z"))
    ap.add_argument("--cond-cache", type=str, default=str(NB_ROOT / "outputs/gem/cond_cache"))
    ap.add_argument("--split-json", type=str, default=str(NB_ROOT / "outputs/leakfree/split.json"))
    ap.add_argument("--alphas", type=str, default="0,0.01,0.1,1,10,100,1000")
    ap.add_argument("--emit", type=str, default="",
                    help="directory to write the ridge test predictions as generation-ready "
                         "condition banks (raw + concentration-calibrated). These are the "
                         "honest EEG-only structure conditions: leak-free, no GT anywhere.")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    sid = args.subject
    alphas = [float(a) for a in args.alphas.split(",")]
    cc = Path(args.cond_cache)
    z_dir = Path(args.z_root) / f"sub-{sid:02d}"
    ztr = l2n(np.load(z_dir / "shared_r_train.npy").astype(np.float32))
    zte = l2n(np.load(z_dir / "shared_r_test.npy").astype(np.float32))

    sp = json.loads(Path(args.split_json).read_text())
    fit_i = np.asarray(sp["fit_rows"], dtype=int)
    val_i = np.asarray(sp.get("val_b_rows") or sp["val_a_rows"], dtype=int)
    wc_i = np.asarray(sp["val_a_rows"], dtype=int)     # separate from the alpha-selection rows
    reps = int(sp["reps"])
    n_con = int(sp["n_concepts"])
    cid_tr = concept_ids(n_con, reps, len(ztr))
    # concept id for the within-concept split, indexed to its own local concept list
    wc_concepts = np.repeat(np.arange(len(sp["val_a_concepts"])), len(wc_i) // len(sp["val_a_concepts"]))
    if len(wc_concepts) < len(wc_i):
        wc_concepts = np.concatenate([wc_concepts, np.full(len(wc_i) - len(wc_concepts),
                                                           wc_concepts[-1])])

    print(f"[probe] sub-{sid}  EEG {ztr.shape}->{zte.shape}  fit={len(fit_i)} "
          f"val={len(val_i)}  within-concept rows={len(wc_i)} "
          f"({len(sp['val_a_concepts'])} concepts x {len(wc_i)//len(sp['val_a_concepts'])})  "
          f"alphas={alphas}")

    # ---- targets ------------------------------------------------------------
    targets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, fn in (("img_clip", "clip_img1024"), ("depth_clip", "clip_depth1024"),
                     ("edge_clip", "clip_edge1024")):
        tr = np.load(cc / f"{fn}_train.npy").astype(np.float32)
        te = np.load(cc / f"{fn}_test.npy").astype(np.float32)
        targets[name] = (l2n(tr), l2n(te))
    # what our shipped A8 path achieved, for the comparison that matters
    eeg_d = NB_ROOT / "outputs/nw5_s08/conds" / f"eeg_depth1024_cal_sub-{sid:02d}_test.npy"
    eeg_e = NB_ROOT / "outputs/nw5_s08/conds" / f"eeg_edge1024_cal_sub-{sid:02d}_test.npy"

    results: dict[str, dict] = {}
    ridge_w: dict[str, np.ndarray] = {}
    for name, (ytr, yte) in targets.items():
        const = constant_baseline(ytr, yte)
        best = None
        for a in alphas:
            zf = np.concatenate([ztr[fit_i], np.ones((len(fit_i), 1), np.float32)], 1)
            g = zf.T @ zf
            if a > 0:
                pen = np.eye(g.shape[0], dtype=np.float32) * a
                pen[-1, -1] = 0.0                       # never penalise the intercept
                g = g + pen
            w = np.linalg.solve(g, zf.T @ ytr[fit_i])
            # alpha chosen on val ONLY
            v = evaluate(ridge_apply(w, ztr[val_i]), ytr[val_i])
            if best is None or v["top1"] > best[1]["top1"]:
                best = (a, v, w)
        a, val, w = best
        ridge_w[name] = w
        te = evaluate(ridge_apply(w, zte), yte)
        te["diag_cos_excess"] = round(te["diag_cos"] - const["diag_cos"], 4)
        wc = within_concept_top1(ridge_apply(w, ztr[wc_i]), ytr[wc_i], wc_concepts)
        # reference points for the within-concept scale
        wc_ref = {
            "gt_self": within_concept_top1(ytr[wc_i], ytr[wc_i], wc_concepts),
            "concept_prior": within_concept_top1(
                np.stack([ytr[wc_i][wc_concepts == c].mean(0)
                          for c in range(len(sp["val_a_concepts"]))])[wc_concepts],
                ytr[wc_i], wc_concepts),
        }
        results[name] = {"alpha": a, "val": val, "test": te, "constant": const,
                         "within_concept": wc, "within_concept_refs": wc_ref}
        chance = 1.0 / len(yte)
        print(f"\n[{name}] alpha={a:g}  (selected on val, top1={val['top1']:.4f})")
        print(f"   constant baseline   diag_cos {const['diag_cos']:.4f}  "
              f"(this is what a USELESS condition scores)")
        print(f"   TEST  diag_cos {te['diag_cos']:.4f}  excess over constant "
              f"{te['diag_cos_excess']:+.4f}  top1 {te['top1']:.4f} "
              f"({te['top1']/chance:.1f}x chance {chance:.4f})  rsa {te['rsa']:.4f}")
        print(f"   within-concept identification (own rep among a concept's {reps}): "
              f"{wc['within_top1']:.4f}   [chance {wc['chance']:.2f}, "
              f"concept-prior {wc_ref['concept_prior']['within_top1']:.4f}, "
              f"GT {wc_ref['gt_self']['within_top1']:.4f}]")

    # ---- the comparison against what we ship today --------------------------
    print("\n" + "=" * 78)
    print("WHAT THE ARCHITECTURE MUST BEAT (our shipped EEG-derived structure)")
    print("=" * 78)
    for label, p, fn in (("depth", eeg_d, "clip_depth1024"), ("edge", eeg_e, "clip_edge1024")):
        if not p.is_file():
            print(f"  {label}: {p} missing"); continue
        yte = targets[f"{label}_clip"][1]
        m = evaluate(l2n(np.load(p).astype(np.float32)), yte)
        print(f"  shipped {label:<6} diag_cos {m['diag_cos']:.4f}  top1 {m['top1']:.4f}  "
              f"rsa {m['rsa']:.4f}")

    print("\n" + "=" * 78)
    print("VERDICT  (top1 is the only axis a constant cannot fake: chance = 0.0050)")
    print("=" * 78)
    for label in ("depth", "edge"):
        key = f"{label}_clip"
        lin = results[key]["test"]
        wc, ref = results[key]["within_concept"], results[key]["within_concept_refs"]
        p = eeg_d if label == "depth" else eeg_e
        print(f"  {label}")
        print(f"    ridge (leak-free EEG)  top1 {lin['top1']:.4f} "
              f"({lin['top1']/0.005:.1f}x chance)  rsa {lin['rsa']:.4f}  "
              f"diag_cos_excess {lin['diag_cos_excess']:+.4f}")
        if p.is_file():
            cur = evaluate(l2n(np.load(p).astype(np.float32)),
                           targets[f"{label}_clip"][1])
            print(f"    shipped (UCK -> Canny) top1 {cur['top1']:.4f} "
                  f"({cur['top1']/0.005:.1f}x chance)  rsa {cur['rsa']:.4f}  "
                  f"diag_cos_excess "
                  f"{cur['diag_cos'] - results[key]['constant']['diag_cos']:+.4f}")
        print(f"    within-concept {wc['within_top1']:.4f} vs concept-prior "
              f"{ref['concept_prior']['within_top1']:.4f} (chance {wc['chance']:.2f}) "
              f"-> {'trial-specific' if wc['within_top1'] > 0.35 else 'MOSTLY CONCEPT-LEVEL'}")
        print(f"    => {'ridge structure is a real signal worth feeding' if lin['top1'] > 5*0.005 else 'ridge structure is near chance; do not expect generation gains'}")

    if args.emit:
        out = Path(args.emit)
        out.mkdir(parents=True, exist_ok=True)
        wrote = []
        for label, key, fn in (("depth", "depth_clip", "clip_depth1024"),
                               ("edge", "edge_clip", "clip_edge1024")):
            w = ridge_w[key]
            # ---- honest EEG-only condition: leak-free ridge, fit on `fit`, alpha on `val`
            pred_te = l2n(ridge_apply(w, zte)).astype(np.float32)
            pred_tr = l2n(ridge_apply(w, ztr)).astype(np.float32)   # for calibration ref
            p_te = out / f"ridge_{label}1024_sub-{sid:02d}_test.npy"
            p_tr = out / f"ridge_{label}1024_sub-{sid:02d}_trainbank.npy"
            np.save(p_te, pred_te); np.save(p_tr, pred_tr)
            wrote += [p_te.name, p_tr.name]
        # semantic reference, so an "all-ridge" arm is possible too
        w = ridge_w["img_clip"]
        p_te = out / f"ridge_img1024_sub-{sid:02d}_test.npy"
        np.save(p_te, l2n(ridge_apply(w, zte)).astype(np.float32))
        wrote.append(p_te.name)
        print(f"\n[emit] {len(wrote)} raw condition banks -> {out}")
        for n in wrote:
            print(f"   {n}")
        print("   NOTE: these are RAW (uncalibrated). Calibrate with gem_calib.py "
              "--ref <GT train bank> before generation.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"subject": sid, "results": results},
                                             indent=2), encoding="utf-8")
        print(f"\n[wrote] {args.out}")


if __name__ == "__main__":
    main()
