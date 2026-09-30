#!/usr/bin/env python3
"""Does the condition actually come from the EEG, and is it above TRIVIAL?

This is the measurement stage that `run_gem_intra.sh` calls at the END of every
run, after generation and scoring.  It exists because the three numbers it
computes were, in the sub-08 GVM run, the difference between a "condition at
cos +0.66, a decent result" and the truth:

  1. THE CENTRELINE.  A condition can reach cos +0.66 to the true image embedding
     while carrying no row-specific information at all, by sitting near the MEAN
     of the target distribution.  The number to compare against is therefore
     `cos(mean(T), T_i)` -- measured at +0.6275 on sub-08 -- not zero.  Against
     that line the reported +0.6602 is a +0.0327 margin.

  2. ARM AGREEMENT, ROW BY ROW.  The `noise` arm is meant to answer "does any of
     this come from the EEG".  Its exported conditions agreed with the `full`
     arm's row for row at +0.9938 (ip_clip), +0.9905 (ip_sem), +0.9724
     (ip_fused).  A control that moves the condition by 1% has controlled for
     nothing.  (Root cause was found and fixed in `gem_train.py`: the ablation was
     applied to 1 of the model's 2 EEG inputs at 2 of 8 call sites, and not at all
     at test time.  `--noise-arm` now silences both streams through one entry
     point.)  This check stays as the regression test for that class of bug: if it
     ever reports a high agreement again, the control is broken, not the model.

  3. RETRIEVAL, WHICH IS THE HONEST HEADLINE.  Row-identity was reported as
     0.0000 for every condition including the oracle, which cannot distinguish
     "the condition is over-concentrated" from "the condition is uninformative".
     So it is measured directly here, on BOTH the full condition and its
     constant-removed residual: the residual is the only part that can identify a
     trial, and its retrieval accuracy upper-bounds everything downstream.

Never raises: every section is guarded and reports `null` when its inputs are
missing, so a diagnostic can no longer kill a job before the metrics are written.
That failure mode is exactly how job 571020 lost its evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np

ROWS_DEFAULT = ("ip_clip_test", "ip_sem_test", "ip_fused_auto_test")


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8, None)


def rowcos(x: np.ndarray) -> float:
    """Mean pairwise cosine between DISTINCT rows.  High = rows are alike."""
    z = l2(x)
    g = z @ z.T
    n = len(z)
    return float((g.sum() - np.trace(g)) / max(n * (n - 1), 1))


def retrieval(q: np.ndarray, bank: np.ndarray, labels: list[str] | None
              ) -> dict:
    """Row identity and same-concept top-1/top-5 of `q` against `bank`.

    Row identity is the strict test: query i should retrieve bank row i.  With N
    rows chance is 1/N.  Same-concept is the official THINGS-EEG2-style protocol
    and is easier, because it only asks for the right concept.
    """
    qz, bz = l2(q), l2(bank)
    sim = qz @ bz.T
    n = len(qz)
    order = np.argsort(-sim, axis=1)
    top1 = order[:, 0]
    row_id = float((top1 == np.arange(n)).mean())
    top5 = float(np.mean([i in order[i, :5] for i in range(n)]))
    out = {"n": n, "chance_row_identity": 1.0 / max(n, 1), "row_identity_top1": row_id,
           "row_identity_top5": top5, "mean_top1_cos": float(sim[np.arange(n), top1].mean())}
    if labels is not None and len(labels) == n:
        ok1 = ok5 = 0
        for i in range(n):
            want = labels[i]
            ranked = [labels[j] for j in order[i, :5]]
            ok1 += int(ranked[0] == want)
            ok5 += int(want in ranked)
        out["same_concept_top1"] = ok1 / n
        out["same_concept_top5"] = ok5 / n
        # 1654-way gallery: the concept is "retrieved" if ANY train image of it
        # ranks above the best image of any other concept.  Rows here are test
        # rows, so this is the same-concept rate within the test bank.
    return out


def strip_constant(p: np.ndarray) -> np.ndarray:
    """Remove the component along the row-mean direction, then renormalise.

    The removed part is a CONSTANT across rows -- it is the same vector for every
    trial, so it cannot identify a trial, yet it inflates every raw cosine.  What
    survives is the only part that carries a trial.
    """
    z = l2(p)
    mu = l2(z.mean(0, keepdims=True))
    proj = (z * mu).sum(1, keepdims=True)
    r = z - proj * mu
    return r / np.clip(np.linalg.norm(r, axis=-1, keepdims=True), 1e-8, None)


def build(args) -> dict:
    rep: dict = {"out_dir": str(args.out_dir), "rows": list(args.rows),
                 "bank": str(args.bank), "errors": {}}

    T = np.load(args.bank).astype(np.float32)
    bank = l2(T)
    rep["n_test_rows"] = int(len(T))

    # ---------------------------------------------------------------- section 1
    try:
        centre = l2(T.mean(0, keepdims=True))
        c = float((centre * bank).sum(1).mean())
        best, bj = -2.0, -1
        for j in range(len(T)):
            v = float((bank[j] * bank).sum(1).mean())
            if v > best:
                best, bj = v, j
        rep["centreline"] = {"cos_mean_to_each": c, "best_single_constant": best,
                             "best_single_row": int(bj)}
        rep["centreline"]["interpretation"] = (
            f"cos(mean of true image embeddings, each true embedding) = {c:+.4f}. "
            f"This is a ROW-INDEPENDENT constant with row-identity "
            f"{1.0/len(T):.4f} by construction. Any condition whose cos is not "
            f"meaningfully above it is explained by the target mean, not by the EEG. "
            f"The best single REAL image embedding used as a constant reaches "
            f"{best:+.4f}, so a condition below that line is beaten by a constant.")
        print(f"[measure] centreline cos(mean,each) = {c:+.4f} | "
              f"best constant = {best:+.4f} (row {bj})")
    except Exception as e:                                             # noqa: BLE001
        rep["errors"]["centreline"] = f"{type(e).__name__}: {e}"
        print(f"[measure] centreline FAILED: {e}")

    # ---------------------------------------------------------------- section 2
    # ARM AGREEMENT.  This is the regression test for the broken control: the noise
    # arm must NOT reproduce the full arm row by row.
    try:
        agree: dict = {}
        for nm in args.rows:
            per: dict = {}
            pa = args.out_dir / "full" / "conds" / f"{nm}.npy"
            if not pa.is_file():
                continue
            A = np.load(pa).astype(np.float32)
            for arm in ("nofront", "noise"):
                pb = args.out_dir / arm / "conds" / f"{nm}.npy"
                if not pb.is_file():
                    continue
                B = np.load(pb).astype(np.float32)
                if B.shape != A.shape:
                    per[arm] = {"error": f"shape {B.shape} != {A.shape}"}
                    continue
                per[arm] = {
                    "row_wise_cos": float((l2(A) * l2(B)).sum(1).mean()),
                    "rowcos_full": rowcos(A), "rowcos_arm": rowcos(B)}
            agree[nm] = per
        rep["arm_agreement"] = agree
        worst = 0.0
        for nm, per in agree.items():
            for arm, v in per.items():
                if "row_wise_cos" in v:
                    print(f"[measure] {nm:<20} full vs {arm:<8} row-wise cos "
                          f"{v['row_wise_cos']:+.4f}")
                    worst = max(worst, abs(v["row_wise_cos"]))
        if worst > 0.95:
            msg = (f"CONTROL BROKEN: an arm that should be a null reproduces the "
                   f"`full` arm's condition row for row (max |cos| {worst:.4f}). "
                   f"Check that the ablation is applied to EVERY input the model "
                   f"consumes, at TRAIN, VALIDATION and EXPORT.")
            rep["arm_agreement_verdict"] = "BROKEN"
            rep["arm_agreement_warning"] = msg
            print(f"[measure] [FAIL] {msg}")
        elif agree:
            rep["arm_agreement_verdict"] = "ok"
            print(f"[measure] [ok] arm agreement is low (max |cos| {worst:.4f}): "
                  f"the arms are distinguishable, so the control does something.")
    except Exception as e:                                             # noqa: BLE001
        rep["errors"]["arm_agreement"] = f"{type(e).__name__}: {e}"
        print(f"[measure] arm agreement FAILED: {e}")
        traceback.print_exc(file=sys.stdout)

    # ---------------------------------------------------------------- section 3
    # The honest headline: retrieval, on the full condition and on the residual.
    try:
        labels = None
        if args.captions is not None and args.captions.is_dir():
            # `.jsonl` FIRST: the captions are one JSON object per line, and
            # `json.load` on such a file raises "Extra data" on the second record.
            js = [p for pat in ("captions_test.jsonl", "*test*.jsonl", "*test*.json",
                                "test*.json")
                  for p in sorted(args.captions.glob(pat))]
            if js:
                caps = []
                for line in js[0].read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line:
                        caps.append(json.loads(line))
                if not caps:
                    caps = json.loads(js[0].read_text(encoding="utf-8"))
                if isinstance(caps, dict):
                    caps = caps.get("captions", caps.get("data", []))
                labels = [Path(c["path"]).parent.name.split("_", 1)[1].replace("_", " ")
                          for c in caps]
                if len(labels) != len(T):
                    print(f"[measure] caption labels {len(labels)} != bank {len(T)}; "
                          f"same-concept numbers suppressed")
                    labels = None
                else:
                    rep["concept_labels_from"] = str(js[0])

        per_row: dict = {}
        for nm in args.rows:
            p = args.out_dir / "full" / "conds" / f"{nm}.npy"
            if not p.is_file():
                continue
            P = np.load(p).astype(np.float32)
            if len(P) != len(T):
                per_row[nm] = {"error": f"{len(P)} rows vs bank {len(T)}"}
                continue
            full = retrieval(P, T, labels)
            res = retrieval(strip_constant(P), T, labels)
            rm = float(np.linalg.norm(P - (l2(P).mean(0, keepdims=True)), axis=1).mean())
            cos_t = float((l2(P) * bank).sum(1).mean())
            cont = float((l2(P) * l2(P.mean(0, keepdims=True))).sum(1).mean())
            per_row[nm] = {
                "cos_to_target": cos_t,
                "cos_to_own_mean": cont,
                "rowcos": rowcos(P),
                "above_centreline_by": cos_t - rep.get("centreline", {})
                                             .get("cos_mean_to_each", float("nan")),
                "full": full, "residual_only": res,
                "residual_row_identity_gain": (res["row_identity_top1"]
                                               - full["row_identity_top1"])}
            print(f"[measure] {nm:<20} cos {cos_t:+.4f} | row-id full "
                  f"{full['row_identity_top1']:.4f} vs residual "
                  f"{res['row_identity_top1']:.4f} (chance "
                  f"{full['chance_row_identity']:.4f}) | same-concept top1 "
                  f"{full.get('same_concept_top1', float('nan')):.4f}")
        rep["condition_retrieval"] = per_row
    except Exception as e:                                             # noqa: BLE001
        rep["errors"]["condition_retrieval"] = f"{type(e).__name__}: {e}"
        print(f"[measure] retrieval FAILED: {e}")
        traceback.print_exc(file=sys.stdout)

    # ---------------------------------------------------------------- verdict
    try:
        v: list[str] = []
        cl = rep.get("centreline", {})
        pr = rep.get("condition_retrieval", {})
        # HEAD = the condition with the highest raw cosine, so the verdict is about
        # the best case and cannot be softened by a single mis-scaled row.
        head_nm, head = None, {}
        for nm, d in pr.items():
            if "error" in d:
                continue
            if not head or d["cos_to_target"] > head["cos_to_target"]:
                head_nm, head = nm, d
        if cl and head:
            margin = head.get("above_centreline_by")
            if margin is not None and margin < 0.0:
                v.append(f"BEST condition ({head_nm}, cos {head['cos_to_target']:+.4f}) "
                         f"does not even reach the row-independent centreline "
                         f"({cl['cos_mean_to_each']:+.4f}): its cosine is fully "
                         f"explained by where the target distribution sits, not by "
                         f"the EEG.  This is a SPACE/OFFSET problem, not necessarily "
                         f"an information problem -- read the row-identity numbers "
                         f"below before concluding the signal is absent.")
            elif margin is not None and margin < 0.05:
                v.append(f"The best condition ({head_nm}) is only {margin:+.4f} above "
                         f"a row-independent constant. Most of its cosine is the "
                         f"target mean, not the EEG.")
        if head:
            ri = head.get("full", {}).get("row_identity_top1", 0.0)
            rr = head.get("residual_only", {}).get("row_identity_top1", 0.0)
            chance = head.get("full", {}).get("chance_row_identity", 1.0)
            if ri and ri > 3 * chance:
                v.append(f"Row-identity retrieval is {ri:.4f} against chance "
                         f"{chance:.4f} ({ri/chance:.1f}x): the condition DOES carry "
                         f"genuine trial-specific information.")
            elif ri and ri > 1.5 * chance:
                v.append(f"Row-identity retrieval {ri:.4f} vs chance {chance:.4f}: "
                         f"real but small row-specific signal.")
            else:
                v.append(f"Row-identity retrieval {ri:.4f} is at chance "
                         f"{chance:.4f}: the condition does not identify the trial.")
            # THE ACTIONABLE NUMBER.  If deleting the row-constant raises retrieval,
            # the constant is not just uninformative, it is actively destroying the
            # condition's usability, and removing it is a free improvement.
            if rr > 1.25 * max(ri, 1e-9):
                v.append(f"REMOVING THE ROW-CONSTANT RAISES row-identity from "
                         f"{ri:.4f} to {rr:.4f} ({rr/max(ri,1e-9):.1f}x). The "
                         f"constant is therefore not neutral: it is crowding out the "
                         f"trial-specific part that IP-Adapter needs. Centre (or "
                         f"whiten) the condition before injection -- this is a free "
                         f"gain that costs no retraining.")
        if rep.get("arm_agreement_verdict") == "BROKEN":
            v.append("An ablation arm is indistinguishable from the full arm, so "
                     "every paired arm comparison in this run is INVALID.")
        rep["verdict"] = v
        for line in v:
            print(f"[measure] verdict: {line}")
    except Exception as e:                                             # noqa: BLE001
        rep["errors"]["verdict"] = f"{type(e).__name__}: {e}"

    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="GEM run root; <arm>/conds/<row>.npy is read from here")
    ap.add_argument("--bank", type=Path, required=True,
                    help="true test image embeddings, the retrieval gallery")
    ap.add_argument("--captions", type=Path, default=None,
                    help="captions dir; used only to derive concept labels")
    ap.add_argument("--rows", nargs="+", default=list(ROWS_DEFAULT))
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    print("=" * 78)
    print("CONTROL AND RETRIEVAL MEASUREMENT -- the numbers cos alone cannot give")
    print("=" * 78)
    rep = build(args)

    out = args.json or (args.out_dir / "reports" / "measure.json")
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep, indent=2, default=float), encoding="utf-8")
        print(f"[measure] wrote {out}")
    except Exception as e:                                             # noqa: BLE001
        print(f"[measure] could not write {out}: {e}")
    # ALWAYS exit 0.  This stage is a measurement, and a measurement that crashes a
    # job after the metrics exist is worse than a measurement that reports null.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
