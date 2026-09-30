#!/usr/bin/env python3
"""Cross-subject aggregation for the CF-MSF joint-training chain.

WHY A SEPARATE STAGE
--------------------
Everything so far was measured on sub-08 alone.  A single subject cannot separate
"this method works" from "this subject happens to be easy": the literature does not
report single-subject numbers either, it reports the MEAN OVER 10 SUBJECTS
(73.5% CORTIVA, 86.3% multi-blur+EVNet, 91.3% SAMGA).  So the headline number has to
be a mean over subjects, and the evidence for any improvement has to be a PAIRED test
across subjects -- which is only possible once every subject has been run under every
arm in the same chain.

WHAT IT REPORTS
---------------
1. Per-subject table, one row per (subject, encoder arm), for a FIXED fusion rule
   named on the command line.  The rule is never chosen by looking at these numbers;
   picking the best rule on the test set is exactly the bias this project removed.
2. Mean +/- std over subjects per arm, which is the literature-comparable row.
3. Wilcoxon signed-rank test on the per-subject pairs (joint vs frozen).  Chosen over
   a paired t-test because n=10 with a bounded accuracy is not safely Gaussian; the
   exact binomial sign test is reported alongside as a distribution-free check.

A NOTE ON USING SINKHORN AS THE HEADLINE
----------------------------------------
Sinkhorn uses the public structure of the test set (200 distinct concepts, one image
each), i.e. it is TRANSDUCTIVE.  It is a large gain (40.0 -> 50.0 -> 71.5 on sub-08)
and it is legitimate to report it as long as it is reported AS transductive and the
inductive number is given beside it.  Both are carried through here so the choice
stays visible at the table level rather than in prose.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cfmsf_joint_summary import RULES, fuse_arm, load_arm  # noqa: E402


def wilcoxon(a: np.ndarray, b: np.ndarray) -> dict:
    """Two-sided Wilcoxon signed-rank on paired per-subject values, exact.

    Implemented here rather than pulled from scipy so the number is reproducible
    without a scipy version dependency.  Exact null = all 2^n sign patterns.
    """
    from itertools import product
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    d = d[d != 0.0]
    n = len(d)
    if n == 0:
        return {"n": 0, "stat": None, "p": 1.0, "note": "all differences zero"}
    order = np.argsort(np.abs(d))
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)
    w_plus = float(ranks[d > 0].sum())
    # exact distribution of W+ over all sign assignments
    obs = abs(w_plus - n * (n + 1) / 4.0)
    total = 0
    extreme = 0
    for signs in product((1.0, -1.0), repeat=n):
        w = float((ranks * np.asarray(signs)).sum())
        w = (w + n * (n + 1) / 2.0) / 2.0        # W+ for this sign pattern
        total += 1
        if abs(w - n * (n + 1) / 4.0) >= obs - 1e-9:
            extreme += 1
    return {"n": n, "stat": w_plus, "p": extreme / total, "exact": True}


def sign_test(a: np.ndarray, b: np.ndarray) -> dict:
    """Two-sided exact binomial sign test; the distribution-free cross-check."""
    from math import comb
    d = np.asarray(a) - np.asarray(b)
    n_plus = int((d > 0).sum())
    n_minus = int((d < 0).sum())
    n = n_plus + n_minus
    if n == 0:
        return {"n": 0, "p": 1.0}
    k = min(n_plus, n_minus)
    p = 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return {"n": n, "n_joint_better": n_plus, "n_frozen_better": n_minus,
            "p": min(p, 1.0)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="parent dir containing sub-XX/<arm>/{joint_report,probe,...}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--subjects", type=str, default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--arms", type=str, default="joint,frozen")
    ap.add_argument("--pick", type=str, default="lvl5+agg",
                    help="fixed fusion rule (named, never searched)")
    ap.add_argument("--baseline-probe", type=str,
                    default="/project/peilab/why/NeuroBridge/outputs/cfmsf_probe",
                    help="sub-08 job 581602 probe (frozen-encoder reference)")
    args = ap.parse_args()

    root = Path(args.root)
    subjects = [int(s) for s in args.subjects.split(",") if s.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    per_subject: dict[str, dict[str, dict]] = {}
    missing: list[str] = []
    for sid in subjects:
        sdir = root / f"sub-{sid:02d}"
        jr = sdir / "joint_report.json"
        if not jr.is_file():
            missing.append(f"sub-{sid:02d}: no joint_report.json")
            continue
        joint = json.loads(jr.read_text())
        row: dict[str, dict] = {}
        for arm in arms:
            probedir = sdir / arm / "probe"
            a = load_arm(probedir)
            if a is None:
                missing.append(f"sub-{sid:02d}/{arm}: no probe output")
                row[arm] = None
                continue
            names = sorted({k.split("__")[0] for k in a["npz"].files})
            rules = dict(RULES)
            rules["all13"] = names
            ns = [n for n in rules[args.pick] if n in names]
            if len(ns) < 2:
                missing.append(f"sub-{sid:02d}/{arm}: rule {args.pick} has {len(ns)} routes")
                row[arm] = None
                continue
            f = fuse_arm(a, ns)
            best_name = max(names, key=lambda n: a["report"]["targets"][n]["mlp"]["top1"])
            row[arm] = {
                "rule_routes": ns, "n_routes_probed": len(names),
                "best_single_route": best_name,
                "best_single_route_top1": a["report"]["targets"][best_name]["mlp"]["top1"],
                "fuse_raw_top1": f["raw"]["top1"], "fuse_raw_top5": f["raw"]["top5"],
                "fuse_csls_top1": f["csls"]["top1"], "fuse_csls_top5": f["csls"]["top5"],
                "fuse_csls_meanrank": f["csls"]["mean_rank"],
                "fuse_csls_sinkhorn_top1": f["csls"]["sinkhorn_top1"],
                "fuse_raw_sinkhorn_top1": f["raw"]["sinkhorn_top1"],
                "encoder": {k: joint["arms"].get(arm, {}).get(k)
                            for k in ("best", "two_way", "test200_top1",
                                      "test200_top5", "init", "freeze_encoder")},
            }
        per_subject[f"sub-{sid:02d}"] = row

    ok = {s: r for s, r in per_subject.items() if all(r.get(a) for a in arms)}
    print(f"\n{'='*110}")
    print(f"CF-MSF 联合训练 跨被试汇总   rule={args.pick}   "
          f"(complete subjects: {len(ok)}/{len(subjects)})")
    print(f"{'='*110}")
    if not ok:
        print("[FATAL] no complete subject rows")
        for m in missing:
            print("  [MISS]", m)
        sys.exit(1)

    print(f"{'subject':<10}" + "".join(
        f"{a[:6]+' t1':>10}{a[:6]+' t5':>10}{a[:6]+' sk':>10}" for a in arms))
    for s in sorted(ok):
        line = f"{s:<10}"
        for a in arms:
            r = ok[s][a]
            line += (f"{r['fuse_csls_top1']:>10.4f}{r['fuse_csls_top5']:>10.4f}"
                     f"{r['fuse_csls_sinkhorn_top1']:>10.4f}")
        print(line)

    summary: dict = {"rule": args.pick, "n_subjects_complete": len(ok),
                     "subjects": sorted(ok), "arms": {}, "missing": missing,
                     "per_subject": {s: ok[s] for s in sorted(ok)},
                     "transductive_note": ("sinkhorn columns are transductive: they use "
                                           "the 200-distinct-concepts structure of the "
                                           "test set; inductive = the t1 columns")}
    print(f"\n{'-'*110}\nmean +/- std over subjects")
    for a in arms:
        v = {k: np.array([ok[s][a][k] for s in ok]) for k in
             ("fuse_csls_top1", "fuse_csls_top5", "fuse_csls_sinkhorn_top1",
              "best_single_route_top1")}
        entry = {k: {"mean": float(x.mean()), "std": float(x.std(ddof=1)),
                     "min": float(x.min()), "max": float(x.max())}
                 for k, x in v.items()}
        summary["arms"][a] = entry
        print(f"  {a:<8} csls_t1={entry['fuse_csls_top1']['mean']:.4f}"
              f"+/-{entry['fuse_csls_top1']['std']:.4f}"
              f"  csls_t5={entry['fuse_csls_top5']['mean']:.4f}"
              f"  +sink={entry['fuse_csls_sinkhorn_top1']['mean']:.4f}"
              f"+/-{entry['fuse_csls_sinkhorn_top1']['std']:.4f}"
              f"  best_single={entry['best_single_route_top1']['mean']:.4f}")

    if len(arms) == 2:
        j = np.array([ok[s]["joint"]["fuse_csls_top1"] for s in ok])
        f = np.array([ok[s]["frozen"]["fuse_csls_top1"] for s in ok])
        summary["paired_test"] = {
            "comparison": f"{arms[0]} vs {arms[1]}", "metric": "fuse_csls_top1",
            "wilcoxon": wilcoxon(j, f), "sign_test": sign_test(j, f),
            "mean_delta": float((j - f).mean()),
        }
        w, st = summary["paired_test"]["wilcoxon"], summary["paired_test"]["sign_test"]
        print(f"\n配对检验 ({arms[0]} vs {arms[1]}, csls_t1): "
              f"delta={summary['paired_test']['mean_delta']:+.4f}  "
              f"Wilcoxon W+={w['stat']} p={w['p']:.4g} (n={w['n']})  "
              f"sign-test {st['n_joint_better']}/{st['n']} better, p={st['p']:.4g}")

    # sub-08 cross-check against job 581602: the `frozen` arm must reproduce the
    # probe's single-route number on the same target, or the chains disagree.
    ref_f = Path(args.baseline_probe) / "sub-08" / "route_probe.json"
    if ref_f.is_file() and "sub-08" in ok:
        base = json.loads(ref_f.read_text())
        tr = json.loads((root / "sub-08" / "joint_report.json").read_text()).get("target", "")
        m = {"image": "vith_image", "lowresolution": "vith_lowresolution",
             "levels_mean": "vith_levels_mean", "cat3": "vith_cat3", "cat5": "vith_cat5"}
        if m.get(tr) in base["targets"]:
            pv = base["targets"][m[tr]]["mlp"]["top1"]
            fv = ok["sub-08"]["frozen"]["best_single_route_top1"]
            summary["sub08_probe_consistency"] = {
                "probe_top1": pv, "frozen_arm_best_single": fv, "delta": fv - pv}
            print(f"[consistency] sub-08 frozen arm vs job 581602 probe on {m[tr]}: "
                  f"probe={pv:.4f} frozen={fv:.4f} delta={fv - pv:+.4f}")

    Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[aggregate] -> {args.out}")


if __name__ == "__main__":
    main()
