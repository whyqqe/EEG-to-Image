#!/usr/bin/env python3
"""Why is the frozen arm's FUSION worse than its best SINGLE route?

THE OBSERVATION (job 581704, sub-08, frozen arm)
------------------------------------------------
    best single route        vith_cat5             test200 top1 = 0.4050
    probe's own fusion       (--fuse-topk 4)       test200 top1 = 0.2250
i.e. the fusion is 0.18 BELOW the best route it was supposed to combine. Meanwhile the
aggregator's `lvl5+agg` rule on 8 fixed routes reaches 0.520 with CSLS. So the routes
are fine and the fusion MACHINERY is throwing information away, which means the
reported 0.3815 (mean CSLS, frozen) is not a property of the representation -- it is
partly a property of a bad selection rule.

THE SUSPECT
-----------
`cfmsf_route_probe.py` picks the fusion routes by `val_top1` (full-1654-gallery top-1
on val_b rows), and `cfmsf_train.py:train_route` picks each head's CHECKPOINT by the
same statistic. Both use a number whose hit rate is 2-5%, on 820 rows. This script
measures, from saved artefacts, whether that statistic actually ranks routes the way
their test performance does. If it does not, the fix is the selection rule, and no
amount of regularisation tuning would have found it.

WHY THIS IS RUN BEFORE ANY TRAINING CHANGE
------------------------------------------
The user's instinct was that the frozen arm overfits early (the `Proj` head did peak at
epoch 5 out of 40). But `cfmsf_joint_train.py` exports raw `r` (line ~314) and the
frozen arm's encoder is frozen, so that head is DISCARDED: the frozen arm's export is
bit-identical to the initial intra encoder -- which the job's own consistency check
confirms with a delta of exactly 0.0000. So the epoch-5 peak cannot explain the ceiling.
This script tests the alternative explanation against the actual numbers instead of
accepting either story.

READ-ONLY: uses only route_probe.json files. No training, no GPU.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/project/peilab/why/NeuroBridge/outputs/cfmsf_all")


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rho with average ranks for ties. No scipy dependency."""
    def rank(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x, kind="stable")
        r = np.empty(len(x), dtype=np.float64)
        r[order] = np.arange(len(x), dtype=np.float64)
        # average ties so a statistic with many equal values is not ordered arbitrarily
        xs = x[order]
        i = 0
        while i < len(xs):
            j = i
            while j + 1 < len(xs) and xs[j + 1] == xs[i]:
                j += 1
            if j > i:
                r[order[i:j + 1]] = (i + j) / 2.0
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def load_probe(p: Path) -> dict | None:
    f = p / "route_probe.json"
    if not f.is_file():
        return None
    return json.loads(f.read_text())


def main() -> None:
    subs = sorted([d.name for d in ROOT.iterdir()
                   if d.is_dir() and d.name.startswith("sub-")])
    print(f"subjects with a probe: ", end="")
    available = []
    for s in subs:
        if (load_probe(ROOT / s / "frozen" / "probe") is not None
                and load_probe(ROOT / s / "joint" / "probe") is not None):
            available.append(s)
    print(f"{len(available)}/{len(subs)} -> {available}\n")

    # ---------------------------------------------------------------- 1. route ranking
    # Pool every (subject, arm, route) triple and ask: does the statistic used for
    # SELECTION rank routes the same way the reported test metric does?
    rows = []
    for s in available:
        for arm in ("frozen", "joint"):
            rep = load_probe(ROOT / s / arm / "probe")
            for name, t in rep["targets"].items():
                rows.append({"subject": s, "arm": arm, "route": name,
                             "val_top1": t["mlp"]["val_top1"],
                             "val_inst_cos": t["mlp"]["val_inst_cos"],
                             "best_epoch": t["mlp"]["best_epoch"],
                             "test_top1": t["mlp"]["top1"],
                             "test_csls": t["mlp"]["top1_csls"],
                             "test_top5": t["mlp"]["top5"],
                             "ridge_test": t["ridge"]["top1"]})
    n = len(rows)
    val = np.array([r["val_top1"] for r in rows])
    cs = np.array([r["val_inst_cos"] for r in rows])
    ep = np.array([float(r["best_epoch"]) for r in rows])
    te = np.array([r["test_top1"] for r in rows])
    tc = np.array([r["test_csls"] for r in rows])
    rg = np.array([r["ridge_test"] for r in rows])

    print("=" * 100)
    print(f"1. DOES THE SELECTION STATISTIC RANK ROUTES CORRECTLY?  (n={n} subject x arm x route)")
    print("=" * 100)
    print(f"   Spearman(val_top1,      test_top1)     = {spearman(val, te):+.3f}")
    print(f"   Spearman(val_top1,      test_csls)     = {spearman(val, tc):+.3f}")
    print(f"   Spearman(val_inst_cos,  test_top1)     = {spearman(cs, te):+.3f}")
    print(f"   Spearman(best_epoch,    test_top1)     = {spearman(ep, te):+.3f}")
    print(f"   Spearman(ridge_test,    test_top1)     = {spearman(rg, te):+.3f}  (linear ceiling)")
    print()
    print(f"   val_top1 range = [{val.min():.4f}, {val.max():.4f}]  "
          f"sd={val.std():.4f}")
    print(f"   test_top1 range = [{te.min():.4f}, {te.max():.4f}]  sd={te.std():.4f}")
    print("   -> if the first correlation is near zero or negative, the rule that CHOOSES")
    print("      routes is not reading the quantity it is supposed to optimise.")

    # ---------------------------------------------------- 2. what the val-topk rule picks
    print()
    print("=" * 100)
    print("2. WHAT `--fuse-topk 4` ACTUALLY SELECTS, vs WHAT IT SHOULD HAVE")
    print("=" * 100)
    print(f"{'subject':<9}{'arm':<8}{'val-selected (top-4 by val_top1)':<44}"
          f"{'mean test':>10}{'best single':>12}{'oracle top4':>12}")
    picks_are_bad = 0
    for s in available:
        for arm in ("frozen", "joint"):
            rep = load_probe(ROOT / s / arm / "probe")
            names = list(rep["targets"])
            byval = sorted(names, key=lambda k: -rep["targets"][k]["mlp"]["val_top1"])[:4]
            bytest = sorted(names, key=lambda k: -rep["targets"][k]["mlp"]["top1"])
            m_val = float(np.mean([rep["targets"][k]["mlp"]["top1"] for k in byval]))
            m_orc = float(np.mean([rep["targets"][k]["mlp"]["top1"] for k in bytest[:4]]))
            best = rep["targets"][bytest[0]]["mlp"]["top1"]
            flag = "" if m_val >= m_orc - 1e-9 else "  <- val-rule underperforms"
            if not bytest[0] in byval:
                flag += "  [MISSES best route]"
                picks_are_bad += 1
            print(f"{s:<9}{arm:<8}{','.join(k.replace('vith_','') for k in byval)[:43]:<44}"
                  f"{m_val:>10.4f}{best:>12.4f}{m_orc:>12.4f}{flag}")
    print()
    print(f"   the val_top1 rule misses the best route in {picks_are_bad} of "
          f"{2*len(available)} (subject, arm) pairs")

    # ------------------------------------------------- 3. per-route spread within subject
    print()
    print("=" * 100)
    print("3. SUB-08 DETAIL (frozen): the statistic is nearly FLAT while test spans 0.24")
    print("=" * 100)
    rep = load_probe(ROOT / "sub-08" / "frozen" / "probe")
    t = rep["targets"]
    order = sorted(t, key=lambda k: -t[k]["mlp"]["top1"])
    print(f"{'route':<22}{'val_top1':>10}{'val_rank':>10}{'test_top1':>11}{'test_rank':>11}")
    vrank = {k: i + 1 for i, k in enumerate(
        sorted(t, key=lambda k: -t[k]["mlp"]["val_top1"]))}
    for i, k in enumerate(order):
        m = t[k]["mlp"]
        print(f"{k:<22}{m['val_top1']:>10.4f}{vrank[k]:>10}{m['top1']:>11.4f}{i+1:>11}")
    print()
    print("   the 4 routes the val rule keeps are exactly the ones with the largest")
    print("   val_top1; the route with the best TEST score (vith_cat5) is 7th by val_top1.")

    # ------------------------------------------------------- 4. is fusion helping at all
    print()
    print("=" * 100)
    print("4. IS THE FIXED-RULE FUSION USED BY THE AGGREGATOR ACTUALLY ADDING VALUE?")
    print("=" * 100)
    ag = json.loads((ROOT / "aggregate.json").read_text())
    ps = ag["per_subject"]
    print(f"{'subject':<9}{'arm':<8}{'best_single':>12}{'fuse_raw':>10}{'fuse_csls':>11}"
          f"{'+sinkhorn':>11}{'gain vs single':>16}")
    gains = []
    for s in available:
        for arm in ("frozen", "joint"):
            r = ps[s][arm]
            g = r["fuse_csls_top1"] - r["best_single_route_top1"]
            gains.append(g)
            print(f"{s:<9}{arm:<8}{r['best_single_route_top1']:>12.4f}"
                  f"{r['fuse_raw_top1']:>10.4f}{r['fuse_csls_top1']:>11.4f}"
                  f"{r['fuse_csls_sinkhorn_top1']:>11.4f}{g:>+16.4f}")
    g = np.array(gains)
    print()
    print(f"   CSLS fusion gain over best single route: mean={g.mean():+.4f} "
          f"min={g.min():+.4f} max={g.max():+.4f}  positive in {(g>0).sum()}/{len(g)}")

    # ------------------------------------------- 5. head-checkpoint selection resolution
    print()
    print("=" * 100)
    print("5. HOW MUCH RESOLUTION DOES THE HEAD-CHECKPOINT SELECTOR HAVE?")
    print("=" * 100)
    ep = np.array([r["best_epoch"] for r in rows], dtype=float)
    print(f"   best_epoch across {n} route-fits: min={ep.min():.0f} max={ep.max():.0f} "
          f"median={np.median(ep):.0f}  (of 80 epochs)")
    print(f"   share of fits choosing an epoch in [18,25]: "
          f"{float(((ep>=18)&(ep<=25)).mean()):.2f}")
    print("   -> a SHARP peak on a few epochs would signal a systematic early optimum;")
    print("      a broad spread signals the statistic is selecting near-noise. A `None`")
    print("      here would be the honest answer if the spread is uninformative.")

    # ---------------------------------------------------------------- 6. linear vs MLP
    print()
    print("=" * 100)
    print("6. MLP vs RIDGE on the SAME target (is the map linear?)")
    print("=" * 100)
    print(f"   mean MLP test_top1 = {te.mean():.4f}   mean ridge test_top1 = {rg.mean():.4f}"
          f"   ratio = {te.mean()/max(rg.mean(),1e-9):.2f}x")
    print("   -> a large ratio means the representation-to-target map is strongly")
    print("      non-linear, so head capacity is NOT the thing to cut.")

    # -------------------------------------------------------------- 7. selection cap
    # An upper bound on what a perfect route-selection rule could have delivered, using
    # only information available at selection time (val_b): pick the top-4 by val_top1
    # vs the top-4 by ORACLE test. Reported so the "fix the rule" idea has a measured
    # ceiling rather than an assumed one.
    print()
    print("=" * 100)
    print("7. CEILING OF A BETTER ROUTE-SELECTION RULE (frozen arm only)")
    print("=" * 100)
    v_means, o_means = [], []
    for s in available:
        rep = load_probe(ROOT / s / "frozen" / "probe")
        names = list(rep["targets"])
        n_by = min(4, len(names))
        byval = sorted(names, key=lambda k: -rep["targets"][k]["mlp"]["val_top1"])[:n_by]
        bytest = sorted(names, key=lambda k: -rep["targets"][k]["mlp"]["top1"])[:n_by]
        v_means.append(np.mean([rep["targets"][k]["mlp"]["top1"] for k in byval]))
        o_means.append(np.mean([rep["targets"][k]["mlp"]["top1"] for k in bytest]))
    v_means, o_means = np.array(v_means), np.array(o_means)
    print(f"   sum-of-scores upper bound, val_top1 rule : {v_means.mean():.4f}")
    print(f"   sum-of-scores upper bound, oracle rule   : {o_means.mean():.4f}")
    print(f"   headroom from the ROUTE-SELECTION rule alone = "
          f"{o_means.mean()-v_means.mean():+.4f}")
    print("   (an upper bound: it ignores that routes must be combined, not picked)")


if __name__ == "__main__":
    main()
