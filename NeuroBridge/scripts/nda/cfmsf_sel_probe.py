#!/usr/bin/env python3
"""How much does the head's CHECKPOINT SELECTION cost, and does a better rule recover it?

THE MEASURED PROBLEM (audit of job 581704's 260 route-fits, `cfmsf_select_audit.py`)
-------------------------------------------------------------------------------
The route heads are selected by `val_top1`: full-1654-gallery top-1 over val_b rows.
That statistic has sd=0.0075 and lives in [0.0073, 0.0488] -- an 8x smaller spread than
the `test_top1` it is supposed to predict (sd=0.0591, range [0.11, 0.405]), and its
Spearman correlation with test is only +0.509. Meanwhile `cfmsf_train.py` keeps only the
last 5 epochs of history, so the per-epoch curve that would show whether the selector is
picking a bad epoch has NEVER been recorded for any route in this project.

That is the gap this script closes, and it is deliberately the FIRST thing to measure:
if the val_top1-selected epoch is typically far from the test-optimal epoch, then the
reported numbers are partly an artefact of a near-noise selection rule and the fix is a
better rule. If the selected epoch is already close to optimal, then checkpoint
selection is NOT the bottleneck and effort should go elsewhere. Either answer is
useful; guessing between them would not be.

THE THREE CANDIDATE FIXES, RANKED BY HOW MUCH THEY COULD COST
------------------------------------------------------------
1. A HIGHER-RESOLUTION SELECTOR.  `val_top1` asks the head to find the right concept
   among 1654; a 2-5% hit rate is a tiny number with tiny variance. Restricting the
   gallery to the concepts actually present in the val rows (83 concepts, 10 rows each)
   asks a strictly easier question whose answer is highly correlated with the hard one,
   but with a usable dynamic range. Same rows, same head, same data -- only the number
   of competing columns changes. `mini_top1` is therefore not a new model, it is a
   better MEASUREMENT of the same quantity.
2. SWA (stochastic weight averaging) over the last K epochs.  Needs no selection at all,
   which is attractive precisely because selection is what is broken. If it matches the
   oracle, the whole selection problem can be deleted rather than improved.
3. A fixed early epoch.  The cheapest possible rule; included so the comparison has a
   trivial baseline that cannot be accused of being clever.

THE ORACLE IS A DIAGNOSTIC, NOT A RESULT
----------------------------------------
Per-epoch test scores are computed so the selector's regret can be quantified. They are
printed in an explicitly-labelled oracle column and are NEVER used to select anything:
the reported metric always comes from a rule that saw only val data. Conflating the two
would reintroduce exactly the test-set selection this project has removed everywhere
else.

LEAK-FREE: gradients on `fit`; every selection statistic on `val_b` (the probe's own
selection set, so results are comparable to job 581704) with `val_a` reported as an
independent confirmation; the 200 test concepts are read once per epoch for the oracle
column only.

READ-ONLY with respect to existing artefacts: writes to its own --out, and reuses
`cfmsf_train`/`cfmsf_route_probe` helpers by import so the heads are byte-identical in
construction to the ones that produced the numbers being challenged.
"""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

NB_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NB_ROOT))
sys.path.insert(0, str(NB_ROOT / "scripts" / "nda"))

from cfmsf_train import Head, build_concept_mean, cos_loss, gallery_nce  # noqa: E402
from cfmsf_route_probe import build_targets, csls, metrics_200, sinkhorn  # noqa: E402
from ocf_train import build_concept_bank, l2n, l2t  # noqa: E402
import leakfree as LF  # noqa: E402

SUBJECT = 8
SELECTORS = ("full_top1", "mini_top1", "mini_csls", "two_way", "inst_cos")


def mini_bank(gal: np.ndarray, cid: np.ndarray, rows: np.ndarray):
    """Gallery restricted to the concepts present in `rows` + row->column labels."""
    cons = np.unique(cid[rows])
    return gal[cons], np.searchsorted(cons, cid[rows]), cons


def two_way(q: np.ndarray, bank: np.ndarray, labels: np.ndarray, n_neg: int,
            seed: int) -> float:
    """Correct column vs `n_neg` random distractors, same row."""
    sim = q @ bank.T
    rng = np.random.default_rng(seed)
    n = len(q)
    neg = rng.integers(0, bank.shape[0], size=(n, n_neg))
    correct = sim[np.arange(n), labels][:, None]
    negs = np.take_along_axis(sim, neg, axis=1)
    wins = (correct > negs).sum() + 0.5 * (correct == negs).sum()
    return float(wins / (n * n_neg))


def val_stats(q: np.ndarray, gal: np.ndarray, cid_rows: np.ndarray, mb, ml, tgt_rows,
              n_neg: int, seed: int) -> dict:
    """All selection candidates, computed from val data only."""
    qn = l2n(q)
    full = qn @ l2n(gal).T
    mini = qn @ l2n(mb).T
    return {
        "full_top1": float((full.argmax(1) == cid_rows).mean()),
        "mini_top1": float((mini.argmax(1) == ml).mean()),
        "mini_csls": float((csls(mini).argmax(1) == ml).mean()),
        "two_way": two_way(qn, l2n(gal), cid_rows, n_neg, seed),
        "inst_cos": float((qn * l2n(tgt_rows)).sum(-1).mean()),
    }


def train_route_full(name: str, tgt_tr: np.ndarray, tgt_te: np.ndarray,
                     ztr: np.ndarray, zte: np.ndarray, cid: np.ndarray,
                     fit_i: np.ndarray, val_b: np.ndarray, val_a: np.ndarray,
                     args, dev: torch.device) -> dict:
    """Train one route head, recording the FULL per-epoch curve for every selector.

    The only structural difference from `cfmsf_train.train_route` is that this keeps
    all epochs and evaluates every selector each epoch.  Loss, optimiser, schedule,
    batching, seeds and the head class are identical, so the numbers are comparable
    with job 581704's.
    """
    torch.manual_seed(args.seed)
    out_dim = tgt_tr.shape[1]
    gal = build_concept_mean(tgt_tr, cid, int(cid.max()) + 1)
    mb, ml, _ = mini_bank(gal, cid, val_b)
    ab, al, _ = mini_bank(gal, cid, val_a)

    head = Head(dim=ztr.shape[1], depth=args.depth, drop=args.drop,
                out_dim=out_dim).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    z_t = torch.from_numpy(ztr).to(dev)
    t_t = torch.from_numpy(tgt_tr).to(dev)
    g_t = torch.from_numpy(gal).to(dev)
    cid_t = torch.from_numpy(cid.astype(np.int64)).to(dev)
    fit_t = torch.from_numpy(fit_i.astype(np.int64)).to(dev)

    # SWA accumulators, one per window.  Averaging WEIGHTS (not predictions) keeps the
    # result a single model, so it can be exported exactly like a selected checkpoint.
    swa: dict[int, dict] = {k: {} for k in args.swa_k}
    curve: list[dict] = []
    best = {s: {"epoch": -1, "value": -1e18} for s in SELECTORS}
    ckpts: dict[str, dict] = {}

    for ep in range(args.epochs):
        head.train()
        perm = fit_t[torch.randperm(len(fit_t), device=dev)]
        tot, nb = 0.0, 0
        for s in range(0, len(perm), args.batch_size):
            idx = perm[s:s + args.batch_size]
            opt.zero_grad(set_to_none=True)
            q = head(z_t[idx])
            loss = gallery_nce(q, g_t, cid_t[idx], args.tau)
            if args.inst_weight > 0:
                loss = loss + args.inst_weight * cos_loss(q, t_t[idx])
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            nb += 1
        sch.step()

        for k in args.swa_k:
            if ep >= args.epochs - k:
                # Accumulate a RUNNING SUM rather than a list of copies.  At out_dim
                # 5120 a single head is ~24 MB, so keeping 20 snapshots per route would
                # cost ~0.5 GB per route and ~4 GB across the route set for no benefit:
                # only the mean is ever used.
                for n, p in head.state_dict().items():
                    if n not in swa[k]:
                        swa[k][n] = p.detach().float().clone()
                        swa[k][f"__count__{n}"] = 1
                    else:
                        swa[k][n] += p.detach().float()
                        swa[k][f"__count__{n}"] += 1

        head.eval()
        with torch.no_grad():
            qb = head(z_t[torch.from_numpy(val_b.astype(np.int64)).to(dev)]
                      ).cpu().numpy().astype(np.float32)
            qa = head(z_t[torch.from_numpy(val_a.astype(np.int64)).to(dev)]
                      ).cpu().numpy().astype(np.float32)
            qte = head(torch.from_numpy(zte).to(dev)).cpu().numpy().astype(np.float32)

        sb = val_stats(qb, gal, cid[val_b], mb, ml, tgt_tr[val_b], args.n_neg, args.seed)
        sa = val_stats(qa, gal, cid[val_a], ab, al, tgt_tr[val_a], args.n_neg, args.seed)
        # ORACLE COLUMN -- diagnostic only, never a selection input.
        oracle = metrics_200(qte @ tgt_te.T)["top1"]

        row = {"epoch": ep, "loss": tot / max(nb, 1), "oracle_test_top1": oracle,
               **{f"valb_{k}": v for k, v in sb.items()},
               **{f"vala_{k}": v for k, v in sa.items()}}
        curve.append(row)

        for s in SELECTORS:
            if sb[s] > best[s]["value"]:
                best[s] = {"epoch": ep, "value": sb[s], "vala": sa[s],
                           "oracle_at_epoch": oracle}
                ckpts[s] = {n: p.detach().clone()
                            for n, p in head.state_dict().items()}

    def score_weights(sd: dict) -> dict:
        h = Head(dim=ztr.shape[1], depth=args.depth, drop=args.drop,
                 out_dim=out_dim).to(dev)
        h.load_state_dict(sd)
        h.eval()
        with torch.no_grad():
            q = l2n(h(torch.from_numpy(zte).to(dev)).cpu().numpy().astype(np.float32))
        sim = q @ tgt_te.T
        m = metrics_200(sim)
        m["sinkhorn_top1"] = float((sinkhorn(sim) == np.arange(200)).mean())
        m["csls_sinkhorn_top1"] = float((sinkhorn(csls(sim)) == np.arange(200)).mean())
        return m

    # (1) each selector's chosen checkpoint, scored on test
    sel_rows = {}
    for s in SELECTORS:
        if s in ckpts:
            sel_rows[s] = {"epoch": best[s]["epoch"], "valb_value": best[s]["value"],
                           "vala_value": best[s]["vala"],
                           "oracle_test_top1_at_epoch": best[s]["oracle_at_epoch"],
                           **score_weights(ckpts[s])}

    # (2) SWA over the last K epochs -- no selection involved at all
    swa_rows = {}
    for k, acc in swa.items():
        if not acc:
            continue
        keys = [n for n in acc if not n.startswith("__count__")]
        avg = {n: acc[n] / acc[f"__count__{n}"] for n in keys}
        swa_rows[f"swa{k}"] = {"epochs": f"{args.epochs-k}..{args.epochs-1}",
                               "n_averaged": int(acc[f"__count__{keys[0]}"]),
                               **score_weights(avg)}

    # (3) the trivial fixed-epoch rules
    fixed_rows = {}
    for e in args.fixed_epochs:
        e = min(e, args.epochs - 1)
        fixed_rows[f"fixed{e}"] = {"epoch": e,
                                   "oracle_test_top1_at_epoch": curve[e]["oracle_test_top1"]}

    # (4) the oracle ceiling: the best epoch that exists at all
    oracle_curve = [r["oracle_test_top1"] for r in curve]
    oi = int(np.argmax(oracle_curve))
    oracle_row = {"best_epoch": oi, "best_test_top1": oracle_curve[oi]}

    return {"route": name, "dim": int(out_dim), "curve": curve,
            "selectors": sel_rows, "swa": swa_rows, "fixed": fixed_rows,
            "oracle": oracle_row, "n_params": int(sum(p.numel() for p in head.parameters()))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--test-subject", type=int, default=8)
    ap.add_argument("--z-root", type=str, required=True)
    ap.add_argument("--routes", type=str, default="",
                    help="comma list; default = the lvl5+agg route set")
    ap.add_argument("--cond-cache", type=str, default="outputs/gem/cond_cache")
    ap.add_argument("--clip-text-dir", type=str, default="")
    ap.add_argument("--captions-jsonl", type=str,
                    default=str(NB_ROOT / "outputs/g2/captions/captions_train.jsonl"))
    ap.add_argument("--split-json", type=str,
                    default=str(NB_ROOT / "outputs/leakfree/split.json"))
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--inst-weight", type=float, default=0.2)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--n-neg", type=int, default=64)
    ap.add_argument("--swa-k", type=str, default="5,10,20")
    ap.add_argument("--fixed-epochs", type=str, default="5,9,15,30")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    global SUBJECT
    SUBJECT = args.test_subject
    args.swa_k = [int(s) for s in args.swa_k.split(",") if s.strip()]
    args.fixed_epochs = [int(s) for s in args.fixed_epochs.split(",") if s.strip()]
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    sid = f"{SUBJECT:02d}"
    args.clip_text_dir = str(NB_ROOT / f"outputs/nda_ss/sub-{sid}/clip_text")

    ztr = l2n(np.load(Path(args.z_root) / f"sub-{sid}" / "shared_r_train.npy"
                      ).astype(np.float32))
    zte = l2n(np.load(Path(args.z_root) / f"sub-{sid}" / "shared_r_test.npy"
                      ).astype(np.float32))
    split = LF.load(args.split_json)
    fit_i = LF.rows_for(split, "fit", len(ztr))
    vb = LF.rows_for(split, "val_b", len(ztr))
    va = LF.rows_for(split, "val_a", len(ztr))
    for x, y in ((fit_i, vb), (fit_i, va), (vb, va)):
        assert not (set(x.tolist()) & set(y.tolist())), "split overlap"

    _, cid, phrases = build_concept_bank(Path(args.clip_text_dir),
                                         Path(args.captions_jsonl))
    if len(cid) != len(ztr):
        raise SystemExit(f"[FATAL] cid {len(cid)} vs z {len(ztr)}")
    n_cls = len(phrases)
    assert n_cls == int(cid.max()) + 1, (n_cls, int(cid.max()) + 1)

    args.cond_cache = str(NB_ROOT / args.cond_cache)
    tgts = build_targets(args)
    routes = ([s.strip() for s in args.routes.split(",") if s.strip()]
              if args.routes else
              ["vith_cat5", "vith_mixall", "vith_levels_mean", "vith_cat3",
               "vith_gaussiannoise", "vith_mosaic", "vith_lowresolution",
               "vith_image"])
    missing = [r for r in routes if r not in tgts]
    if missing:
        raise SystemExit(f"[FATAL] unknown routes {missing}")

    print(f"[sel] sub-{sid} z={ztr.shape} concepts={n_cls} routes={len(routes)} "
          f"fit={len(fit_i)} valB={len(vb)} valA={len(va)} epochs={args.epochs} "
          f"dev={dev}", flush=True)

    report: dict = {
        "pipeline": "cfmsf_sel_probe", "subject": f"sub-{sid}", "z_root": args.z_root,
        "why": ("cfmsf_select_audit measured Spearman(val_top1, test_top1)=+0.51 with "
                "sd(val_top1)=0.0075 against sd(test_top1)=0.0591, and cfmsf_train keeps "
                "only the last 5 epochs of history, so the per-epoch curve has never been "
                "recorded. This run records it and prices the selection rule."),
        "params": {k: getattr(args, k) for k in
                   ("epochs", "lr", "weight_decay", "tau", "inst_weight", "batch_size",
                    "depth", "drop", "seed", "swa_k", "fixed_epochs")},
        "protocol": {"fit": "gradients", "selection": "val_b", "confirmation": "val_a",
                     "test": "oracle column only, never a selection input"},
        "routes": {},
    }

    for r in routes:
        tgt_tr, tgt_te = tgts[r]
        info = train_route_full(r, tgt_tr, tgt_te, ztr, zte, cid, fit_i, vb, va,
                                args, dev)
        report["routes"][r] = info
        sel = info["selectors"]
        cur = sel.get("full_top1", {})
        mini = sel.get("mini_top1", {})
        swa = info["swa"].get(f"swa{args.swa_k[-1] if args.swa_k else 10}", {})
        print(f"[{r:<20}] oracle_ep={info['oracle']['best_epoch']:<3}"
              f" oracle_t1={info['oracle']['best_test_top1']:.4f} | "
              f"full_top1 ep{cur.get('epoch','-'):<3} t1={cur.get('top1',float('nan')):.4f} | "
              f"mini_top1 ep{mini.get('epoch','-'):<3} t1={mini.get('top1',float('nan')):.4f} | "
              f"swa t1={swa.get('top1',float('nan')):.4f}", flush=True)
        (out / "sel_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # ---------------- summary: price each rule against the oracle ----------------
    summ: dict = {"per_rule": {}, "oracle_mean_top1": float(np.mean(
        [v["oracle"]["best_test_top1"] for v in report["routes"].values()]))}
    rule_names = (list(SELECTORS) + [f"swa{k}" for k in args.swa_k]
                  + [f"fixed{e}" for e in args.fixed_epochs])
    for rule in rule_names:
        vals, gains, miss = [], [], 0
        for r, v in report["routes"].items():
            if rule in v["selectors"]:
                row = v["selectors"][rule]
            elif rule in v["swa"]:
                row = v["swa"][rule]
            elif rule in v["fixed"]:
                e = int(v["fixed"][rule]["epoch"])
                # fixed-epoch rules read the recorded per-epoch curve instead of a saved
                # checkpoint.  That is the same quantity because the training loop is
                # seeded and deterministic, so epoch `e` of this run IS the model that
                # rule would have kept -- no separate forward pass is needed.
                vals.append(v["curve"][e]["oracle_test_top1"])
                gains.append(v["curve"][e]["oracle_test_top1"]
                             - v["oracle"]["best_test_top1"])
                continue
            else:
                continue
            vals.append(row["top1"])
            gains.append(row["top1"] - v["oracle"]["best_test_top1"])
            if abs(row.get("epoch", -1) - v["oracle"]["best_epoch"]) > 10:
                miss += 1
        if vals:
            summ["per_rule"][rule] = {
                "mean_test_top1": float(np.mean(vals)),
                "mean_regret_vs_oracle": float(np.mean(gains)),
                "n": len(vals), "far_epoch_gt10": miss}
    summ["note"] = ("`fixed*` rows read the recorded per-epoch curve rather than a saved "
                    "checkpoint, so they are the same quantity as the other rows only "
                    "because every checkpoint here is deterministic given its epoch; the "
                    "oracle column is an upper bound, not an achievable number.")
    report["summary"] = summ
    (out / "sel_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 92)
    print("RULE vs ORACLE (test200 top1, mean over routes)")
    print("=" * 92)
    print(f"{'rule':<14}{'mean test t1':>14}{'regret vs oracle':>18}{'n':>5}"
          f"{'|epoch-oracle|>10':>18}")
    for rule, s in sorted(summ["per_rule"].items(), key=lambda kv: -kv[1]["mean_test_top1"]):
        print(f"{rule:<14}{s['mean_test_top1']:>14.4f}{s['mean_regret_vs_oracle']:>+18.4f}"
              f"{s['n']:>5}{s['far_epoch_gt10']:>18}")
    print(f"{'ORACLE':<14}{summ['oracle_mean_top1']:>14.4f}{0.0:>+18.4f}")
    print(f"\n[sel] wrote {out}")


if __name__ == "__main__":
    main()
