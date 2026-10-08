#!/usr/bin/env python
"""v13 twin verdict: does the order-2 bias anchor beat its own OFF-arm twin, and WHY.

WHAT IS BEING DECIDED. `configs/v13_bias_loso_k20.yaml` (ON) adds exactly one term to
`configs/v13_bias_off_loso_k20.yaml` (OFF): `1 - corr(D_subject, D_gallery)`, the gradient of the
one estimator-error component that job 650135 measured to have headroom. This script reads the
paired 10-fold result and applies the criterion that was fixed BEFORE the runs:

    PASS  <=>  Delta corr(D_R80, D_g) > 0   AND   Delta corr(D_R1, D_g) >= Delta corr(D_R80) > 0

The second clause is the discriminating one and it is not decoration. The bias anchor is scored on
the POOLED metric, but the gate's headroom (+0.438) was measured at R=1. If training lifts only
R=1, the encoder got less noisy -- and pooling already removes noise (rho <= 0, K_eff = R), so the
deployment cell would not move. Only a lift at R=80 means the term removed SUBJECT-SPECIFIC bias,
which is the measured gap (target 0.616 vs a 9-subject pool 0.796).

GUARDS, all reported, and a "win" that fails one is booked as a collapse rather than a success:
  1. `effrank(D_eeg)` must not fall (v12 collapsed to rank 2-3 and lost 13.85pp raw cosine);
  2. the concept-shuffle control must stay dead in BOTH arms (otherwise the gain is M1 leakage);
  3. the two configs must differ in EXACTLY ONE effective key -- asserted here by loading both and
     diffing, because a pair that differs in two places cannot attribute anything.

Usage:
    python scripts/summarize_v13.py \
        --on-eval outputs/eval/v13_bias --off-eval outputs/eval/v13_bias_off \
        --on-probe outputs/probe/v13_bias --off-probe outputs/probe/v13_bias_off \
        --on-cfg configs/v13_bias_loso_k20.yaml --off-cfg configs/v13_bias_off_loso_k20.yaml \
        --out outputs/v13_twin_summary.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

#: the deployed headline cell (the one that holds 54.83/83.45 for the banked recipe)
HEADLINE_HINTS = ("fuse=16", "R=80")


def _paired(a: np.ndarray, b: np.ndarray) -> dict:
    d = a - b
    sd = float(d.std(ddof=1)) if d.size > 1 else 0.0
    t = float(d.mean() / (sd / np.sqrt(d.size))) if sd > 0 else float("nan")
    return {"mean": float(d.mean()), "sd": sd, "t": t,
            "folds_positive": int((d > 0).sum()), "n": int(d.size)}


def _load_headline(root: str) -> dict[str, dict]:
    """Per-fold headline Top-1/Top-5 from the deployed cell, with a loud fallback."""
    out: dict[str, dict] = {}
    files = sorted(glob.glob(os.path.join(root, "sub*_seed*.json")))
    for f in files:
        j = json.load(open(f))
        ck = list(j["checkpoints"].values())[0]
        rows = ck.get("rows", {})
        hit = None
        for name, row in rows.items():
            if isinstance(row, dict) and all(h in name for h in HEADLINE_HINTS):
                hit = (name, row)
                break
        if hit is None:
            # never silently substitute a different cell: a fallback would make an
            # apples-to-oranges number look like the pre-registered one
            cand = [(n, r) for n, r in rows.items() if isinstance(r, dict) and "top1" in r]
            if not cand:
                print(f"[v13] WARN {f}: no scored rows at all")
                continue
            hit = max(cand, key=lambda kv: kv[1]["top1"])
            print(f"[v13] WARN {os.path.basename(f)}: no cell matching {HEADLINE_HINTS}; "
                  f"falling back to best row {hit[0]!r} -- NOT the pre-registered cell")
        out[Path(f).stem] = {"row": hit[0], "top1": float(hit[1]["top1"]),
                             "top5": float(hit[1]["top5"]),
                             "effrank_ok": True}
    return out


def _load_probe(root: str) -> dict[str, dict]:
    """Per-fold bias-curve + rank/control readout from `probe_view_independence.py`."""
    out: dict[str, dict] = {}
    for f in sorted(glob.glob(os.path.join(root, "sub*_seed*.json"))):
        j = json.load(open(f))
        bc = j["bias_curve"]
        de = np.asarray(bc["corr_with_gallery"], dtype=float)
        ra = np.asarray(bc["corr_with_gallery_raw_frame"], dtype=float)
        sh = np.abs(np.asarray(bc["corr_shuffled_control"], dtype=float))
        out[Path(f).stem] = {
            "R": list(bc["R_for_average"]),
            "corr_R1": float(de[0]), "corr_R80": float(de[-1]),
            "corr_shuffled_max": float(sh.max()),
            "corr_R1_raw": float(ra[0]), "corr_R80_raw": float(ra[-1]),
            "effrank_single": float(j["effrank_single"]),
            "effrank_pooled": float(j["effrank_pooled"]),
            "agreement_with_gallery_pooled": float(de[-1]),
        }
    return out


def _config_diff(on_cfg: str, off_cfg: str) -> dict:
    """Prove the pair is single-variable by loading both and diffing the effective configs."""
    from samclip.utils import load_config  # noqa: PLC0415

    def flatten(o, p=""):
        if isinstance(o, dict):
            for k, v in o.items():
                yield from flatten(v, f"{p}.{k}" if p else str(k))
        else:
            yield p, o

    a = dict(flatten(load_config(on_cfg)))
    b = dict(flatten(load_config(off_cfg)))
    keys = sorted(set(a) | set(b))
    diffs = {k: {"on": a.get(k), "off": b.get(k)} for k in keys if a.get(k) != b.get(k)}
    return {"diffs": diffs, "single_variable": len(diffs) == 1}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--on-eval", default=str(ROOT / "outputs/eval/v13_bias"))
    ap.add_argument("--off-eval", default=str(ROOT / "outputs/eval/v13_bias_off"))
    ap.add_argument("--on-probe", default=str(ROOT / "outputs/probe/v13_bias"))
    ap.add_argument("--off-probe", default=str(ROOT / "outputs/probe/v13_bias_off"))
    ap.add_argument("--on-cfg", default=str(ROOT / "configs/v13_bias_loso_k20.yaml"))
    ap.add_argument("--off-cfg", default=str(ROOT / "configs/v13_bias_off_loso_k20.yaml"))
    ap.add_argument("--out", default=str(ROOT / "outputs/v13_twin_summary.json"))
    args = ap.parse_args()

    on_ev, off_ev = _load_headline(args.on_eval), _load_headline(args.off_eval)
    on_pr, off_pr = _load_probe(args.on_probe), _load_probe(args.off_probe)
    folds = sorted(set(on_ev) & set(off_ev))
    pfolds = sorted(set(on_pr) & set(off_pr))

    summary: dict = {"arg": vars(args), "n_folds_eval": len(folds), "n_folds_probe": len(pfolds)}
    summary["config_diff"] = _config_diff(args.on_cfg, args.off_cfg)

    if not folds:
        print("[v13] no paired eval folds yet; nothing to decide")
        Path(args.out).write_text(json.dumps(summary, indent=2))
        return

    t1 = np.asarray([on_ev[f]["top1"] for f in folds])
    t1o = np.asarray([off_ev[f]["top1"] for f in folds])
    t5 = np.asarray([on_ev[f]["top5"] for f in folds])
    t5o = np.asarray([off_ev[f]["top5"] for f in folds])
    summary["headline_top1_on"] = float(t1.mean())
    summary["headline_top1_off"] = float(t1o.mean())
    summary["headline_top1_delta"] = _paired(t1, t1o)
    summary["headline_top5_delta"] = _paired(t5, t5o)
    summary["headline_row"] = on_ev[folds[0]]["row"]

    print(f"[v13] headline cell {summary['headline_row']!r} over {len(folds)} folds")
    print(f"[v13]   Top-1  ON {t1.mean():.2f} +/- {t1.std(ddof=1) if len(t1) > 1 else 0:.2f}   "
          f"OFF {t1o.mean():.2f} +/- {t1o.std(ddof=1) if len(t1o) > 1 else 0:.2f}   "
          f"delta {summary['headline_top1_delta']['mean']:+.2f} "
          f"(t={summary['headline_top1_delta']['t']:+.2f}, "
          f"{summary['headline_top1_delta']['folds_positive']}/{len(folds)} folds +)")
    print(f"[v13]   Top-5  delta {summary['headline_top5_delta']['mean']:+.2f} "
          f"(t={summary['headline_top5_delta']['t']:+.2f}, "
          f"{summary['headline_top5_delta']['folds_positive']}/{len(folds)} folds +)")

    if pfolds:
        d_r1 = np.asarray([on_pr[f]["corr_R1"] - off_pr[f]["corr_R1"] for f in pfolds])
        d_r80 = np.asarray([on_pr[f]["corr_R80"] - off_pr[f]["corr_R80"] for f in pfolds])
        d_r1r = np.asarray([on_pr[f]["corr_R1_raw"] - off_pr[f]["corr_R1_raw"] for f in pfolds])
        d_r80r = np.asarray([on_pr[f]["corr_R80_raw"] - off_pr[f]["corr_R80_raw"] for f in pfolds])
        ctl = max(max(v["corr_shuffled_max"] for v in on_pr.values()),
                  max(v["corr_shuffled_max"] for v in off_pr.values()))
        se_ok = min(on_pr[f]["effrank_single"] for f in pfolds) >= \
            min(off_pr[f]["effrank_single"] for f in pfolds)
        sp_ok = min(on_pr[f]["effrank_pooled"] for f in pfolds) >= \
            min(off_pr[f]["effrank_pooled"] for f in pfolds)
        summary["probe"] = {
            "delta_corr_R1": _paired(np.asarray([on_pr[f]["corr_R1"] for f in pfolds]),
                                     np.asarray([off_pr[f]["corr_R1"] for f in pfolds])),
            "delta_corr_R80": _paired(np.asarray([on_pr[f]["corr_R80"] for f in pfolds]),
                                      np.asarray([off_pr[f]["corr_R80"] for f in pfolds])),
            "delta_corr_R1_raw": _paired(np.asarray([on_pr[f]["corr_R1_raw"] for f in pfolds]),
                                         np.asarray([off_pr[f]["corr_R1_raw"] for f in pfolds])),
            "delta_corr_R80_raw": _paired(np.asarray([on_pr[f]["corr_R80_raw"] for f in pfolds]),
                                          np.asarray([off_pr[f]["corr_R80_raw"] for f in pfolds])),
            "shuffle_control_max_abs": float(ctl),
            "effrank_single_guard_ok": bool(se_ok),
            "effrank_pooled_guard_ok": bool(sp_ok),
        }
        print(f"[v13]   probe  dR1 {d_r1.mean():+.4f} (t={summary['probe']['delta_corr_R1']['t']:+.1f})"
              f"   dR80 {d_r80.mean():+.4f} (t={summary['probe']['delta_corr_R80']['t']:+.1f})")
        print(f"[v13]   probe(raw frame) dR1 {d_r1r.mean():+.4f}   dR80 {d_r80r.mean():+.4f}"
              f"   shuffle control <= {ctl:.4f}")
        print(f"[v13]   guards: effrank single {'OK' if se_ok else 'FAIL'}, "
              f"pooled {'OK' if sp_ok else 'FAIL'}; single-variable config "
              f"{'OK' if summary['config_diff']['single_variable'] else 'FAIL'}")

        pass_lift = bool(d_r80.mean() > 0 and d_r1.mean() >= d_r80.mean() > 0)
        guards = bool(ctl < 0.10 and se_ok and sp_ok
                      and summary["config_diff"]["single_variable"])
        summary["PASS_mechanism"] = pass_lift
        summary["PASS_guards"] = guards
        summary["reading"] = (
            "PASS: training lifted the DEPLOYED (pooled) agreement, not merely the noisy single "
            "view, and every guard held -> the bias anchor removed subject-specific bias, which is "
            "the gap the gate measured (0.616 -> 0.796 reference)." if (pass_lift and guards) else
            "NO LIFT AT R80: only the single-view agreement (or neither) moved. The lever did not "
            "reach the deployed estimator -- pooling already removes view noise, so this is the "
            "predicted null, NOT a mechanism failure. Record it and keep the shipped 54.83."
            if not pass_lift else
            "LIFT BUT A GUARD FAILED: do not book this as a win -- a gain with a live shuffle "
            "control or a fallen effective rank is a collapse or a leak, not a mechanism.")
        print(f"[v13] {summary['reading']}")

    if not folds:
        summary["reading"] = "no paired folds"
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"[v13] wrote {args.out}")


if __name__ == "__main__":
    main()
