#!/usr/bin/env python
"""v14 O2-MVE verdict: is the multi-view order-2 estimator better than the shipped one?

WHAT IS BEING DECIDED. `calibration.mve_scores` pools SEVERAL partition families of the target's own
repetition cloud as independent metric views into ONE structural reference:

    cont    -- B contiguous blocks (this is the SHIPPED `--fgw-struct-fuse` object)
    stride  -- block b takes repetitions b::B; slow equipment drift is spread across every block
               instead of being given its own offset per block, so this is a different view family
               rather than a relabelling of `cont`
    rand    -- the seeded control between the two

The estimator error is `IMSE = bias^2 + sigma^2 / K_eff` with `K_eff = K / (1 + (K-1) rho)`. THIS
project measured `rho <= 0` on 10/10 folds (job 650112), so `K_eff = K`: each added view cuts the
variance and the gain does NOT saturate. That measurement is the entire licence for pooling families.

THE PRE-REGISTERED CRITERIA, all reported, and the guards are not decoration:

  * KILL SWITCH -- `cont-only` must reproduce the shipped `fuse=16` cell. If it does not, the MVE
    pipeline is a reimplementation that already disagrees with the object it claims to generalise,
    and NOTHING from this run may be read. This clause exists because "an argument that silently
    never arrives" is this project's most expensive recurring bug: a fusion that is mis-wired looks
    exactly like a fusion that does not help, and the shipped cell is the only available witness.
  * H1 -- multi-family beats `cont-only`, paired over folds. This is the estimator claim.
  * H2 -- the multi-family structural-OFF twin must be BIT-IDENTICAL to the cont-only
    structural-off twin. With `alpha=0` the structural term is off, so the injected reference cannot
    reach the output by any path; an INEQUALITY would mean the fusion is leaking through the
    cross-modal cost, which is a different and unlicensed mechanism.
  * H3 -- Top-5 must not fall. The v11 fusion's +0.90pp Top-5 came from the same block structure,
    and a Top-1 gain bought with Top-5 is not the trade the shipped recipe makes.

Usage:
    python scripts/summarize_v14_mve.py \
        --off outputs/eval/v14_mve_off --raw outputs/eval/v14_mve_raw \
        --whit outputs/eval/v14_mve_whit --base outputs/eval/v14_anchor_off \
        --out outputs/v14_summary_mve.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _paired(a: np.ndarray, b: np.ndarray) -> dict:
    d = a - b
    sd = float(d.std(ddof=1)) if d.size > 1 else 0.0
    t = float(d.mean() / (sd / np.sqrt(d.size))) if sd > 0 else float("nan")
    return {"mean": float(d.mean()), "sd": sd, "t": t,
            "folds_positive": int((d > 0).sum()), "n": int(d.size)}


def _rows(path: str) -> dict[str, dict]:
    """All scored rows of the single checkpoint in an eval json."""
    j = json.load(open(path))
    ck = list(j["checkpoints"].values())[0]
    return {n: r for n, r in ck.get("rows", {}).items()
            if isinstance(r, dict) and "top1" in r}


def _pick(rows: dict[str, dict], *, need=(), forbid=()) -> tuple[str, dict] | None:
    for name, row in rows.items():
        if all(s in name for s in need) and not any(s in name for s in forbid):
            return name, row
    return None


def _collect(root: str, picker) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for f in sorted(glob.glob(os.path.join(root, "sub*_seed*.json"))):
        got = picker(_rows(f))
        if got is None:
            print(f"[mve] WARN {os.path.basename(f)}: no row matching the selector in {root}")
            continue
        out[Path(f).stem] = {"row": got[0], "top1": float(got[1]["top1"]),
                             "top5": float(got[1]["top5"])}
    return out


def _multi(rows: dict[str, dict]):
    return _pick(rows, need=("views=",), forbid=("structural-off", "cont-only"))


def _multi_off(rows: dict[str, dict]):
    return _pick(rows, need=("views=", "structural-off"), forbid=("cont-only",))


def _cont_only(rows: dict[str, dict]):
    return _pick(rows, need=("cont-only",), forbid=("structural-off",))


def _cont_only_off(rows: dict[str, dict]):
    return _pick(rows, need=("cont-only", "structural-off"))


def _fuse16(rows: dict[str, dict]):
    return _pick(rows, need=("fuse=16", "R=80"), forbid=("structural-off",))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--off", default=str(ROOT / "outputs/eval/v14_mve_off"))
    ap.add_argument("--raw", default=str(ROOT / "outputs/eval/v14_mve_raw"))
    ap.add_argument("--whit", default=str(ROOT / "outputs/eval/v14_mve_whit"))
    ap.add_argument("--whit05", default=None,
                    help="optional fourth MVE arm; omitted from the verdict if absent.")
    ap.add_argument("--base", default=str(ROOT / "outputs/eval/v14_anchor_off"),
                    help="the BASE ladder dir, used for the shipped `fuse=16` witness.")
    ap.add_argument("--out", default=str(ROOT / "outputs/v14_summary_mve.json"))
    args = ap.parse_args()

    arms = {"off": args.off, "raw": args.raw, "whit": args.whit}
    if args.whit05:
        arms["whit05"] = args.whit05
    multi = {a: _collect(d, _multi) for a, d in arms.items()}
    cont = {a: _collect(d, _cont_only) for a, d in arms.items()}
    moff = {a: _collect(d, _multi_off) for a, d in arms.items()}
    coff = {a: _collect(d, _cont_only_off) for a, d in arms.items()}
    base16 = _collect(args.base, _fuse16)

    summary: dict = {"arg": vars(args),
                     "n_folds": {a: len(multi[a]) for a in arms},
                     "n_folds_base_fuse16": len(base16)}
    print(f"[mve] folds: " + ", ".join(f"{a}={len(multi[a])}" for a in arms)
          + f" | base fuse=16 folds={len(base16)}")

    if not any(multi.values()):
        print("[mve] no MVE rows on disk yet; nothing to decide")
        Path(args.out).write_text(json.dumps(summary, indent=2))
        return

    # ---- KILL SWITCH: cont-only must reproduce the shipped fuse=16 cell ---------------------
    ks_folds = sorted(set(cont["off"]) & set(base16))
    if ks_folds:
        c = np.asarray([cont["off"][f]["top1"] for f in ks_folds])
        b = np.asarray([base16[f]["top1"] for f in ks_folds])
        d = c - b
        summary["kill_switch_cont_only_vs_fuse16"] = _paired(c, b)
        summary["kill_switch_ok"] = bool(np.abs(d).max() < 1e-6)
        print(f"[mve] KILL SWITCH: cont-only {c.mean():.2f} vs shipped fuse=16 {b.mean():.2f} "
              f"| max|delta| {np.abs(d).max():.2e} -> "
              f"{'OK (identical object)' if summary['kill_switch_ok'] else 'FAIL -- do NOT read this run'}")
        if not summary["kill_switch_ok"]:
            print("[mve] the MVE `cont` path is NOT the shipped operator. Every number below is "
                  "unbookable until that is explained.")
    else:
        summary["kill_switch_ok"] = False
        print("[mve] KILL SWITCH: no paired folds (need both the MVE dir and --base). Cannot "
              "certify that cont-only reproduces the shipped cell.")

    # ---- H2: the fusion may only enter through the structural term --------------------------
    h2_folds = sorted(set(moff["off"]) & set(coff["off"]))
    if h2_folds:
        dm = np.asarray([moff["off"][f]["top1"] - coff["off"][f]["top1"] for f in h2_folds])
        summary["h2_structural_off_twin_identical"] = bool(np.abs(dm).max() < 1e-9)
        print(f"[mve] H2 structural-off twins: max|delta| {np.abs(dm).max():.2e} -> "
              f"{'OK (bit-identical, reference cannot leak)' if summary['h2_structural_off_twin_identical'] else 'FAIL (reference leaks through the cross-modal cost)'}")
    else:
        summary["h2_structural_off_twin_identical"] = False

    # ---- H1: the estimator claim, per arm and pooled -----------------------------------------
    per_arm = {}
    for a in arms:
        folds = sorted(set(multi[a]) & set(cont[a]))
        if not folds:
            continue
        m1 = np.asarray([multi[a][f]["top1"] for f in folds])
        c1 = np.asarray([cont[a][f]["top1"] for f in folds])
        m5 = np.asarray([multi[a][f]["top5"] for f in folds])
        c5 = np.asarray([cont[a][f]["top5"] for f in folds])
        per_arm[a] = {"n_folds": len(folds),
                      "multi_top1": float(m1.mean()), "cont_only_top1": float(c1.mean()),
                      "delta_top1": _paired(m1, c1),
                      "delta_top5": _paired(m5, c5),
                      "row_multi": multi[a][folds[0]]["row"]}
        print(f"[mve] arm {a:4s} ({len(folds)} folds): multi-family {m1.mean():.2f} vs cont-only "
              f"{c1.mean():.2f} | Top-1 {per_arm[a]['delta_top1']['mean']:+.2f} "
              f"(t={per_arm[a]['delta_top1']['t']:+.2f}, "
              f"{per_arm[a]['delta_top1']['folds_positive']}/{len(folds)} +) | "
              f"Top-5 {per_arm[a]['delta_top5']['mean']:+.2f}")
    summary["per_arm"] = per_arm

    if per_arm:
        base_arm = "off" if "off" in per_arm else sorted(per_arm)[0]
        d = per_arm[base_arm]["delta_top1"]
        d5 = per_arm[base_arm]["delta_top5"]
        summary["H1_pass"] = bool(d["mean"] > 0 and d["t"] > 2.0
                                 and d["folds_positive"] > d["n"] / 2)
        summary["H3_pass"] = bool(d5["mean"] >= -0.05)
        summary["reading"] = (
            f"PASS: pooling {per_arm[base_arm]['row_multi']!r} beats its own cont-only twin by "
            f"{d['mean']:+.2f}pp (t={d['t']:+.2f}, {d['folds_positive']}/{d['n']} folds) with Top-5 "
            f"{d5['mean']:+.2f}pp and the kill switch and structural-off guard both held. The extra "
            f"view families bought real variance, which is what rho<=0 licenses."
            if (summary["H1_pass"] and summary["H3_pass"]
                and summary.get("kill_switch_ok")
                and summary.get("h2_structural_off_twin_identical")) else
            "NOT A WIN. Either the multi-family pool did not beat its own cont-only control "
            "(the added families were duplicates, not new views), or a guard failed -- and a gain "
            "with a live structural-off twin or an unreproduced kill switch is unbookable. The "
            "shipped 54.83/83.45 stands unchanged either way.")
        print(f"[mve] {summary['reading']}")

    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"[mve] wrote {args.out}")


if __name__ == "__main__":
    main()
