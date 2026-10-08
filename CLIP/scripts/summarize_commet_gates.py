#!/usr/bin/env python
"""Summarise the COMMET gates G0 (metric adapter) and G1 (bilateral gallery view).

Both gates are PAIRED per fold and read through the pre-registered criterion written into the
architecture doc (`docs/eeg2image_commet_architecture.md` §6). This script reports the criterion
and the verdict; it does not choose one after seeing the numbers.

G0 rows (one JSON per fold, `outputs/probe/g0/`): a dict of adapter arm -> {top1, top5,
metric_agreement}. The arms are `identity` (the deployed row), `within` (the adapter), `total`
(unsupervised control), `shuffled` (label-destroying control) and `lda_k`.

G1 rows (one run_eval JSON per fold, `outputs/eval/g1/`): the paired cells are the query-block
fusion `fuse=16` and its gallery-view twin `fuse=16,gv`, each with a structural-off sibling that
MUST stay flat (the injected gallery metric can only route through the structural term).

Run:
    python scripts/summarize_commet_gates.py --g0 'outputs/probe/g0/*.json' \
        --g1 'outputs/eval/g1/*.json' --out outputs/commet_gates_summary.json
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _paired(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.asarray(a, float) - np.asarray(b, float)
    if d.size == 0:
        return {"n": 0}
    sd = float(d.std(ddof=1)) if d.size > 1 else 0.0
    t = float(d.mean() / (sd / np.sqrt(d.size))) if sd > 0 else float("nan")
    return {"n": int(d.size), "mean": float(a.mean()), "base": float(b.mean()),
            "delta": float(d.mean()), "sd": sd, "t": t,
            "pos": int((d > 0).sum())}


def _fmt(r: dict) -> str:
    if r.get("n", 0) == 0:
        return "  (no rows)"
    return (f"{r['mean']:6.2f} +- {r['sd']:<5.2f} vs {r['base']:6.2f}  "
            f"delta {r['delta']:+6.2f}pp  t={r['t']:5.2f}  {r['pos']:>2}/{r['n']} pos")


# ------------------------------------------------------------------------- G0
def summarise_g0(pattern: str) -> dict:
    files = sorted(glob.glob(pattern))
    print("=" * 92)
    print(f"G0 -- metric adapter fitted on SOURCE subjects, applied to the held-out subject")
    print(f"     {len(files)} folds: {[Path(f).stem for f in files]}")
    print("=" * 92)
    if not files:
        return {"n_folds": 0}

    arms = ["identity", "within", "total", "shuffled", "lda_k"]
    top1 = {a: [] for a in arms}
    agree = {a: [] for a in arms}
    used = []
    for f in files:
        g = json.loads(Path(f).read_text())["rows"]
        if not all(a in g for a in arms):
            print(f"  [skip] {Path(f).name}: arms {sorted(g)} incomplete")
            continue
        for a in arms:
            top1[a].append(g[a]["top1"])
            agree[a].append(g[a]["metric_agreement"])
        used.append(Path(f).stem)
    out = {"n_folds": len(used), "folds": used}
    base = np.asarray(top1["identity"], float)
    print(f"\n  {'arm':<10} {'Top-1 (paired vs identity)':<52} {'delta agreement':>16}")
    for a in arms:
        pr = _paired(np.asarray(top1[a], float), base)
        out[a] = pr
        da = np.mean(np.asarray(agree[a], float) - np.asarray(agree["identity"], float)) \
            if agree[a] else float("nan")
        out[a]["delta_agreement"] = float(da)
        print(f"  {a:<10} {_fmt(pr):<52} {da:>+16.4f}")

    w, t, s = out["within"], out["total"], out["shuffled"]
    passed = (w.get("delta", 0) > 0 and w.get("pos", 0) >= (w.get("n", 1) + 1) // 2
              and w.get("delta", 0) > t.get("delta", 0)
              and w.get("delta", 0) > s.get("delta", 0))
    out["verdict"] = {
        "pass": bool(passed),
        "criterion": "within > 0 on a majority of folds AND within > total AND within > shuffled",
        "reading": (
            "G0 PASSES: the source-fitted within-stimulus adapter beats the deployed row AND "
            "both controls, so a learned metric head (I5) has headroom -> build it."
            if passed else
            "G0 FAILS: the BEST-CASE linear adapter does not separate from its controls. A "
            "learned phi cannot do better than the optimal linear map, so I5 is closed at "
            "the price of this job."),
    }
    print(f"\n  G0 VERDICT: {out['verdict']['reading']}")
    return out


# ------------------------------------------------------------------------- G1
TAG = "T2 R=80,a=0.75,t=0.03,fuse=16"


def summarise_g1(pattern: str) -> dict:
    files = sorted(glob.glob(pattern))
    print("\n" + "=" * 92)
    print("G1 -- bilateral structural reference: gallery-side layer-view metric fusion")
    print(f"     {len(files)} folds: {[Path(f).stem for f in files]}")
    print("=" * 92)
    if not files:
        return {"n_folds": 0}

    on, gv, off, gvoff, wts = [], [], [], [], []
    used = []
    for f in files:
        d = json.loads(Path(f).read_text())
        ck = list(d["checkpoints"].values())[0]
        rows = ck["rows"]
        names = {
            "on": f"+ {TAG}", "gv": f"+ {TAG},gv",
            "off": f"+ {TAG} (structural-off)", "gvoff": f"+ {TAG},gv (structural-off)",
        }
        if not all(n in rows for n in names.values()):
            print(f"  [skip] {Path(f).name}: missing {[n for n in names.values() if n not in rows]}")
            continue
        on.append(rows[names["on"]]["top1"])
        gv.append(rows[names["gv"]]["top1"])
        off.append(rows[names["off"]]["top1"])
        gvoff.append(rows[names["gvoff"]]["top1"])
        gd = ck.get("gallery_view_diag") or {}
        if "weights" in gd:
            wts.append(gd["weights"])
        used.append(Path(f).stem)

    out = {"n_folds": len(used), "folds": used}
    if not used:
        return out
    r_gv = _paired(np.asarray(gv, float), np.asarray(on, float))
    r_off = _paired(np.asarray(gvoff, float), np.asarray(off, float))
    out["gv_vs_twin"] = r_gv
    out["gvoff_vs_off"] = r_off
    out["mean_gallery_weights"] = (np.mean(np.asarray(wts, float), axis=0).tolist()
                                   if wts else None)
    print(f"\n  gallery-view row  : {_fmt(r_gv)}")
    print(f"  structural-off    : {_fmt(r_off)}   (MUST be flat: the di_ref only routes "
          f"through the structural term)")
    if wts:
        print(f"  mean gallery weights K={len(wts[0])}: "
              f"{[round(x, 3) for x in out['mean_gallery_weights']]}")

    passed = (r_gv.get("delta", 0) > 0 and r_gv.get("pos", 0) >= (r_gv.get("n", 1) + 1) // 2
              and abs(r_off.get("delta", 1)) < 0.5)
    out["verdict"] = {
        "pass": bool(passed),
        "criterion": "gv beats its fuse=16 twin on a majority of folds AND the structural-off "
                     "twin is flat (|delta| < 0.5pp)",
        "reading": (
            "G1 PASSES: denoising the GALLERY side of the FGW coupling helps, so the bilateral "
            "multi-view estimator (I4) is real."
            if passed else
            "G1 FAILS: the gallery-side view ensemble does not beat the single fused gallery. "
            "The gallery metric is already clean enough that pooling its layer views adds "
            "nothing -- I4's second half is closed."),
    }
    print(f"\n  G1 VERDICT: {out['verdict']['reading']}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--g0", default=str(ROOT / "outputs/probe/g0/*.json"))
    ap.add_argument("--g1", default=str(ROOT / "outputs/eval/g1/*.json"))
    ap.add_argument("--out", default=str(ROOT / "outputs/commet_gates_summary.json"))
    args = ap.parse_args()

    res = {"g0": summarise_g0(args.g0), "g1": summarise_g1(args.g1)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"\n[commet] wrote {args.out}")
    g0p, g1p = res["g0"].get("verdict", {}).get("pass"), res["g1"].get("verdict", {}).get("pass")
    print(f"[commet] G0={'PASS' if g0p else 'FAIL'}  G1={'PASS' if g1p else 'FAIL'}")
    if g0p or g1p:
        print("[commet] at least one gate passed -> configure and submit the G3 grid for it.")
    else:
        print("[commet] neither gate passed -> do NOT submit G3; the architecture is unchanged.")


if __name__ == "__main__":
    main()
