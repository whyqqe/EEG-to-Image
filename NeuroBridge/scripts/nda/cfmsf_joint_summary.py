#!/usr/bin/env python3
"""CF-MSF Stage 0 summary: did joint encoder training buy anything?

Reads, for each encoder arm, the probe output (`route_probe.json` +
`probe_queries.npz`) and the joint-training row, and prints ONE table in which the
only thing that differs between rows is the encoder.

THE REFERENCE POINTS THIS HAS TO BE READ AGAINST
------------------------------------------------
  job 581546  the 4-route CF-MSF on the pretrained encoder:
              CSLS 40.0% / +Sinkhorn 50.0%
  job 581602  13-route probe on the SAME frozen encoder:
              single route 40.5%, fuse-13 CSLS 50.0%, +Sinkhorn 71.5%

FUSION RULES ARE REPORTED AS FIXED, NAMED SETS, NOT AS A SEARCH
--------------------------------------------------------------
The probe's own fusion picks its subset by val_top1, and job 581602 measured that
this WEAKLY ANTI-correlates with test performance (Spearman 0.379; the val-picked
4-subset scored 49.0% while using all thirteen scored 50.0%).  So the subsets here
are chosen by a rule that cannot see any label:
    all13      every arm the probe built
    pert4      the four corrupted views (blur / lowres / mosaic / noise)
    lvl5       the five raw levels
    lvl5+agg   those five plus the mean and the two concats
Selecting the best of these on the TEST column would be exactly the bias the
project removed, so `--pick` must be named explicitly and the chosen rule is
recorded in the output JSON with that caveat.

Usage:
    cfmsf_joint_summary.py --root <joint out dir> --out <summary.json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

RULES = {
    "all13": None,   # filled at runtime with whatever the probe produced
    "pert4": ["vith_gaussianblur", "vith_lowresolution", "vith_mosaic", "vith_gaussiannoise"],
    "lvl5": ["vith_image", "vith_gaussianblur", "vith_lowresolution", "vith_mosaic",
             "vith_gaussiannoise"],
    "lvl5+agg": ["vith_image", "vith_gaussianblur", "vith_lowresolution", "vith_mosaic",
                 "vith_gaussiannoise", "vith_levels_mean", "vith_cat3", "vith_cat5"],
}


def l2n(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8, None)


def csls(s: np.ndarray, k: int = 10) -> np.ndarray:
    q = np.sort(s, 1)[:, -k:].mean(1, keepdims=True)
    b = np.sort(s, 0)[-k:, :].mean(0, keepdims=True)
    return 2.0 * s - q - b


def sinkhorn(s: np.ndarray, iters: int = 50, tau: float = 0.07) -> np.ndarray:
    lk = s.astype(np.float64) / max(tau, 1e-6)
    lk -= lk.max(1, keepdims=True)
    k = np.exp(lk)
    for _ in range(iters):
        k /= np.clip(k.sum(1, keepdims=True), 1e-12, None)
        k /= np.clip(k.sum(0, keepdims=True), 1e-12, None)
    return k.argmax(1)


def metrics(s: np.ndarray) -> dict:
    n = len(s)
    order = np.argsort(-s, 1)
    ok = order[:, 0] == np.arange(n)
    return {"top1": float(ok.mean()),
            "top5": float(np.mean([i in order[i, :5] for i in range(n)])),
            "mean_rank": float(np.mean([np.where(order[i] == i)[0][0] + 1 for i in range(n)]))}


def mcnemar(a_ok: np.ndarray, b_ok: np.ndarray) -> dict:
    """Paired comparison on the same 200 test concepts (exact binomial on discordants)."""
    from math import comb
    b = int(((~a_ok) & b_ok).sum())
    c = int((a_ok & (~b_ok)).sum())
    n = b + c
    if n == 0:
        return {"b_only": 0, "a_only": 0, "p": 1.0}
    k = min(b, c)
    p = 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return {"b_only": b, "a_only": c, "p": min(p, 1.0)}


def load_arm(src: Path) -> dict | None:
    """`src` is a directory containing route_probe.json + probe_queries.npz."""
    p, q = src / "route_probe.json", src / "probe_queries.npz"
    if not (p.is_file() and q.is_file()):
        return None
    return {"report": json.loads(p.read_text()), "npz": np.load(q)}


def fuse_arm(a: dict, names: list[str]) -> dict:
    S = {n: np.asarray(l2n(a["npz"][f"{n}__mlp_q"]) @ l2n(a["npz"][f"{n}__bank"]).T,
                       dtype=np.float64) for n in names}
    raw = sum(S.values())
    c = csls(raw)
    return {"routes": names,
            "raw": {**metrics(raw), "sinkhorn_top1": float((sinkhorn(raw) == np.arange(200)).mean())},
            "csls": {**metrics(c), "sinkhorn_top1": float((sinkhorn(c) == np.arange(200)).mean())},
            "_ok_raw": (raw.argmax(1) == np.arange(200)),
            "_ok_csls": (c.argmax(1) == np.arange(200))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--baseline-probe", type=str,
                    default="/project/peilab/why/NeuroBridge/outputs/cfmsf_probe",
                    help="parent of sub-08 from job 581602 (the frozen-encoder reference)")
    ap.add_argument("--pick", type=str, default="lvl5+agg",
                    help="which fixed rule to carry forward (must be named, not searched)")
    ap.add_argument("--tol", type=float, default=0.10,
                    help="allowed |frozen-arm - probe| gap on the same target")
    args = ap.parse_args()

    root = Path(args.root)
    arms = [d.name for d in sorted(root.iterdir()) if (d / "probe").is_dir()]
    if (Path(args.baseline_probe) / "route_probe.json").is_file():
        arms = ["frozen_581602"] + arms

    loaded: dict[str, dict] = {}
    for arm in arms:
        src = Path(args.baseline_probe) / "sub-08" if arm == "frozen_581602" else root / arm / "probe"
        a = load_arm(src)
        if a is None:
            print(f"[skip] {arm}: no probe output at {src}")
            continue
        loaded[arm] = a

    joint = json.loads((root / "joint_report.json").read_text()) \
        if (root / "joint_report.json").is_file() else {"arms": {}}

    # ---- consistency check against the probe -------------------------------
    # The `frozen` arm is the probe's setting (frozen encoder, multi-level
    # target), so its single-target test Top-1 must land near the probe's number
    # for the SAME target.  If it does not, the two pipelines disagree and every
    # joint-vs-frozen comparison below is uninterpretable -- so check it instead
    # of assuming it.
    TARGET_TO_PROBE = {"image": "vith_image", "lowresolution": "vith_lowresolution",
                       "levels_mean": "vith_levels_mean", "cat3": "vith_cat3",
                       "cat5": "vith_cat5"}
    consistency: dict = {}
    ref = joint.get("reference", {}).get("probe_581602", {})
    probe_arm = TARGET_TO_PROBE.get(joint.get("target", ""), "")
    frozen_row = joint.get("arms", {}).get("frozen", {})
    if frozen_row and probe_arm and (Path(args.baseline_probe) / "sub-08" / "route_probe.json").is_file():
        base = json.loads((Path(args.baseline_probe) / "sub-08" / "route_probe.json").read_text())
        if probe_arm in base["targets"]:
            probe_v = base["targets"][probe_arm]["mlp"]["top1"]
            frozen_v = frozen_row.get("test200_top1", float("nan"))
            delta = frozen_v - probe_v
            ok = abs(delta) <= args.tol
            consistency = {"probe_arm": probe_arm, "probe_top1": probe_v,
                           "frozen_arm_top1": frozen_v, "delta": delta,
                           "tolerance": args.tol, "pass": ok}
            print(f"\n[consistency] frozen arm reproduces the probe on the same target "
                  f"({probe_arm}): probe={probe_v:.4f} frozen={frozen_v:.4f} "
                  f"delta={delta:+.4f} tol={args.tol} -> "
                  f"{'PASS' if ok else 'WARN (pipelines disagree; treat deltas with care)'}")

    out: dict = {"root": str(root), "pick": args.pick,
                 "pick_caveat": ("selected by name, never by its test value; choosing "
                                 "the best rule on test would reintroduce the bias this "
                                 "project removed"),
                 "consistency": consistency,
                 "arms": {}, "encoders": {}}
    print(f"\n{'='*96}\n联合训练 vs 冻结编码器  (sub-08, official 200-way zero-shot)\n{'='*96}")
    print(f"{'encoder':<16}{'valA 2way':>10}{'test 2way':>10}{'best single':>12}"
          f"{'top1':>8}{'t5':>8}{'csls':>8}{'csls t5':>9}{'sink':>8}{'#1 ok':>7}")
    ok_by_arm: dict[str, dict] = {}
    for arm, a in loaded.items():
        names = sorted({k.split("__")[0] for k in a["npz"].files})
        best = max(a["report"]["targets"][n]["mlp"]["top1"] for n in names)
        bestname = max(names, key=lambda n: a["report"]["targets"][n]["mlp"]["top1"])
        rules = dict(RULES)
        rules["all13"] = names
        fused = {rn: fuse_arm(a, [n for n in ns if n in names]) for rn, ns in rules.items()}
        je = joint["arms"].get(arm, {})
        print(f"{arm:<16}{je.get('best', {}).get('val_a', {}).get('two_way', float('nan')):>10.4f}"
              f"{je.get('two_way', float('nan')):>10.4f}{bestname[:12]:>12}"
              f"{best:>8.4f}{a['report']['targets'][bestname]['mlp']['top5']:>8.4f}"
              f"{fused[args.pick]['csls']['top1']:>8.4f}"
              f"{fused[args.pick]['csls']['top5']:>9.4f}"
              f"{fused[args.pick]['csls']['sinkhorn_top1']:>8.4f}"
              f"{int(fused[args.pick]['csls']['top1']*200):>7}")
        out["arms"][arm] = {
            "n_arms_probed": len(names), "best_single_route": bestname,
            "best_single_route_top1": best,
            "single_routes": {n: a["report"]["targets"][n]["mlp"] for n in names},
            "fusion_rules": {rn: {k: v for k, v in f.items() if not k.startswith("_")}
                             for rn, f in fused.items()},
            "joint_row": je,
        }
        ok_by_arm[arm] = {rn: f["_ok_csls"] for rn, f in fused.items()}

    # ---- paired tests: each encoder arm against the frozen reference ----
    if "frozen_581602" in ok_by_arm:
        sig: dict = {}
        for arm in ok_by_arm:
            if arm == "frozen_581602":
                continue
            sig[arm] = {rn: mcnemar(ok_by_arm["frozen_581602"][rn], ok_by_arm[arm][rn])
                        for rn in ok_by_arm[arm]}
            print(f"\n配对比对 {arm} vs frozen_581602 (rule={args.pick}): "
                  f"{sig[arm][args.pick]}")
        out["paired_vs_frozen"] = sig

    print(f"\n[reference] 581546 (4 routes, pretrained encoder): csls 0.4000 sink 0.5000")
    print(f"[reference] 581602 (13 routes, SAME frozen encoder): csls 0.5000 sink 0.7150")
    if consistency and not consistency["pass"]:
        print("[WARN] the frozen arm does not reproduce the probe on the same target; "
              "the joint-vs-frozen delta below is not attributable to training alone")
    print(f"[carried forward] rule={args.pick}")
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[summary] -> {args.out}")


if __name__ == "__main__":
    main()
