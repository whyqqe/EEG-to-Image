#!/usr/bin/env python3
"""G3F readout: 10-subject averages, PAIRED deltas, and the sub-08 detail table.

The only comparison that is causally interpretable is a paired one, because the
generator, sampler, seed, prompt and structural anchor are held fixed inside each
pair. This script therefore leads with paired deltas rather than with averages,
and it labels each metric's direction (swav / effnet / fid are DISTANCES, so a
decrease is an improvement).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

METRICS = ["pixcorr", "ssim", "alex2", "alex5", "inception", "clip", "swav", "effnet", "fid"]
LOWER_BETTER = {"swav", "effnet", "fid"}

PAIRS = [
    ("g3f_ll_gen", "hcma_deploy", "CLEAN: only the SEMANTIC CONDITION differs (same generic prompt, anchor, low-level path)"),
    ("g3f_ll_self", "g3f_ll_gen", "CLEAN: only the PROMPT differs (same condition, anchor, low-level path)"),
    ("g3f_mem_self", "g3f_ll_self", "CLEAN: only the IP head differs (memory vs fused)"),
    ("g3f_cn_self", "g3f_ll_self", "CLEAN: only the LOW-LEVEL PATHWAY differs (Depth-CN vs LL-SDEdit)"),
    ("g3f_intra_self", "g3f_ll_self", "CLEAN: only the TRAINING SET differs (sub-08 only vs 9 others)"),
    ("g3f_ll_self", "sdedit_ll", "MIXED: condition + prompt + low-level path all differ; context only, NOT causal"),
]


def split(tag: str) -> tuple[str, str]:
    base, _, fold = tag.rpartition("_sub-")
    return base, f"sub-{fold}" if fold else "?"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--std7", type=str, default="")
    args = ap.parse_args()
    out = Path(args.out)
    std7 = Path(args.std7) if args.std7 else out / "standard7" / "results.json"
    if not std7.is_file():
        raise SystemExit(f"[FATAL] missing {std7}")

    rows = json.loads(std7.read_text(encoding="utf-8"))["rows"]
    by_var: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        v, fold = split(r["tag"])
        by_var[v][fold] = r
    have = {v: len(d) for v, d in by_var.items()}
    print("[g3f] variants present: " + ", ".join(f"{v}={n}" for v, n in sorted(have.items())))

    def avg(v: str, folds: list[str] | None = None) -> dict[str, float] | None:
        d = by_var.get(v)
        if not d:
            return None
        sel = [d[f] for f in (folds or sorted(d)) if f in d]
        if not sel:
            return None
        return {k: float(np.mean([r[k] for r in sel if isinstance(r.get(k), (int, float))]))
                for k in METRICS} | {"_n": len(sel)}

    order = ["g3f_ll_self", "g3f_ll_gen", "g3f_mem_self", "g3f_cn_self",
             "g3f_intra_self", "g3f_intra_gen", "hcma_deploy", "g2f_mem", "sdedit_ll"]
    loso = [f"sub-{i:02d}" for i in range(1, 11)]

    print("\n=== 10-subject LOSO averages (standard-7, official protocol) ===")
    print(f"{'variant':<17}{'n':>3}" + "".join(f"{k[:8]:>10}" for k in METRICS))
    for v in order:
        a = avg(v, loso)
        if not a or a["_n"] < 2:
            continue
        print(f"{v:<17}{int(a['_n']):>3}" + "".join(f"{a[k]:>10.4f}" for k in METRICS))
    print("note: swav / effnet / fid are DISTANCES (lower is better)")

    summary: dict = {"rows_present": have, "loso_averages": {}, "paired": {}, "sub08": {}}
    for v in order:
        a = avg(v, loso)
        if a:
            summary["loso_averages"][v] = a

    print("\n=== PAIRED deltas over the 10 LOSO subjects (identical setup inside each pair) ===")
    for a_v, b_v, why in PAIRS:
        if a_v not in by_var or b_v not in by_var:
            continue
        folds = sorted(set(by_var[a_v]) & set(by_var[b_v]) & set(loso))
        if len(folds) < 2:
            continue
        print(f"\n  [{a_v}] - [{b_v}]   n={len(folds)}   # {why}")
        print(f"    {'metric':<10}{'mean_delta':>12}{'better/win':>14}")
        rec = {}
        for k in METRICS:
            d = [by_var[a_v][f][k] - by_var[b_v][f][k] for f in folds]
            win = sum(1 for x in d if (x < 0 if k in LOWER_BETTER else x > 0))
            print(f"    {k:<10}{np.mean(d):>+12.4f}{f'{win}/{len(d)}':>14}")
            rec[k] = {"mean_delta": float(np.mean(d)), "wins": win, "n": len(d),
                      "lower_is_better": k in LOWER_BETTER}
        summary["paired"][f"{a_v}|{b_v}"] = rec

    print("\n=== sub-08 detail (all rows that exist for sub-08) ===")
    print(f"{'variant':<17}" + "".join(f"{k[:8]:>10}" for k in METRICS))
    for v in order + [k for k in sorted(by_var) if k not in order]:
        r = by_var.get(v, {}).get("sub-08")
        if not r:
            continue
        print(f"{v:<17}" + "".join(
            (f"{r[k]:>10.4f}" if isinstance(r.get(k), (int, float)) else f"{'-':>10}")
            for k in METRICS))
        summary["sub08"][v] = {k: r.get(k) for k in METRICS}

    pf = {}
    for p in sorted(out.glob("fid_pooled_*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(d.get("pooled_fid_unique_gt"), (int, float)) and d.get("n_fake_total", 0):
                pf[d.get("tag", p.stem)] = d["pooled_fid_unique_gt"]
        except Exception:                                          # noqa: BLE001
            pass
    if pf:
        print("\n=== pooled FID (2000 fake vs 200 unique GT) ===")
        for k, v in sorted(pf.items(), key=lambda x: x[1]):
            print(f"  {k:<17}{v:>9.2f}")
        summary["pooled_fid"] = pf

    (out / "g3f_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[g3f] summary -> {out / 'g3f_summary.json'}")


if __name__ == "__main__":
    main()
