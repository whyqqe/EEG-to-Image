#!/usr/bin/env python3
"""Assemble the G2 comparison table: new rows next to the existing SOTA rows.

Why a separate script exists
----------------------------
eval_standard7.py emits its own results.json for whatever set of rows it was handed.
It does not know about previously evaluated models, and it owns the file it writes.
Rather than mutate that table (or let two eval jobs overwrite each other), this script
only READS the SOTA table and the G2 table and emits a third, combined artifact.

What it reports
---------------
The seven standard metrics (PixCorr, SSIM, AlexNet(2), AlexNet(5), Inception, CLIP,
SwAV) plus FID, for
  * the G2 rows, split by protocol (intra vs LOSO) and by condition variant, and
  * the existing SOTA / baseline rows for reference.
Means are computed per (protocol, variant) group over the available folds, and the
fold count is printed with every mean so a partially finished sweep can never be
misread as a complete one. Pooled FID files, when present, are attached as a separate
block because pooled FID is a distribution-level number and is not the mean of the
per-row FIDs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

METRICS = ["pixcorr", "ssim", "alex2", "alex5", "inception", "clip", "swav", "fid"]
# for these, larger is better; for fid and the 2-way distances reported as errors, smaller
HIGHER_IS_BETTER = {"pixcorr": True, "ssim": True, "alex2": True, "alex5": True,
                    "inception": True, "clip": True, "swav": True, "fid": False}


def load_rows(p: Path) -> list[dict]:
    if not p.is_file():
        return []
    d = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(d, list):
        return d
    return d.get("rows", d.get("results", []))


def group_key(row: dict) -> str:
    """Collapse a row tag into the (protocol, variant) group it belongs to."""
    proto = row.get("protocol")
    var = row.get("variant")
    if proto is None:
        # fall back to parsing the tag, e.g. g2_sub-05_direct / g2_sub-08_inter_cfm0
        t = row.get("tag", "")
        var = "unknown"
        proto = "unknown"
        for v in ("direct_nc0", "direct_np", "direct", "cfm0"):
            if t.endswith("_" + v) or f"_{v}_" in t:
                var = v
                break
        if "intra" in t:
            proto = "intra"
        elif "inter" in t or "loso" in t:
            proto = "inter_loso"
    return f"{proto}/{var}"


def mean_of(rows: list[dict], k: str) -> float | None:
    vals = [float(r[k]) for r in rows if isinstance(r.get(k), (int, float))]
    return sum(vals) / len(vals) if vals else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--g2", required=True)
    ap.add_argument("--sota", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fid-glob", default="")
    args = ap.parse_args()

    g2_rows = load_rows(Path(args.g2))
    sota_rows = load_rows(Path(args.sota)) if args.sota else []

    groups: dict[str, list[dict]] = {}
    for r in g2_rows:
        groups.setdefault(group_key(r), []).append(r)
    groups = {k: v for k, v in groups.items() if k.startswith("intra") or k.startswith("inter")}

    summary: dict[str, dict] = {}
    print(f"\n{'group':<26}{'n':>4}" + "".join(f"{m:>10}" for m in METRICS))
    print("-" * (30 + 10 * len(METRICS)))
    for k in sorted(groups):
        rows = groups[k]
        summary[k] = {"n_rows": len(rows), "n_folds": len({r.get('fold') for r in rows})}
        line = f"{k:<26}{len(rows):>4}"
        for m in METRICS:
            v = mean_of(rows, m)
            summary[k][m] = v
            line += f"{v:>10.4f}" if v is not None else f"{'-':>10}"
        print(line)

    # reference: the best existing row per metric, so the G2 numbers have a baseline
    ref: dict[str, dict] = {}
    if sota_rows:
        print(f"\nreference SOTA table ({len(sota_rows)} rows): best value per metric")
        for m in METRICS:
            vals = [(float(r[m]), r.get("tag", "?")) for r in sota_rows
                    if isinstance(r.get(m), (int, float))]
            if not vals:
                continue
            best = max(vals) if HIGHER_IS_BETTER[m] else min(vals)
            ref[m] = {"value": best[0], "tag": best[1]}
            print(f"  {m:<10} {best[0]:>10.4f}  ({best[1]})")

    pooled: dict[str, dict] = {}
    if args.fid_glob:
        for f in sorted(Path(args.fid_glob).parent.glob(Path(args.fid_glob).name)):
            d = json.loads(f.read_text(encoding="utf-8"))
            pooled[f.stem] = {"pooled_fid_unique_gt": d.get("pooled_fid_unique_gt"),
                              "n_fake_total": d.get("n_fake_total"),
                              "n_gt_unique": d.get("n_gt_unique")}
            print(f"\n{ f.stem }: pooled FID {d.get('pooled_fid_unique_gt')} "
                  f"(fake {d.get('n_fake_total')} vs GT {d.get('n_gt_unique')})")

    out = {"g2_groups": summary, "sota_best_per_metric": ref, "pooled_fid": pooled,
           "n_g2_rows": len(g2_rows), "n_sota_rows": len(sota_rows),
           "caveat": ("means are over the folds that actually completed; n_folds is "
                      "reported per group and a partially finished sweep must not be "
                      "read as a complete one")}
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[OK] wrote {args.out}")


if __name__ == "__main__":
    main()
