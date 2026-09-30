#!/usr/bin/env python3
"""Rank Alex2-first candidates with semantic gates (no PixCorr/SSIM selection)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-dir", type=str, required=True)
    ap.add_argument("--ref-tag", type=str, default="ref_hcma_full_a40")
    ap.add_argument("--output-json", type=str, required=True)
    ap.add_argument("--output-md", type=str, required=True)
    ap.add_argument("--clip-drop", type=float, default=0.01)
    ap.add_argument("--alex5-drop", type=float, default=0.01)
    ap.add_argument("--inc-drop", type=float, default=0.01)
    ap.add_argument("--swav-slack", type=float, default=0.02)
    ap.add_argument("--fid-slack", type=float, default=15.0)
    ap.add_argument("--alex2-target", type=float, default=0.776)
    args = ap.parse_args()

    mdir = Path(args.metrics_dir)
    rows = []
    for p in sorted(mdir.glob("*_bundle.json")):
        rows.append(json.loads(p.read_text()))
    if not rows:
        raise SystemExit(f"no *_bundle.json under {mdir}")

    by = {r["tag"]: r for r in rows}
    if args.ref_tag not in by:
        raise SystemExit(f"missing ref tag {args.ref_tag}")
    ref = by[args.ref_tag]

    ranked = []
    for r in rows:
        gate = (
            float(r["clip"]) >= float(ref["clip"]) - args.clip_drop
            and float(r["alex5"]) >= float(ref["alex5"]) - args.alex5_drop
            and float(r["inception"]) >= float(ref["inception"]) - args.inc_drop
            and float(r["swav"]) <= float(ref["swav"]) + args.swav_slack
            and (r.get("fid") is None or float(r["fid"]) <= float(ref.get("fid") or 999) + args.fid_slack)
        )
        ranked.append(
            {
                **r,
                "pass_gate": bool(gate),
                "delta_alex2": float(r["alex2"]) - float(ref["alex2"]),
                "delta_clip": float(r["clip"]) - float(ref["clip"]),
                "delta_alex5": float(r["alex5"]) - float(ref["alex5"]),
                "delta_inc": float(r["inception"]) - float(ref["inception"]),
                "delta_swav": float(r["swav"]) - float(ref["swav"]),
                "gap_to_atm_alex2": float(r["alex2"]) - args.alex2_target,
            }
        )
    ranked.sort(key=lambda x: (x["pass_gate"], x["alex2"], -x["swav"]), reverse=True)
    gated = [r for r in ranked if r["pass_gate"]]
    best = gated[0] if gated else ranked[0]

    summary = {
        "pipeline": "alex2_first",
        "ref": {k: ref[k] for k in ["tag", "alex2", "alex5", "inception", "clip", "swav", "fid"]},
        "gate_rule": {
            "clip": f">=ref-{args.clip_drop}",
            "alex5": f">=ref-{args.alex5_drop}",
            "inception": f">=ref-{args.inc_drop}",
            "swav": f"<=ref+{args.swav_slack}",
            "fid": f"<=ref+{args.fid_slack}",
            "select": "max Alex2 among gated; PixCorr/SSIM ignored",
        },
        "alex2_target_atm": args.alex2_target,
        "best_gated": best if best.get("pass_gate") else None,
        "best_overall_alex2": max(ranked, key=lambda x: x["alex2"]),
        "n_pass": len(gated),
        "n": len(ranked),
        "all_ranked": ranked,
    }
    Path(args.output_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Alex2-first overnight — sub-08",
        "",
        f"Ref `{args.ref_tag}`: Alex2={ref['alex2']:.3f} Alex5={ref['alex5']:.3f} "
        f"Inc={ref['inception']:.3f} CLIP={ref['clip']:.3f} SwAV={ref['swav']:.3f} FID={ref.get('fid')}",
        f"ATM Alex2 target: {args.alex2_target}",
        f"Gate: CLIP/A5/Inc ≥ ref−{args.clip_drop}; SwAV ≤ ref+{args.swav_slack}; FID ≤ ref+{args.fid_slack}",
        "**Selection ignores PixCorr/SSIM.**",
        "",
        "| tag | pass | Alex2 | ΔA2 | CLIP | Alex5 | Inc | SwAV | FID |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in ranked:
        fid = f"{r['fid']:.1f}" if r.get("fid") is not None else "—"
        lines.append(
            f"| `{r['tag']}` | {int(r['pass_gate'])} | {r['alex2']:.3f} | {r['delta_alex2']:+.3f} | "
            f"{r['clip']:.3f} | {r['alex5']:.3f} | {r['inception']:.3f} | {r['swav']:.3f} | {fid} |"
        )
    if best.get("pass_gate"):
        lines += [
            "",
            f"## Best gated: `{best['tag']}`",
            f"- Alex2 **{best['alex2']:.3f}** (Δ {best['delta_alex2']:+.3f}; gap to ATM {best['gap_to_atm_alex2']:+.3f})",
            f"- CLIP {best['clip']:.3f} / Alex5 {best['alex5']:.3f} / Inc {best['inception']:.3f} / SwAV {best['swav']:.3f}",
        ]
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"best_gated": summary["best_gated"], "n_pass": len(gated)}, indent=2))


if __name__ == "__main__":
    main()
