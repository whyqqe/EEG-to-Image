#!/usr/bin/env python3
"""Aggregate per-subject metrics into publication tables (mean ± std)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_metric(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_mean_std(vals: list[float]) -> str:
    if not vals:
        return "—"
    return f"{np.mean(vals):.4f} ± {np.std(vals):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", type=str, default="outputs/erdc/w16_loso_metrics")
    parser.add_argument(
        "--ours-prefix",
        type=str,
        default="w16_loso_fuse",
        help="metric file stem prefix, e.g. w16_loso_fuse_sub-08",
    )
    parser.add_argument(
        "--official-prefix",
        type=str,
        default="w16_official_prior",
        help="official baseline stem prefix",
    )
    parser.add_argument("--output-md", type=str, default="outputs/erdc/paper_ten_subject_table.md")
    parser.add_argument("--output-tex", type=str, default="outputs/erdc/paper_ten_subject_table.tex")
    args = parser.parse_args()

    met_dir = Path(args.metrics_dir)
    if not met_dir.is_absolute():
        met_dir = ROOT / met_dir

    rows = []
    ours_pix, ours_clip, off_pix, off_clip = [], [], [], []

    md = [
        "# 十被试扩展表（per-subject train/test）",
        "",
        "> 每被试独立训练 S1→S3，Turbo + fuse λ=0.15；Official 为 per-subject prior_atm + brain。",
        "",
        "| Subject | Official Pix | Official CLIP | Ours Pix | Ours CLIP | ΔPix | ΔCLIP |",
        "|---------|--------------|-----------------|----------|-----------|------|-------|",
    ]
    tex_lines = [
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Subject & Off. Pix & Off. CLIP & Ours Pix & Ours CLIP & $\\Delta$Pix & $\\Delta$CLIP \\\\",
        "\\midrule",
    ]

    for i in range(1, 11):
        sub = f"sub-{i:02d}"
        off = load_metric(met_dir / f"{args.official_prefix}_{sub}.json")
        ours = load_metric(met_dir / f"{args.ours_prefix}_{sub}.json")
        if not off and not ours:
            md.append(f"| {sub} | — | — | — | — | — | — |")
            continue

        op = off["pixcorr"] if off else None
        oc = off["clip_cosine"] if off else None
        up = ours["pixcorr"] if ours else None
        uc = ours["clip_cosine"] if ours else None

        if op is not None:
            off_pix.append(op)
            off_clip.append(oc)
        if up is not None:
            ours_pix.append(up)
            ours_clip.append(uc)

        dp = f"{up - op:+.4f}" if up is not None and op is not None else "—"
        dc = f"{uc - oc:+.4f}" if uc is not None and oc is not None else "—"
        op_s = f"{op:.4f}" if op is not None else "—"
        oc_s = f"{oc:.4f}" if oc is not None else "—"
        up_s = f"{up:.4f}" if up is not None else "—"
        uc_s = f"{uc:.4f}" if uc is not None else "—"
        md.append(f"| {sub} | {op_s} | {oc_s} | {up_s} | {uc_s} | {dp} | {dc} |")
        tex_lines.append(
            f"{sub} & {op_s} & {oc_s} & {up_s} & {uc_s} & {dp} & {dc} \\\\"
        )
        rows.append({"subject": sub, "official": off, "ours": ours})

    md.extend(
        [
            "",
            "## 汇总（n=" + str(len(ours_pix)) + "）",
            "",
            f"| 指标 | Official | Ours | Ours 超 Official (Pix/CLIP) |",
            f"|------|----------|------|-----------------------------|",
        ]
    )
    if ours_pix and off_pix:
        pix_pass = sum(u >= o for u, o in zip(ours_pix, off_pix))
        clip_pass = sum(u >= o for u, o in zip(ours_clip, off_clip))
        md.append(
            f"| PixCorr | {fmt_mean_std(off_pix)} | {fmt_mean_std(ours_pix)} | {pix_pass}/10 |"
        )
        md.append(
            f"| CLIP | {fmt_mean_std(off_clip)} | {fmt_mean_std(ours_clip)} | {clip_pass}/10 |"
        )

    out_md = Path(args.output_md)
    if not out_md.is_absolute():
        out_md = ROOT / out_md
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    tex_lines.extend(["\\bottomrule", "\\end{tabular}"])
    out_tex = Path(args.output_tex)
    if not out_tex.is_absolute():
        out_tex = ROOT / out_tex
    out_tex.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")

    print(out_md.read_text(encoding="utf-8"))
    print(f"[OK] {out_md}")
    print(f"[OK] {out_tex}")


if __name__ == "__main__":
    main()
