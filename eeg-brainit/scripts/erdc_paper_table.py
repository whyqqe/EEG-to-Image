#!/usr/bin/env python3
"""Build publication-ready main table (Markdown + optional LaTeX) from ERDC metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Friendly names for paper table
DISPLAY = {
    "official_atm_gen": "Official ATM",
    "w12_atm_brain": "Ours: Turbo + ATM + brain",
    "w12_official_atm_brain": "Ours: Turbo + ATM + brain",
    "w13_merged_brain": "Ours: merge brain",
    "w14_merged_ctrl_retrieve": "Mech: retrieve (fuse)",
    "w14_merged_ctrl_shuffle": "Mech: shuffle struct",
    "w14_merged_ctrl_misalign": "Mech: misalign struct",
    "w14_merged_ctrl_zero": "Mech: zero struct",
    "w13_merged_fuse_l0p15": "Ours: merge fuse λ=0.15 (main)",
    "w13_merged_brain": "Ours: merge brain",
    "w15_mega_brain": "Ours: mega bank + brain",
    "w15_mega_fuse_l0p10": "Ours: mega fuse λ=0.10",
    "w15_mega_fuse_l0p15": "Ours: mega fuse λ=0.15",
    "w15_mega_fuse_l0p20": "Ours: mega fuse λ=0.20",
    "w15_mega_topk2_l0p15": "Ours: mega top-2 struct",
    "w15_mega_topk3_l0p15": "Ours: mega top-3 struct",
    "w15_dual_atm_bit_fuse": "Ours: dual-path fuse",
    "w10_sdxl_fuse": "Ablation: SDXL fuse",
    "w12_fuse_bit_l0p25": "W12 bit fuse λ=0.25",
}


def load_json(paths: list[Path]) -> dict | None:
    for p in paths:
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
    return None


def fmt3(x: float) -> str:
    return f"{x:.3f}"


def bold_if_best(val: float, best: float, s: str) -> str:
    if abs(val - best) < 1e-5:
        return f"**{s}**"
    return s


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dirs", type=str, nargs="+", default=["outputs/erdc/w15_metrics"])
    parser.add_argument("--tags", type=str, nargs="+", required=True)
    parser.add_argument("--baseline-tag", type=str, default="official_atm_gen")
    parser.add_argument("--output-md", type=str, default="outputs/erdc/paper_main_table.md")
    parser.add_argument("--output-tex", type=str, default="outputs/erdc/paper_main_table.tex")
    args = parser.parse_args()

    bases = [ROOT / d for d in args.metrics_dirs]
    rows_data = []
    for tag in args.tags:
        m = load_json([b / f"{tag}.json" for b in bases])
        if not m:
            continue
        fid = load_json([b / f"{tag}_fid.json" for b in bases])
        tw = load_json([b / f"{tag}_2wc.json" for b in bases])
        rows_data.append(
            {
                "tag": tag,
                "name": DISPLAY.get(tag, tag),
                "pix": m["pixcorr"],
                "clip": m["clip_cosine"],
                "ssim": m.get("ssim", 0),
                "fid": fid.get("fid") if fid else None,
                "twc": tw["twoway"]["clip"] * 100 if tw else None,
            }
        )

    base = next((r for r in rows_data if r["tag"] == args.baseline_tag), None)
    b_pix = base["pix"] if base else 0.0
    b_clip = base["clip"] if base else 0.0

    best_pix = max(r["pix"] for r in rows_data)
    best_clip = max(r["clip"] for r in rows_data)
    fids = [r["fid"] for r in rows_data if r["fid"] is not None]
    best_fid = min(fids) if fids else None
    twcs = [r["twc"] for r in rows_data if r["twc"] is not None]
    best_twc = max(twcs) if twcs else None

    md = [
        "# 论文主表（自动生成）",
        "",
        "> Pix/CLIP 相对 Official ATM 的 Δ；**粗体**为列内最优。",
        "",
        "| Method | PixCorr | CLIP | Δ Pix | Δ CLIP | FID↓ | 2WC (CLIP) |",
        "|--------|---------|------|-------|--------|------|------------|",
    ]
    tex_rows = []
    for r in rows_data:
        dp = r["pix"] - b_pix
        dc = r["clip"] - b_clip
        fid_s = f"{r['fid']:.1f}" if r["fid"] is not None else "—"
        tw_s = f"{r['twc']:.1f}%" if r["twc"] is not None else "—"
        pix_cell = bold_if_best(r["pix"], best_pix, fmt3(r["pix"]))
        clip_cell = bold_if_best(r["clip"], best_clip, fmt3(r["clip"]))
        if r["fid"] is not None and best_fid is not None:
            fid_cell = bold_if_best(r["fid"], best_fid, fid_s) if r["fid"] == best_fid else fid_s
        else:
            fid_cell = fid_s
        if r["twc"] is not None and best_twc is not None:
            tw_cell = bold_if_best(r["twc"], best_twc, tw_s) if r["twc"] == best_twc else tw_s
        else:
            tw_cell = tw_s
        md.append(
            f"| {r['name']} | {pix_cell} | {clip_cell} | {dp:+.3f} | {dc:+.3f} | {fid_cell} | {tw_cell} |"
        )
        tex_rows.append(
            f"{r['name']} & {fmt3(r['pix'])} & {fmt3(r['clip'])} & {dp:+.3f} & {dc:+.3f} & {fid_s} & {tw_s} \\\\"
        )

    md.append("")
    md.append("## 推荐 headline 行")
    # Pix+CLIP both >= baseline
    dual = [
        r for r in rows_data
        if r["pix"] >= b_pix - 0.001 and r["clip"] >= b_clip and r["tag"] != args.baseline_tag
    ]
    for r in sorted(dual, key=lambda x: x["clip"] + x["pix"], reverse=True)[:3]:
        md.append(f"- **{r['name']}**: Pix={fmt3(r['pix'])}, CLIP={fmt3(r['clip'])}")

    out_md = Path(args.output_md)
    if not out_md.is_absolute():
        out_md = ROOT / out_md
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    tex = [
        "% Auto-generated — requires booktabs",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "Method & Pix & CLIP & $\\Delta$Pix & $\\Delta$CLIP & FID & 2WC \\\\",
        "\\midrule",
        *tex_rows,
        "\\bottomrule",
        "\\end{tabular}",
    ]
    out_tex = Path(args.output_tex)
    if not out_tex.is_absolute():
        out_tex = ROOT / out_tex
    out_tex.write_text("\n".join(tex) + "\n", encoding="utf-8")

    print(out_md.read_text(encoding="utf-8"))
    print(f"[OK] {out_md}")
    print(f"[OK] {out_tex}")


if __name__ == "__main__":
    main()
