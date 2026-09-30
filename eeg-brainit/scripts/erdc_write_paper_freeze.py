#!/usr/bin/env python3
"""Write outputs/erdc/w16_paper_freeze.md from collected metrics."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MET = ROOT / "outputs/erdc/w15_metrics"
W14 = ROOT / "outputs/erdc/w14_metrics"

rows = [
    "# W16 论文定稿汇总",
    "",
    "## sub-08 主结果",
    "",
    "| 方法 | Pix | CLIP | FID | 2WC | Bootstrap Pix | Bootstrap CLIP |",
    "|------|-----|------|-----|-----|-----------------|----------------|",
]


def load(tag: str) -> dict | None:
    for base in [MET, W14]:
        p = base / f"{tag}.json"
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
    return None


tags = [
    "official_atm_gen",
    "w12_atm_brain",
    "w13_merged_fuse_l0p15",
    "w13_merged_brain",
    "w15_mega_brain",
    "w15_mega_fuse_l0p15",
    "w15_dual_atm_bit_fuse",
]
for tag in tags:
    d = load(tag)
    if not d:
        continue
    fid_p = MET / f"{tag}_fid.json"
    tw_p = MET / f"{tag}_2wc.json"
    if not tw_p.is_file():
        tw_p = W14 / f"{tag}_2wc.json"
    boot_p = MET / f"{tag}_bootstrap.json"
    fid = json.loads(fid_p.read_text(encoding="utf-8"))["fid"] if fid_p.is_file() else None
    tw = (
        json.loads(tw_p.read_text(encoding="utf-8"))["twoway"]["clip"] * 100
        if tw_p.is_file()
        else None
    )
    boot = json.loads(boot_p.read_text(encoding="utf-8")) if boot_p.is_file() else None
    bp = (
        f"{boot['pixcorr']['mean']:.4f} [{boot['pixcorr']['ci95_lo']:.4f},{boot['pixcorr']['ci95_hi']:.4f}]"
        if boot
        else "—"
    )
    bc = (
        f"{boot['clip_cosine']['mean']:.4f} [{boot['clip_cosine']['ci95_lo']:.4f},{boot['clip_cosine']['ci95_hi']:.4f}]"
        if boot
        else "—"
    )
    fid_s = f"{fid:.1f}" if fid is not None else "—"
    tw_s = f"{tw:.1f}%" if tw is not None else "—"
    rows.append(
        f"| {tag} | {d['pixcorr']:.4f} | {d['clip_cosine']:.4f} | "
        f"{fid_s} | {tw_s} | {bp} | {bc} |"
    )

ten = ROOT / "outputs/erdc/paper_ten_subject_table.md"
if ten.is_file():
    rows.extend(["", "## 十被试", "", ten.read_text(encoding="utf-8")])

main = ROOT / "outputs/erdc/paper_main_table.md"
if main.is_file():
    rows.extend(["", "## 主表", "", main.read_text(encoding="utf-8")])

out = ROOT / "outputs/erdc/w16_paper_freeze.md"
out.write_text("\n".join(rows) + "\n", encoding="utf-8")
print(out.read_text(encoding="utf-8"))
print(f"[OK] {out}")
