#!/usr/bin/env python3
"""Freeze SCI main-table numbers from existing JOB_COMPLETE files (no training)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "顶会主表_数字冻结.txt"


def load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def pct(x) -> str:
    if x is None:
        return "n/a"
    return f"{float(x) * 100:.2f}%"


def main() -> None:
    lines = [
        "=" * 72,
        "SCI 主表数字冻结（自动汇总，勿手改实验目录）",
        f"生成自 scripts/freeze_sci_main_table.py",
        "=" * 72,
        "",
    ]

    # Abs-only
    abs_only = load(ROOT / "outputs/aria/abs_only_v2/JOB_COMPLETE.json")
    lines.append("[Abs-only / 无 W_s]")
    lines.append(f"  absolute Top-1 = {pct(abs_only['results']['absolute']['top1_mean'])}")
    lines.append("")

    # Identity audit
    audit = load(ROOT / "outputs/aria/identity_audit_v1/JOB_COMPLETE.json")
    lines.append("[ILA 新鲜身份探针]")
    for run in audit["runs"]:
        s = run["summary"]
        lines.append(
            f"  {run['name']:10s} probe={pct(s['identity_probe_fresh_mean'])} "
            f"abs={pct(s['absolute_top1_mean'])} (chance≈{pct(s['chance'])})"
        )
    for c in audit.get("compare") or []:
        lines.append(
            f"  Δ {c['this']} vs {c['baseline']}: "
            f"probe={c['delta_probe']*100:+.1f}pp abs={c['delta_abs']*100:+.1f}pp"
        )
    lines.append("")

    # Negatives
    for name, path in [
        ("abs_id GRL", "outputs/aria/abs_id_v1/JOB_COMPLETE.json"),
        ("abs_center", "outputs/aria/abs_center_v1/JOB_COMPLETE.json"),
    ]:
        m = load(ROOT / path)
        lines.append(f"[{name}]")
        lines.append(f"  absolute Top-1 = {pct(m['results']['absolute']['top1_mean'])}")
        if m.get("compare_to_abs_only"):
            d = m["compare_to_abs_only"]["delta_abs_top1"]
            lines.append(f"  Δ vs abs_only = {d*100:+.2f}pp")
        idp = m["results"].get("identity_probe_acc")
        if idp:
            lines.append(f"  id_probe (run-time) = {pct(idp.get('mean'))}")
        lines.append("")

    # Geometry H1
    g = load(ROOT / "outputs/aria/diagnostics/geometry_h1.json")
    lines.append("[P0 teacher 几何]")
    lines.append(f"  cross RDM = {g['cross_subject_eeg_rdm_corr']['mean']:.3f}")
    lines.append(f"  cross abs cos = {g['cross_subject_eeg_abs_diag_cos']['mean']:.3f}")
    lines.append(f"  ATM abs retrieval = {pct(g['atm_retrieval_absolute']['top1_mean'])}")
    lines.append(f"  ATM rel retrieval = {pct(g['atm_retrieval_relative']['top1_mean'])}")
    lines.append("")

    # Align
    for ver in ["v3", "v4"]:
        p = ROOT / f"outputs/subject_align_loso/loso_{ver}/JOB_COMPLETE.json"
        if not p.is_file():
            continue
        m = load(p)
        lines.append(f"[Subject-Align / SCI 有标签 · loso_{ver}]")
        res = m.get("results") or {}
        # v3 flat; v4 nested by ablation
        if "n0" in res:
            lines.append(f"  n0={pct(res['n0']['top1_mean'])} n_all={pct(res['n_all']['top1_mean'])}")
        else:
            for abl, block in res.items():
                if not isinstance(block, dict) or "n0" not in block:
                    continue
                lines.append(
                    f"  {abl}: n0={pct(block['n0']['top1_mean'])} "
                    f"n_all={pct(block['n_all']['top1_mean'])}"
                )
        lines.append("")

    lines.append("=" * 72)
    lines.append("下一步实验：ST-GATE（无标签坐标接口）→ outputs/st_gate/loso_v1/")
    lines.append("=" * 72)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
