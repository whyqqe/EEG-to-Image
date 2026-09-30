#!/usr/bin/env python3
"""Aggregate standard-7 results into a SOTA-style comparison table.

Reads results.json (eval_standard7) + manifest_full10.json (which lists rows by
group and desired avg_rows with optional pooled FID) and writes a Markdown table.

Auto averages each declared avg_rows group over the manifest rows in that group.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

METRICS = [
    ("pixcorr", "PixCorr\u2191", "{:.4f}"),
    ("ssim", "SSIM\u2191", "{:.4f}"),
    ("alex2", "AlexNet(2)\u2191", "{:.4f}"),
    ("alex5", "AlexNet(5)\u2191", "{:.4f}"),
    ("inception", "Inception\u2191", "{:.4f}"),
    ("clip", "CLIP\u2191", "{:.4f}"),
    ("swav", "SwAV\u2193", "{:.4f}"),
    ("fid", "FID\u2193", "{:.2f}"),
]

REFERENCE_ROWS = [
    ("CognitionCapturer (all) AAAI'25 (lit.)", 0.150, 0.347, 0.754, 0.623, 0.669, 0.715, 0.590, None),
    ("META-MEG Benchetrit 2024 (lit.)", 0.090, 0.341, 0.774, 0.876, 0.703, 0.811, 0.567, None),
    ("MindEye-fMRI Scotti 2024 (lit.)", 0.309, 0.323, 0.947, 0.978, 0.938, 0.941, 0.367, None),
]


def fmt(row: dict, key: str, spec: str) -> str:
    if key not in row or row[key] is None:
        return "\u2014"
    return spec.format(row[key])


def avg_rows_by_group(group: str, manifest_rows: list[dict], by_tag: dict) -> dict | None:
    """Mean metrics over manifest rows in this group that already have results."""
    tags = [r["tag"] for r in manifest_rows if r.get("group") == group and r["tag"] in by_tag]
    if not tags:
        return None
    agg = {}
    for k, _h, _s in METRICS:
        vals = [by_tag[t][k] for t in tags if k in by_tag[t] and by_tag[t][k] is not None]
        agg[k] = float(np.mean(vals)) if vals else None
    agg["_n"] = len(tags)
    agg["_tags"] = tags
    return agg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-json", type=str, required=True)
    ap.add_argument("--manifest", type=str, required=True)
    ap.add_argument("--output-md", type=str, required=True)
    args = ap.parse_args()

    results = json.loads(Path(args.results_json).read_text(encoding="utf-8"))
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    rows = results["rows"]
    by_tag = {r["tag"]: r for r in rows}

    # ---------------- build display list ----------------
    # 1) declared avg groups first (top rows)
    avg_notes = []
    display = []
    for spec in manifest.get("avg_rows", []):
        group = spec["group"]
        agg = avg_rows_by_group(group, manifest["rows"], by_tag)
        if agg is None:
            continue
        pooled = spec.get("fid_pooled")
        row = {
            "display": spec["display"],
            "pixcorr": agg["pixcorr"],
            "ssim": agg["ssim"],
            "alex2": agg["alex2"],
            "alex5": agg["alex5"],
            "inception": agg["inception"],
            "clip": agg["clip"],
            "swav": agg["swav"],
            "fid": pooled if pooled is not None else agg["fid"],
        }
        if pooled is not None:
            avg_notes.append(f"*{spec['display']}: pooled FID {pooled:.2f} "
                             f"(fake={spec.get('n_fake', 2000)}, real=200); "
                             f"per-subject mean FID={agg['fid']:.2f} ({agg['_n']} subjects)*")
        else:
            avg_notes.append(f"*{spec['display']}: per-subject mean FID={agg['fid']:.2f} "
                             f"over {agg['_n']} subjects (no pooled FID reported)*")
        display.append(row)

    # 2) rows NOT in any declared avg group, in manifest order (sub-08 rows etc.)
    avg_groups = {s["group"] for s in manifest.get("avg_rows", [])}
    seen = set()
    for cfg in manifest["rows"]:
        t = cfg["tag"]
        if t in by_tag and cfg.get("group") not in avg_groups and t not in seen:
            seen.add(t)
            display.append({**by_tag[t], "display": cfg.get("display", t)})

    # 3) individual rows of avg groups (per-subject details) after singles
    lines_tail = []
    for spec in manifest.get("avg_rows", []):
        group = spec["group"]
        tags = [r["tag"] for r in manifest["rows"] if r.get("group") == group and r["tag"] in by_tag]
        if len(tags) >= 3:
            lines_tail.append("")
            lines_tail.append(f"### {spec['display']} \u2014 per-subject")
            lines_tail.append("| Method | PixCorr | SSIM | AlexNet(2) | AlexNet(5) | Inception | CLIP | SwAV | FID |")
            lines_tail.append("|---|---|---|---|---|---|---|---|---|")
            for t in tags:
                r = by_tag[t]
                disp = next((c.get("display", t) for c in manifest["rows"] if c["tag"] == t), t)
                vals = [fmt(r, m[0], m[2]) for m in METRICS]
                lines_tail.append(f"| {disp} | " + " | ".join(vals) + " |")

    # ---------------- markdown ----------------
    header = "| Method | " + " | ".join(h for _, h, _ in METRICS) + " |"
    sep = "|---|" + "---|" * len(METRICS)
    lines = [
        "# Standard-7 + FID protocol table (10-subject, aligned to SOTA)",
        "",
        "> Protocol family: Ozcelik & VanRullen / ATM (NeurIPS'24) / MindEye 2-way; "
        "CogCap (AAAI'25) & META-MEG metrics per Benchetrit 2024. "
        "CLIP backbone OpenCLIP ViT-H/14 (same family as MindEye / CogCap).",
        "",
        header,
        sep,
    ]
    for r in display:
        cells = [fmt(r, m[0], m[2]) for m in METRICS]
        lines.append(f"| {r['display']} | " + " | ".join(cells) + " |")
    lines += [""] + avg_notes

    lines += [
        "",
        "## 文献参考值（不是本表实测；用于上下文）",
        "",
        header.replace("Method", "\u6587\u732e"),
        sep,
    ]
    for label, pix, ss, a2, a5, inc, cl, sw, _fid in REFERENCE_ROWS:
        vals = [f"{pix:.4f}", f"{ss:.4f}", f"{a2:.4f}", f"{a5:.4f}", f"{inc:.4f}", f"{cl:.4f}", f"{sw:.4f}", "\u2014"]
        lines.append(f"| {label} | " + " | ".join(vals) + " |")

    lines += lines_tail

    lines += [
        "",
        "## Protocol details",
        "",
        "- **PixCorr**: Pearson on flattened RGB; both images resized to 425\u00d7425 BILINEAR (official).",
        "- **SSIM**: skimage on grayscale 425\u00d7425; `gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0` (official).",
        "- **2-way ID**: Pearson-correlation percent-correct (chance \u2248 50%) on AlexNet `features.4`(A2) / `features.11`(A5), Inception `avgpool`, CLIP final.",
        "- **SwAV**: mean 1-Pearson on SwAV-RN50 `avgpool` @224. **EffNet-B1 distance** also computed but not shown.",
        "- **FID**: torchmetrics `FrechetInceptionDistance(normalize=True)` @299, 200 fake vs 200 real (identical to all historical FIDs).",
        "- 所有行均针对同一组 200 张测试图 (THINGS-EEG test set)，逐索引配对。",
        "- `sdedit_ll` 全受试行 = 每受试用其 HCMA z_ret 训练 EEG\u2192VAE 低层头 + HCMA embed 的 sdedit_ll(LL init, s=0.82) 解码。",
    ]

    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote {args.output_md}")


if __name__ == "__main__":
    main()
