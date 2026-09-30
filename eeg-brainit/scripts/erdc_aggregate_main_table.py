#!/usr/bin/env python3
"""Aggregate ERDC main-paper table from w7/w8/overnight metric JSONs."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
M7 = ROOT / "outputs/erdc/w7_metrics"
M8 = ROOT / "outputs/erdc/w8_metrics"
MON = ROOT / "outputs/erdc/overnight_metrics"
OUT = ROOT / "outputs/erdc/主表_过夜冻结.txt"


def load(tag: str):
    for base in (MON, M8, M7):
        p = base / f"{tag}.json"
        if p.is_file():
            return json.loads(p.read_text())
    return None


def row(tag: str) -> str:
    d = load(tag)
    if d is None:
        return f"{tag:36s} MISSING"
    tw = ""
    # optional companion 2WC file
    for base in (MON, M8, M7):
        p = base / f"{tag}_2wc.json"
        if p.is_file():
            twod = json.loads(p.read_text())["twoway"]
            tw = (
                f" | 2WC CLIP={twod['clip']*100:.1f}% "
                f"A2={twod['alex2']*100:.1f}% A5={twod['alex5']*100:.1f}% "
                f"Inc={twod['inception']*100:.1f}%"
            )
            break
    return (
        f"{tag:36s} "
        f"Pix={d.get('pixcorr', float('nan')):.4f} "
        f"SSIM={d.get('ssim', float('nan')):.4f} "
        f"CLIP={d.get('clip_cosine', float('nan')):.4f} "
        f"A2={d.get('alexnet2', float('nan')):.4f} "
        f"A5={d.get('alexnet5', float('nan')):.4f} "
        f"Inc={d.get('inception', float('nan')):.4f} "
        f"Eff={d.get('effnet_b1', float('nan')):.4f}"
        f"{tw}"
    )


def main() -> None:
    tags = [
        "w1_b0_atm",
        "w6_b0_bit",
        "w7_bit_k8_brain",
        "official_atm_gen",
        "w8_ras_bit_brain",
        "w8_ras_bit_random",
        "w8_ras_bit_shuffle_brain",
        "w8_ras_bit_lowstr_brain",
        "w8_ras_atm_sub-08_brain",
        "w8_ras_atm_sub-01_brain",
        "w8_ras_atm_sub-02_brain",
    ]
    # add overnight multi-subject if present
    for i in range(1, 11):
        tags.append(f"overnight_ras_atm_sub-{i:02d}_brain")
        tags.append(f"overnight_ras_atm_sub-{i:02d}_random")

    lines = ["ERDC main table freeze", "=" * 72]
    lines.extend(row(t) for t in tags)

    # multi-subject mean for overnight atm brain
    brains = []
    for i in range(1, 11):
        d = load(f"overnight_ras_atm_sub-{i:02d}_brain") or load(f"w8_ras_atm_sub-{i:02d}_brain")
        if d:
            brains.append(d)
    if brains:
        import numpy as np

        def mean_key(k):
            return float(np.mean([b[k] for b in brains]))

        lines.append("")
        lines.append(
            f"MULTI_SUBJ_ATM_RAS_BRAIN_n={len(brains)} "
            f"Pix={mean_key('pixcorr'):.4f} CLIP={mean_key('clip_cosine'):.4f} "
            f"A2={mean_key('alexnet2'):.4f} Inc={mean_key('inception'):.4f}"
        )

    # pass checks
    lines.append("")
    b = load("w8_ras_bit_brain")
    r = load("w8_ras_bit_random")
    sh = load("w8_ras_bit_shuffle_brain")
    w7 = load("w7_bit_k8_brain")
    off = load("official_atm_gen")
    if b and r:
        lines.append(f"PASS_ras_gt_random CLIP={b['clip_cosine']>r['clip_cosine']} Pix={b['pixcorr']>r['pixcorr']}")
    if b and sh:
        lines.append(f"PASS_ras_gt_shuffle Pix={b['pixcorr']>sh['pixcorr']} CLIP={b['clip_cosine']>sh['clip_cosine']}")
    if b and w7:
        lines.append(f"RAS_vs_W7 dCLIP={b['clip_cosine']-w7['clip_cosine']:+.4f} dPix={b['pixcorr']-w7['pixcorr']:+.4f}")
    if b and off:
        lines.append(f"RAS_vs_official dCLIP={b['clip_cosine']-off['clip_cosine']:+.4f} dPix={b['pixcorr']-off['pixcorr']:+.4f}")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT.read_text())
    print("[OK]", OUT)


if __name__ == "__main__":
    main()
