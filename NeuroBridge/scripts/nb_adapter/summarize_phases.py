#!/usr/bin/env python3
"""Aggregate metrics from all phases into one summary JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict | None:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=str, required=True)
    ap.add_argument("--output-json", type=str, required=True)
    args = ap.parse_args()

    root = Path(args.out_root)
    summary = {"out_root": str(root), "phases": {}, "generation": {}, "clip_fid": None}

    summary["phases"]["0_rag"] = load_json(root / "phase0" / "rag_report.json")
    summary["phases"]["1_prior_scratch"] = load_json(root / "phase1" / "prior_scratch_report.json")
    summary["phases"]["1_prior_pretrained"] = load_json(root / "phase1" / "prior_pretrained_report.json")
    summary["phases"]["2_dual"] = load_json(root / "phase2" / "dual_teacher" / "dual_report.json")
    summary["phases"]["3_ensemble"] = load_json(root / "phase3" / "ensemble_report.json")

    gen_root = root / "generation_full200"
    if gen_root.is_dir():
        for tag_dir in sorted(gen_root.iterdir()):
            if not tag_dir.is_dir():
                continue
            m = load_json(tag_dir / "metrics.json")
            if m:
                summary["generation"][tag_dir.name] = m

    fid = load_json(root / "clip_fid_metrics_full200.json")
    if fid:
        summary["clip_fid"] = fid

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[OK] {out}")


if __name__ == "__main__":
    main()
