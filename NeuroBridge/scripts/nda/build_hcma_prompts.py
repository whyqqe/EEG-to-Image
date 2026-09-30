#!/usr/bin/env python3
"""Build HCMA hierarchical prompts: subject / detail / background roles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


# coarse scene prior by simple keyword buckets (no VLM needed)
SCENE_RULES = [
    (("ocean", "boat", "ship", "carrier", "fish", "seal", "whale", "calamari", "beaver"), "on water"),
    (("bench", "bike", "unicycle", "cart", "buggy", "road", "car"), "outdoors"),
    (("cake", "bread", "cheese", "sausage", "banana", "cashew", "bok choy", "basil", "bun"), "on a table"),
    (("cat", "dog", "cheetah", "antelope", "bug", "caterpillar", "grasshopper", "bat"), "in a natural setting"),
    (("basketball", "baseball", "balance beam", "baton"), "in a sports setting"),
]


def scene_for(concept: str) -> str:
    c = concept.lower()
    for keys, scene in SCENE_RULES:
        if any(k in c for k in keys):
            return scene
    return "in a clean studio setting"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concepts-json", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    args = ap.parse_args()

    concepts = json.loads(Path(args.concepts_json).read_text(encoding="utf-8"))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    prompts_subj = [f"a photo of a {c}" for c in concepts]
    prompts_subj_det = [
        f"a photo of a {c}, clearly showing its shape, color, and distinctive parts, highly detailed"
        for c in concepts
    ]
    prompts_full = [
        f"a photo of a {c}, clearly showing its shape, color, and distinctive parts, {scene_for(c)}, natural lighting"
        for c in concepts
    ]
    # detail-only residual phrase (for logging / future weighted compose)
    prompts_det = [f"detailed {c} with true-to-class appearance" for c in concepts]
    prompts_bg = [scene_for(c) for c in concepts]

    (out / "prompts_subj_test.json").write_text(json.dumps(prompts_subj, indent=2), encoding="utf-8")
    (out / "prompts_subj_det_test.json").write_text(json.dumps(prompts_subj_det, indent=2), encoding="utf-8")
    (out / "prompts_full_hcma_test.json").write_text(json.dumps(prompts_full, indent=2), encoding="utf-8")
    (out / "prompts_det_test.json").write_text(json.dumps(prompts_det, indent=2), encoding="utf-8")
    (out / "prompts_bg_test.json").write_text(json.dumps(prompts_bg, indent=2), encoding="utf-8")
    report = {
        "n": len(concepts),
        "roles": ["subj", "subj_det", "full_hcma", "det", "bg"],
        "note": "Hierarchical text roles for HCMA; no saliency R branch",
    }
    (out / "hcma_prompts_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
