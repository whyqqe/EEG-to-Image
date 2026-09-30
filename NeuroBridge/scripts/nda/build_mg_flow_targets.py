#!/usr/bin/env python3
"""Build MG-Flow coarse/fine text targets from existing assets (+ optional template enrich)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def l2(x: np.ndarray) -> np.ndarray:
    return (x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-8)).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-text-root", type=str, required=True, help="nda_ss/.../clip_text")
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--images-per-concept-train", type=int, default=10)
    args = ap.parse_args()

    root = Path(args.clip_text_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # train
    t_concept_tr = l2(np.load(root / "train" / "text_concept_clip.npy"))  # (1654, D)
    t_flat_tr = l2(np.load(root / "train" / "text_flat_clip.npy"))  # (16540, D)
    n_flat = t_flat_tr.shape[0]
    ipc = args.images_per_concept_train
    concept_idx = np.arange(n_flat) // ipc
    t_coarse_tr = t_concept_tr[concept_idx]
    # fine = 0.6*flat + 0.4*coarse (attribute + class lock)
    t_fine_tr = l2(0.6 * t_flat_tr + 0.4 * t_coarse_tr)

    # test
    t_concept_te = l2(np.load(root / "test" / "text_concept_clip.npy"))  # (200, D)
    t_flat_te = l2(np.load(root / "test" / "text_flat_clip.npy")) if (root / "test" / "text_flat_clip.npy").is_file() else t_concept_te
    if t_flat_te.shape[0] != t_concept_te.shape[0]:
        t_flat_te = t_concept_te
    t_coarse_te = t_concept_te
    t_fine_te = l2(0.6 * t_flat_te + 0.4 * t_coarse_te)

    concepts_tr = json.loads((root / "train" / "concept_phrases.json").read_text(encoding="utf-8"))
    concepts_te = json.loads((root / "test" / "concept_phrases.json").read_text(encoding="utf-8"))

    prompts_c_te = [f"a photo of {c}, highly detailed" for c in concepts_te]
    prompts_f_te = [
        f"a photo of {c}, clearly showing its shape, color, and distinctive parts"
        for c in concepts_te
    ]
    # dual prompt for generation: coarse class lock + fine attributes cue
    prompts_dual_te = [
        f"a photo of {c}, highly detailed, natural lighting, true-to-class appearance"
        for c in concepts_te
    ]

    np.save(out / "t_coarse_train.npy", t_coarse_tr)
    np.save(out / "t_fine_train.npy", t_fine_tr)
    np.save(out / "t_coarse_test.npy", t_coarse_te)
    np.save(out / "t_fine_test.npy", t_fine_te)
    (out / "prompts_coarse_test.json").write_text(json.dumps(prompts_c_te, indent=2), encoding="utf-8")
    (out / "prompts_fine_test.json").write_text(json.dumps(prompts_f_te, indent=2), encoding="utf-8")
    (out / "prompts_dual_test.json").write_text(json.dumps(prompts_dual_te, indent=2), encoding="utf-8")
    (out / "concepts_test.json").write_text(json.dumps(concepts_te, indent=2), encoding="utf-8")

    report = {
        "t_coarse_train": list(t_coarse_tr.shape),
        "t_fine_train": list(t_fine_tr.shape),
        "t_coarse_test": list(t_coarse_te.shape),
        "t_fine_test": list(t_fine_te.shape),
        "n_concepts_train": len(concepts_tr),
        "n_concepts_test": len(concepts_te),
        "fine_mix": "0.6*flat + 0.4*coarse",
        "note": "No extra VLM download; reuse nda_ss clip_text banks",
    }
    (out / "targets_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
