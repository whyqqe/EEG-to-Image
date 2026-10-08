#!/usr/bin/env python
"""Sweep the label-free calibration rungs for each checkpoint, over their free knobs.

Why this exists
---------------
The eval report's ladder (`raw -> SAW whiten -> CSLS -> whiten+CSLS`) is the deployable
number, and `saw_whiten`'s `shrink` / `max_cond` are FIXED constants in `calibration.py`.
That is fine while comparing arms of one representation, and it is a confound the moment
two checkpoints have different covariance spectra: the rung then measures "how well does
shrink=0.1 suit this embedding", not "how good is this embedding".

This probe separates the two. It reports, per checkpoint, the whole shrink curve rather
than the single default point, so a ladder difference can be attributed to the
representation or to the untuned knob.

Run:  python scripts/probe_calibration_rungs.py --ckpts a/last.pt b/last.pt \
          --target-subject 8 --mvnn test
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from samclip import calibration, config, evaluate  # noqa: E402
from samclip.data import things_eeg  # noqa: E402
from samclip.data.targets import load_target_stack  # noqa: E402
from samclip.models import build_model  # noqa: E402


def extract(ckpt_path: Path, target_subject: int, mvnn: str, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_cfg = ckpt["cfg"]
    channel_set = model_cfg.get("channel_set", "all63")
    channels = (config.CHANNELS_OCCIPITO_PARIETAL
                if channel_set == "occipital17" else None)
    img = model_cfg.get("image", {}) or {}
    _, test = things_eeg.load_subject_std(target_subject, channels, mvnn=mvnn)
    test = np.asarray(test)
    targets_te = load_target_stack(img.get("feature_set", "clip_h14_multilevel"),
                                   img.get("layers"), "test")
    model = build_model(model_cfg, targets_te.shape[2], targets_te.shape[-1]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    loader = DataLoader(things_eeg.TestDataset(test, targets_te), batch_size=200,
                        shuffle=False, collate_fn=things_eeg.collate)
    feats = evaluate.extract_features(model, loader, device)
    return feats["eeg"], feats["img"], model_cfg


def raw_cos_top1(q: np.ndarray, g: np.ndarray) -> float:
    qn = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8)
    gn = g / np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-8)
    return calibration.report_with_scores(qn @ gn.T)["top1"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--target-subject", type=int, required=True)
    ap.add_argument("--mvnn", default="test")
    ap.add_argument("--csls-k", type=int, default=10)
    ap.add_argument("--shrinks", type=float, nargs="*",
                    default=[0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7])
    args = ap.parse_args()

    device = torch.device("cpu")
    print(f"[probe] target=sub-{args.target_subject:02d} mvnn={args.mvnn} "
          f"n={config.N_TEST_CONCEPTS}-way")

    for c in args.ckpts:
        p = Path(c)
        tag = f"{p.parent.parent.name}/{p.parent.name}"
        q, g, cfg = extract(p, args.target_subject, args.mvnn, device)
        raw = raw_cos_top1(q, g)
        csls_only = calibration.report_with_scores(
            calibration.csls_scores(q, g, k=args.csls_k))["top1"]

        print(f"\n{'=' * 78}\n[{tag}]  fusion={cfg.get('target_fusion')}\n{'=' * 78}")
        print(f"  raw cosine        top1 {raw:6.2f}")
        print(f"  + CSLS (no whiten) top1 {csls_only:6.2f}")
        print(f"\n  {'shrink':>8} {'whiten':>9} {'whiten+CSLS':>13} {'cond':>9}")
        best = (None, -1.0)
        for sh in args.shrinks:
            qw, wd = calibration.saw_whiten(q, shrink=sh)
            w1 = raw_cos_top1(qw, g)
            w2 = calibration.report_with_scores(
                calibration.csls_scores(qw, g, k=args.csls_k))["top1"]
            mark = "  <- default" if abs(sh - 0.1) < 1e-9 else ""
            print(f"  {sh:>8.2f} {w1:>9.2f} {w2:>13.2f} {wd['cond']:>9.1f}{mark}")
            if w2 > best[1]:
                best = (sh, w2)
        print(f"  best whiten+CSLS: shrink={best[0]:.2f} -> {best[1]:.2f} "
              f"(default 0.10 -> {[calibration.report_with_scores(calibration.csls_scores(calibration.saw_whiten(q, shrink=0.1)[0], g, k=args.csls_k))['top1']][0]:.2f})")


if __name__ == "__main__":
    main()
