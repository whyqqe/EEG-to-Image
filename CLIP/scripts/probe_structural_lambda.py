#!/usr/bin/env python
"""Sweep the structural expert's fusion weight `lam` on a frozen checkpoint.

WHY A SWEEP AND NOT A SINGLE VALUE. The structural expert (SATTC's mutual-NN /
bidirectional-rank / popularity signals) is a *weak* signal on this fold: the diagnostic
`mutual_topk_enrichment` measures how much more often a pair is mutual-top-k than a random
ranking would give, and it is only ~8x. An unweighted fusion therefore loses badly
(measured: 35.50 -> 18.00 Top-1). The question is not "does it work" but "is there a
non-zero weight at which it pays for itself", and that is a one-dimensional sweep.

`lam=0` must reproduce the pure geometric score exactly, and is asserted below: if it does
not, the fusion is not a clean interpolation and nothing else here can be trusted.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from samclip import calibration, evaluate  # noqa: E402
from samclip.data import things_eeg  # noqa: E402
from samclip.data.targets import load_target_stack  # noqa: E402
from samclip.models import build_model  # noqa: E402
from run_eval import _load_fold_arrays  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--target-subject", type=int, default=8)
    ap.add_argument("--mvnn", default="test")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--lams", type=float, nargs="+",
                    default=[0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3])
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    for ckpt_path in args.ckpts:
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        mc = ck["cfg"]
        img = mc.get("image", {}) or {}
        _, test = _load_fold_arrays(args.target_subject, None, args.mvnn)
        tg = load_target_stack(img.get("feature_set", "clip_hinternvit_multilevel"),
                               img.get("layers"), "test")
        m = build_model(mc, tg.shape[2], tg.shape[-1]).to(device)
        m.load_state_dict(ck["model"])
        m.eval()
        ld = DataLoader(things_eeg.TestDataset(test, tg), batch_size=200,
                        shuffle=False, collate_fn=things_eeg.collate)
        f = evaluate.extract_features(m, ld, device)
        q = np.asarray(f["eeg"])
        g = np.asarray(f["img"])

        qw, _ = calibration.saw_whiten(q)
        qr, _ = calibration.coordinate_recovery(qw, g, k=args.k)

        name = Path(ckpt_path).parent.name
        print(f"=== {name}  (n={q.shape[0]}, d={q.shape[1]}) ===")
        for frame, qq in (("whiten", qw), ("recovery", qr)):
            s = calibration.csls_scores(qq, g, k=args.k)
            base = calibration.report_with_scores(s)["top1"]
            parts = [f"{frame:<9} base={base:>5.1f}"]
            for lam in args.lams:
                sf, d = calibration.structural_scores(s, k=args.k, lam=lam)
                t1 = calibration.report_with_scores(sf)["top1"]
                if lam == 0.0:
                    assert abs(t1 - base) < 1e-9, (
                        f"lam=0 must reproduce the geometric score exactly "
                        f"({t1} vs {base})")
                parts.append(f"l{lam:g}={t1:>5.1f}")
            print("   " + " | ".join(parts))
            print(f"   enrichment={d['mutual_topk_enrichment']:.2f}x  "
                  f"rate={d['mutual_topk_rate']:.4f}  chance={d['mutual_topk_chance']:.4f}  "
                  f"max_deg/(k*nq)={d['max_degree_over_k_nq']:.4f}")
        print()


if __name__ == "__main__":
    main()
