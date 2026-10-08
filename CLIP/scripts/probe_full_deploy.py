#!/usr/bin/env python
"""Full DEPLOY ladder on a frozen checkpoint, including the structural expert.

WHY THIS EXISTS
---------------
`run_eval.py` measures the *training-side* ladder (raw cosine up to CSLS). This script
measures the *deployment* stack, which is a different question and now the one that
matters: with the encoders frozen, how far does the label-free calibration carry a
checkpoint that was never trained for it?

It exists because of a measured result on 2026-10-04. Applying SCORE's deploy-time
orthogonal recovery to `v5-a1-k20` lifted Top-1 from 27.00 to **35.50** on sub-08, with
no retraining -- past SAMGA's LOSO average (34.4) and SVTL's no-adaptation number (35.3).
That single fact changes the priority order: the deploy stack is not a postscript to the
architecture, it is where the next ten points live.

Ladder (each rung is label-free and transductive over the query set only):

    raw cosine
    + CSLS
    + centre + CSLS
    + whiten + CSLS
    + whiten + CSLS + recovery            <- SCORE's coordinate recovery
    + whiten + CSLS + structural          <- SATTC's structural expert (PoE)
    + whiten + CSLS + recovery + structural

Run:
  python scripts/probe_full_deploy.py --ckpt outputs/stage1/v5-a1-k20/last.pt \
      --target-subject 8 --mvnn test --device cuda
"""
from __future__ import annotations

import argparse
import json
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--target-subject", type=int, default=8)
    ap.add_argument("--mvnn", default="test", choices=["train", "test", "off"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--struct-k", type=int, default=10)
    ap.add_argument("--hub-alpha", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_cfg = ckpt["config"]

    channel_set = model_cfg.get("channel_set", "all63")
    channels = (config.CHANNELS_OCCIPITO_PARIETAL
                if channel_set == "occipital17" else None)
    img = model_cfg.get("image", {}) or {}
    feature_set = img.get("feature_set", "clip_h14_multilevel")
    layers = img.get("layers")

    # the fold loader lives in run_eval and encodes the train-split z-score convention;
    # re-deriving it here would be a second place for that convention to drift.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_eval import _load_fold_arrays  # type: ignore

    _, test = _load_fold_arrays(args.target_subject, channels, args.mvnn)
    targets_te = load_target_stack(feature_set, layers, "test")

    model = build_model(model_cfg, targets_te.shape[2], targets_te.shape[-1]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    loader = DataLoader(things_eeg.TestDataset(test, targets_te), batch_size=200,
                        shuffle=False, collate_fn=things_eeg.collate)
    feats = evaluate.extract_features(model, loader, device)
    q = feats["eeg"].numpy() if torch.is_tensor(feats["eeg"]) else np.asarray(feats["eeg"])
    g = (feats["img"].numpy() if torch.is_tensor(feats["img"])
         else np.asarray(feats["img"]))

    def score_of(qq, use_csls=True):
        if use_csls:
            return calibration.csls_scores(qq, g, k=args.k)
        qn = qq / np.maximum(np.linalg.norm(qq, axis=-1, keepdims=True), 1e-8)
        gn = g / np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-8)
        return qn @ gn.T

    rows: dict = {}
    rows["raw cosine"] = calibration.report_with_scores(score_of(q, False))

    q_c, _ = calibration.center_queries(q)
    q_w, _ = calibration.saw_whiten(q)
    q_rec, rec_diag = calibration.coordinate_recovery(q_w, g, k=args.k)

    base = {
        "+ CSLS": q,
        "+ centre + CSLS": q_c,
        "+ whiten + CSLS": q_w,
        "+ whiten + CSLS + recovery": q_rec,
    }
    for name, qq in base.items():
        rows[name] = calibration.report_with_scores(score_of(qq, True))

    # structural expert, fused with the geometric score by Product-of-Experts, on top of
    # the two best coordinate frames (plain whiten, and recovery).
    s_geom = score_of(q_w, True)
    s_fused, s_diag = calibration.structural_scores(
        s_geom, k=args.struct_k, hub_alpha=args.hub_alpha)
    rows["+ whiten + CSLS + structural"] = calibration.report_with_scores(s_fused)

    s_geom_r = score_of(q_rec, True)
    s_fused_r, s_diag_r = calibration.structural_scores(
        s_geom_r, k=args.struct_k, hub_alpha=args.hub_alpha)
    rows["+ whiten + CSLS + recovery + structural"] = calibration.report_with_scores(
        s_fused_r)

    name = ckpt_path.parent.name
    print(f"[full-deploy] {ckpt_path}  epoch={ckpt.get('epoch')}")
    print(f"  n_queries={q.shape[0]}  d={q.shape[1]}  gallery={g.shape}")
    print(f"  {'row':<42}{'Top-1':>7}{'Top-5':>7}{'meanrank':>10}")
    for k, v in rows.items():
        print(f"  {k:<42}{v['top1']:>7.2f}{v['top5']:>7.2f}{v['mean_rank']:>10.2f}")
    print(f"\n  structural diag (whiten): {json.dumps(s_diag)}")
    print(f"  recovery diag: {json.dumps({k: v for k, v in rec_diag.items() if not isinstance(v, np.ndarray)}, default=str)[:300]}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"ckpt": str(ckpt_path), "target_subject": args.target_subject,
             "mvnn": args.mvnn, "rows": rows,
             "structural_diag": s_diag, "structural_diag_recovery": s_diag_r,
             "recovery_diag": {k: (v if not isinstance(v, np.ndarray) else None)
                               for k, v in rec_diag.items()}},
            indent=2, default=str))
        print(f"\n[full-deploy] wrote {args.out}")


if __name__ == "__main__":
    main()
