#!/usr/bin/env python3
"""P0 diagnostic for ARIA: is cross-subject geometry conserved better than absolute coords?

Computes, using ATM teacher EEG embeddings (or raw proxy):
  H1a: within each subject, RDM(EEG) corr RDM(CLIP)
  H1b: across subjects, mean pairwise RDM(EEG_s, EEG_t) vs absolute embedding alignment
  H1c: relative-profile retrieval vs absolute cosine retrieval (ATM features, no training)

Writes: outputs/aria/diagnostics/geometry_h1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eeg_brainit.models.aria import build_class_anchors, relative_from_anchors
import torch


ALL_SUBJECTS = [f"sub-{i:02d}" for i in range(1, 11)]


def l2(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


def rdm_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson corr of upper-triangular pairwise cosine similarities."""
    a = l2(a)
    b = l2(b)
    sa = a @ a.T
    sb = b @ b.T
    iu = np.triu_indices(a.shape[0], k=1)
    va, vb = sa[iu], sb[iu]
    va = va - va.mean()
    vb = vb - vb.mean()
    den = np.linalg.norm(va) * np.linalg.norm(vb) + 1e-8
    return float((va * vb).sum() / den)


def abs_alignment(a: np.ndarray, b: np.ndarray) -> float:
    """Mean diagonal cosine after independent L2 (same index = same stimulus)."""
    a = l2(a)
    b = l2(b)
    n = min(a.shape[0], b.shape[0])
    return float((a[:n] * b[:n]).sum(1).mean())


def retrieval(q: np.ndarray, g: np.ndarray) -> dict[str, float]:
    q, g = l2(q), l2(g)
    sim = q @ g.T
    ranks = np.argmax(np.argsort(-sim, axis=1) == np.arange(sim.shape[0])[:, None], axis=1)
    return {"top1": float((ranks < 1).mean()), "top5": float((ranks < 5).mean())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--atm-bridge-dir", default="outputs/atm_bridge")
    p.add_argument("--output-dir", default="outputs/aria/diagnostics")
    p.add_argument("--n-test", type=int, default=200)
    p.add_argument("--anchor-k", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    bridge = ROOT / args.atm_bridge_dir
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    gallery = np.load(bridge / "clip_img_test_1024.npy").astype(np.float32)
    clip_train = np.load(bridge / "clip_img_train_1024.npy").astype(np.float32)
    anchors = build_class_anchors(clip_train)
    rng = np.random.RandomState(args.seed)
    if args.anchor_k < anchors.shape[0]:
        idx = np.sort(rng.choice(anchors.shape[0], size=args.anchor_k, replace=False))
        anchors = anchors[idx]

    # per-subject test ATM embeddings
    eeg = {}
    for sub in ALL_SUBJECTS:
        path = bridge / f"{sub}_test_eeg_1024.npy"
        if path.is_file():
            eeg[sub] = np.load(path).astype(np.float32)[: args.n_test]

    within_rdm = {sub: rdm_corr(eeg[sub], gallery[: eeg[sub].shape[0]]) for sub in eeg}
    abs_to_clip = {sub: abs_alignment(eeg[sub], gallery[: eeg[sub].shape[0]]) for sub in eeg}

    # cross-subject RDM vs absolute
    subs = list(eeg.keys())
    cross_rdm, cross_abs = [], []
    for i, s in enumerate(subs):
        for t in subs[i + 1 :]:
            n = min(eeg[s].shape[0], eeg[t].shape[0])
            cross_rdm.append(rdm_corr(eeg[s][:n], eeg[t][:n]))
            cross_abs.append(abs_alignment(eeg[s][:n], eeg[t][:n]))

    # relative vs absolute retrieval (ATM teacher features)
    anchors_t = torch.from_numpy(anchors)
    rel_gallery = relative_from_anchors(torch.from_numpy(gallery), anchors_t).numpy()
    abs_ret, rel_ret = {}, {}
    for sub, e in eeg.items():
        abs_ret[sub] = retrieval(e, gallery[: e.shape[0]])
        rel_e = relative_from_anchors(torch.from_numpy(e), anchors_t).numpy()
        rel_ret[sub] = retrieval(rel_e, rel_gallery[: e.shape[0]])

    report = {
        "hypothesis": "H1 geometry conservation (ATM teacher features)",
        "within_subject_rdm_corr_eeg_clip": {
            "mean": float(np.mean(list(within_rdm.values()))),
            "std": float(np.std(list(within_rdm.values()))),
            "per_subject": within_rdm,
        },
        "within_subject_abs_diag_cos_eeg_clip": {
            "mean": float(np.mean(list(abs_to_clip.values()))),
            "std": float(np.std(list(abs_to_clip.values()))),
            "per_subject": abs_to_clip,
        },
        "cross_subject_eeg_rdm_corr": {
            "mean": float(np.mean(cross_rdm)),
            "std": float(np.std(cross_rdm)),
        },
        "cross_subject_eeg_abs_diag_cos": {
            "mean": float(np.mean(cross_abs)),
            "std": float(np.std(cross_abs)),
        },
        "atm_retrieval_absolute": {
            "top1_mean": float(np.mean([v["top1"] for v in abs_ret.values()])),
            "top5_mean": float(np.mean([v["top5"] for v in abs_ret.values()])),
            "per_subject": abs_ret,
        },
        "atm_retrieval_relative": {
            "top1_mean": float(np.mean([v["top1"] for v in rel_ret.values()])),
            "top5_mean": float(np.mean([v["top5"] for v in rel_ret.values()])),
            "per_subject": rel_ret,
        },
        "h1_support": {
            "cross_rdm_gt_cross_abs": bool(np.mean(cross_rdm) > np.mean(cross_abs)),
            "note": "If cross RDM >> cross abs, geometry is more conserved than absolute coords.",
        },
        "anchor_k": int(anchors.shape[0]),
    }
    path = out_dir / "geometry_h1.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["h1_support"], indent=2))
    print(
        f"cross RDM={report['cross_subject_eeg_rdm_corr']['mean']:.4f} "
        f"cross abs={report['cross_subject_eeg_abs_diag_cos']['mean']:.4f}"
    )
    print(
        f"ATM abs top1={report['atm_retrieval_absolute']['top1_mean']*100:.2f}% "
        f"rel top1={report['atm_retrieval_relative']['top1_mean']*100:.2f}%"
    )
    print("[OK]", path)


if __name__ == "__main__":
    main()
