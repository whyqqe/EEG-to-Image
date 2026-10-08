#!/usr/bin/env python
"""PREDICTION 1 (the go/no-go gate for v11): does POOLING structure views beat one view?

THE CLAIM BEING TESTED
----------------------
v11 (docs/eeg2image_v11_unified_architecture.md) reframes cross-subject retrieval as multi-view
DENOISING: the target subject, the (independent) source subjects and the image gallery are noisy
views of one shared concept manifold, and the task is to estimate THAT structure rather than to
fit and invert a map between views. The reframe is only worth building if pooling estimates
actually raises the structure's SNR. Its single gate is Prediction 1:

    pooled-metric agreement with the gallery metric  >  single-view agreement (0.593),
    monotonically in the number of pooled views K,
    and NOT reproduced when the concept correspondence is destroyed.

The control is the same one that killed M1 (job 645593): shuffle the concept axis so the geometry
survives but the correspondence does not. A pool that gains without separating from that control
is M1 again, and this script must say so.

TWO REFERENCE ARMS, BECAUSE ONE ALONE WOULD BE CIRCULAR
------------------------------------------------------
  * within-EEG leave-one-view-out -- pool K of the 9 source subjects, score against the
    HELD-OUT 9th. Needs no image features, no cross-space assumption and no argument about which
    gallery fusion the deployed model used. This is the clean SNR test.
  * cross-modal -- pool K source subjects, score against the image gallery metric (0.593's arm).
    The gallery here is the layer-mean of the cached l2-normalised features rather than the
    model's routed fusion, so the ABSOLUTE value differs from 0.593 by construction; the
    internal comparison (pooled vs single, same di) is what is read, and it is paired per fold.

WHY THE LOGIN NODE IS NOT USED: submitted via slurm (see slurm/probe_fusion.sbatch). The algebra
is small, but this repo's rule is that nothing runs on the shared login node (AGENTS.md §3.1).

Writes outputs/probe/fusion/fusion_probe.json + a text table; CPU or CUDA, chosen automatically.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
OUT_DIR = os.path.join(ROOT, "outputs", "probe", "fusion")

try:
    import torch
    _DEV = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:                                     # pragma: no cover
    torch = None
    _DEV = "cpu"


def _mm_np(x: np.ndarray) -> np.ndarray:
    """Per-view moment match: the normalisation the deployed frame applies before a metric."""
    return (x - x.mean(0, keepdims=True)) / (x.std(0, keepdims=True) + 1e-8)


def _metric(x: np.ndarray) -> np.ndarray:
    """Standardised squared chordal distance, i.e. EXACTLY calibration._sq_cos_dist.

    Reimplemented here rather than imported so the probe can run under plain numpy on a compute
    node without dragging in the package's import graph; the definition is pinned by
    self--test at startup against the real one when the package is importable.
    """
    n = x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9)
    d = (n @ n.T - 1.0) ** 2
    return (d - d.mean()) / max(float(d.std()), 1e-9)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    iu = np.triu_indices(a.shape[0], 1)
    return float(np.corrcoef(a[iu], b[iu])[0, 1])


def load_sources(pattern: str) -> dict:
    """One npz per fold; each holds that fold's 9 source subjects' concept means in ONE encoder's
    output space, so the 9 clouds are directly comparable points."""
    out = {}
    for f in sorted(glob.glob(pattern)):
        z = np.load(f)
        out[os.path.basename(f)[:-5]] = np.asarray(z["src_means"], dtype=np.float64)
    return out


def load_gallery(path: str) -> np.ndarray:
    """Cached multi-layer image features -> one fused gallery embedding per concept.

    The stack is (C, I, K, D) with K layers; fused by the layer-and-image MEAN of the already
    l2-normalised features. This is a stand-in for the model's routed fusion and is labelled as
    such in the output: it is used only as a common target for the pooled-vs-single comparison.
    """
    a = np.load(path, mmap_mode="r")
    a = np.asarray(a, dtype=np.float64)
    while a.ndim > 2:
        a = a.mean(axis=1)
    a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-9)
    return a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-glob",
                    default=os.path.join(ROOT, "outputs/src_metric/v8/sub*_seed2025.npz"))
    ap.add_argument("--gallery", default=os.path.join(
        ROOT, "data/cache/targets/internvit_multilevel_test_L20-24-28-32-36_l2.npy"))
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[fusion] device={_DEV} (torch={'yes' if torch else 'no'})")
    if torch is not None and _DEV == "cuda":
        print(f"[fusion] gpu={torch.cuda.get_device_name(0)}")

    # fidelity: the local metric must equal the package's, or none of this is the same object
    try:
        from samclip.calibration import _sq_cos_dist  # noqa: PLC0415
        r = np.random.default_rng(0).standard_normal((7, 5))
        assert np.allclose(_metric(r), _sq_cos_dist(r)), "metric definition drifted"
        print("[fusion] metric self-test vs samclip.calibration._sq_cos_dist: OK")
    except ImportError:
        print("[fusion] WARNING: samclip not importable; metric self-test skipped")

    folds = load_sources(args.src_glob)
    if args.limit:
        folds = dict(sorted(folds.items())[: args.limit])
    if not folds:
        raise SystemExit(f"no npz matched {args.src_glob}")
    gal = load_gallery(args.gallery)
    print(f"[fusion] {len(folds)} folds | gallery fused to {gal.shape}")
    di = _metric(gal)

    rng = np.random.default_rng(args.seed)
    rows = []
    for name in sorted(folds):
        means = folds[name]
        s, c, d = means.shape
        views = [_metric(_mm_np(means[i])) for i in range(s)]
        ks = list(range(1, s))
        cm_pool, cm_single, cm_shuf, loo_pool, loo_single = [], [], [], [], []
        for k in ks:
            pool = np.mean(views[:k], axis=0)
            cm_pool.append(_corr(pool, di))
            cm_single.append(_corr(views[0], di))
            # ---- M1-style control: destroy the concept correspondence, keep the geometry ----
            sh = []
            for _r in range(5):
                vv = [v[np.ix_(p, p)] for v in views[:k]
                      for p in [rng.permutation(c)]]
                sh.append(_corr(np.mean(vv, axis=0), di))
            cm_shuf.append(float(np.mean(sh)))
            # ---- within-EEG leave-one-view-out: score the pool against a HELD OUT subject ----
            held = views[k]
            loo_pool.append(_corr(pool, held))
            loo_single.append(_corr(views[0], held))
        rows.append({
            "fold": name,
            "k": ks,
            "crossmodal_pool": cm_pool,
            "crossmodal_single": cm_single,
            "crossmodal_shuffled": cm_shuf,
            "loo_pool": loo_pool,
            "loo_single": loo_single,
        })

    def stack(key):
        return np.asarray([r[key] for r in rows], dtype=float)     # (folds, K)

    P, S, H = stack("crossmodal_pool"), stack("crossmodal_single"), stack("crossmodal_shuffled")
    LP, LS = stack("loo_pool"), stack("loo_single")
    ks = rows[0]["k"]

    def pm(v, axis=0):
        return v.mean(axis=axis), (v.std(axis=axis, ddof=1) if v.shape[axis] > 1 else 0.0)

    summary = {"n_folds": len(rows), "device": _DEV, "k": ks, "table": []}
    print(f"\n[fusion] {'K':>3} {'xmodal pool':>18} {'xmodal single':>16} {'shuffled':>16} "
          f"{'LOO pool':>16} {'LOO single':>16}")
    for idx, k in enumerate(ks):
        prm, prs = pm(P[:, idx])
        psm, pss = pm(S[:, idx])
        hrm, _ = pm(H[:, idx])
        lrm, _ = pm(LP[:, idx])
        lsm, _ = pm(LS[:, idx])
        delta = P[:, idx] - S[:, idx]
        t = (delta.mean() / (delta.std(ddof=1) / np.sqrt(len(delta)))
             if delta.std(ddof=1) > 0 else float("nan"))
        summary["table"].append({
            "K": int(k),
            "crossmodal_pool_mean": float(prm), "crossmodal_pool_sd": float(prs),
            "crossmodal_single_mean": float(psm), "crossmodal_shuffled_mean": float(hrm),
            "loo_pool_mean": float(lrm), "loo_single_mean": float(lsm),
            "pool_minus_single": float(delta.mean()), "paired_t": float(t),
            "folds_positive": int((delta > 0).sum()),
        })
        print(f"    {k:>3} {prm:>10.4f}±{prs:<6.4f} {psm:>8.4f}       {hrm:>8.4f}"
              f"       {lrm:>8.4f}       {lsm:>8.4f}   Δ={delta.mean():+.4f} "
              f"t={t:+.1f} {int((delta>0).sum())}/{len(delta)}")

    # ---- the pre-registered decision ------------------------------------------------------
    kbest = int(np.argmax(P.mean(axis=0)))
    best_pool = float(P.mean(axis=0)[kbest])
    base_single = float(S.mean(axis=0)[0])
    best_k = ks[kbest]
    gap = best_pool - base_single
    shuf_gap = best_pool - float(H.mean(axis=0)[kbest])
    mono = bool(np.all(np.diff(P.mean(axis=0)) > -0.005))       # allow tiny numerical wobble
    loo_gain = float(LP.mean(axis=0)[kbest] - LS.mean(axis=0)[0])
    passed = bool(gap > 0.01 and shuf_gap > 0.01 and mono and best_k > 1)
    summary["verdict"] = {
        "baseline_single_view": base_single,
        "best_pooled": best_pool, "best_K": best_k,
        "pool_minus_single": gap,
        "pool_minus_shuffled": shuf_gap,
        "monotone_in_K": mono,
        "loo_pool_gain": loo_gain,
        "prediction1_passed": passed,
        "reading": ("PREDICTION 1 PASSES: pooling raises structure SNR, monotonically in K, and "
                    "separates from the correspondence-destroying control -> build L2-L5"
                    if passed else
                    "PREDICTION 1 FAILS: pooled structure does not beat one view with a "
                    "control-separated, K-monotone margin -> architecture is dead, do not build"),
    }
    p = os.path.join(OUT_DIR, "fusion_probe.json")
    with open(p, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\n[fusion] wrote {p}")
    print(f"[fusion] single-view {base_single:.4f} -> pooled {best_pool:.4f} at K={best_k} "
          f"(+{gap:.4f}); vs shuffled +{shuf_gap:.4f}; LOO gain {loo_gain:+.4f}")
    print(f"[fusion] VERDICT: {summary['verdict']['reading']}")


if __name__ == "__main__":
    main()
