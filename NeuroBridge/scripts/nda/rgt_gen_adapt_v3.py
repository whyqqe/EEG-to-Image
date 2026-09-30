#!/usr/bin/env python3
"""Clean generation fusion (no train-RAG leakage) + adaptive schedules.

Fusion search uses concept-held-out validation on (cfm, nda) only; mem is capped.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def l2(x: np.ndarray) -> np.ndarray:
    return (x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-8)).astype(np.float32)


def concept_holdout_idx(n: int, n_img: int, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    n_concepts = n // n_img
    rng = np.random.default_rng(seed)
    n_val = max(1, int(round(n_concepts * val_frac)))
    val_c = set(rng.choice(n_concepts, size=n_val, replace=False).tolist())
    tr, va = [], []
    for c in range(n_concepts):
        ids = list(range(c * n_img, (c + 1) * n_img))
        (va if c in val_c else tr).extend(ids)
    return np.asarray(tr), np.asarray(va)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfm-train", type=str, required=True)
    ap.add_argument("--cfm-test", type=str, required=True)
    ap.add_argument("--nda-train", type=str, required=True)
    ap.add_argument("--nda-test", type=str, required=True)
    ap.add_argument("--mem-train", type=str, required=True)
    ap.add_argument("--mem-test", type=str, required=True)
    ap.add_argument("--clip-train", type=str, required=True)
    ap.add_argument("--clip-test", type=str, required=True)
    ap.add_argument("--z-ret-test", type=str, required=True)
    ap.add_argument("--z-ret-train-gallery", type=str, required=True)
    ap.add_argument("--neighbor-idx", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--min-cfm", type=float, default=0.15)
    ap.add_argument("--max-mem", type=float, default=0.45)
    ap.add_argument("--base-strength", type=float, default=0.42)
    ap.add_argument("--base-ip-scale", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfm_tr = l2(np.load(args.cfm_train).astype(np.float32))
    nda_tr = l2(np.load(args.nda_train).astype(np.float32))
    mem_tr = l2(np.load(args.mem_train).astype(np.float32))
    gt_tr = l2(np.load(args.clip_train).astype(np.float32))
    n = min(len(cfm_tr), len(nda_tr), len(mem_tr), len(gt_tr))
    cfm_tr, nda_tr, mem_tr, gt_tr = cfm_tr[:n], nda_tr[:n], mem_tr[:n], gt_tr[:n]
    tr_idx, va_idx = concept_holdout_idx(n, 10, 0.1, args.seed)

    def score(w, a, b, c, g, idx):
        w = np.asarray(w, dtype=np.float32)
        w = w / max(float(w.sum()), 1e-8)
        mix = l2(w[0] * a[idx] + w[1] * b[idx] + w[2] * c[idx])
        return float((mix * g[idx]).sum(1).mean()), w

    grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    best = (-1.0, None)
    for wc in grid:
        for wn in grid:
            for wm in grid:
                if wc + wn + wm < 1e-6:
                    continue
                w_raw = np.asarray([wc, wn, wm], dtype=np.float32)
                w_norm = w_raw / w_raw.sum()
                # enforce constraints AFTER normalization (v3 bug: pre-norm leaked mem≈0.67)
                if w_norm[0] < args.min_cfm:
                    continue
                if w_norm[2] > args.max_mem:
                    continue
                sc, w = score(w_raw, cfm_tr, nda_tr, mem_tr, gt_tr, va_idx)
                if sc > best[0]:
                    best = (sc, w)
    if best[1] is None:
        # fallback safe mix
        best = (-1.0, np.asarray([0.35, 0.35, 0.30], dtype=np.float32))
        print("[WARN] no grid point satisfied constraints; using fallback weights")

    w = best[1]
    cfm_te = l2(np.load(args.cfm_test).astype(np.float32))
    nda_te = l2(np.load(args.nda_test).astype(np.float32))
    mem_te = l2(np.load(args.mem_test).astype(np.float32))
    gt_te = l2(np.load(args.clip_test).astype(np.float32))
    m = min(len(cfm_te), len(nda_te), len(mem_te), len(gt_te))
    fused = l2(w[0] * cfm_te[:m] + w[1] * nda_te[:m] + w[2] * mem_te[:m])
    # also nda-heavy and cfm-heavy blends for ablation
    nda_cfm = l2(0.45 * cfm_te[:m] + 0.55 * nda_te[:m])
    nda_mem = l2(0.55 * nda_te[:m] + 0.45 * mem_te[:m])  # close to NDA-SS winner
    np.save(out / "z_gen_adapt_test.npy", fused.astype(np.float32))
    np.save(out / "blend_nda_cfm.npy", nda_cfm.astype(np.float32))
    np.save(out / "blend_nda_mem.npy", nda_mem.astype(np.float32))

    z_ret = l2(np.load(args.z_ret_test).astype(np.float32))[:m]
    gal = l2(np.load(args.z_ret_train_gallery).astype(np.float32))
    neigh = np.load(args.neighbor_idx)
    if neigh.ndim == 1:
        neigh = neigh.reshape(-1, 1)
    conf = np.array([float(z_ret[i] @ gal[int(neigh[i, 0])]) for i in range(m)], dtype=np.float32)
    conf_n = (conf - conf.min()) / (conf.max() - conf.min() + 1e-8)
    # with text prompts we often want slightly higher strength (more IP/text freedom)
    strength = np.clip(args.base_strength + 0.12 * (1.0 - conf_n), 0.30, 0.58).astype(np.float32)
    ip_scale = np.clip(args.base_ip_scale + 0.20 * conf_n, 0.70, 1.15).astype(np.float32)
    np.save(out / "strength_adapt.npy", strength)
    np.save(out / "ip_scale_adapt.npy", ip_scale)

    report = {
        "val_cos": best[0],
        "weights": {"cfm": float(w[0]), "nda": float(w[1]), "mem": float(w[2])},
        "constraints": {"min_cfm": args.min_cfm, "max_mem": args.max_mem},
        "test_cos": {
            "cfm": float((cfm_te[:m] * gt_te[:m]).sum(1).mean()),
            "nda": float((nda_te[:m] * gt_te[:m]).sum(1).mean()),
            "mem": float((mem_te[:m] * gt_te[:m]).sum(1).mean()),
            "fused": float((fused * gt_te[:m]).sum(1).mean()),
            "nda_cfm": float((nda_cfm * gt_te[:m]).sum(1).mean()),
            "nda_mem": float((nda_mem * gt_te[:m]).sum(1).mean()),
        },
        "strength_mean": float(strength.mean()),
        "ip_scale_mean": float(ip_scale.mean()),
        "note": "concept-held-out val; mem capped; min_cfm enforced",
    }
    (out / "gen_adapt_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
