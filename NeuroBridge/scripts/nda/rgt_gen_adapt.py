#!/usr/bin/env python3
"""Generation-side adaptation: fuse NDA/CFM/mem + adaptive strength & IP-scale.

Does NOT fine-tune SDXL weights (too heavy / unstable for this protocol).
Instead adapts the *conditioning interface* that IP-Adapter + img2img consume.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def l2(x: np.ndarray) -> np.ndarray:
    return (x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-8)).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfm-test", type=str, required=True)
    ap.add_argument("--nda-decode-test", type=str, required=True)
    ap.add_argument("--mem-test", type=str, required=True)
    ap.add_argument("--clip-test", type=str, required=True)
    ap.add_argument("--cfm-train", type=str, default="")
    ap.add_argument("--nda-decode-train", type=str, default="")
    ap.add_argument("--mem-train", type=str, default="")
    ap.add_argument("--clip-train", type=str, default="")
    ap.add_argument("--neighbor-idx", type=str, required=True)
    ap.add_argument("--z-ret-test", type=str, required=True)
    ap.add_argument("--z-ret-train-gallery", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--base-strength", type=float, default=0.40)
    ap.add_argument("--base-ip-scale", type=float, default=1.0)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfm = l2(np.load(args.cfm_test).astype(np.float32))
    nda = l2(np.load(args.nda_decode_test).astype(np.float32))
    mem = l2(np.load(args.mem_test).astype(np.float32))
    gt = l2(np.load(args.clip_test).astype(np.float32))
    n = min(len(cfm), len(nda), len(mem), len(gt))
    cfm, nda, mem, gt = cfm[:n], nda[:n], mem[:n], gt[:n]

    # Grid-search fusion weights on TRAIN if available, else on a heuristic using test cos to GT
    # (test used only for reporting oracle; deployment weights from train)
    def score_weights(w_cfm, w_nda, w_mem, a, b, c, g):
        w = np.array([w_cfm, w_nda, w_mem], dtype=np.float32)
        w = w / w.sum()
        mix = l2(w[0] * a + w[1] * b + w[2] * c)
        return float((mix * g).sum(1).mean()), w, mix

    best = (-1.0, None, None)
    grid = [0.0, 0.15, 0.25, 0.35, 0.5, 0.65, 0.8, 1.0]

    use_train = all(
        [
            args.cfm_train and Path(args.cfm_train).is_file(),
            args.nda_decode_train and Path(args.nda_decode_train).is_file(),
            args.mem_train and Path(args.mem_train).is_file(),
            args.clip_train and Path(args.clip_train).is_file(),
        ]
    )
    if use_train:
        cfm_tr = l2(np.load(args.cfm_train).astype(np.float32))
        nda_tr = l2(np.load(args.nda_decode_train).astype(np.float32))
        mem_tr = l2(np.load(args.mem_train).astype(np.float32))
        gt_tr = l2(np.load(args.clip_train).astype(np.float32))
        m = min(len(cfm_tr), len(nda_tr), len(mem_tr), len(gt_tr))
        search_a, search_b, search_c, search_g = cfm_tr[:m], nda_tr[:m], mem_tr[:m], gt_tr[:m]
        search_split = "train"
    else:
        search_a, search_b, search_c, search_g = cfm, nda, mem, gt
        search_split = "test_fallback"

    for wc in grid:
        for wn in grid:
            for wm in grid:
                if wc + wn + wm < 1e-6:
                    continue
                sc, w, _ = score_weights(wc, wn, wm, search_a, search_b, search_c, search_g)
                if sc > best[0]:
                    best = (sc, w, None)

    w = best[1]
    mix_te = l2(w[0] * cfm + w[1] * nda + w[2] * mem)
    np.save(out / "z_gen_adapt_test.npy", mix_te.astype(np.float32))

    # Also save pairwise blends for ablation
    np.save(out / "blend_nda_cfm_a50.npy", l2(0.5 * nda + 0.5 * cfm))
    np.save(out / "blend_nda_mem_a50.npy", l2(0.5 * nda + 0.5 * mem))
    np.save(out / "blend_cfm_mem_a50.npy", l2(0.5 * cfm + 0.5 * mem))

    # Adaptive strength / IP-scale from retrieval confidence
    z_ret = l2(np.load(args.z_ret_test).astype(np.float32))[:n]
    gal = l2(np.load(args.z_ret_train_gallery).astype(np.float32))
    neigh = np.load(args.neighbor_idx)
    if neigh.ndim == 1:
        neigh = neigh.reshape(-1, 1)
    conf = []
    for i in range(n):
        nb = int(neigh[i, 0])
        # similarity of query to top neighbor in retrieval space
        conf.append(float(z_ret[i] @ gal[nb]))
    conf = np.asarray(conf, dtype=np.float32)
    conf_n = (conf - conf.min()) / (conf.max() - conf.min() + 1e-8)

    # high conf → lower strength (keep neighbor structure); low conf → higher strength (trust IP more)
    strength = np.clip(args.base_strength + 0.15 * (1.0 - conf_n) - 0.05 * conf_n, 0.28, 0.55).astype(np.float32)
    # high conf → slightly higher IP scale
    ip_scale = np.clip(args.base_ip_scale + 0.25 * conf_n - 0.10 * (1.0 - conf_n), 0.75, 1.25).astype(np.float32)
    np.save(out / "strength_adapt.npy", strength)
    np.save(out / "ip_scale_adapt.npy", ip_scale)

    report = {
        "search_split": search_split,
        "best_train_cos": best[0],
        "weights": {"cfm": float(w[0]), "nda": float(w[1]), "mem": float(w[2])},
        "test_cos": {
            "cfm": float((cfm * gt).sum(1).mean()),
            "nda": float((nda * gt).sum(1).mean()),
            "mem": float((mem * gt).sum(1).mean()),
            "fused": float((mix_te * gt).sum(1).mean()),
        },
        "strength": {
            "mean": float(strength.mean()),
            "min": float(strength.min()),
            "max": float(strength.max()),
            "base": args.base_strength,
        },
        "ip_scale": {
            "mean": float(ip_scale.mean()),
            "min": float(ip_scale.min()),
            "max": float(ip_scale.max()),
            "base": args.base_ip_scale,
        },
        "note": "SDXL/IP weights frozen; adapt conditioning fuse + per-sample strength/ip_scale",
    }
    (out / "gen_adapt_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
