#!/usr/bin/env python3
"""Build DSDA embeddings: PoE fusion, confidence blend, adaptive img2img strengths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)


def blend(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    return l2(alpha * a + (1.0 - alpha) * b)


def poe(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Product-of-experts in embedding space (normalized element-wise product)."""
    return l2(a * b)


def conf_blend(mem: np.ndarray, proj: np.ndarray, alpha_base: float, k: float) -> np.ndarray:
    """More projection weight when mem and proj disagree (semantic gap)."""
    cos = np.sum(l2(mem) * l2(proj), axis=1, keepdims=True)
    # cos in [-1,1]; low cos -> higher proj weight
    alpha = alpha_base + (1.0 - alpha_base) * np.clip((1.0 - cos) / 2.0, 0.0, 1.0) * k
    alpha = np.clip(alpha, 0.15, 0.85)
    out = alpha * mem + (1.0 - alpha) * proj
    return l2(out.astype(np.float32)), alpha.squeeze(-1).astype(np.float32)


def adaptive_strength(mem: np.ndarray, proj: np.ndarray, s_hi: float, s_lo: float) -> np.ndarray:
    """Low mem-proj agreement -> lower img2img strength (more semantic freedom)."""
    cos = np.sum(l2(mem) * l2(proj), axis=1)
    cos = np.clip(cos, 0.0, 1.0)
    return (s_lo + (s_hi - s_lo) * cos).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mem-npy", type=str, required=True)
    ap.add_argument("--proj-npy", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--alpha-base", type=float, default=0.5)
    ap.add_argument("--conf-k", type=float, default=1.0)
    ap.add_argument("--s-hi", type=float, default=0.45)
    ap.add_argument("--s-lo", type=float, default=0.32)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    mem = l2(np.load(args.mem_npy).astype(np.float32))
    proj = l2(np.load(args.proj_npy).astype(np.float32))
    n = min(len(mem), len(proj))
    mem, proj = mem[:n], proj[:n]

    fixed_blend = blend(mem, proj, args.alpha_base)
    poe_emb = poe(mem, proj)
    conf_emb, alpha_per = conf_blend(mem, proj, args.alpha_base, args.conf_k)
    strength = adaptive_strength(mem, proj, args.s_hi, args.s_lo)

    np.save(out / "dsda_fixed_blend_a50.npy", fixed_blend)
    np.save(out / "dsda_poe_mem_proj.npy", poe_emb)
    np.save(out / "dsda_conf_blend.npy", conf_emb)
    np.save(out / "dsda_adaptive_strength.npy", strength)
    np.save(out / "dsda_conf_alpha.npy", alpha_per)

    report = {
        "n": n,
        "mean_cos_mem_proj": float(np.mean(np.sum(mem * proj, axis=1))),
        "strength_mean": float(strength.mean()),
        "strength_std": float(strength.std()),
        "alpha_mean": float(alpha_per.mean()),
        "outputs": {
            "fixed_blend": str(out / "dsda_fixed_blend_a50.npy"),
            "poe": str(out / "dsda_poe_mem_proj.npy"),
            "conf_blend": str(out / "dsda_conf_blend.npy"),
            "strength": str(out / "dsda_adaptive_strength.npy"),
        },
    }
    (out / "dsda_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
