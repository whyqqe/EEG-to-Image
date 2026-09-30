"""Audit the per-query informativeness of a conditioning vector bank.

Why this exists
---------------
`uck_train.memory()` returns a top-k *soft mixture* of gallery concept means. A
flat similarity profile then collapses the mixture toward the bank mean, so the
200 rows of `ip_mem_test.npy` can be nearly the same vector -- i.e. the IP
condition carries almost no per-query information, and the generator is being
asked to render "something generic". Cosine-to-bank-mean cannot see this; the
row-to-row spread can.

This script measures, for a (N, D) condition bank:

  row2mean   mean_i cos(z_i, mean(z))     -> 1.0 means every row is identical
  offdiag    mean_{i<j} cos(z_i, z_j)     -> row-to-row redundancy
  erank      exp(entropy(sigma^2))        -> effective # of independent directions
  top1       N-way argmax accuracy vs gallery (row i <-> concept i)
  top1_shuf  same after a circular row shift of the gallery = chance control

A bank that is *informative* must sit clearly below `offdiag` of a constant
vector (1.0) and its `top1` must beat `top1_shuf`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def l2n(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def rows_to_mean_cos(z: np.ndarray) -> float:
    zn = l2n(z)
    mu = l2n(zn.mean(0, keepdims=True))
    return float((zn * mu).sum(1).mean())


def offdiag_cos(z: np.ndarray) -> float:
    s = l2n(z) @ l2n(z).T
    n = s.shape[0]
    off = s[~np.eye(n, dtype=bool)]
    return float(off.mean())


def effective_rank(z: np.ndarray) -> float:
    z = np.asarray(z, dtype=np.float64)
    z = z - z.mean(0, keepdims=True)
    sv = np.linalg.svd(z, compute_uv=False)
    p = sv ** 2
    s = p.sum()
    if s <= 0:
        return 0.0
    p = p / s
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def top1(z: np.ndarray, gallery: np.ndarray) -> float:
    sim = l2n(z) @ l2n(gallery).T
    return float((sim.argmax(1) == np.arange(len(z))).mean())


def row_norm_spread(z: np.ndarray) -> float:
    n = np.linalg.norm(np.asarray(z, dtype=np.float64), axis=1)
    return float(n.std() / max(n.mean(), 1e-12))


def audit(name: str, z: np.ndarray, gallery: np.ndarray | None) -> dict:
    z = np.asarray(z, dtype=np.float32)
    r = {
        "name": name,
        "shape": list(z.shape),
        "row2mean": round(rows_to_mean_cos(z), 4),
        "offdiag": round(offdiag_cos(z), 4),
        "erank": round(effective_rank(z), 1),
        "norm_cv": round(row_norm_spread(z), 4),
        "std_overall": float(z.std()),
    }
    if gallery is not None and len(gallery) == len(z):
        r["top1"] = round(top1(z, gallery), 4)
        # negative control: break the row<->concept correspondence
        r["top1_shuf"] = round(top1(z, np.roll(gallery, 1, axis=0)), 4)
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery", type=str,
                    default="/project/peilab/why/NeuroBridge/outputs/g2/targets/sem_image_test.npy",
                    help="(200, D) GT concept-mean bank for the test split; row i == concept i")
    ap.add_argument("--bank", action="append", default=[], help="name=path, repeatable")
    ap.add_argument("--json-out", type=str, default="")
    args = ap.parse_args()

    gpath = Path(args.gallery)
    gal = None
    if gpath.is_file():
        gal = np.load(gpath).astype(np.float32)
        if gal.shape[0] != 200:
            print(f"[warn] gallery rows={gal.shape[0]} != 200; top1 skipped")
            gal = None
    else:
        print(f"[warn] gallery missing: {gpath}")

    rows = []
    if gal is not None:
        rows.append(audit("GT_concept_mean(ceiling)", gal, gal))

    for spec in args.bank:
        name, _, path = spec.partition("=")
        p = Path(path)
        if not p.is_file():
            rows.append({"name": name, "error": f"missing {p}"})
            continue
        z = np.load(p).astype(np.float32)
        rows.append(audit(name, z, gal))

    # constant-vector reference: what a fully collapsed condition looks like
    if gal is not None:
        const = np.repeat(gal.mean(0, keepdims=True), len(gal), axis=0)
        rows.append(audit("__constant(floor)", const, gal))

    w = max(len(r["name"]) for r in rows)
    hdr = f"{'bank':<{w}}  {'row2mean':>8} {'offdiag':>8} {'erank':>7} {'normcv':>7} {'top1':>7} {'shuf':>6}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if "error" in r:
            print(f"{r['name']:<{w}}  {r['error']}")
            continue
        print(f"{r['name']:<{w}}  {r['row2mean']:>8.4f} {r['offdiag']:>8.4f} "
              f"{r['erank']:>7.1f} {r['norm_cv']:>7.4f} "
              f"{r.get('top1', float('nan')):>7.4f} {r.get('top1_shuf', float('nan')):>6.4f}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\n[json] {args.json_out}")


if __name__ == "__main__":
    main()
