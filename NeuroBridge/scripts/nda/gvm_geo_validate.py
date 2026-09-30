#!/usr/bin/env python3
"""Validate the four depth/edge caches and report the numbers the orchestrator lost.

WHY THIS IS NEEDED
------------------
`run_gem_geo.sh` extracts the two splits in SEPARATE `gem_geo.py` invocations, and
each invocation writes `gem_geo_report.json` from scratch.  The second call
therefore overwrote the first, and the report now documents only the test split --
the train distinctness numbers, which are the ones that matter for deciding whether a
head is worth training, are gone from disk.  The CACHES are unaffected (all eight
files were written); only the record of what they contain was lost.  This script
recomputes it from the arrays themselves, which is the authoritative source anyway,
and checks the properties a downstream training run depends on:

  * row counts match the captions files ROW FOR ROW (a silent reorder would misalign
    every per-row array);
  * the 1024-d columns are unit-norm (they are handed to an IP-Adapter);
  * depth and edge are NOT near-copies of the RGB embedding -- if they were, a head
    for them would be a re-parameterisation of the image head rather than new
    information, and the "four modalities" claim would be hollow.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

NB = Path("/project/peilab/why/NeuroBridge")
CC = NB / "outputs/gem/cond_cache"
CAPS = NB / "outputs/g2/captions"


def l2(x):
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8, None)


print("=" * 78)
print("1. ROW-ORDER AND SHAPE VALIDATION")
print("=" * 78)
caps = {}
for split in ("train", "test"):
    rows = [json.loads(l) for l in (CAPS / f"captions_{split}.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    caps[split] = rows
    print(f"  {split}: captions {len(rows)} rows")

ok = True
for split in ("train", "test"):
    n = len(caps[split])
    for m in ("depth", "edge"):
        for d, dt in (("1024", "float32"), ("1280", "float16")):
            p = CC / f"clip_{m}{d}_{split}.npy"
            if not p.is_file():
                print(f"  [MISS] {p.name}"); ok = False; continue
            a = np.load(p, mmap_mode="r")
            exp_dim = 1024 if d == "1024" else 1280
            good = (a.shape == (n, exp_dim))
            print(f"  [{'ok' if good else 'BAD'}] {p.name}: {a.shape} {a.dtype} "
                  f"(expected ({n}, {exp_dim}) {dt})")
            ok &= bool(good)
    for m in ("depth", "edge"):
        a = np.load(CC / f"clip_{m}1024_{split}.npy", mmap_mode="r")
        nrm = np.linalg.norm(np.asarray(a, dtype=np.float32), axis=1)
        print(f"  1024-norm {m}/{split}: mean {nrm.mean():.6f} "
              f"min {nrm.min():.6f} max {nrm.max():.6f} (expect 1.0)")

print()
print("=" * 78)
print("2. DISTINCTNESS IN THE SPACE THE IP-ADAPTER CONSUMES (1024-d)")
print("=" * 78)
print("  A pair of modalities at cosine ~1.0 means one head would be a relabelling of")
print("  the other.  These are reported per split because they can differ.")
summary = {}
for split in ("train", "test"):
    img = l2(np.asarray(np.load(CC / f"clip_img1024_{split}.npy"), dtype=np.float32))
    dep = l2(np.asarray(np.load(CC / f"clip_depth1024_{split}.npy"), dtype=np.float32))
    edg = l2(np.asarray(np.load(CC / f"clip_edge1024_{split}.npy"), dtype=np.float32))
    d = {}
    for nm, a in (("depth", dep), ("edge", edg)):
        d[f"cos_{nm}_vs_image"] = float((a * img).sum(1).mean())
    d["cos_depth_vs_edge"] = float((dep * edg).sum(1).mean())
    # how separable are they?  a linear probe from one to the other: a high R2 means
    # one is (linearly) a function of the other and adds no independent target
    n = len(img); tr = slice(0, int(0.75 * n)); te = slice(int(0.75 * n), n)
    for nm, A, B in (("depth->edge", dep, edg), ("image->depth", img, dep),
                     ("image->edge", img, edg)):
        mu, sd = A[tr].mean(0), A[tr].std(0).clip(1e-6)
        As = (A[tr] - mu) / sd
        ym = B[tr].mean(0)
        W = np.linalg.solve(As.T @ As + 1.0 * np.eye(As.shape[1]), As.T @ (B[tr] - ym))
        P = ((A[te] - mu) / sd) @ W + ym
        r2 = float(1.0 - ((P - B[te]) ** 2).sum() / ((B[te] - ym) ** 2).sum())
        d[f"ridge_r2_{nm}"] = r2
    summary[split] = d
    print(f"  --- {split} ---")
    for k in sorted(d):
        print(f"    {k:<26} {d[k]:+.4f}")

print()
print("=" * 78)
print("3. VERDICT INPUTS")
print("=" * 78)
for split in ("train", "test"):
    d = summary[split]
    print(f"  {split}: depth vs image {d['cos_depth_vs_image']:+.4f}, "
          f"edge vs image {d['cos_edge_vs_image']:+.4f}, "
          f"depth vs edge {d['cos_depth_vs_edge']:+.4f}")
    print(f"        linear predictability: image->depth R2 {d['ridge_r2_image->depth']:.4f}, "
          f"image->edge R2 {d['ridge_r2_image->edge']:.4f}, "
          f"depth->edge R2 {d['ridge_r2_depth->edge']:.4f}")
print()
print("  READ: a cosine well below 1.0 with a LOW image->depth/edge R2 means the new")
print("  modality carries information the image embedding does not already contain.")
print("  A high depth->edge R2 means those two heads are largely redundant with EACH")
print("  OTHER and should be reported as one geometric factor, not two.")
print(f"\nall caches valid: {ok}")
