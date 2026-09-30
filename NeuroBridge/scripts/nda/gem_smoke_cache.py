#!/usr/bin/env python3
"""Build a SMALL, shape-correct stand-in for the train image-feature cache.

Smoke tests only.  The placeholder is a resampled copy of the test features plus
noise, so it is deliberately NOT the real target distribution: it exists so the
training script can be exercised end to end for shapes, dtypes, the ordering of
the teacher-forced decode, and the preflight gradient check.  Any run that uses
this cache must be limited to the same number of rows (--limit-train) and must
never have its metrics reported.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

src = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/gem/cond_cache")
dst = Path(sys.argv[2] if len(sys.argv) > 2 else "outputs/gem/smoke_cache")
n = int(sys.argv[3]) if len(sys.argv) > 3 else 640

dst.mkdir(parents=True, exist_ok=True)
f = np.load(src / "clip_img1024_test.npy").astype(np.float32)
rng = np.random.default_rng(0)
big = np.repeat(f[: max(1, n // 8)], 8, axis=0)[:n] + rng.normal(0, 0.02, (n, f.shape[1]))
big /= np.clip(np.linalg.norm(big, axis=1, keepdims=True), 1e-8, None)
np.save(dst / "clip_img1024_train.npy", big.astype(np.float32))
np.save(dst / "clip_img1024_test.npy", f.astype(np.float32))
print(f"[smoke] train-stand-in {big.shape} -> {dst/'clip_img1024_train.npy'}")
print(f"[smoke] test copied {f.shape}")
print("[smoke] NOT a real target: smoke runs must use --limit-train and must not be "
      "reported")
