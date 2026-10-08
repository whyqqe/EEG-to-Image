"""Test score-level fusion rules on the BANKED score matrices. No training, no GPU.

Motivation from the literature:
  * SATTC (CVPR 2026) fuses a GEOMETRIC expert (adaptive whitening + adaptive CSLS) with a
    STRUCTURAL expert (mutual NN + bidirectional top-k ranks + class popularity) via
    Product-of-Experts, on the raw similarity matrix of frozen encoders.
  * CORTIVA (2026) argues for fusing at the CANDIDATE-SCORE level rather than consolidating
    embeddings, because early consolidation "imposes one similarity geometry on every
    candidate order and removes encoder-specific disagreements from the final ranking".

Our deployed ladder currently fuses T1 and T2 with a normalised SUM, and that fusion is
HARMFUL: T2 alone 53.80 vs the fused row 52.72. So the fusion rule is a live, untested lever
and it is the cheapest one -- it needs no training at all.
"""
import glob
import sys
from pathlib import Path

import numpy as np

T1 = "row::+ CSLS + recovery"
T2 = "row::+ T2 reps"
D = sys.argv[1] if len(sys.argv) > 1 else "outputs/scores/v8scr_a0.75_t0.03"


def top1(m):
    return float(np.mean(m.argmax(1) == np.arange(m.shape[0])) * 100)


def zs(m):
    """per-row z-score, the standard way to make two score sources comparable in scale"""
    return (m - m.mean(1, keepdims=True)) / m.std(1, keepdims=True).clip(1e-9)


def softmax(m, t=1.0):
    e = np.exp((m - m.max(1, keepdims=True)) / t)
    return e / e.sum(1, keepdims=True)


def csls(m, k=10):
    """hubness correction ON THE FUSION, mirroring `calibration.csls_scores`"""
    idx1 = np.argsort(-m, axis=1)[:, :k]                 # each row's top-k
    fwd = np.take_along_axis(m, idx1, axis=1).mean(1, keepdims=True)
    idx0 = np.argsort(-m, axis=0)[:k, :]                 # each column's top-k
    bwd = np.take_along_axis(m, idx0, axis=0).mean(0, keepdims=True)
    return 2 * m - fwd - bwd


def structural_expert(m, k=10):
    """SATTC-style structural expert: mutual-NN + bidirectional rank agreement."""
    n = m.shape[0]
    fwd = np.argsort(-m, axis=1)[:, :k]          # EEG i -> its top-k images
    bwd = np.argsort(-m, axis=0)[:k, :]          # image j -> its top-k EEG
    mutual = np.zeros_like(m)
    for i in range(n):
        row = np.zeros(n)
        row[fwd[i]] += 1.0
        mutual[i] += row
    for j in range(n):
        mutual[bwd[:, j], j] += 1.0
    return mutual


FOLDS = sorted({Path(p).stem.split("_seed")[0] for p in glob.glob(D + "/sub*_seed*.npz")})
RULES = {}


def accum(name, v):
    RULES.setdefault(name, []).append(v)


hdr = None
for fold in FOLDS:
    for p in sorted(glob.glob(f"{D}/{fold}_seed*.npz")):
        z = np.load(p)
        t1, t2 = z[T1], z[T2]
        cands = {
            "T1 alone": t1,
            "T2 alone (current best)": t2,
            "sum (current deployed)": zs(t1) + zs(t2),
            "PoE tau=1": np.log(softmax(t1, 1.0) * softmax(t2, 1.0) + 1e-12),
            "PoE z-logit tau=1": np.log(softmax(zs(t1), 1.0) * softmax(zs(t2), 1.0) + 1e-12),
            "PoE + CSLS": csls(np.log(softmax(zs(t1), 1.0) * softmax(zs(t2), 1.0) + 1e-12)),
            "T2 + structural expert": zs(t2) + 0.5 * zs(structural_expert(zs(t2))),
            "PoE(T2, struct)": np.log(softmax(zs(t2), 1.0)
                                      * softmax(zs(structural_expert(zs(t2))), 1.0) + 1e-12),
            "T2 + CSLS": csls(zs(t2)),
        }
        for k, v in cands.items():
            accum(k, v)


print(f"dir = {D}")
print("%-26s %8s %8s" % ("fusion rule", "Top-1", "n"))
res = {k: np.mean([top1(m) for m in v]) for k, v in RULES.items()}
for k in sorted(res, key=lambda x: -res[x]):
    print("%-26s %8.2f %8d" % (k, res[k], len(RULES[k])))
print()
print("SCORE = 53.23")
