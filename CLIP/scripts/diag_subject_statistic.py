#!/usr/bin/env python
"""Which subject statistic actually survives the preprocessing?

`diag_anchor_swamp.py` found that the per-channel moments channel (`[mean_c(S); std_c(S)]`,
which `SupportSetEncoder.moments` computes) is subject-agnostic: its cross-subject
separation was +0.000011, i.e. zero. The obvious explanation is the preprocessing: every
subject's data is z-scored with its OWN statistics, so each channel ends at mean 0 and
std 1 for every subject -- the two numbers the anchor is built from are removed by
construction.

If that is right, the fix is not to abandon moment matching (SATTC measures
subject-adaptive whitening as the largest cross-subject lever, Latent Alignment conditions
on per-subject statistics with zero subject-specific parameters, EA alone gives +14.8
points) but to use the statistic that SURVIVES per-channel standardisation. Centring and
scaling each channel individually preserves the CORRELATION STRUCTURE between channels,
so the spatial covariance `C = XX^T / T` -- what EA, MVNN, and SATTC's SAW all use --
should still separate subjects.

This measures the separation of several candidate statistics directly, so the choice is
made on evidence rather than on the fact that they are all called "statistics".

  python scripts/diag_subject_statistic.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from samclip import config  # noqa: E402
from samclip.data import things_eeg  # noqa: E402
from samclip.utils import load_config  # noqa: E402


def separation(v: np.ndarray, subs: list[int]) -> tuple[float, float, float]:
    """Cross-subject separation of a per-draw statistic.

    Returns (mean cos same-subject, mean cos different-subject, difference). The
    difference is the quantity that matters: 0 means subject-agnostic.
    """
    x = torch.from_numpy(v).float()
    x = F.normalize(x - x.mean(0, keepdim=True), dim=-1)   # centre, then compare directions
    cos = (x @ x.T).numpy()
    same, diff = [], []
    for i in range(len(subs)):
        for j in range(i + 1, len(subs)):
            (same if subs[i] == subs[j] else diff).append(cos[i, j])
    return float(np.mean(same)), float(np.mean(diff)), float(np.mean(same) - np.mean(diff))


def stats_of(x: np.ndarray) -> dict[str, np.ndarray]:
    """`x`: (n, C, T) -> candidate per-draw subject statistics, all C- or C^2-dimensional."""
    n, c, t = x.shape
    xc = x - x.mean(-1, keepdims=True)                     # per-channel centring
    scale = xc.std(-1) + 1e-6                              # (n, C)

    per_ch = np.concatenate([x.mean(-1), x.std(-1)], axis=-1)          # 2C  (what .moments gives)
    per_ch_centred = np.concatenate([xc.mean(-1), xc.std(-1)], axis=-1)

    cov = np.einsum("nct,ndt->ncd", xc, xc) / t                        # C x C spatial covariance
    cov_corr = cov / (scale[:, :, None] * scale[:, None, :])           # correlation, scale-free
    log_cov = np.log(cov + 1e-6 * np.eye(c)[None])

    # EA / SAW whitener: the matrix that whitens the subject's own covariance.
    # Symmetric inverse square root, via eigendecomposition.
    evals, evecs = np.linalg.eigh(cov + 1e-6 * np.eye(c)[None])
    evals = np.clip(evals, 1e-6, None)
    whiten = np.einsum("nck,nk,ndk->ncd", evecs, evals ** -0.5, evecs)

    return {
        "per-channel [mean_c; std_c]  (2C=126)": per_ch,
        "per-channel on centred x     (2C=126)": per_ch_centred,
        "log spatial covariance       (C^2=3969)": log_cov.reshape(n, -1),
        "spatial covariance           (C^2=3969)": cov.reshape(n, -1),
        "spatial correlation          (C^2=3969)": cov_corr.reshape(n, -1),
        "EA whitener                   (C^2=3969)": whiten.reshape(n, -1),
        "flattened raw trial           (C*T=15750)": x.reshape(n, -1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(config.CONFIGS / "loso_sub08.yaml"))
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    src = cfg.get("source_subjects") or [s for s in config.all_subjects()
                                         if s != int(cfg["target_subject"])]
    channels = (config.CHANNELS_OCCIPITO_PARIETAL
                if cfg.get("channel_set", "all63") == "occipital17" else None)
    data = things_eeg.load_loso(src, int(cfg["target_subject"]), channels,
                               mvnn=cfg.get("mvnn", "off"))
    arr = [np.asarray(data.tr_eeg[s]).reshape(-1, *np.asarray(data.tr_eeg[s]).shape[-2:])
           for s in range(data.n_subjects)]
    n_ch, n_t = arr[0].shape[-2:]

    print(f"[data] mvnn={cfg.get('mvnn', 'off')} | {data.n_subjects} subjects | "
          f"trial ({n_ch}, {n_t})")
    print(f"[data] per-channel mean/std of the pooled data: "
          f"mean {np.mean([a.mean(-1).mean() for a in arr]):+.4f}, "
          f"std {np.mean([a.std(-1).mean() for a in arr]):.4f} "
          f"(a per-subject z-score forces these to 0 and 1 for EVERY subject)")

    rng = np.random.default_rng(args.seed)
    stacks, subs = [], []
    for s in range(data.n_subjects):
        for _ in range(args.draws):
            stacks.append(arr[s][rng.choice(len(arr[s]), size=args.k, replace=False)])
            subs.append(s)
    x = np.stack(stacks).reshape(len(stacks), n_ch, -1)     # (n, C, K*T)

    print(f"\n{'statistic':<42} {'same':>8} {'diff':>8} {'sep':>10}")
    print("-" * 72)
    rows = {}
    for name, v in stats_of(x).items():
        same, diff, sep = separation(v, subs)
        rows[name] = sep
        flag = "  <-- subject-agnostic" if abs(sep) < 0.02 else ""
        print(f"{name:<42} {same:>8.4f} {diff:>8.4f} {sep:>+10.5f}{flag}")

    print("\nRanked by separation:")
    for name, sep in sorted(rows.items(), key=lambda kv: -abs(kv[1])):
        print(f"  {abs(sep):.5f}  {name}")


if __name__ == "__main__":
    main()
