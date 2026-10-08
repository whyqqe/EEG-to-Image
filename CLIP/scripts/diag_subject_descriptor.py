#!/usr/bin/env python
"""Pick the low-dimensional subject descriptor: how much of the whitener's signal survives?

`diag_subject_statistic.py` found the full EA whitener separates subjects 5.4x better than
anything else (0.1425 vs 0.0263), while the per-channel moments it replaces are
subject-agnostic (-0.0090). But the whitener is C^2 = 3969-dimensional, which is too wide to
be `z_s`. This ranks practical condensations of it against each other, and reports how much
of the separation each retains, so the descriptor is chosen on measured retention.

The base geometry is the same in all cases -- the support set's own spatial covariance -- so
this is a choice of READOUT, not of quantity.

  python scripts/diag_subject_statistic.py   # the base ranking this refines
  python scripts/diag_subject_descriptor.py  # this
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


def separation(v: np.ndarray, subs: list[int]) -> float:
    x = torch.from_numpy(np.asarray(v, dtype=np.float64)).float()
    if not np.all(np.isfinite(x.numpy())):
        return float("nan")
    x = F.normalize(x - x.mean(0, keepdim=True), dim=-1)
    cos = (x @ x.T).numpy()
    same, diff = [], []
    for i in range(len(subs)):
        for j in range(i + 1, len(subs)):
            (same if subs[i] == subs[j] else diff).append(cos[i, j])
    return float(np.mean(same) - np.mean(diff))


def whiten_and_cov(xc: np.ndarray, shrink: float = 0.05):
    """`xc` (n, C, T) -> (covariance, whitener), both (n, C, C), shrunk toward the identity."""
    n, c, t = xc.shape
    cov = np.einsum("nct,ndt->ncd", xc, xc) / t
    eye = np.eye(c)[None]
    trace = np.trace(cov, axis1=1, axis2=2)[:, None, None] / c
    cov_s = (1 - shrink) * cov + shrink * trace * eye          # Ledoit-Wolf-style targeting
    evals, evecs = np.linalg.eigh(cov_s)
    evals = np.clip(evals, 1e-8, None)
    whiten = np.einsum("nck,nk,ndk->ncd", evecs, evals ** -0.5, evecs)
    return cov_s, whiten, evals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(config.CONFIGS / "loso_sub08.yaml"))
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--mode", default="cov", choices=["cov", "raw", "centred"])
    ap.add_argument("--mvnn", default=None,
                    help="override the config's mvnn setting. The control that matters: "
                         "MVNN whitens with the subject's OWN noise covariance, so if the "
                         "eigenvalue spectrum's separation comes only from how well MVNN "
                         "whitened, 'off' will collapse it.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.mvnn is not None:
        cfg = dict(cfg, mvnn=args.mvnn)
    src = cfg.get("source_subjects") or [s for s in config.all_subjects()
                                         if s != int(cfg["target_subject"])]
    channels = (config.CHANNELS_OCCIPITO_PARIETAL
                if cfg.get("channel_set", "all63") == "occipital17" else None)
    data = things_eeg.load_loso(src, int(cfg["target_subject"]), channels,
                               mvnn=cfg.get("mvnn", "off"))
    arr = [np.asarray(data.tr_eeg[s]).reshape(-1, *np.asarray(data.tr_eeg[s]).shape[-2:])
           for s in range(data.n_subjects)]
    n_ch, n_t = arr[0].shape[-2:]

    rng = np.random.default_rng(args.seed)
    stacks, subs = [], []
    for s in range(data.n_subjects):
        for _ in range(args.draws):
            stacks.append(arr[s][rng.choice(len(arr[s]), size=args.k, replace=False)])
            subs.append(s)
    x = np.stack(stacks).reshape(len(stacks), n_ch, -1)         # (n, C, K*T)
    xc = x - x.mean(-1, keepdims=True)
    cov, whiten, evals = whiten_and_cov(xc)
    # repackage to a common (n, -1) where each descriptor is a readout of (cov, whiten)

    print(f"[data] mvnn={cfg.get('mvnn', 'off')} | {data.n_subjects} subj x {args.draws} "
          f"draws x K={args.k} | trial ({n_ch}, {n_t})")

    def spec(m):
        return np.concatenate([np.log(m), np.log(m)[:, ::-1]], axis=-1)

    cands: dict[str, np.ndarray] = {
        "log eigenvalue spectrum of C      (2C)": spec(evals),
        "log eigenvalue spectrum of W      (2C)": spec(1.0 / evals),
        "log diag(C)                       (C)": np.log(np.diagonal(cov, axis1=1, axis2=2)),
        "log trace of C                    (1)": np.log(np.trace(cov, axis1=1, axis2=2))[:, None],
        "upper-tri of log C               (C(C+1)/2)":
            np.stack([np.log(cov[i])[np.triu_indices(n_ch)] for i in range(len(cov))]),
        "EVAL baseline: full whitener W    (C^2)": whiten.reshape(len(cov), -1),
        "EVAL baseline: full log C         (C^2)": np.log(cov).reshape(len(cov), -1),
    }
    for rank in (8, 16, 32):
        v = np.stack([np.linalg.eigh(cov[i])[1][:, -rank:] for i in range(len(cov))])
        cands[f"top-{rank} eigenvectors of C        ({rank}C)"] = v.reshape(len(cov), -1)
        # sign-invariant readout: the projector onto the top-r subspace
        cands[f"top-{rank} subspace projector of C ({rank}C)"] = np.einsum(
            "nck,ndk->ncd", v, v).reshape(len(cov), -1)

    print(f"\n{'descriptor':<46} {'sep':>10} {'retained':>10}")
    print("-" * 70)
    base = separation(cands["EVAL baseline: full whitener W    (C^2)"], subs)
    rows = {}
    for name, v in cands.items():
        s = separation(v, subs)
        rows[name] = s
        keep = f"{s / base:>9.1%}" if base and np.isfinite(s) else "       n/a"
        print(f"{name:<46} {s:>+10.5f} {keep}")

    print(f"\n[reference] full whitener separation = {base:+.5f}; "
          f"per-channel moments = "
          f"{separation(np.concatenate([x.mean(-1), x.std(-1)], -1), subs):+.5f}")
    print("\nBest practical (<= 2C-dim) descriptor by separation:")
    narrow = {k: v for k, v in rows.items() if "(C^2)" not in k and np.isfinite(v)}
    if narrow:
        best = max(narrow.items(), key=lambda kv: abs(kv[1]))
        print(f"  {best[1]:+.5f}  {best[0]}")


if __name__ == "__main__":
    main()
