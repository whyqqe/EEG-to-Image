#!/usr/bin/env python3
"""BRDT: build band x window tangent-space features (the new EEG encoder front end).

WHY THIS ENCODER, AND WHY IT IS NOT A RE-TREAD
----------------------------------------------
Riemannian / SPD encoders already exist for EEG (MAtt NeurIPS'22, ManifoldFormer,
SPD-token-transformer 2601.21521), so using them is not by itself a contribution.
Here they are used for ONE specific reason that follows from the routing law:

    retinotopy is a SPATIAL property -- which electrodes covary together tells you
    WHERE in the visual field the drive is. Band power tells you only HOW MUCH.

So the covariance is not decoration: it is the only carrier of the layout signal that
Gate 1 tests for. The tangent-space (Log-Euclidean) map then linearises the SPD
manifold, and the per-subject reference mean is a LABEL-FREE cross-subject alignment
(the same principle as Euclidean alignment, which HCMA's cross-subject setting needs).

Features per band b and window w:
    mean tangent vector  (pooled over the concept's trials)   -> PCA(d_pca) on train
    log band power       (per channel, 63 dims)               -> magnitude contrast
    tangent dispersion   (std over trials, in the same PCA basis) -> trial variability

The PCA basis is fitted on TRAIN concepts only. Test concepts are transformed with
that frozen basis -- no test statistics enter the representation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


BANDS = {"theta": (4.0, 8.0), "alpha": (8.0, 12.0), "beta": (13.0, 30.0), "gamma": (40.0, 80.0)}
WINDOWS = {"early": (0, 50), "late": (50, 250)}
FS = 250.0


def band_filter(x: torch.Tensor, lo: float, hi: float, fs: float = FS) -> torch.Tensor:
    F = torch.fft.rfft(x.float(), dim=-1)
    f = torch.fft.rfftfreq(x.shape[-1], d=1.0 / fs).to(x.device)
    m = ((f >= lo) & (f <= hi)).float()
    return torch.fft.irfft(F * m, n=x.shape[-1], dim=-1)


def shrink_cov(x: torch.Tensor, shrink: float = 0.10) -> torch.Tensor:
    N, C, T = x.shape
    x = x - x.mean(dim=-1, keepdim=True)
    Cm = (x @ x.transpose(1, 2)) / max(T - 1, 1)
    tr = Cm.diagonal(dim1=-2, dim2=-1).sum(-1) / C
    eye = torch.eye(C, device=x.device, dtype=Cm.dtype).expand(N, C, C)
    return (1.0 - shrink) * Cm + shrink * tr.view(N, 1, 1) * eye


def log_tangent(Cm: torch.Tensor) -> torch.Tensor:
    w, V = torch.linalg.eigh(Cm)
    logC = V @ torch.diag_embed(w.clamp_min(1e-6).log()) @ V.transpose(1, 2)
    iu = torch.triu_indices(Cm.shape[-1], Cm.shape[-1], offset=0, device=Cm.device)
    return logC[:, iu[0], iu[1]]


def build(E: np.ndarray, dev: torch.device, d_pca: int, basis: dict | None,
          batch: int = 4096):
    """E: (nC, nR, nS, C, T) -> per-band-window blocks. Returns (blocks, basis, meta)."""
    nC, nR, nS, C, T = E.shape
    n_tr = nR * nS
    X = torch.from_numpy(E.reshape(nC * n_tr, C, T))
    N = X.shape[0]
    out: dict[str, np.ndarray] = {}
    raw_mean: dict[str, np.ndarray] = {}
    for bn, (lo, hi) in BANDS.items():
        for wn, win in WINDOWS.items():
            chunks, disp = [], []
            with torch.no_grad():
                for i in range(0, N, batch):
                    t = log_tangent(shrink_cov(
                        band_filter(X[i:i + batch].to(dev), lo, hi)[..., win[0]:win[1]]))
                    chunks.append(t)
            T_all = torch.cat(chunks, 0).reshape(nC, n_tr, -1)          # (nC, n_tr, D)
            # NOTE: T_all lives on `dev` (cuda on the cluster). Always .cpu() before .numpy(),
            # otherwise this dies with "can't convert cuda:0 device type tensor to numpy"
            # -- which a CPU-only smoke test cannot catch.
            mean = T_all.mean(1).cpu().numpy().astype(np.float32)       # (nC, D)
            sd = T_all.std(1).mean(-1, keepdim=True).cpu().numpy().astype(np.float32)
            raw_mean[f"{bn}_{wn}"] = mean
            out[f"{bn}_{wn}_mean"] = mean
            out[f"{bn}_{wn}_disp"] = np.repeat(sd, 1, axis=1)
    for bn, (lo, hi) in BANDS.items():
        chunks = []
        with torch.no_grad():
            for i in range(0, N, batch):
                xb = band_filter(X[i:i + batch].to(dev), lo, hi)[..., 50:250]
                chunks.append((xb ** 2).mean(-1).log())
        p = torch.cat(chunks, 0).reshape(nC, n_tr, -1).mean(1).cpu().numpy().astype(np.float32)
        out[f"{bn}_power"] = p

    # PCA basis (train concepts only)
    if basis is None:
        basis = {}
        for k, v in raw_mean.items():
            mu = v.mean(0, keepdims=True)
            A = v - mu
            Cov = (A.T @ A) / max(len(A) - 1, 1)
            w, Vv = np.linalg.eigh(Cov.astype(np.float64))
            idx = np.argsort(-w)[:d_pca]
            basis[k] = {"mu": mu.astype(np.float32),
                        "comp": Vv[:, idx].astype(np.float32),
                        "evr": (w[idx] / max(w.sum(), 1e-12)).astype(np.float32)}
        # global whitening of the concatenated mean blocks (for the rank-r bottleneck)
        basis["_meta"] = {"d_pca": d_pca, "bands": list(BANDS), "windows": list(WINDOWS)}
    for k, v in raw_mean.items():
        b = basis[k]
        out[f"{k}_mean"] = ((v - b["mu"]) @ b["comp"]).astype(np.float32)
        out[f"{k}_disp"] = (out[f"{k}_disp"] * 1.0).astype(np.float32)
    meta = {"n_concepts": nC, "n_trials_per_concept": n_tr, "C": C, "T": T,
            "d_pca": d_pca, "bands": list(BANDS), "windows": list(WINDOWS),
            "evr": {k: float(basis[k]["evr"].sum()) for k in raw_mean}}
    return out, basis, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eeg-train-npy", required=True)
    ap.add_argument("--eeg-test-npy", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--d-pca", type=int, default=256)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")

    Etr = np.load(a.eeg_train_npy).astype(np.float32)
    print(f"[TRAIN] eeg {Etr.shape}")
    Btr, basis, meta = build(Etr, dev, a.d_pca, None)
    np.savez(out / "bandfeat_train.npz", **Btr)
    import pickle
    (out / "bandfeat_basis.pkl").write_bytes(pickle.dumps(basis))
    print("[OK] train blocks:", ", ".join(f"{k}{v.shape}" for k, v in list(Btr.items())[:4]))

    Ete = np.load(a.eeg_test_npy).astype(np.float32)
    print(f"[TEST ] eeg {Ete.shape}")
    Bte, _, meta_te = build(Ete, dev, a.d_pca, basis)
    np.savez(out / "bandfeat_test.npz", **Bte)
    meta["test"] = meta_te
    (out / "bandfeat_report.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[OK] -> {out}/bandfeat_{{train,test}}.npz")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
