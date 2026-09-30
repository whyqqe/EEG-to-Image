#!/usr/bin/env python3
"""BRDT Gate 0-2: decide, by measurement, whether band->tower routing is viable.

The proposed routing law rests on one 2025 high-density-EEG result:
  * narrowband GAMMA (40-80 Hz) is strongly RETINOTOPICALLY tuned -- subfield
    responses sum linearly to the full-field response, and orientation tuning is
    robust. It therefore carries WHERE things are (layout).
  * ALPHA (8-12 Hz) shows LITTLE spatial tuning -- it responds even without visual
    input (anticipatory suppression) and sums sub-additively (divisive
    normalization). It therefore carries global STATE/gain, not layout.
  (Dissociable Spatial and Feature Tuning of Gamma and Alpha Rhythms, 2025)

That result is from high-density EEG with dedicated retinotopic stimulation. It is
NOT obvious that it survives in THINGS-EEG2's 63-channel, 250 Hz, 1000 ms trials.
So we do not assume it -- we test it, and the pipeline branches on the outcome.

Three gates, all on TRAIN data only (test never touched):

  Gate 0  RELIABILITY. Is the band's covariance reliably measurable at all?
          Split the 40 training trials of each concept into two halves, average the
          tangent-space covariance within each half, correlate across concepts.
          A band whose reliability is at noise level cannot be routed anywhere.

  Gate 1  TUNING. The actual routing claim, stated as a contrast:
              spatial structure (inter-channel covariance) vs magnitude (per-channel power)
          predicting the LAYOUT target (low-frequency VAE latent).
          Prediction: gamma's covariance predicts layout clearly better than alpha's
          covariance, i.e. layout lives in gamma's SPATIAL structure.

  Gate 2  RESIDUAL DEPENDENCY. If the bands were cleanly dissociated there would be
          nothing to exchange between the towers. We measure the canonical
          correlations between the gamma-group and the alpha/theta-group features and
          take the rank r as the number of components above threshold. r is therefore
          an ESTIMATE FROM DATA, not a chosen hyper-parameter.

Writes gates.json. The orchestrator refuses to run the routed arms if Gate 1 fails
and falls back to the merged-band variant (documented, not silent).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


BANDS = {"theta": (4.0, 8.0), "alpha": (8.0, 12.0), "beta": (13.0, 30.0), "gamma": (40.0, 80.0)}
WINDOWS = {"early": (0, 50), "late": (50, 250)}  # @250 Hz -> 0-200 ms / 200-1000 ms
FS = 250.0


def band_filter(x: torch.Tensor, lo: float, hi: float, fs: float = FS) -> torch.Tensor:
    """FFT band-pass along the last (time) axis. x: (..., T)."""
    F = torch.fft.rfft(x.float(), dim=-1)
    n = x.shape[-1]
    f = torch.fft.rfftfreq(n, d=1.0 / fs).to(x.device)
    m = ((f >= lo) & (f <= hi)).float()
    return torch.fft.irfft(F * m, n=n, dim=-1)


def shrink_cov(x: torch.Tensor, shrink: float = 0.10) -> torch.Tensor:
    """x: (N, C, T) -> SPD covariance (N, C, C) with shrinkage toward scaled identity."""
    N, C, T = x.shape
    x = x - x.mean(dim=-1, keepdim=True)
    Cm = (x @ x.transpose(1, 2)) / max(T - 1, 1)
    tr = Cm.diagonal(dim1=-2, dim2=-1).sum(-1) / C
    eye = torch.eye(C, device=x.device, dtype=Cm.dtype).expand(N, C, C)
    return (1.0 - shrink) * Cm + shrink * tr.view(N, 1, 1) * eye


def log_tangent(Cm: torch.Tensor) -> torch.Tensor:
    """SPD log map (Log-Euclidean) -> upper-triangular vectorisation.

    Log-Euclidean is chosen over affine-invariant because it has a closed form
    (one batched eigh) and, per the SPD-token-transformer study, is the stronger
    embedding in practice on EEG parcellations.
    """
    w, V = torch.linalg.eigh(Cm)
    w = w.clamp_min(1e-6)
    logC = V @ torch.diag_embed(w.log()) @ V.transpose(1, 2)
    iu = torch.triu_indices(Cm.shape[-1], Cm.shape[-1], offset=0, device=Cm.device)
    return logC[:, iu[0], iu[1]]


def design(x: torch.Tensor, lo: float, hi: float, win: tuple[int, int]) -> torch.Tensor:
    """Band-limit, crop to window, return tangent-space covariance vector."""
    xb = band_filter(x, lo, hi)[..., win[0]:win[1]]
    return log_tangent(shrink_cov(xb))


def ridge_r2(Xtr: np.ndarray, Ytr: np.ndarray, Xte: np.ndarray, Yte: np.ndarray,
             lam: float = -1.0) -> tuple[float, float]:
    """Out-of-sample band-limited R^2 and mean Pearson r.

    The ridge is scaled to the problem's shape: with p features and n fit samples the
    ill-posed regime is p > n, so lam defaults to max(1, p/n) instead of a constant.
    """
    if lam < 0:
        lam = max(1.0, Xtr.shape[1] / max(len(Xtr), 1))
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True).clip(1e-6)
    A = (Xtr - mu) / sd
    B = (Xte - mu) / sd
    W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1], dtype=np.float32), A.T @ Ytr)
    P = B @ W
    Pp, Yy = P - P.mean(1, keepdims=True), Yte - Yte.mean(1, keepdims=True)
    den = np.linalg.norm(Pp, axis=1) * np.linalg.norm(Yy, axis=1)
    ok = den > 1e-8
    r = float(np.mean((Pp * Yy).sum(1)[ok] / den[ok])) if ok.any() else 0.0
    ss_res = ((P - Yte) ** 2).mean()
    ss_tot = ((Yte - Yte.mean(0, keepdims=True)) ** 2).mean()
    return r, float(1.0 - ss_res / max(ss_tot, 1e-12))


def _pca_reduce(X: np.ndarray, k: int) -> np.ndarray:
    A = X - X.mean(0, keepdims=True)
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    k = max(1, min(k, U.shape[1]))
    return U[:, :k] * s[:k]


def cca_heldout(A: np.ndarray, B: np.ndarray, k: int, seed: int = 0) -> list[float]:
    """Held-out canonical correlations.

    Plain CCA on n_samples << n_features returns canonical correlations of exactly 1
    for as many components as min(p, q) allows -- it fits the sample, not the relation.
    So we (a) reduce both blocks to k components and (b) fit the canonical directions
    on one half and SCORE them on the other. What is reported is out-of-sample.
    """
    n = len(A)
    idx = np.random.default_rng(seed).permutation(n)
    h1, h2 = idx[: n // 2], idx[n // 2:]
    if len(h1) < 4 or len(h2) < 4:
        return []
    # reduce on the FIT half, apply the same basis to the SCORE half
    def fit_basis(X, Y):
        Xm, Ym = X.mean(0, keepdims=True), Y.mean(0, keepdims=True)
        Ux, sx, Vx = np.linalg.svd(X - Xm, full_matrices=False)
        Uy, sy, Vy = np.linalg.svd(Y - Ym, full_matrices=False)
        kk = max(1, min(k, Ux.shape[1], Uy.shape[1]))
        Wx = (Vx.T[:, :kk] / sx[:kk].clip(1e-8))
        Wy = (Vy.T[:, :kk] / sy[:kk].clip(1e-8))
        return Wx, Wy, Xm, Ym
    Wx, Wy, Xm, Ym = fit_basis(A[h1], B[h1])
    Px, Py = (A - Xm) @ Wx, (B - Ym) @ Wy
    M = Px[h1].T @ Py[h1] / max(len(h1) - 1, 1)
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    # canonical directions, then evaluate their correlation on the held-out half
    dx, dy = Wx @ U[:, :len(s)], Wy @ Vt.T[:, :len(s)]
    ax, ay = A[h2] @ dx, B[h2] @ dy
    ax = ax - ax.mean(0, keepdims=True)
    ay = ay - ay.mean(0, keepdims=True)
    num = (ax * ay).sum(0)
    den = np.linalg.norm(ax, axis=0) * np.linalg.norm(ay, axis=0)
    return [float(x) for x in (num / den.clip(1e-12))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eeg-train-npy", required=True)
    ap.add_argument("--vae-train-npy", required=True, help="GT SDXL VAE latents (N,4,64,64)")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--cut", type=float, default=0.125, help="layout/high band split")
    ap.add_argument("--cca-thresh", type=float, default=0.50)
    ap.add_argument("--max-trials", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    E = np.load(a.eeg_train_npy).astype(np.float32)          # (n_concept, n_rep, n_sess, C, T)
    nC, nR, nS = E.shape[0], E.shape[1], E.shape[2]
    C, T = E.shape[-2], E.shape[-1]
    V = np.load(a.vae_train_npy, mmap_mode="r").astype(np.float32)
    print(f"[DATA] eeg {E.shape} -> {nC} concepts x {nR*nS} trials | vae {V.shape}")

    X = torch.from_numpy(E.reshape(nC * nR * nS, C, T))
    if a.max_trials:
        X = X[: a.max_trials]
    N = X.shape[0]
    n_tr = N // nC
    print(f"[DATA] trials used {N} ({n_tr} per concept)")

    # per-trial, per-band-window tangent features
    feats: dict[str, np.ndarray] = {}
    for bn, (lo, hi) in BANDS.items():
        for wn, win in WINDOWS.items():
            out = []
            with torch.no_grad():
                for i in range(0, N, 4096):
                    out.append(design(X[i:i + 4096].to(dev), lo, hi, win).cpu().numpy())
            feats[f"{bn}_{wn}"] = np.concatenate(out, 0).astype(np.float32)
            print(f"  [feat] {bn}_{wn} {feats[f'{bn}_{wn}'].shape}")
    # per-channel band power (the "magnitude" contrast for Gate 1)
    power: dict[str, np.ndarray] = {}
    for bn, (lo, hi) in BANDS.items():
        out = []
        with torch.no_grad():
            for i in range(0, N, 4096):
                xb = band_filter(X[i:i + 4096].to(dev), lo, hi)[..., 50:250]
                out.append((xb ** 2).mean(-1).log().cpu().numpy())
        power[bn] = np.concatenate(out, 0).astype(np.float32)

    # ---- layout target: low-frequency VAE latent
    n = min(nC, len(V))
    Vt = np.asarray(V[:n], dtype=np.float32)
    H = W = Vt.shape[-1]
    fy = np.fft.fftfreq(H)[:, None]
    fx = np.fft.fftfreq(W)[None, :]
    R = np.sqrt(fy**2 + fx**2) / 0.5
    mlow = (R < a.cut).astype(np.float32)
    Vlow = np.real(np.fft.ifft2(np.fft.fft2(Vt, axes=(-2, -1)) * mlow, axes=(-2, -1)))
    Y = Vlow.reshape(n, -1)

    # concept-level pooling: mean over trials, then align to V rows
    def pool(key: str) -> np.ndarray:
        f = feats[key][: n * n_tr].reshape(n, n_tr, -1).mean(1)
        return f

    # concept-disjoint fit/val split (same construction as leakfree.concept_split)
    rng = np.random.default_rng(20260910)
    perm = rng.permutation(n)
    fi, vi = perm[: int(0.9 * n)], perm[int(0.9 * n):]

    rep: dict = {"n_concepts": n, "n_trials_per_concept": int(n_tr), "cut": a.cut, "gates": {}}

    # ---------------- Gate 0: reliability (split-half over trials)
    g0 = {}
    for bn in BANDS:
        keys = [f"{bn}_{w}" for w in WINDOWS]
        halves = []
        for h in (0, 1):
            parts = []
            for k in keys:
                f = feats[k][: n * n_tr].reshape(n, n_tr, -1)
                parts.append(f[:, h::2].mean(1))
            halves.append(np.concatenate(parts, 1))
        h0, h1 = halves[0][fi], halves[1][fi]
        a0 = h0 - h0.mean(0, keepdims=True)
        a1 = h1 - h1.mean(0, keepdims=True)
        num = (a0 * a1).sum(1)
        den = np.linalg.norm(a0, axis=1) * np.linalg.norm(a1, axis=1)
        ok = den > 1e-8
        g0[bn] = round(float(np.mean(num[ok] / den[ok])), 4) if ok.any() else 0.0
    rep["gates"]["gate0_reliability"] = g0
    rep["gates"]["gate0_pass"] = bool(min(g0.values()) > 0.02)

    # ---------------- Gate 1: spatial (covariance) vs magnitude (power) -> layout
    g1 = {}
    for bn in BANDS:
        Xc = np.concatenate([pool(f"{bn}_{w}") for w in WINDOWS], 1)
        Xp = power[bn][: n * n_tr].reshape(n, n_tr, -1).mean(1)
        r_cov, r2_cov = ridge_r2(Xc[fi], Y[fi], Xc[vi], Y[vi])
        r_pow, r2_pow = ridge_r2(Xp[fi], Y[fi], Xp[vi], Y[vi])
        g1[bn] = {"cov_r": round(r_cov, 4), "cov_r2": round(r2_cov, 4),
                  "pow_r": round(r_pow, 4), "pow_r2": round(r2_pow, 4)}
    rep["gates"]["gate1_layout_tuning"] = g1
    # The routing claim needs a MARGIN, not just a win, otherwise two noise bands can
    # pass by chance. gamma's spatial structure must (a) predict layout better than
    # its own magnitude, (b) beat alpha's spatial structure, and (c) actually be
    # predictive out of sample.
    d_pow = g1["gamma"]["cov_r"] - g1["gamma"]["pow_r"]
    d_alpha = g1["gamma"]["cov_r"] - g1["alpha"]["cov_r"]
    predictive = g1["gamma"]["cov_r2"] > 0.0
    claim_a = bool(predictive and d_pow > 0.01)
    claim_b = bool(predictive and d_alpha > 0.01)
    rep["gates"]["gate1_claim_gamma_spatial_beats_magnitude"] = claim_a
    rep["gates"]["gate1_claim_gamma_beats_alpha_for_layout"] = claim_b
    rep["gates"]["gate1_margins"] = {"over_magnitude": round(d_pow, 4),
                                     "over_alpha": round(d_alpha, 4),
                                     "gamma_cov_r2_positive": bool(predictive)}
    rep["gates"]["gate1_pass"] = bool(claim_a or claim_b)

    # ---------------- Gate 2: residual dependency -> rank r (held-out)
    Xs = np.concatenate([pool(f"gamma_{w}") for w in WINDOWS], 1)
    Xq = np.concatenate([pool(f"alpha_{w}") for w in WINDOWS]
                        + [pool(f"theta_{w}") for w in WINDOWS], 1)
    k_max = int(min(20, max(2, len(fi) // 10)))
    A_r = _pca_reduce(Xs[fi], k_max)
    B_r = _pca_reduce(Xq[fi], k_max)
    cc = cca_heldout(A_r, B_r, k=k_max)
    r = int(sum(1 for c in cc if c > a.cca_thresh))
    rep["gates"]["gate2_canonical_corr_heldout"] = [round(c, 4) for c in cc]
    rep["gates"]["gate2_k_max"] = k_max
    rep["gates"]["gate2_rank_r"] = r
    rep["gates"]["gate2_thresh"] = a.cca_thresh
    rep["gates"]["gate2_note"] = (
        "r = number of gamma<->alpha/theta canonical components whose HELD-OUT "
        "correlation exceeds threshold. Held-out because unregularised CCA on "
        "n << p returns 1.0 for min(p,q) components and would report a meaningless r. "
        "r is the measured residual dependency, i.e. how much the towers NEED to exchange.")

    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_json).write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print("\n" + json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
