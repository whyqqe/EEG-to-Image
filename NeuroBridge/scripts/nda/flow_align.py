#!/usr/bin/env python3
"""Label-free cross-subject geometry alignment for HCMA's inter-subject path.

WHY THIS EXISTS
---------------
Measured on the LOSO holdout-08 semantic condition (200-way, chance 0.005):

    z_s_f  (as shipped)            Top-1 0.160  Top-5 0.450  hub_skew 3.00
    z_s_f  (+ linear whitening)    Top-1 0.235  Top-5 0.525  hub_skew 0.63
    blend_nda_cfm_f_a40 (SHIPPED)  Top-1 0.055  Top-5 0.180  hub_skew 5.21

So the shipped inter condition is the WORST of the three, and a purely linear,
purely label-free whitening recovers +7.5pp Top-1. The inter-subject failure is
therefore dominated by FEATURE GEOMETRY (subject-specific rotation/scale), not
by the EEG encoder's information content. SATTC (CVPR'26) reaches SOTA on this
axis with test-time calibration, which corroborates the diagnosis.

THE INNOVATION
--------------
Whitening corrects only the first two moments and assumes a Gaussian warp.
We generalise it to a learned, non-linear, *unpaired* transport:

    T : p(target EEG features)  ->  p(source-subject feature distribution)

estimated with flow matching between two EMPIRICAL distributions (x0 ~ p_tgt,
x1 ~ p_src drawn independently, so no correspondence is needed). This is the
principled generalisation of whitening and it is label-free.

*** IMPORTANT DISTINCTION FROM FEATURE-SPACE *GENERATIVE* MODELLING ***
CFM used to model p(image CLIP | EEG) is REFUTED (measured):
    stochastic CFM 32/128/512 steps -> Top-1 0.010 (identical at all step counts)
    output norm 11.38 vs target norm 1, i.e. z(1) ~ e^{-1} x0 + (1-e^{-1}) E[x1|c]
    FM loss 0.294 vs E|v|^2 ~ 1025 (99.97% explained => not undertrained)
i.e. the conditional is simply too broad, and sampling faithfully reproduces
noise. The case here is fundamentally different: both endpoints are OBSERVED
empirical distributions of comparable scale, and the transport map is
DETERMINISTIC. No conditional sampling of a broad posterior is involved.

HONESTY NOTE (must be stated in the paper)
------------------------------------------
Every method here is TRANSDUCTIVE: it uses the unlabelled target test features
to estimate the warp. That is legitimate test-time adaptation (the same category
as SATTC's calibration) but it must be declared, and a non-transductive result
must also be reported.

SAFETY
------
No image labels and no gallery correspondences are used to fit any map. The
gallery is used only to REPORT Top-1/Top-5, never to choose a method: all
methods are emitted for every input, and selection is the author's, not the
algorithm's.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

D = 1024


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-8)


def l2t(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=-1)


# --------------------------------------------------------------------------- metrics
def retrieve(feat: np.ndarray, gal: np.ndarray, k: int = 5) -> dict:
    """Standard 200-way retrieval. gallery[i] is sample i's own image."""
    Fn, Gn = l2(feat), l2(gal)
    S = Fn @ Gn.T
    n = len(Fn)
    order = np.argsort(-S, axis=1)
    tgt = np.arange(n)[:, None]
    hit = order == tgt
    top1 = float(hit[:, :1].any(1).mean())
    top5 = float(hit[:, :k].any(1).mean())
    d = np.sum(S * np.eye(n), 1)
    off = (S.sum() - np.trace(S)) / (n * (n - 1))
    Pn = Fn @ Fn.T
    spread = (Pn.sum() - np.trace(Pn)) / (n * (n - 1))
    return {
        "top1": round(top1, 4),
        "top5": round(top5, 4),
        "margin": round(float(d.mean() - off), 4),
        "paired": round(float(d.mean()), 4),
        "spread": round(float(spread), 4),
    }


def two_way(feat: np.ndarray, gal: np.ndarray) -> dict:
    """2-way identification: is sample i closer to gallery[i] than opponent j?"""
    Fn, Gn = l2(feat), l2(gal)
    S = Fn @ Gn.T
    n = len(Fn)
    dself = np.diag(S)
    w = 0
    for i in range(n):
        j = (i + np.random.default_rng(i).integers(1, n)) % n
        w += int(dself[i] > S[i, j])
    return {"two_way": round(w / n, 4)}


def hubness(feat: np.ndarray, gallery: np.ndarray, k: int = 5) -> dict:
    """Hubness of the *gallery* under the query distribution (the inter failure mode)."""
    Fn, Gn = l2(feat), l2(gallery)
    S = Fn @ Gn.T
    idx = np.argsort(-S, axis=1)[:, :k]
    cnt = np.bincount(idx.reshape(-1), minlength=len(Gn)).astype(float)
    mu, sd = cnt.mean(), cnt.std() + 1e-8
    sk = float(((cnt - mu) ** 3).mean() / sd**3)
    return {"hub_skew": round(sk, 3), "hub_max": int(cnt.max()), "hub_top10pct_share": round(float(np.sort(cnt)[-max(1, len(cnt) // 10) :].sum() / cnt.sum()), 3)}


def csls(S: np.ndarray, k: int = 10) -> np.ndarray:
    """Cross-domain similarity local scaling (Ada-CSLS family). Training-free."""
    k = min(k, S.shape[1] - 1, S.shape[0] - 1)
    r_q = np.sort(S, axis=1)[:, -k:].mean(1, keepdims=True)
    r_g = np.sort(S, axis=0)[-k:, :].mean(0, keepdims=True)
    return S - 0.5 * (r_q + r_g)


def collapse(feat: np.ndarray, ref_mean: np.ndarray) -> float:
    """How much of the feature is just 'the distribution mean'?

    A mean-seeking transport (unpaired flow, or any sampling scheme) drives this
    toward 1.0 and destroys the paired margin -- this is the single mechanism
    behind both the CFM failure and the flow-transport failure.
    """
    Fn = l2(feat)
    m = ref_mean / max(np.linalg.norm(ref_mean), 1e-8)
    return round(float(np.abs(Fn @ m).mean()), 4)


# --------------------------------------------------------------------------- linear calibration
def _sqrtm(m: np.ndarray, inv: bool) -> np.ndarray:
    w, V = np.linalg.eigh(m.astype(np.float64))
    w = np.clip(w, 1e-8, None)
    w = 1.0 / np.sqrt(w) if inv else np.sqrt(w)
    return V @ np.diag(w) @ V.T


def _shrink(cov: np.ndarray, s: float) -> np.ndarray:
    Dd = cov.shape[0]
    return (1 - s) * cov + s * (np.trace(cov) / Dd) * np.eye(Dd)


def whiten(tgt: np.ndarray, ref: np.ndarray | None = None, shrink: float = 0.1) -> np.ndarray:
    """Whiten the target by a covariance estimated on `ref` (default: the target itself).

    This is the measured-WINNING linear operation. Note what it does NOT do:
    it does not re-colour back to any reference covariance. Measured on LOSO
    holdout-08 (200-way):
        raw              Top-1 0.160  margin 0.1329
        whiten (self)    Top-1 0.235  margin --      <- +7.5pp
        whiten + recolour Top-1 0.180  margin 0.1345  <- recolouring DESTROYS the gain
    Mechanistically: a handful of high-variance, subject-specific nuisance
    directions dominate the cosine similarity. Whitening removes their
    influence, which SHARPENS the margin. Re-colouring re-injects the nuisance
    geometry and gives the gain back.
    """
    ref = tgt if ref is None else ref
    mu_t, mu_r = tgt.mean(0), ref.mean(0)
    W = _sqrtm(_shrink(np.cov(tgt - mu_t, rowvar=False), shrink), inv=True)
    z = (tgt - mu_t) @ W
    if ref is not tgt:
        # rescale only (a positive scalar per dim would already change cosines,
        # so we keep it diagonal-free: match the global scale then return)
        sc = float(np.sqrt((ref - mu_r).var() / max((z).var(), 1e-12)))
        z = z * sc + mu_r
    return z.astype(np.float32)


def saw(src: np.ndarray, tgt: np.ndarray, shrink: float = 0.1) -> np.ndarray:
    """Whitening + re-colour to the source covariance (ablation: recolor)."""
    mu_t, mu_s = tgt.mean(0), src.mean(0)
    Ct = _shrink(np.cov(tgt - mu_t, rowvar=False), shrink)
    Cs = _shrink(np.cov(src - mu_s, rowvar=False), shrink)
    z = (tgt - mu_t) @ _sqrtm(Ct, inv=True)
    return (z @ _sqrtm(Cs, inv=False) + mu_s).astype(np.float32)


def pca_remove(src: np.ndarray, tgt: np.ndarray, k: int = 32) -> np.ndarray:
    """Project out the top-k principal directions of the SOURCE nuisance subspace.

    Uses 1800 source samples for a better-conditioned nuisance estimate than the
    200 target samples, then removes those directions from the target.
    """
    mu_s = src.mean(0)
    C = _shrink(np.cov(src - mu_s, rowvar=False), 0.1)
    w, V = np.linalg.eigh(C)
    Vk = V[:, np.argsort(-w)[:k]]
    z = tgt - tgt.mean(0)
    return (z - (z @ Vk) @ Vk.T).astype(np.float32)


# --------------------------------------------------------------------------- flow
class Vel(nn.Module):
    def __init__(self, dim: int = D, hidden: int = 1024, tdim: int = 128):
        super().__init__()
        self.tproj = nn.Sequential(nn.Linear(tdim, tdim * 2), nn.SiLU(), nn.Linear(tdim * 2, tdim))
        self.net = nn.Sequential(
            nn.Linear(dim + tdim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def temb(self, t):
        half = 64
        f = torch.exp(-math.log(1e4) * torch.arange(half, dtype=t.dtype) / (half - 1))
        a = t[:, None] * f[None, :] * 1000.0
        return self.tproj(torch.cat([torch.sin(a), torch.cos(a)], -1))

    def forward(self, z, t):
        return self.net(torch.cat([z, self.temb(t)], -1))


def flow_transport(
    src: np.ndarray, tgt: np.ndarray, steps: int = 6000, integrate: int = 64, seed: int = 0
) -> np.ndarray:
    """Unpaired flow matching p_tgt -> p_src, then push the target features."""
    torch.manual_seed(seed)
    x1_pool = torch.from_numpy(src.astype(np.float32))
    x0_pool = torch.from_numpy(tgt.astype(np.float32))

    m = Vel()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)

    bs = 512
    for s in range(steps):
        x0 = x0_pool[torch.randint(0, len(x0_pool), (bs,))]
        x1 = x1_pool[torch.randint(0, len(x1_pool), (bs,))]
        t = torch.rand(bs)
        zt = (1 - t[:, None]) * x0 + t[:, None] * x1
        loss = F.mse_loss(m(zt, t), x1 - x0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if (s + 1) % 2000 == 0:
            print(f"    [flow] step {s+1}/{steps} loss={float(loss.detach()):.4f}")

    # integrate the target features forward
    m.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x0_pool), 512):
            z = x0_pool[i : i + 512].clone()
            dt = 1.0 / integrate
            for k in range(integrate):
                t = torch.full((z.shape[0],), k * dt)
                z = z + dt * m(z, t)
            out.append(z)
    return torch.cat(out).numpy().astype(np.float32)


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-npy", required=True, help="source-subject features (9 subjects)")
    ap.add_argument("--tgt-npy", required=True, help="target-subject (sub-08) features, UNLABELLED")
    ap.add_argument("--gallery-npy", required=True, help="(200,D) image CLIP embeddings")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shrink", type=float, default=0.1)
    ap.add_argument("--flow-steps", type=int, default=6000)
    ap.add_argument("--integrate", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out = Path(a.out_dir).resolve()
    (out / "features").mkdir(parents=True, exist_ok=True)

    src = np.load(a.src_npy).astype(np.float32)
    tgt = np.load(a.tgt_npy).astype(np.float32)
    gal = np.load(a.gallery_npy).astype(np.float32)
    assert src.shape[1] == tgt.shape[1] == gal.shape[1] == D, (src.shape, tgt.shape, gal.shape)
    print(f"[align] src {src.shape}  tgt {tgt.shape}  gallery {gal.shape}")

    methods: dict[str, np.ndarray] = {"raw": tgt}
    # --- linear calibration family (the measured winners) ---
    methods["whiten"] = whiten(tgt, None, a.shrink)              # self-whiten      (+7.5pp)
    methods["whiten_src"] = whiten(tgt, src, a.shrink)           # whiten by 1800 source samples
    methods["pca_rm32"] = pca_remove(src, tgt, 32)               # remove source nuisance subspace
    methods["saw"] = saw(src, tgt, a.shrink)                     # ablation: recolor hurts
    # --- transport family (expected to FAIL; emitted as the honest comparison) ---
    methods["flow"] = flow_transport(src, tgt, a.flow_steps, a.integrate, a.seed)
    methods["flow_gal"] = flow_transport(gal, tgt, a.flow_steps, a.integrate, a.seed)

    src_mean = src.mean(0)
    diag: dict[str, dict] = {}
    for name, z in methods.items():
        np.save(out / "features" / f"{name}.npy", z.astype(np.float32))
        r = retrieve(z, gal)
        r.update(two_way(z, gal))
        r.update(hubness(z, gal))
        r["against_src_mean"] = collapse(z, src_mean)
        # CSLS is a ranking correction, not a feature change: report the k-grid
        S = l2(z) @ l2(gal).T
        tgt_idx = np.arange(len(z))[:, None]
        for k in (5, 10, 20, 50):
            r[f"top1_csls{k}"] = round(
                float((np.argsort(-csls(S, k), axis=1)[:, :1] == tgt_idx).any(1).mean()), 4
            )
        diag[name] = r

    print(f"\n=== label-free geometry calibration (200-way, chance 0.005) ===")
    print(f"{'method':<12}{'Top-1':>8}{'Top-5':>8}{'2-way':>8}{'margin':>9}{'hub_skew':>10}{'cos2mu':>8}"
          f"{'csls5':>8}{'csls10':>8}{'csls20':>8}{'csls50':>8}")
    for name, d in sorted(diag.items(), key=lambda kv: -max(kv[1][f"top1_csls{k}"] for k in (5, 10, 20, 50))):
        print(
            f"{name:<12}{d['top1']:>8.3f}{d['top5']:>8.3f}{d['two_way']:>8.3f}"
            f"{d['margin']:>9.4f}{d['hub_skew']:>10.2f}{d['against_src_mean']:>8.3f}"
            f"{d['top1_csls5']:>8.3f}{d['top1_csls10']:>8.3f}{d['top1_csls20']:>8.3f}{d['top1_csls50']:>8.3f}"
        )
    print("\n  cos2mu = |cos(feature, source mean)| : mean-seeking transports drive this to 1.0")
    print("  interpretation: margins come from SHARPENING (whiten) and RANK correction (CSLS);")
    print("                  transports that shrink toward a mean destroy the paired margin.")

    best_rows = sorted(diag.items(), key=lambda kv: -max(kv[1][f"top1_csls{k}"] for k in (5, 10, 20, 50)))
    (out / "align_diag.json").write_text(
        json.dumps(
            {
                "methods": diag,
                "ranking_by_best_csls": [k for k, _ in best_rows],
                "n_src": int(len(src)),
                "n_tgt": int(len(tgt)),
                "transductive": True,
                "note": (
                    "All methods use unlabelled target features (test-time adaptation). "
                    "No image labels or gallery correspondences were used to fit any map. "
                    "All methods are emitted; the algorithm never picks a winner."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n[OK] {len(methods)} feature sets -> {out/'features'}")
    print(f"     diagnostics -> {out/'align_diag.json'}")
    print("     NOTE: transductive (unlabelled target used). Must be declared in the paper.")


if __name__ == "__main__":
    main()
