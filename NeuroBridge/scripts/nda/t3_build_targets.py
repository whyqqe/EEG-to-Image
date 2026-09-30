#!/usr/bin/env python3
"""T3 targets: explicit frequency decomposition + DISTRIBUTIONAL (texture) targets.

MEASURED MOTIVATION (sub-08)
----------------------------
Per-band correlation between the EEG-predicted SDXL-VAE latent and the GT latent,
and the GT energy fraction carried by each band:

    band r            energy_frac_GT     corr(pred, GT)
    0.000-0.0625          0.583             0.5955    <- EEG is informative
    0.0625-0.125          0.051             0.1703
    0.125-0.25            0.039             0.0988
    0.25-0.5              0.068             0.0718
    0.5-Nyquist           0.259             0.0355    <- EEG carries ~nothing

Low-passing the anchor raises usable fidelity by 1.72x:

    cut=0.0625 -> latent pearson +0.5696, std_ratio 0.692  (vs full-band 0.3317/0.390)

So the two bands must NOT be supervised as one target with equal weight -- which is
exactly what the shipped `train_eeg_vae_head.py` does (a single L1 over (4,64,64)).

WHY THE HIGH BAND NEEDS STATISTICS, NOT PIXELS
----------------------------------------------
L1 regression is a conditional-median estimator. For a band EEG cannot resolve, the
optimal L1 prediction is (approximately) the conditional mean, so the predicted HF
field collapses toward ZERO. Measured: shipped spread ratio 0.428 vs 0.4636 for a
plain linear baseline. This collapse CANNOT be fixed by loss re-weighting or by a
separate head: it is caused by conditional uncertainty, not by supervision
granularity.

A high-pass field has zero DC, so "collapse to the conditional mean" for it means
"collapse to nothing". We therefore supervise the high band with DISTRIBUTIONAL
targets -- radial power spectrum, angular (orientation) spectrum, and a low-res Gram
matrix -- for which the TYPICAL realisation IS the correct answer. This is the one
place where a mean-like target is not a bug.

This is also the only setting where conditional flow matching is well-posed: SP-FM
(arXiv 2601.11827) shows single-Gaussian conditional bases make SAMPLE recovery
ill-posed, while a distributional target is exactly what a transport map should
recover. CFM failed earlier in this project as an *alignment* module (mean-vs-sample
conflict, Top-1 0.010 from 0.160); the mismatch was the task, not only the module.

Outputs per split
-----------------
    lf_latent_{split}.npy  (N,4,64,64) float16   low band   (structure tower target)
    hf_latent_{split}.npy  (N,4,64,64) float16   high band  (texture pixel reference)
    hf_stats_{split}.npy   (N,S)       float32   texture statistics target
    mask.npy               (64,64)     bool      low-band indicator (shared by train/test)

Statistics layout (per sample, S = C*(R + A) + C*C)
    [0        : C*R )              log radial power, per channel, HF region only
    [C*R      : C*(R+A))           log angular power, per channel, HF region only
    [C*(R+A)  : C*(R+A)+C*C)       Gram of HF downsampled to G x G (spatial correlation)

All terms are MEAN-REMOVED and ENERGY-RELATIVE, so the target is a *shape* rather
than an absolute amplitude; absolute HF amplitude is re-imposed at assembly time by
energy matching inside build_spectral_anchor.py. This keeps the head from being
penalised for the global contrast of the stimulus, which EEG does not carry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# frequency helpers (train/test MUST share the identical mask)
# --------------------------------------------------------------------------- #
def radial(H: int, W: int) -> np.ndarray:
    """Normalised radial frequency in [0, ~1.41]; 0 = DC, 0.5 = Nyquist on an axis."""
    fy = np.fft.fftfreq(H)[:, None]
    fx = np.fft.fftfreq(W)[None, :]
    return np.sqrt(fy**2 + fx**2) / 0.5


def angle(H: int, W: int) -> np.ndarray:
    """Orientation in [0, pi); sign of the frequency vector is irrelevant for power."""
    fy = np.fft.fftfreq(H)[:, None]
    fx = np.fft.fftfreq(W)[None, :]
    return np.arctan2(fy, fx) % np.pi


def band_split(x: np.ndarray, r: np.ndarray, cut: float) -> tuple[np.ndarray, np.ndarray]:
    """Orthogonal low/high split along the last two dims. lo + hi == x exactly.

    numpy's FFT always promotes to complex128, so the outputs are cast back to the
    input dtype; otherwise every downstream tensor silently becomes float64/float64
    (which showed up as "Found dtype Double but expected Float" in torch).
    """
    dt = x.dtype if np.issubdtype(x.dtype, np.floating) else np.float32
    F = np.fft.fft2(x.astype(np.float32), axes=(-2, -1))
    m = (r < cut).astype(np.float32)
    lo = np.real(np.fft.ifft2(F * m, axes=(-2, -1)))
    hi = np.real(np.fft.ifft2(F * (1.0 - m), axes=(-2, -1)))
    return lo.astype(dt), hi.astype(dt)


def hf_stats(x: np.ndarray, r: np.ndarray, ang: np.ndarray, cut: float,
             n_radial: int, n_ang: int, gram_res: int) -> np.ndarray:
    """Distributional texture descriptor of the HIGH band of a batch.

    x: (N, C, H, W) float32
    returns: (N, S) float32
    """
    n, c, h, w = x.shape
    # power spectrum of the whole sample, then restrict to the high band
    F = np.fft.fft2(x.astype(np.float32), axes=(-2, -1))
    P = (F.real**2 + F.imag**2)                       # (N,C,H,W)
    hf = (r >= cut).astype(np.float32)                 # (H,W)
    P = P * hf

    r_edges = np.linspace(cut, r.max() + 1e-6, n_radial + 1)
    a_edges = np.linspace(0.0, np.pi + 1e-6, n_ang + 1)
    ri = np.clip(np.digitize(r, r_edges) - 1, 0, n_radial - 1)
    ai = np.clip(np.digitize(ang, a_edges) - 1, 0, n_ang - 1)

    n_hi = max(float(hf.sum()), 1.0)
    # radial: mean power per annulus, normalised by the mean over the HF region
    rad = np.zeros((n, c, n_radial), dtype=np.float32)
    angp = np.zeros((n, c, n_ang), dtype=np.float32)
    cnt_r = np.zeros(n_radial, dtype=np.float32)
    cnt_a = np.zeros(n_ang, dtype=np.float32)
    for k in range(n_radial):
        m = hf * (ri == k)
        cnt_r[k] = m.sum()
        if cnt_r[k] > 0:
            rad[:, :, k] = (P * m).sum(axis=(2, 3)) / cnt_r[k]
    for k in range(n_ang):
        m = hf * (ai == k)
        cnt_a[k] = m.sum()
        if cnt_a[k] > 0:
            angp[:, :, k] = (P * m).sum(axis=(2, 3)) / cnt_a[k]

    scale = P.sum(axis=(2, 3)) / n_hi                          # (N,C) mean HF power
    amp = np.log(np.maximum(scale, 1e-12))                     # (N,C)
    scale = np.maximum(scale, 1e-12)[:, :, None]
    rad = rad / scale
    angp = angp / scale
    # log-compress: texture energy spans orders of magnitude, and this makes the
    # target closer to Gaussian (so L1 and L2 stop disagreeing about what matters)
    rad = np.log1p(rad * 10.0)
    angp = np.log1p(angp * 10.0)

    # Gram of the HF field, downsampled: spatial correlation / "grain" of the texture
    hi = np.real(np.fft.ifft2(F * (1.0 - hf), axes=(-2, -1)))   # (N,C,H,W)
    if gram_res < h:
        f = h // gram_res
        hi_ds = hi.reshape(n, c, gram_res, f, gram_res, f).mean(axis=(3, 5))
    else:
        hi_ds = hi
    hi_f = hi_ds.reshape(n, c, -1)
    hi_f = hi_f - hi_f.mean(axis=2, keepdims=True)
    denom = np.sqrt((hi_f**2).sum(axis=2, keepdims=True)).clip(min=1e-8)
    hi_f = hi_f / denom
    gram = (hi_f @ hi_f.transpose(0, 2, 1)).reshape(n, c * c)   # (N,C*C)

    # AMPLITUDE TERM. Without this the target is exactly scale-invariant -- radial
    # and angular shapes are divided by the sample's own mean HF power, and the Gram
    # rows are unit-normalised -- so a head can match the SHAPE perfectly while the
    # field's amplitude decays toward zero. Measured on the CPU harness: the
    # statistics-supervised head reached stats_l1 0.048 with std ratio 0.135, i.e.
    # it fit the target beautifully and still collapsed. log mean HF power per
    # channel pins the absolute scale and is itself a distributional statistic.
    out = np.concatenate([amp.reshape(n, -1), rad.reshape(n, -1),
                          angp.reshape(n, -1), gram], axis=1)
    return out.astype(np.float32)


def stats_layout(c: int, n_radial: int, n_ang: int) -> dict:
    return {
        "amp": [0, c],
        "radial": [c, c + c * n_radial],
        "angular": [c + c * n_radial, c + c * (n_radial + n_ang)],
        "gram": [c + c * (n_radial + n_ang), c + c * (n_radial + n_ang) + c * c],
        "total": c + c * (n_radial + n_ang) + c * c,
    }


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae-train-npy", required=True)
    ap.add_argument("--vae-test-npy", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--cut", type=float, default=0.0625,
                    help="LF/HF split radius. 0.0625 = the band the EEG prediction is "
                         "actually informative in (corr 0.5955 vs 0.0355 above 0.5)")
    ap.add_argument("--n-radial", type=int, default=16)
    ap.add_argument("--n-ang", type=int, default=8)
    ap.add_argument("--gram-res", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--channels", type=int, default=0, help="truncate channels (0 = keep all)")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tr = np.load(args.vae_train_npy, mmap_mode="r")
    te = np.load(args.vae_test_npy, mmap_mode="r")
    assert tr.shape[1:] == te.shape[1:], f"train {tr.shape} vs test {te.shape} channel/res mismatch"
    c, h, w = tr.shape[1], tr.shape[2], tr.shape[3]
    if args.channels:
        c = min(c, args.channels)
    print(f"[INFO] vae train={tr.shape} test={te.shape} c={c} h={h} w={w} cut={args.cut}")

    r = radial(h, w)
    ang = angle(h, w)
    mask = (r < args.cut)
    np.save(out / "mask.npy", mask)
    lay = stats_layout(c, args.n_radial, args.n_ang)

    report: dict = {
        "cut": args.cut, "n_radial": args.n_radial, "n_ang": args.n_ang,
        "gram_res": args.gram_res, "stats_layout": lay, "splits": {},
    }

    for split, arr in (("train", tr), ("test", te)):
        n = arr.shape[0]
        lf = np.zeros((n, c, h, w), dtype=np.float16)
        hf = np.zeros((n, c, h, w), dtype=np.float16)
        st = np.zeros((n, lay["total"]), dtype=np.float32)
        e_lf = e_hf = e_all = 0.0
        for i in range(0, n, args.batch):
            x = np.asarray(arr[i : i + args.batch], dtype=np.float32)
            if args.channels:
                x = x[:, :c]
            lo, hi = band_split(x, r, args.cut)
            st[i : i + x.shape[0]] = hf_stats(x, r, ang, args.cut,
                                              args.n_radial, args.n_ang, args.gram_res)
            lf[i : i + x.shape[0]] = lo.astype(np.float16)
            hf[i : i + x.shape[0]] = hi.astype(np.float16)
            e_lf += float((lo**2).sum())
            e_hf += float((hi**2).sum())
            e_all += float((x**2).sum())
        np.save(out / f"lf_latent_{split}.npy", lf)
        np.save(out / f"hf_latent_{split}.npy", hf)
        np.save(out / f"hf_stats_{split}.npy", st)
        # orthogonal-split sanity: lo + hi must reproduce the input energy exactly
        ratio = (e_lf + e_hf) / max(e_all, 1e-12)
        assert abs(ratio - 1.0) < 1e-3, f"{split}: lo+hi energy {ratio:.6f} != 1 (split not orthogonal)"
        report["splits"][split] = {
            "n": int(n),
            "energy_frac_lf": round(e_lf / max(e_all, 1e-12), 4),
            "energy_frac_hf": round(e_hf / max(e_all, 1e-12), 4),
            "split_energy_ratio": round(ratio, 6),
            "lf_std": round(float(lf.std().astype(np.float32)), 5),
            "hf_std": round(float(hf.std().astype(np.float32)), 5),
            "stats_mean_abs": round(float(np.abs(st).mean()), 5),
        }
        print(f"[OK] {split}: n={n} LF energy {report['splits'][split]['energy_frac_lf']:.3f} "
              f"HF energy {report['splits'][split]['energy_frac_hf']:.3f} stats={st.shape}")

    report["note"] = (
        "LF = structure tower target (EEG informative). HF = texture tower target, "
        "supervised as DISTRIBUTIONAL statistics because a pixel-level L1 target "
        "collapses to zero in the band EEG cannot resolve.")
    (out / "t3_targets_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
