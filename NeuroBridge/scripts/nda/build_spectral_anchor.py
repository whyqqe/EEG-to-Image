#!/usr/bin/env python3
"""Spectral Assembly: build a structural anchor from band-complementary evidence.

MEASURED MOTIVATION (sub-08, intra; 200 test samples)
-----------------------------------------------------
Per-band correlation between the EEG-predicted SDXL-VAE latent and the GT latent:

    band r        energy_frac_GT   corr(pred, GT)
    0.000-0.0625      0.583           0.5955     <-- EEG is informative here
    0.0625-0.125      0.051           0.1703
    0.125-0.25        0.039           0.0988
    0.25-0.5          0.068           0.0718
    0.5-Nyquist       0.259           0.0355     <-- EEG carries NO information

Low band (r<0.125) corr 0.528 vs high band (r>0.25) corr 0.045 -> 11.8x.

So structure is NOT a scalar: it is frequency-resolved, and EEG only supplies the
low-frequency layout. Low-passing the anchor raises its usable fidelity a lot:

    cut=0.0625 -> latent pearson +0.5696, std_ratio 0.692   (vs full-band 0.3317/0.390)
    cut=0.125  -> latent pearson +0.5002, std_ratio 0.608

Meanwhile a scalar `strength` puts the whole pipeline on a Pareto frontier
(measured over 3 independent cn_scale values):

    strength 0.82 -> 0.86 :  SSIM -0.009, PixCorr -0.010, Alex2 +0.009, FID -8.5

This script builds the anchor that breaks that trade-off by giving each band its
own evidence source:

    LF  (r < cut)  <- EEG prediction        (layout; the only band EEG knows)
    HF  (r > cut)  <- real image / prior    (texture; the only source with real detail)

The two sources are ERROR-COMPLEMENTARY: EEG errs by lacking texture, retrieval
errs by proposing a possibly-wrong layout -- and we discard the latter's layout.

This is NOT FBSDiff/FCDiffusion-style band substitution of a *source image*:
there is no source image here. The bands come from heterogeneous evidence
(brain signal + retrieved exemplar) and are selected by a measured information
profile rather than tuned for controllability.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def radial(H: int, W: int) -> np.ndarray:
    fy = np.fft.fftfreq(H)[:, None]
    fx = np.fft.fftfreq(W)[None, :]
    return np.sqrt(fy**2 + fx**2) / 0.5


def split_bands(x: np.ndarray, r: np.ndarray, cut: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (low-pass, high-pass) of x along the last two dims."""
    F = np.fft.fft2(x.astype(np.float32), axes=(-2, -1))
    m = (r < cut).astype(np.float32)
    lo = np.real(np.fft.ifft2(F * m, axes=(-2, -1)))
    hi = np.real(np.fft.ifft2(F * (1.0 - m), axes=(-2, -1)))
    return lo, hi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eeg-latent-npy", required=True, help="EEG-predicted VAE latent (N,4,64,64)")
    ap.add_argument("--train-latent-npy", default="", help="train GT VAE latents (for HF source)")
    ap.add_argument("--neighbor-idx-npy", default="", help="(N,K) retrieval neighbours into train rows")
    ap.add_argument("--output-npy", required=True)
    ap.add_argument("--report-json", default="")
    ap.add_argument("--mode", default="lf_eeg",
                    choices=["full_eeg", "lf_eeg", "lf_eeg_hf_retr"],
                    help="full_eeg = monolithic anchor (control); lf_eeg = LF only; "
                         "lf_eeg_hf_retr = LF from EEG + HF from retrieved real image")
    ap.add_argument("--cut", type=float, default=0.0625)
    ap.add_argument("--retr-k", type=int, default=1, help="how many neighbours feed the HF band")
    ap.add_argument("--hf-scale", type=float, default=1.0,
                    help="rescale retrieved HF to this fraction of the EEG anchor's HF energy")
    args = ap.parse_args()

    A = np.load(args.eeg_latent_npy).astype(np.float32)
    n, C, H, W = A.shape
    r = radial(H, W)
    lo_eeg, hi_eeg = split_bands(A, r, args.cut)

    report: dict = {
        "mode": args.mode, "cut": args.cut, "n": n,
        "source_energy": {
            "eeg_lf": round(float((lo_eeg**2).mean()), 5),
            "eeg_hf": round(float((hi_eeg**2).mean()), 5),
        },
    }

    if args.mode == "full_eeg":
        anchor = A.copy()
    elif args.mode == "lf_eeg":
        anchor = lo_eeg
        report["note"] = ("LF hard-anchored from EEG; HF left to the diffusion prior "
                          "(expected: break the strength Pareto frontier)")
    else:
        if not (args.train_latent_npy and args.neighbor_idx_npy):
            raise ValueError("lf_eeg_hf_retr needs --train-latent-npy and --neighbor-idx-npy")
        Tr = np.load(args.train_latent_npy, mmap_mode="r")
        idx = np.load(args.neighbor_idx_npy)
        k = max(1, min(args.retr_k, idx.shape[1]))
        # average the retrieved exemplars' HF -> consensus texture statistics
        hi_retr = np.zeros_like(hi_eeg)
        for j in range(k):
            nb = np.asarray(Tr[np.asarray(idx[:, j], dtype=np.int64)], dtype=np.float32)
            _, h = split_bands(nb, r, args.cut)
            hi_retr += h / k
        # match HF energy to the EEG anchor's own HF scale so the composite is balanced
        e_retr = float((hi_retr**2).mean())
        e_eeg = float((hi_eeg**2).mean())
        scale = (np.sqrt(e_eeg / max(e_retr, 1e-12)) * args.hf_scale)
        hi_retr = hi_retr * scale
        anchor = lo_eeg + hi_retr
        report["retrieval"] = {"k": k, "hf_rescale": round(float(scale), 4),
                               "retr_hf_energy": round(e_retr, 5)}
        report["note"] = ("LF from EEG (layout) + HF from retrieved real exemplars (texture); "
                          "retrieval layout is discarded, so retrieval errors cannot hurt layout")

    # diagnostics: how faithful is the anchor to GT-looking statistics?
    report["anchor_stats"] = {
        "std": round(float(anchor.std()), 5),
        "per_sample_spatial_std_mean": round(float(anchor.std(axis=(1, 2, 3)).mean()), 5),
        "cross_sample_spread": round(float(anchor.std(axis=0).mean()), 5),
    }
    np.save(args.output_npy, anchor.astype(np.float32))
    print(f"[OK] anchor({args.mode}) cut={args.cut} -> {args.output_npy}  shape={anchor.shape}")
    print(json.dumps(report, indent=2))
    if args.report_json:
        Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_json).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
