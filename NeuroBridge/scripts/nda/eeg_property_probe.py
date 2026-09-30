"""Measure the EEG properties that should DICTATE the encoder's front end.

Five properties, all cheap, all with a direct architectural consequence:

1. MAINS / LINE NOISE.  The encoder's fifth band is 30-80 Hz at 250 Hz sampling
   (Nyquist 125 Hz), so a 50 Hz mains peak sits INSIDE "gamma".  THINGS-EEG2 was
   recorded in Europe, where mains is 50 Hz.  If the peak survives preprocessing,
   the gamma token is partly an electrical artefact and every gamma-based mechanism
   (DLA, DNG, gamma linearity) is measured through it.

2. APERIODIC 1/f SLOPE.  EEG spectra are aperiodic (1/f^chi) plus a few peaks.  The
   exponent tracks excitation/inhibition balance, differs across subjects, and is not
   stimulus-locked, i.e. it is NUISANCE variance for a visual decoder.  If the slope
   is steep, raw band power is dominated by the aperiodic floor rather than by
   oscillatory response.

3. EFFECTIVE SPATIAL RANK.  Volume conduction makes neighbouring electrodes nearly
   redundant, so the 17x17 band covariance is dominated by a few components.  The
   participation ratio says how many of the 17 x 17 = 289 entries of a per-band
   unmixing matrix are actually estimable from the data.

4. EVOKED vs INDUCED.  Total power = phase-locked (evoked) + non-phase-locked
   (induced).  The evoked part is reproducible across repetitions; the induced part is
   not.  Their relative size decides whether the affordable encoder should read the
   ERP waveform or band power.

5. PHASE FEASIBILITY.  samples/cycle = 250/f.  Reliable single-trial phase needs
   roughly 8 samples/cycle, which caps phase-based mechanisms at about 30 Hz.
"""
from __future__ import annotations

import os
import pathlib
import sys

import numpy as np
from scipy.signal import welch

NB_ROOT = os.environ.get("NB_ROOT", "/project/peilab/why/NeuroBridge")
sys.path.insert(0, f"{NB_ROOT}/scripts/nda")

SFREQ = 250.0
BANDS = ((0.0, 4.0), (4.0, 8.0), (8.0, 12.0), (12.0, 30.0), (30.0, 80.0))
SUB = int(os.environ.get("SUBJECT", "8"))

from tdm_gate0 import load_eeg                                            # noqa: E402

cache = pathlib.Path(f"{NB_ROOT}/outputs/tdm/cache")
eeg, row = load_eeg(SUB, True, cache)
print(f"[eegprops] sub-{SUB:02d} train EEG {eeg.shape} dtype={eeg.dtype}")
n, C, T = eeg.shape

# ---------------------------------------------------------------- 1 & 2: PSD
f, p = welch(eeg, fs=SFREQ, nperseg=T, axis=-1)      # (n, C, F), one window on 1 s
p = p.mean(0).mean(0)                                # average over trials and channels
band_ok = (f >= 1.0) & (f <= 100.0)

print()
print("=" * 92)
print("1. LINE NOISE: is there a mains peak, and is it inside the 30-80 Hz band?")
print("=" * 92)
for mains in (50.0, 60.0):
    near = np.abs(f - mains) <= 1.5
    flank = p[(np.abs(f - (mains - 6.0)) <= 2.0) | (np.abs(f - (mains + 6.0)) <= 2.0)]
    peak = p[near].max() if near.any() else np.nan
    fl = flank.mean() if flank.size else np.nan
    ratio = peak / fl if fl and np.isfinite(fl) else np.nan
    inside = any(lo <= mains < hi for lo, hi in BANDS)
    print(f"  mains {mains:.0f} Hz: peak {peak:.3e} | nearby floor {fl:.3e} | "
          f"ratio {ratio:.2f}x | inside a band: {inside}")
print("  power at 45/48/50/52/55 Hz: "
      + str([f"{p[np.abs(f - g).argmin()]:.2e}" for g in (45, 48, 50, 52, 55)]))

print()
print("=" * 92)
print("2. APERIODIC 1/f SLOPE (log-log fit over 2-100 Hz, peaks excluded)")
print("=" * 92)
sel = (f >= 2.0) & (f <= 100.0)
excl = np.zeros_like(f, bool)
for c in (8, 10, 12, 50, 60):
    excl |= np.abs(f - c) <= 2.0
m = sel & ~excl
slope, intercept = np.polyfit(np.log10(f[m]), np.log10(p[m]), 1)
print(f"  log10-power = {slope:.3f} * log10-f + {intercept:.3f}")
pred = 10 ** (intercept + slope * np.log10(f[sel]))
resid = np.log10(p[sel]) - np.log10(pred)
print(f"  oscillatory share of log-power variance: "
      f"{1.0 - np.var(resid) / np.var(np.log10(p[sel])):.3f}")
for name, (lo, hi) in zip(("delta", "theta", "alpha", "beta", "gamma"), BANDS):
    bm = (f >= max(lo, 1.0)) & (f < hi)
    if bm.any():
        print(f"  {name:<6} {lo:>5.1f}-{hi:<5.1f} Hz  share of in-band power "
              f"{p[bm].sum() / p[sel].sum():.4f}")

# ---------------------------------------------- 3: effective spatial rank
print()
print("=" * 92)
print("3. EFFECTIVE SPATIAL RANK per band (participation ratio of eigenvalues)")
print("=" * 92)
print("   PR = (sum l)^2 / sum l^2.  This is how many independent spatial components")
print("   exist, i.e. how much of a 17x17 = 289-entry unmixing matrix is estimable.")
X = np.fft.rfft(eeg, axis=-1)
fr = np.fft.rfftfreq(T, d=1.0 / SFREQ)
for name, (lo, hi) in zip(("delta", "theta", "alpha", "beta", "gamma"), BANDS):
    bm = (fr >= lo) & (fr < min(hi, SFREQ / 2))
    xb = np.fft.irfft(X[:, :, bm], n=T, axis=-1)
    flat = xb.transpose(1, 0, 2).reshape(C, -1)
    cov = np.cov(flat)
    ev = np.clip(np.linalg.eigvalsh(cov), 0, None)
    pr = float(ev.sum() ** 2 / (ev ** 2).sum())
    print(f"  {name:<6} PR {pr:5.2f} / {C}   top-1 eigenvalue share {ev[-1] / ev.sum():.3f}")

# ---------------------------------------------- 4: evoked vs induced
print()
print("=" * 92)
print("4. EVOKED vs INDUCED: how much of the trial variance is reproducible?")
print("=" * 92)
stride = max(1, len(row) // 1654)
concept = row // stride
uniq = np.unique(concept)[:200]
ii = {c: np.where(concept == c)[0] for c in uniq}
ii = {c: v for c, v in ii.items() if len(v) > 1}
means = np.stack([eeg[v].mean(0) for v in ii.values()])
var_total = float(eeg.var(axis=(1, 2)).mean())
var_between = float(means.var(axis=(1, 2)).mean())
cors = []
for c, v in ii.items():
    mu = eeg[v].mean(0).ravel()
    for i in v[:3]:
        cors.append(float(np.corrcoef(eeg[i].ravel(), mu)[0, 1]))
print(f"  mean per-trial variance          {var_total:.4e}")
print(f"  mean per-concept-mean variance   {var_between:.4e}")
print(f"  reproducible share               {var_between / var_total:.4f}")
print(f"  mean corr(trial, its concept mean) {np.mean(cors):+.4f}  "
      f"(a purely phase-locked signal would be near 1)")

print()
print("=" * 92)
print("5. PHASE FEASIBILITY at 250 Hz")
print("=" * 92)
for g in (4, 8, 13, 30, 40, 50, 60, 80):
    spc = SFREQ / g
    print(f"  {g:>3} Hz: {spc:5.2f} samples/cycle  "
          f"{'<- phase not reliably estimable' if spc < 8 else ''}")
print("  -> single-trial phase is defensible only at or below ~30 Hz.")
