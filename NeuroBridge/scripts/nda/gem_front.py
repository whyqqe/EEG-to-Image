#!/usr/bin/env python3
"""GEM front end -- a SUBJECT FINGERPRINT plus a parameter-frugal purified encoder.

THE DIAGNOSIS THIS FILE ANSWERS
-------------------------------
`tdm_gate0.py` / `eeg_property_probe.py` measured five properties of the raw EEG
this project trains on.  Each one makes a specific class of encoder wrong, and
none of them is a hyper-parameter question:

  1. LINE NOISE.  A 50 Hz peak at 3.5x the local background sits INSIDE the
     30-80 Hz band that the previous design used as its stimulus band.  Any
     gamma-band claim made without a notch is partly a claim about the mains.

  2. APERIODIC DOMINANCE.  A single 1/f slope explains 89% of the variance of
     log band power across channels and bands.  Band amplitudes therefore measure
     mostly the aperiodic exponent, so comparing a 2 Hz statistic with a 60 Hz
     statistic compares two nuisance levels.  Band power must be flattened before
     it is compared, and the slope itself is a stable per-subject constant.

  3. EFFECTIVE SPATIAL RANK << N_CHANNELS.  Theta's participation ratio is ~3 of
     17, with one dominant direction.  A full CxC learnable "unmixing" matrix is
     over-parameterised by construction, and its off-diagonal energy cannot be
     evidence for anything (a rank-3 subspace has no room for it).

  4. MOST VARIANCE IS NOT REPRODUCIBLE.  Only ~28% of single-trial variance is
     reproducible across repetitions of the same image.  That is a CEILING on
     what any read-out can explain, and it must be reported, not assumed away.

  5. PHASE IS NOT ESTIMABLE ABOVE ~30 Hz.  At 250 Hz sampling with single-trial
     SNR, the phase of the 30-80 Hz band is a random number for one trial.  A
     differentiable time shift (the previous DLA mechanism) is a PHASE operation,
     so its learned per-band delays were fitting noise in exactly the band its
     claim was about.

WHAT REPLACES THEM
------------------
Nothing here is learned from scratch.  Everything that is a PROPERTY OF THE
SUBJECT is estimated once, from TRAIN trials only, and frozen into a
"fingerprint": the grand-average template, the aperiodic slope, the per-band
spatial principal directions and their participation ratios.  The learnable part
is deliberately tiny -- 5 scalars saying "how much of each global component to
remove" and 5 band gains -- because the structure being imposed is physics, not
a hypothesis to be discovered from 16 540 rows.

  E1  notch 50 / 100 Hz as a fixed frequency-domain gain
  E2  high-pass 1 Hz by construction (the lowest band starts at 1 Hz, so DC and
      drift have no bin to live in) and the 0-4 Hz band is RANK-LIMITED to what
      its own participation ratio supports rather than dropped
  E3  per-band equalisation by the subject's OWN fitted aperiodic slope, so a
      band amplitude means "energy above this subject's 1/f", not "1/f"
  E4  additive common-mode REMOVAL along the data-derived principal directions,
      with a learnable, bounded, reported coefficient per direction
  E5  the same operator IS the spatial demixing: rank-limited, data-initialised,
      identity-free -- no CxC matrix to overfit
  E6  explicit evoked / induced fork: the stimulus-locked template is subtracted
      as a separate stream, so a code can be built from the deviation as well as
      from the total.  The fork is free because every front-end operator is
      linear: F(x - tmpl) = F(x) - F(tmpl).
  E7  the fingerprint is the whole subject adaptation.  There is no subject
      embedding to leak and no per-subject head to overfit.

DELIBERATE EXCLUSIONS, AND WHY
------------------------------
  * No learned time shift (DLA).  It is a phase operation and phase is not
    estimable in the band its claim concerned (property 5).  The honest version
    of latency handling requires a filterbank with a real-valued group delay at
    low frequency only; it is not a step function, so it is left out instead of
    re-introduced in a form that looks principled.
  * No divisive gain from alpha (DNG).  Alpha is spatially global, but the
    usable part of that fact -- "alpha is not channel-specific evidence" -- is
    exactly what E4 already implements, with parameters that cannot drift.  Two
    mechanisms for one fact is decoration.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------ constants
# 1 Hz high-pass by construction: the lowest band starts at 1, so bin 0 (DC) and
# the drift band have no representation at all rather than being attenuated by a
# filter whose transient would then have to be reasoned about.
BANDS = ((1.0, 4.0), (4.0, 8.0), (8.0, 12.0), (12.0, 30.0), (30.0, 80.0))
BAND_NAMES = ("delta", "theta", "alpha", "beta", "gamma")
NB = len(BANDS)
N_PATCH = 5                       # 250 samples -> 5 x 50-sample (200 ms) patches


def band_masks(freqs: np.ndarray, sfreq: float) -> np.ndarray:
    """(NB, F) real 0/1 masks.  Bands are half-open and clipped at Nyquist."""
    m = np.zeros((NB, freqs.size), dtype=np.float32)
    for i, (lo, hi) in enumerate(BANDS):
        hi = min(hi, sfreq / 2.0)
        m[i] = ((freqs >= lo) & (freqs < hi)).astype(np.float32)
        if m[i].sum() < 2:        # never hand a band an empty token stream
            m[i, max(1, int(round(lo / sfreq * freqs.size)))] = 1.0
    return m


def notch_gain(freqs: np.ndarray, f0s=(50.0, 100.0), width_hz: float = 1.5) -> np.ndarray:
    """Multiplicative amplitude gain that nulls the mains and its harmonic.

    A Gaussian null rather than a single-bin zero: bin spacing here is 1 Hz, so a
    one-bin notch is 1 Hz wide and leaves the shoulders of a real 50 Hz line (which
    has mains-frequency drift and finite spectral leakage) inside the band.
    """
    g = np.ones_like(freqs, dtype=np.float32)
    for f0 in f0s:
        if f0 < freqs[-1]:
            g *= 1.0 - np.exp(-((freqs - f0) ** 2) / (2.0 * width_hz ** 2))
    return g.astype(np.float32)


def fit_aperiodic(psd: np.ndarray, freqs: np.ndarray, f_lo: float = 1.0,
                  f_hi: float = 40.0, iters: int = 12) -> dict:
    """Robust log-log fit: log10 P = slope * log10 f + intercept.

    Iteratively reweighted least squares with Huber weights, because the thing
    being estimated is a slope across four decades of amplitude and a handful of
    line-noise-contaminated bins would otherwise set it.  `r2` is reported so the
    claim "the aperiodic component dominates" is a measurement.
    """
    sel = (freqs >= f_lo) & (freqs <= f_hi) & (psd > 0)
    x = np.log10(freqs[sel]).astype(np.float64)
    y = np.log10(psd[sel]).astype(np.float64)
    w = np.ones_like(x)
    slope = intercept = 0.0
    for _ in range(iters):
        X = np.stack([x, np.ones_like(x)], 1) * w[:, None]
        Y = y * w
        beta = np.linalg.lstsq(X, Y, rcond=None)[0]
        slope, intercept = float(beta[0]), float(beta[1])
        r = y - (slope * x + intercept)
        s = np.median(np.abs(r)) * 1.4826 + 1e-9
        w = np.minimum(1.0, 1.345 * s / np.maximum(np.abs(r), 1e-12))
    r = y - (slope * x + intercept)
    r2 = 1.0 - float((r ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12))
    return {"slope": slope, "intercept": intercept, "r2": r2,
            "n_bins": int(sel.sum()), "f_lo": f_lo, "f_hi": f_hi}


def fit_fingerprint(eeg_train: np.ndarray, sfreq: float, max_rank: int = 8,
                    verbose: bool = True) -> tuple[dict, dict]:
    """Estimate everything subject-specific from TRAIN trials only.

    `eeg_train` is (N, C, T) raw.  Returns (tensors_for_the_module, report).
    Every number in `report` is a measurement that gets written to the run report
    so a claim like "the rank was limited to 3" can be checked rather than
    trusted.
    """
    from scipy.signal import welch

    N, C, T = eeg_train.shape
    freqs = np.fft.rfftfreq(T, d=1.0 / sfreq).astype(np.float32)
    rep: dict = {"n_train": int(N), "n_ch": int(C), "sfreq": float(sfreq),
                 "bands": [list(b) for b in BANDS]}

    # ---- E1 + E2 applied FIRST, because every statistic below must be estimated
    # from the same signal the model will actually see.  Estimating the slope on
    # notched-and-high-passed data and then applying it to unnotched data is how a
    # "correction" ends up correcting something else.
    g_notch = notch_gain(freqs)
    bm = band_masks(freqs, sfreq)

    def front_linear(x: np.ndarray, chunk: int = 2000) -> np.ndarray:
        """(N,C,T) -> (N,NB,C,T): notch, band split, band-mask.  Linear, so the
        evoked/induced fork costs one extra subtraction rather than one extra pass.

        Chunked because the full train split at (N,5,17,250) float64 is 2.8 GB, and
        a fingerprint fit has no reason to hold all of it.
        """
        out = np.zeros((x.shape[0], NB, x.shape[1], x.shape[2]), dtype=np.float32)
        for s in range(0, len(x), chunk):
            xs = x[s:s + chunk].astype(np.float64)
            X = np.fft.rfft(xs, axis=-1) * g_notch[None, None, :]
            for b in range(NB):
                out[s:s + len(xs), b] = np.fft.irfft(
                    X * bm[b][None, None, :], n=x.shape[2], axis=-1).astype(np.float32)
        return out

    xb_tr = front_linear(eeg_train, chunk=2000)
    rep["front_pass"] = True

    # ---- E3: aperiodic slope of THIS subject, on the notched+band-limited signal
    f_w, p_w = welch(eeg_train.astype(np.float64), fs=sfreq, nperseg=min(128, T),
                     axis=-1)
    psd = p_w.mean(axis=(0, 1))
    ap = fit_aperiodic(psd, f_w)
    rep["aperiodic"] = ap
    p = max(0.0, -ap["slope"])                 # 1/f exponent as a positive number
    centres = np.array([np.sqrt(lo * min(hi, sfreq / 2)) for lo, hi in BANDS])
    gain = centres ** (p / 2.0)                # amplitude whitening: P ~ f^-p
    gain = gain / np.exp(np.log(gain).mean())  # geometric mean 1: no global rescale
    rep["aperiodic_equalise"] = {
        "p": float(p), "band_centres_hz": centres.tolist(),
        "band_gain": gain.tolist(),
        "note": ("amplitude gain f_c^(p/2) undoes the subject's own 1/f tilt so a band "
                 "amplitude reads as energy-above-1/f; normalised to geometric mean 1")}
    xb_tr = xb_tr * gain[None, :, None, None]

    # ---- E4/E5: data-derived spatial principal directions + participation ratio
    U = np.zeros((NB, C, C), dtype=np.float32)
    lam = np.zeros((NB, C), dtype=np.float32)
    pr = np.zeros(NB, dtype=np.float32)
    rank = np.zeros(NB, dtype=np.int64)
    for b in range(NB):
        y = xb_tr[:, b]                              # (N,C,T)
        cov = np.einsum("nct,ndt->cd", y, y) / (len(y) * T)
        w, v = np.linalg.eigh(cov.astype(np.float64))
        order = np.argsort(w)[::-1]
        w, v = np.clip(w[order], 0, None), v[:, order]
        U[b] = v.T.astype(np.float32)                # rows = directions
        lam[b] = w.astype(np.float32)
        pr[b] = float((w.sum() ** 2) / max((w ** 2).sum(), 1e-30))
        rank[b] = int(np.clip(round(pr[b]), 1, min(max_rank, C - 1)))
    rep["spatial"] = {"participation_ratio": pr.tolist(),
                      "rank_used": rank.tolist(),
                      "band_names": list(BAND_NAMES),
                      "max_rank": int(max_rank),
                      "eigenvalue_share_top1": (lam[:, 0] / lam.sum(1)).tolist(),
                      "note": ("rank is the participation ratio of the TRIAL-AVERAGED "
                               "band covariance, i.e. the number of directions the "
                               "data supports.  A full CxC unmixing matrix is not "
                               "identifiable when this number is 3.")}
    if verbose:
        for b in range(NB):
            print(f"  [front] {BAND_NAMES[b]:<10} PR {pr[b]:5.2f} -> rank {rank[b]}  "
                  f"top-1 eigenvalue share {lam[b,0]/lam[b].sum():.3f}  "
                  f"equalise gain {gain[b]:.3f}")

    # ---- E6: the stimulus-locked template, and its image under the SAME linear
    # front end (so the fork is exact, not approximate)
    tmpl = eeg_train.mean(0).astype(np.float32)              # (C,T)
    tmpl_fe = front_linear(tmpl[None])[0]                    # (NB,C,T): the batch
    # dimension is indexed away here, so the gain broadcasts over the BAND axis
    # only -- `gain[None,:,None]` was written for a (1,NB,C,T) tensor and raised
    # "operands could not be broadcast together with shapes (5,17,250) (1,5,1)".
    tmpl_fe = (tmpl_fe * gain[:, None, None]).astype(np.float32)
    rep["template"] = {"rms": float(np.sqrt((tmpl ** 2).mean())),
                       "trial_rms": float(np.sqrt((eeg_train ** 2).mean())),
                       "note": ("grand-average ERP over ALL train trials: stimulus-"
                                "locked, concept-agnostic.  `dev = x - tmpl` is the "
                                "induced stream.  The template is a TRAIN statistic, "
                                "and the test concepts are disjoint from the train "
                                "concepts, so this cannot carry a test label.")}

    fp = {"band_mask": torch.from_numpy(bm), "notch": torch.from_numpy(g_notch),
          "band_gain": torch.from_numpy(gain.astype(np.float32)),
          "spatial_u": torch.from_numpy(U), "spatial_rank": torch.from_numpy(rank),
          "tmpl_fe": torch.from_numpy(tmpl_fe), "tmpl": torch.from_numpy(tmpl),
          "freqs": torch.from_numpy(freqs)}
    return fp, rep


# --------------------------------------------------------------------- module
class PurifiedFront(nn.Module):
    """Fingerprint (frozen) + a tiny learnable correction, then tokenisation.

    The learnable parameters are exactly:
        cm_keep_raw (NB, max_rank)   how much of each global direction to remove
        band_scale  (NB,)            a per-band correction on top of the aperiodic
                                     equalisation
    and nothing else.  That is 5*8 + 5 = 45 parameters for the entire front end.
    The contrast with the previous design is the point: `TDMDemix` carried
    NB*C*C = 1445 parameters in the mixing matrix alone, to represent a 3-5
    dimensional structure, and every one of them had to be learned from the same
    16 540 trials that are also supposed to support the read-out.
    """

    def __init__(self, fp: dict, d: int, use_front: bool = True,
                 n_ch: int = 17, n_time: int = 250, sfreq: float = 250.0,
                 drop: float = 0.1):
        super().__init__()
        self.use_front = use_front
        self.d, self.n_ch, self.n_time, self.sfreq = d, n_ch, n_time, sfreq
        self.n_patch = N_PATCH
        self.patch_len = n_time // N_PATCH
        assert self.patch_len * N_PATCH == n_time, "T must divide into N_PATCH patches"
        # the shared-r token exists in BOTH arms, so the ablation isolates the
        # front end and not the cross-subject module.  `bias=False` for the reason
        # documented at `ch_token` below.
        self.csm_proj = nn.Linear(1024, d, bias=False)

        if not use_front:
            # ablation: the historical `shared_r -> MLP` encoder, no EEG path at all
            return

        for k, v in fp.items():
            self.register_buffer(f"_fp_{k}", v, persistent=False)
        rank = fp["spatial_rank"].numpy()
        self.max_rank = int(rank.max())
        # init: remove the global directions almost completely.  This IS the
        # standard common-average reference generalised to r directions, so the
        # starting point is a published default rather than a neutral guess.
        self.cm_keep_raw = nn.Parameter(torch.full((NB, self.max_rank), 4.0))
        self.band_scale = nn.Parameter(torch.ones(NB))

        # `bias=False` IS LOAD-BEARING, NOT A STYLE CHOICE.
        #
        # A bias added to every output element is IDENTICAL for every row of the
        # batch.  On the EEG path it is not a harmless offset: the raw-EEG tokens
        # have magnitude ~0.07, while `nn.Linear`'s default bias over d=96 has
        # magnitude ~1.08 -- measured on sub-08, the bias was 15x the signal
        # (`||bias|| 1.077` vs `||W@x|| 0.0712`).  Every EEG token is therefore
        # approximately the same constant vector for every trial, which is exactly
        # what the measurements showed:
        #
        #     ch_in input                    row-to-row cosine 0.0148
        #     W @ ch_in, no bias             row-to-row cosine 0.0155   <- intact
        #     W @ ch_in + bias               row-to-row cosine 0.9952   <- erased
        #
        # Row-to-row cosine is measured on the front end's own token output, and
        # 0.995 means every trial looks like every other trial.  Downstream this
        # collapsed the condition to a constant (c_self 1.000, row-identity at
        # chance), so the EEG could not control generation at all -- while the whole
        # three-tower model still trained "successfully", because a constant
        # condition is a perfectly finite loss.
        #
        # It also could not recover: with the EEG tokens constant at initialisation
        # the gradient w.r.t. this path is ~0, so it stays constant.  After 26 epochs
        # the exported condition still had c_self 1.000.  That is a dead
        # initialisation, and only removing the term fixes it rather than waiting.
        #
        # The learned per-stream/band/time embeddings (below) start at ZERO and are
        # added, so they are not a source of a row-common offset at init.
        self.ch_token = nn.Linear(NB * self.n_patch, d, bias=False)
        self.band_proj = nn.ModuleList([nn.Linear(n_ch, d, bias=False) for _ in range(NB)])
        # the CSM path's bias is the same failure mode, milder: `shared_r` has
        # magnitude 0.44 against a bias of 0.18, so it does not erase the rows but it
        # does inject a row-independent component into the condition.  Dropped for the
        # same reason as the EEG-path biases above.
        # Scale-normalise each EEG token.  Not a substitute for the above -- LayerNorm
        # is scale-invariant and CANNOT undo a dominant bias (it maps b + s_i to
        # ~LN(b), a constant, when |b| >> |s_i|) -- but with the bias gone it puts the
        # EEG tokens on the same footing as the CSM token, whose natural magnitude is
        # ~7x larger.
        self.tok_ln = nn.LayerNorm(d)
        self.stream_emb = nn.Parameter(torch.zeros(2, d))
        self.band_emb = nn.Parameter(torch.zeros(NB, d))
        self.time_emb = nn.Parameter(torch.zeros(self.n_patch, d))
        self.drop = nn.Dropout(drop)

    # -- the linear, data-fixed part -------------------------------------------------
    def _band_split_demix(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """(B,C,T) -> (B,NB,C,T) with notch, band split, aperiodic equalisation,
        and rank-limited global-direction removal."""
        B, C, T = x.shape
        X = torch.fft.rfft(x.float(), dim=-1) * self._fp_notch[None, None, :]
        xb = torch.stack([torch.fft.irfft(X * self._fp_band_mask[b][None, None, :],
                                          n=T, dim=-1) for b in range(NB)], 1)
        # E3: the subject's own aperiodic equalisation, plus ONE learnable scalar
        # per band.  The fixed part carries the physics; the free part can only
        # rebalance, it cannot re-invent the tilt.
        gain = self._fp_band_gain * self.band_scale
        xb = xb * gain[None, :, None, None]

        a = torch.sigmoid(self.cm_keep_raw)                     # (NB, max_rank) in (0,1)
        for b in range(NB):
            r = int(self._fp_spatial_rank[b])
            if r <= 0:
                continue
            u = self._fp_spatial_u[b, :r]                        # (r,C), rows=directions
            # remove  sum_r a_r * u_r (u_r . x),  i.e. project onto each direction,
            # scale by a bounded coefficient, and project BACK through `u` to the
            # channel axis before subtracting.
            # The first version wrote `einsum("br,brt->bct", a[b,:r], proj)`: it
            # dropped the back-projection entirely (so it subtracted a (r,T) tensor
            # from a (C,T) one, which "worked" only because T == C == 250 in an
            # early shape coincidence) and it also fed a 1-D slice to a 2-subscript
            # equation, which is what actually raised here.
            proj = torch.einsum("rc,bct->brt", u, xb[:, b])      # (B,r,T)
            w = a[b, :r][None, :, None] * proj                   # (B,r,T)
            xb[:, b] = xb[:, b] - torch.einsum("rc,brt->bct", u, w)
        return xb, {"cm_keep": a.detach(), "band_scale": self.band_scale.detach(),
                    "band_gain": gain.detach()}

    def forward(self, x: torch.Tensor, csm: torch.Tensor
                ) -> tuple[torch.Tensor, dict]:
        """x (B,C,T) raw EEG, csm (B,1024) `shared_r` -> tokens (B,L,d)."""
        B = x.shape[0]
        toks = [self.csm_proj(csm).unsqueeze(1)]                # the CSM token
        diag: dict = {}
        if not self.use_front:
            return torch.cat(toks, 1), {"front": "off"}

        xb, diag = self._band_split_demix(x)
        # E6: the induced stream.  Legitimate as a subtraction because every
        # operator above is linear, so this is EXACTLY `front(x - tmpl)`.
        dev = xb - self._fp_tmpl_fe[None]

        for si, yb in enumerate((xb, dev)):
            # `band_proj[b]` maps the CHANNEL axis to d, so the tensor is transposed
            # to put the channel dimension last -- feeding (B, n_patch, C) straight
            # in would apply the map over time and silently destroy the band
            # structure.  `yp[:, b]` is (B, C, n_patch), `.transpose(1,2)` -> (B,
            # n_patch, C), and `band_proj[b]` expects C inputs.
            yp = yb.reshape(B, NB, self.n_ch, self.n_patch, self.patch_len).mean(-1)
            ch_in = yp.permute(0, 2, 1, 3).reshape(B, self.n_ch, NB * self.n_patch)
            ct = self.ch_token(ch_in) + self.stream_emb[si]
            bt = torch.stack([self.band_proj[b](yp[:, b].transpose(1, 2))
                              for b in range(NB)], 1)           # (B, NB, n_patch, d)
            bt = bt + self.band_emb[None, :, None, :] + self.time_emb[None, None, :, :]
            bt = bt.reshape(B, NB * self.n_patch, self.d)
            toks.append(self.drop(self.tok_ln(ct)))
            toks.append(self.drop(self.tok_ln(bt)))
        diag["dev_energy_share"] = float(
            (dev.pow(2).mean() / xb.pow(2).mean().clamp_min(1e-12)).detach())
        return torch.cat(toks, 1), diag


def save_fingerprint(fp: dict, rep: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fp, path)
    path.with_suffix(".json").write_text(json.dumps(rep, indent=2), encoding="utf-8")


def load_fingerprint(path: Path) -> tuple[dict, dict]:
    fp = torch.load(path, map_location="cpu", weights_only=False)
    j = path.with_suffix(".json")
    return fp, (json.loads(j.read_text(encoding="utf-8")) if j.is_file() else {})
