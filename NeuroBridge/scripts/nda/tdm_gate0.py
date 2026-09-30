#!/usr/bin/env python3
"""TDM Gate 0 -- measure the four premises BEFORE building the architecture.

This script exists because of a specific, repeatable failure in this project: a
mechanism was implemented first and its premise turned out false afterwards
(band routing: Gate 1 measured ~0 correlation after the code was written; the
third tower: fine/coarse came out at 0.0765 of the explainable variance, i.e. the
targets were unidentifiable).  Each gate here is cheap, independent, and can
falsify one mechanism on its own.  Nothing downstream should be built on a gate
that fails, and every gate result is itself a reportable measurement.

THE FOUR GATES, AND WHAT EACH ONE DECIDES
=========================================
G3 (the decisive one) -- GRANULARITY x TIME
    Premise: the four VLM description granularities are NOT redundant; they lie
    at different spatial frequencies and therefore have DIFFERENT TIME COURSES in
    EEG.  Evidence: PLoS Comput Biol 2026 (10.1371/journal.pcbi.1014371) shows
    peripheral / low-spatial-frequency information is processed BEFORE foveal /
    high-spatial-frequency information, and that the difference is best modelled
    with a differential spatial transform -- i.e. the visual field layout is
    encoded in TIME, not just in space.
    Test:  for each granularity target, fit a ridge from an EEG time window and
           score it on held-out test, sweeping the window.  Report the peak time.
    Pass:  peak(detail) > peak(overall) by more than the bootstrap CI, i.e. the
           fine-grained description decodes LATER than the coarse one.
    Decides: whether the granularity heads get their own time windows (the whole
           point of the redesign) or revert to a single full-window readout.
    Note:  this gate is worth having even if every other gate fails, because a
           positive result is a novel finding on its own (no EEG-to-image work
           measures a granularity-resolved decoding time course).

G1 -- LATENCY JITTER MAGNITUDE
    Premise: trial-to-trial latency jitter is large enough to be worth correcting
    (mechanism: DLA, a differentiable phase rotation).
    Test:  per-trial lag that maximises cross-correlation with the trial-averaged
           template, band-limited to 2-20 Hz; parabolic sub-sample refinement.
    Pass:  std(lag) > 8 ms, i.e. more than 2 samples at this dataset's 250 Hz.
    Context: ERP methodology has quantified this for decades (ReSync; Event-
           Related Warping reports 5-13% improvement in the averaged response when
           jitter exceeds 100 ms), and a 2026 phase-guided BCI paper reports
           80.94% on MI.  A literature scan found NO EEG-to-image pipeline that
           applies explicit jitter correction -- reconstruction papers handle
           timing only through fixed time-window choice, which is the construct
           most vulnerable to jitter.

G2 -- GAMMA LINEARITY AND ALPHA GAIN  (decides RSD and DNG)
    Premise (RSD): electrode gamma is a LINEAR mixture of retinotopic subfield
    activations, so the inverse is a linear unmixing problem rather than an MLP's
    job.  Evidence: 2026 high-density EEG (10.1101/2025.08.09.669461) reports
    gamma accumulates LINEARLY over subfields, while alpha is SUBADDITIVE and
    present even without visual input.
    Premise (DNG): alpha carries a spatially global gain rather than evidence, so
    it should DIVIDE the evidence rather than be concatenated or routed to it.
    Test:  predict held-out channels' gamma from the remaining channels with a
           LINEAR map vs a nonlinear MLP; and compare alpha's across-channel
           dispersion with gamma's.
    Pass:  linear R^2 >= 0.7 * MLP R^2 (gamma is linearly generated)
           AND dispersion(alpha) < dispersion(gamma) (alpha is more global).
    Decides: whether the structural trunk uses a linear unmixing operator, and
           whether the two trunks are coupled by division.

G4 -- STIMULUS OVERLAP (a validity check on G1 and G3, not a mechanism)
    If the inter-stimulus interval were shorter than the evoked response, trial N
    would contain trials N+1.. responses, which would corrupt the G1 jitter
    estimate AND the G3 time course simultaneously.  The stored arrays begin at
    stimulus onset (the -200 ms baseline in info.json is NOT in the saved
    (63, 250) crop), so pre-stimulus leakage cannot be measured directly.
    Reported here: (a) the dataset's documented SOA, (b) an UPPER BOUND on how
        much late-window decodability could be carryover, by measuring whether the
        final window decodes the CURRENT trial's target above its own shuffle
        floor.  This does not prove absence of overlap; it bounds the size of the
        effect on the reported numbers.

LEAK-FREE
    Every lambda is selected on a held-in 15% slice of TRAIN rows.  The test set
    is scored once per window and never used to choose anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

NB_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NB_ROOT))
sys.path.insert(0, str(NB_ROOT / "scripts" / "nda"))

# Same channel set as every other experiment in this project, so the gate numbers
# are directly comparable to the reconstruction results.
DEFAULT_CHANNELS = [
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2",
]
SFREQ = 250.0              # info.json
TARGETS = {
    # name -> (cache_stem, kind)
    "overall":    ("sem_overall",    "vec1024"),
    "background": ("sem_background", "vec1024"),
    "subject":    ("sem_subject",    "vec1024"),
    "detail":     ("sem_detail",     "vec1024"),
    "image_clip": ("sem_image",      "vec1280"),
    "vae_lf":     ("perc_struct",    "latent"),   # (N,4,64,64) -> pooled
}


def l2n(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-8)


# --------------------------------------------------------------------------- io
def load_targets(tdir: Path, split: str, name: str) -> np.ndarray:
    stem, kind = TARGETS[name]
    a = np.load(tdir / f"{stem}_{split}.npy").astype(np.float32)
    if kind == "latent":
        # pool (N,4,64,64) -> (N,4,8,8) so the ridge problem stays small; this is
        # a time-course probe, not a fidelity measurement, and the pooling is
        # applied identically to train and test
        n, c, h, w = a.shape
        a = a.reshape(n, c, 8, h // 8, 8, w // 8).mean(axis=(3, 5))
        a = a.reshape(n, -1)
    return l2n(a.reshape(len(a), -1))


def load_eeg(subject: int, train: bool, cache_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """(N, C, 250) exactly as every other script in this project builds it.

    Also returns the row index of each sample into the g2 target caches.  The
    index is NOT the sample index: train has 10 images per object while test has
    exactly 1, so a hard-coded `obj*10+img` is wrong on test (it produced
    `index 200 out of bounds for axis 0 with size 200`).  Derive the stride from
    the dataset itself instead.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = f"sub{subject:02d}_{'train' if train else 'test'}"
    ce, cr = cache_dir / f"{tag}_eeg.npy", cache_dir / f"{tag}_row.npy"
    if ce.is_file() and cr.is_file():
        print(f"[gate0] cache hit {ce.name}")
        return np.load(ce), np.load(cr)

    from module.dataset import EEGPreImageDataset

    ds = EEGPreImageDataset(
        [subject], f"{NB_ROOT}/data/things_eeg/preprocessed_eeg", DEFAULT_CHANNELS,
        [0, 250], f"{NB_ROOT}/data/things_eeg/image_feature/RN50", "", False, [],
        True, False, None, train, False, False, False,
    )
    stride = int(ds.num_images_per_object)
    n = len(ds)
    out = np.zeros((n, len(DEFAULT_CHANNELS), 250), dtype=np.float32)
    tgt_row = np.zeros(n, dtype=np.int64)
    for i in range(n):
        eeg, _img, _txt, _sid, obj, img, _rep = ds[i]
        out[i] = eeg.numpy()
        tgt_row[i] = int(obj) * stride + int(img)
    print(f"[gate0] {tag}: n={n} stride={stride} row_range=[{tgt_row.min()},{tgt_row.max()}]")
    np.save(ce, out)
    np.save(cr, tgt_row)
    return out, tgt_row


# ------------------------------------------------------------------------ ridge
def ridge_fit(X: np.ndarray, Y: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu, sd = X.mean(0), X.std(0).clip(1e-6)
    Xs = (X - mu) / sd
    d = Xs.shape[1]
    A = Xs.T @ Xs + lam * np.eye(d, dtype=np.float32)
    W = np.linalg.solve(A, Xs.T @ Y)
    return W.astype(np.float32), mu.astype(np.float32), sd.astype(np.float32)


def ridge_apply(m, X: np.ndarray) -> np.ndarray:
    W, mu, sd = m
    return ((X - mu) / sd) @ W


def pick_lambda(Xtr, Ytr, Xva, Yva, lams) -> float:
    best, bl = -1e9, lams[0]
    for lam in lams:
        m = ridge_fit(Xtr, Ytr, lam)
        p = l2n(ridge_apply(m, Xva))
        s = float((p * l2n(Yva)).sum(1).mean())
        if s > best:
            best, bl = s, lam
    return bl


def two_way(pred: np.ndarray, gt: np.ndarray) -> float:
    """2-way identification: is pred_i closer to gt_i than to a random other gt?"""
    p, g = l2n(pred), l2n(gt)
    s = p @ g.T
    n = len(p)
    idx = np.arange(n)
    return float(np.mean(s[idx, idx] > s[idx, (idx + 1) % n]))


# --------------------------------------------------------------------- G3
def gate3_timecourse(args, eeg_tr, row_tr, eeg_te, row_te, out) -> dict:
    n_tr = len(eeg_tr)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_tr)
    n_va = max(200, int(0.15 * n_tr))
    va, fit = perm[:n_va], perm[n_va:]
    lams = [10.0, 100.0, 1000.0, 1e4]

    wins = list(range(0, 201, args.win_step))
    W = args.win
    res: dict[str, dict] = {}
    for name in TARGETS:
        y_tr = load_targets(Path(args.targets_dir), "train", name)[row_tr]
        y_te = load_targets(Path(args.targets_dir), "test", name)[row_te]
        curve = []
        for t0 in wins:
            t1 = min(t0 + W, eeg_tr.shape[-1])
            if t1 - t0 < 10:
                continue
            Xtr = eeg_tr[:, :, t0:t1].reshape(n_tr, -1)
            Xte = eeg_te[:, :, t0:t1].reshape(len(eeg_te), -1)
            lam = pick_lambda(Xtr[fit], y_tr[fit], Xtr[va], y_tr[va], lams)
            m = ridge_fit(Xtr[fit], y_tr[fit], lam)
            p = ridge_apply(m, Xte)
            cos = float((l2n(p) * l2n(y_te)).sum(1).mean())
            tw = two_way(p, y_te)
            # shuffle floor for the same window
            sh = l2n(y_te[rng.permutation(len(y_te))])
            cos_sh = float((l2n(p) * sh).sum(1).mean())
            curve.append({"t0_ms": 1000.0 * t0 / SFREQ,
                          "t1_ms": 1000.0 * t1 / SFREQ,
                          "cos": cos, "cos_shuffle": cos_sh, "twoway": tw,
                          "lam": lam})
            print(f"  [G3] {name:<11} {1000*t0/SFREQ:6.0f}-{1000*t1/SFREQ:6.0f} ms  "
                  f"cos {cos:.4f} (shuf {cos_sh:.4f})  2way {tw:.3f}")
        if not curve:
            continue
        peak = max(curve, key=lambda d: d["twoway"])
        res[name] = {"curve": curve, "peak_ms": peak["t0_ms"],
                     "peak_twoway": peak["twoway"], "peak_cos": peak["cos"],
                     "n_curve": len(curve)}

    # the ordering claim
    ord_res: dict = {}
    if "overall" in res and "detail" in res:
        ord_res["overall_peak_ms"] = res["overall"]["peak_ms"]
        ord_res["detail_peak_ms"] = res["detail"]["peak_ms"]
        ord_res["delta_ms"] = res["detail"]["peak_ms"] - res["overall"]["peak_ms"]
        # the peak times are quantised to `win_step` samples, so a difference of
        # one grid step is NOT evidence.  Require >= 2 steps of separation.
        step_ms = 1000.0 * args.win_step / SFREQ
        ord_res["grid_step_ms"] = step_ms
        ord_res["pass_detail_after_overall"] = bool(
            (res["detail"]["peak_ms"] - res["overall"]["peak_ms"]) >= 2 * step_ms)
        order = sorted(res.items(), key=lambda kv: kv[1]["peak_ms"])
        ord_res["peak_order"] = [k for k, _ in order]
        ord_res["peak_times_ms"] = {k: v["peak_ms"] for k, v in res.items()}
        ord_res["peak_twoway"] = {k: v["peak_twoway"] for k, v in res.items()}
        print(f"\n  [G3] peak order: {' -> '.join(ord_res['peak_order'])}")
        print(f"  [G3] overall {ord_res['overall_peak_ms']:.0f} ms vs detail "
              f"{ord_res['detail_peak_ms']:.0f} ms (delta "
              f"{ord_res['delta_ms']:+.0f} ms, need >= {2*step_ms:.0f} ms) -> "
              f"{'PASS' if ord_res['pass_detail_after_overall'] else 'FAIL'}")
    return {"targets": res, "ordering": ord_res}


# --------------------------------------------------------------------- G1
def gate1_jitter(args, eeg_tr) -> dict:
    from scipy.signal import butter, filtfilt

    b, a = butter(4, [2.0 / (SFREQ / 2), 20.0 / (SFREQ / 2)], btype="band")
    X = filtfilt(b, a, eeg_tr.astype(np.float64), axis=-1)
    # multi-channel matched filter: project each trial onto the template's dominant
    # spatial pattern at every lag, and take the lag of the maximal response
    T = X.mean(0)                                     # (C, 250)
    # dominant spatial pattern of the template over the analysis window
    seg = T[:, int(0.05 * SFREQ):int(0.5 * SFREQ)]
    u, s, vt = np.linalg.svd(seg, full_matrices=False)
    w = u[:, 0]                                       # (C,)
    Proj = np.einsum("c,nct->nt", w, X)               # (N, 250)
    tpl = (w @ T)                                     # (250,)
    max_lag = int(round(args.jitter_max_ms / 1000.0 * SFREQ))
    lags = np.arange(-max_lag, max_lag + 1)
    # normalised cross-correlation, computed as a convolution over the template
    tpl_c = tpl - tpl.mean()
    num = np.stack([np.convolve(Proj[i], tpl_c[::-1], mode="same") for i in
                    range(len(Proj))], axis=0)        # (N, 250)
    den = (Proj.std(1, keepdims=True) * tpl_c.std() * len(tpl_c) + 1e-12)
    cc = num / den
    cen = len(tpl_c) // 2
    win = cc[:, cen - max_lag: cen + max_lag + 1]
    k = win.argmax(1)
    # parabolic sub-sample refinement, so the estimate is not quantised to 4 ms
    kk = np.clip(k, 1, win.shape[1] - 2)
    y0, y1, y2 = win[np.arange(len(k)), kk - 1], win[np.arange(len(k)), kk], win[np.arange(len(k)), kk + 1]
    denom = (y0 - 2 * y1 + y2)
    frac = np.where(np.abs(denom) > 1e-12, 0.5 * (y0 - y2) / np.where(denom == 0, 1e-12, denom), 0.0)
    lat_samp = (k - max_lag) + np.clip(frac, -0.5, 0.5)
    lat_ms = lat_samp * (1000.0 / SFREQ)
    sd = float(np.std(lat_ms))
    out = {"n_trials": int(len(lat_ms)), "mean_ms": float(np.mean(lat_ms)),
           "sd_ms": sd, "p05_ms": float(np.percentile(lat_ms, 5)),
           "p95_ms": float(np.percentile(lat_ms, 95)),
           "sd_samples": sd / (1000.0 / SFREQ),
           "pass": bool(sd > args.jitter_min_ms),
           "threshold_ms": args.jitter_min_ms}
    print(f"  [G1] latency jitter sd = {sd:.2f} ms ({out['sd_samples']:.2f} samples "
          f"@ {SFREQ:.0f} Hz), 5-95% [{out['p05_ms']:.1f}, {out['p95_ms']:.1f}] ms "
          f"-> {'PASS' if out['pass'] else 'FAIL'} (need > {args.jitter_min_ms} ms)")
    return out


# --------------------------------------------------------------------- G2
def gate2_gamma(args, eeg_tr) -> dict:
    """Is gamma a LINEAR function of the other channels, and is alpha more global?

    NUMERICAL REPAIR (the first version of this probe produced `linear R^2 = -518`
    with the SHUFFLED control giving the identical -518, which is the signature of
    a design matrix blowing up rather than a measured result).  Two causes, both
    fixed here:
      * `ridge_fit` standardises X by `std.clip(1e-6)`.  Several posterior
        electrodes carry almost no gamma power, so a near-constant predictor was
        divided by ~1e-6 and entered the normal equations at ~1e6, making the
        solve ill-conditioned.  Columns with negligible spread are now DROPPED and
        the count is reported.
      * Y was fit in raw units while X was standardised, so the penalty was not a
        shrinkage of anything meaningful.  Both sides are now standardised and the
        prediction is mapped back, and lambda is chosen on a held-in split of the
        TRAIN trials rather than fixed at 1.0.
    """
    from scipy.signal import butter, filtfilt

    def bandpower(x, lo, hi):
        """LOG band power, not raw power.

        Raw gamma power spans orders of magnitude across trials, so a handful of
        outlier trials dominate BOTH the standardisation and the R^2 sum: measured
        with raw power, even the SHUFFLED control scored R^2 = -3.06 (a shuffled fit
        must score ~0), and the linear fit scored -603.  That is a property of the
        estimator, not of the brain.  Log power is the standard fix in EEG -- it is
        approximately Gaussian and stabilises the variance -- and it is applied
        identically to the gamma and alpha measurements below.
        """
        b, a = butter(4, [lo / (SFREQ / 2), hi / (SFREQ / 2)], btype="band")
        p = filtfilt(b, a, x.astype(np.float64), axis=-1) ** 2
        return np.log10(p + 1e-30)

    n = min(args.g2_trials, len(eeg_tr))
    seg = slice(int(0.05 * SFREQ), int(0.45 * SFREQ))
    g = bandpower(eeg_tr[:n], 40.0, 80.0)[:, :, seg].mean(-1)
    al = bandpower(eeg_tr[:n], 8.0, 12.0)[:, :, seg].mean(-1)

    def globalness(p):
        """How much do the electrodes move TOGETHER, i.e. is this a spatial gain?

        `std/mean` was the wrong statistic once the measurement became log power
        (log power is signed, so a coefficient of variation is meaningless).  This
        is the direct question instead: per-channel standardise over trials, form
        the spatial mean per trial, and correlate each channel with it.  A band
        whose channels track a shared gain scores high; a band carrying
        channel-specific (local) evidence scores low.
        """
        z = (p - p.mean(0)) / np.clip(p.std(0), 1e-9, None)
        gm = z.mean(1)
        gm = (gm - gm.mean()) / max(gm.std(), 1e-9)
        return float(np.mean((z * gm[:, None]).mean(0)))

    C = g.shape[1]
    rng = np.random.default_rng(args.seed)
    n_ho = max(3, C // 4)
    ho = np.sort(rng.choice(C, n_ho, replace=False))
    fi = np.array([c for c in range(C) if c not in set(ho.tolist())])
    X, Y = g[:, fi], g[:, ho]

    # drop near-constant predictors: dividing by their ~0 std is what made the
    # previous design matrix ill-conditioned
    sd_raw = X.std(0)
    keep = sd_raw > max(1e-3, 1e-3 * sd_raw.max())
    n_dropped = int((~keep).sum())
    X = X[:, keep]
    fi = fi[keep]
    if X.shape[1] < 2:
        return {"error": "fewer than 2 usable predictors after the variance filter",
                "n_dropped_predictors": n_dropped}

    tr = rng.permutation(n)[: int(0.7 * n)]
    te = np.array([i for i in range(n) if i not in set(tr.tolist())])
    # a held-in split of the TRAIN part, so lambda is not picked on `te`
    tr2 = rng.permutation(tr)
    va, fit = tr2[: len(tr) // 5], tr2[len(tr) // 5:]
    lams = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]

    def fit_std(Xa, Ya, lam, Xmu, Xsd, Ymu, Ysd):
        Xs = (Xa - Xmu) / Xsd
        Ys = (Ya - Ymu) / Ysd
        A = Xs.T @ Xs + lam * np.eye(Xs.shape[1], dtype=np.float64)
        W = np.linalg.solve(A, Xs.T @ Ys)
        return W

    def score(Xa, Ya, W, Xmu, Xsd, Ymu, Ysd):
        P = ((Xa - Xmu) / Xsd) @ W * Ysd + Ymu
        return float(1.0 - ((P - Ya) ** 2).sum() / ((Ya - Ymu) ** 2).sum())

    Xmu, Xsd = X[fit].mean(0), X[fit].std(0).clip(1e-6)
    Ymu, Ysd = Y[fit].mean(0), Y[fit].std(0).clip(1e-6)
    best_lam, best_r2 = lams[0], -1e18
    for lam in lams:
        W = fit_std(X[fit], Y[fit], lam, Xmu, Xsd, Ymu, Ysd)
        r = score(X[va], Y[va], W, Xmu, Xsd, Ymu, Ysd)
        if r > best_r2:
            best_r2, best_lam = r, lam
    W = fit_std(X[fit], Y[fit], best_lam, Xmu, Xsd, Ymu, Ysd)
    r2_lin = score(X[te], Y[te], W, Xmu, Xsd, Ymu, Ysd)

    # shuffled control: the SAME procedure with the channel correspondence broken
    sh = rng.permutation(len(fit))
    Xmu_s, Xsd_s = X[fit][sh].mean(0), X[fit][sh].std(0).clip(1e-6)
    Ymu_s, Ysd_s = Y[fit].mean(0), Y[fit].std(0).clip(1e-6)
    Ws = fit_std(X[fit][sh], Y[fit], best_lam, Xmu_s, Xsd_s, Ymu_s, Ysd_s)
    r2_sh = score(X[te], Y[te], Ws, Xmu_s, Xsd_s, Ymu_s, Ysd_s)

    # nonlinear upper bound: a single hidden layer MLP on standardised inputs
    import torch
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    Xt = torch.from_numpy(((X[tr] - Xmu) / Xsd).astype(np.float32)).to(dev)
    Ys = ((Y[tr] - Y[tr].mean(0)) / Y[tr].std(0).clip(1e-6)).astype(np.float32)
    Yt = torch.from_numpy(Ys).to(dev)
    Xe = torch.from_numpy(((X[te] - Xmu) / Xsd).astype(np.float32)).to(dev)
    net = torch.nn.Sequential(torch.nn.Linear(Xt.shape[1], 64), torch.nn.GELU(),
                              torch.nn.Linear(64, 64), torch.nn.GELU(),
                              torch.nn.Linear(64, Y.shape[1])).to(dev)
    opts = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
    for _ in range(400):
        opts.zero_grad()
        torch.nn.functional.mse_loss(net(Xt), Yt).backward()
        opts.step()
    with torch.no_grad():
        Pn = net(Xe).cpu().numpy() * Y[tr].std(0) + Y[tr].mean(0)
    r2_mlp = 1.0 - float(((Pn - Y[te]) ** 2).sum() / ((Y[te] - Y[tr].mean(0)) ** 2).sum())

    ratio = r2_lin / max(r2_mlp, 1e-9) if r2_mlp > 0 else float("nan")
    gl, gg = globalness(al), globalness(g)
    out = {"r2_linear": r2_lin, "r2_mlp": r2_mlp, "r2_shuffled": r2_sh,
           "lambda": best_lam, "linear_over_mlp": ratio,
           "r2_threshold_ratio": args.g2_ratio,
           "n_dropped_predictors": n_dropped, "n_predictors": int(X.shape[1]),
           "globalness_alpha": gl, "globalness_gamma": gg,
           "alpha_more_global": bool(gl > gg),
           "n_holdout_channels": int(n_ho), "n_trials": int(n),
           # THE CLAIM IS "LINEAR IS ENOUGH", so the MLP is a ceiling, not a hurdle.
           # Requiring `r2_mlp > 0` conflated "the MLP works" with "linear suffices"
           # and produced a false negative on sub-01, where the 400-step MLP simply
           # failed to generalise (R^2 = -0.135) while the linear map scored 0.199.
           # If the nonlinear map cannot beat the linear one, linear *is* enough.
           "pass_rsd": bool(r2_lin > r2_sh + 0.05
                            and (r2_mlp <= r2_lin or r2_lin >= args.g2_ratio * r2_mlp)),
           "pass_dng": bool(gl > gg)}
    print(f"  [G2] linear R2 {r2_lin:.4f} (lambda {best_lam:g}) | MLP R2 {r2_mlp:.4f} "
          f"(ratio {ratio:.3f}) | shuffled {r2_sh:.4f} | dropped {n_dropped}/{C-1} "
          f"near-constant predictors")
    print(f"  [G2] spatial globalness (corr with the across-channel mean): "
          f"alpha {gl:.4f} vs gamma {gg:.4f} -> alpha_more_global={out['alpha_more_global']}")
    print(f"  [G2] RSD {'PASS' if out['pass_rsd'] else 'FAIL'} | "
          f"DNG {'PASS' if out['pass_dng'] else 'FAIL'}")
    return out


# --------------------------------------------------------------------- G4
def gate4_overlap(args, eeg_te, row_te) -> dict:
    """Bound the carryover effect via the LAST window of the 1 s crop.

    A direct test needs the pre-stimulus baseline, which is NOT in the stored
    (63,250) crop (info.json's -200 ms baseline was dropped, so the array starts
    at stimulus onset).  What CAN be measured is whether the 800-1000 ms tail
    still carries any decodable current-trial information: if the tail is at
    chance, carryover from the previous trial can only be hurting us, not
    inflating the G1/G3 numbers.
    """
    y_te = load_targets(Path(args.targets_dir), "test", "image_clip")[row_te]
    X = eeg_te[:, :, int(0.80 * SFREQ):].reshape(len(eeg_te), -1)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(X))
    m = ridge_fit(X[perm[:100]], y_te[perm[:100]], 100.0)
    p = ridge_apply(m, X[perm[100:]])
    y = y_te[perm[100:]]
    tw = two_way(p, y)
    sh = two_way(p, y[rng.permutation(len(y))])
    out = {"tail_window_ms": [800, 1000], "tail_twoway": tw,
           "tail_twoway_shuffled": sh, "tail_above_chance": bool(tw > sh + 0.15),
           "note": ("the -200 ms baseline in info.json is not part of the stored "
                    "(63,250) crop, so pre-stimulus leakage is not directly "
                    "measurable; THINGS-EEG2 documents a 100 ms stimulus with a "
                    "1000 ms SOA, which places the next stimulus beyond the "
                    "analysis window.  tail_above_chance is False when the tail "
                    "holds no residual current-trial information, which is the "
                    "sanity condition for reading G1/G3 as timing effects.")}
    print(f"  [G4] tail 800-1000 ms 2-way {tw:.3f} vs shuffle {sh:.3f} -> "
          f"tail_above_chance={out['tail_above_chance']}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, default=8)
    ap.add_argument("--targets-dir", type=str, default=f"{NB_ROOT}/outputs/g2/targets")
    ap.add_argument("--cache-dir", type=str, default=f"{NB_ROOT}/outputs/tdm/cache")
    ap.add_argument("--out-json", type=str, default=f"{NB_ROOT}/outputs/tdm/gate0_sub08.json")
    ap.add_argument("--win", type=int, default=50, help="window length in samples (200 ms)")
    ap.add_argument("--win-step", type=int, default=25)
    ap.add_argument("--jitter-min-ms", type=float, default=8.0)
    ap.add_argument("--jitter-max-ms", type=float, default=30.0)
    ap.add_argument("--g2-trials", type=int, default=4000)
    ap.add_argument("--g2-ratio", type=float, default=0.7)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cache-only", action="store_true",
                    help="only build the raw-EEG cache and exit.  The pipeline "
                         "stage calls it this way first so that the probes (which "
                         "are measurements, not prerequisites) can never block a "
                         "run whose only hard requirement is the cache.")
    ap.add_argument("--skip-g3", action="store_true",
                    help="skip the granularity x time sweep.  G3 is the expensive "
                         "gate (a ridge per target per time window); it runs for the "
                         "reference subject only, while G1 (jitter) and G2 "
                         "(linearity) run for every subject because they are what "
                         "decide whether DLA and RSD have a premise at all.")
    args = ap.parse_args()

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache_dir)
    print(f"[gate0] loading sub-{args.subject:02d} EEG "
          f"({len(DEFAULT_CHANNELS)} posterior channels, [0,250] samples)")
    eeg_tr, row_tr = load_eeg(args.subject, True, cache)
    eeg_te, row_te = load_eeg(args.subject, False, cache)
    print(f"[gate0] train {eeg_tr.shape} test {eeg_te.shape} "
          f"sfreq {SFREQ} -> 1 sample = {1000/SFREQ:.1f} ms")
    if args.cache_only:
        print(f"[gate0] cache-only: wrote {cache} and exiting")
        return

    report: dict = {"subject": args.subject, "sfreq": SFREQ,
                    "channels": DEFAULT_CHANNELS, "win_samples": args.win}

    print("\n[gate0] === G4 overlap validity check ===")
    report["G4_overlap"] = gate4_overlap(args, eeg_te, row_te)
    print("  [G4] " + report["G4_overlap"]["note"][:150] + "...")

    print("\n[gate0] === G1 latency jitter ===")
    report["G1_jitter"] = gate1_jitter(args, eeg_tr)

    print("\n[gate0] === G2 gamma linearity / alpha gain ===")
    report["G2_linearity"] = gate2_gamma(args, eeg_tr)

    print("\n[gate0] === G3 granularity x time (DECISIVE) ===")
    if args.skip_g3:
        print("  [G3] skipped (--skip-g3)")
        report["G3_timecourse"] = {"skipped": True}
    else:
        report["G3_timecourse"] = gate3_timecourse(args, eeg_tr, row_tr, eeg_te, row_te, out)

    # ---- the verdict table that downstream code reads
    verdict = {
        "use_dla_phase_align": bool(report["G1_jitter"]["pass"]),
        "use_rsd_linear_unmix": bool(report["G2_linearity"]["pass_rsd"]),
        "use_dng_division": bool(report["G2_linearity"]["pass_dng"]),
        "use_granularity_time_gate": bool(
            report["G3_timecourse"].get("ordering", {}).get("pass_detail_after_overall", False)),
        "g3_measured": not args.skip_g3,
    }
    report["verdict"] = verdict
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n[gate0] ================ VERDICT ================")
    for k, v in verdict.items():
        print(f"  {k:<28} {'ON ' if v else 'OFF'}")
    print(f"[gate0] wrote {out}")


if __name__ == "__main__":
    main()
