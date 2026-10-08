#!/usr/bin/env python
"""GATE for the order-2 training architecture (v13/CROMA): does the pooled-metric estimator
have headroom in EITHER of its two error components?

WHY THIS RUNS BEFORE ANY TRAINING CODE. The architecture in the (unwritten) v13 design says the
system estimates ONE object -- the concept metric D -- and that object's error decomposes into
exactly three terms:

    IMSE(D*) = Bias(D*)^2  +  sigma^2 / K_eff ,      K_eff = 1 / (1 + (K-1) rho)

    * Bias   -- the systematic per-view error against the true shared metric. The best proxy we
                hold for the true metric is the GALLERY metric D_g (measured to be the richest
                view: corr(D_eeg, D_g) = M2 = 0.593). A training term can only help if a
                single view's metric is materially worse than the pooled one.
    * rho    -- the intra-concept view-error correlation. If it is ~0 the view errors are already
                independent, pooling is already efficient, and the variance lever is exhausted.
    * rank   -- degeneracy. A "better" bias number bought by rank collapse is a loss (v12: metric
                rank 2-3, raw cosine -13.85pp, headline -20.15pp).

So there are exactly two questions, both answerable on CACHED geometry with ONE forward pass of a
banked checkpoint, and this script answers both WITHOUT labels and WITHOUT retrieval:

    Q1 (bias)  corr(D_{R'=1}, D_g)  vs  corr(D_{R'=80}, D_g)   -> headroom iff the gap is large
    Q2 (rho)   split-half reliability of two DISJOINT rep halves as a function of how many reps
               each half averages -> rho = slope/intercept of the linearised curve

and one CONTROL on each: destroy the concept correspondence (shuffle the concept axis) and both
curves must collapse to ~0. A gain that survives the shuffle is the M1 index-leakage channel, and
this script must say so rather than report a headroom that is an artefact of the estimator.

THE rho ESTIMATOR, AND WHY IT IS IDENTIFIABLE WITHOUT TRUTH. Let the view error be a shared part c
(var rho*sigma^2, carried by EVERY view) plus an independent part e_i (var (1-rho)*sigma^2). The
average of m views has error variance rho*sigma^2 + (1-rho)*sigma^2/m, so for two DISJOINT halves
A, B of the repetitions -- whose INDEPENDENT parts are independent given the shared component --

    corr(Dbar_m^(A), Dbar_m^(B)) = V_s / ( V_s + rho*sigma^2 + (1-rho)*sigma^2 / m ),

with V_s = Var(mu) the signal. Inverting,

    1/corr(m) - 1 = (rho*sigma^2/V_s) + ((1-rho)*sigma^2/V_s) * (1/m),

which is LINEAR IN 1/m with intercept alpha and slope beta. Hence rho = alpha / (alpha + beta): a
one-variable least-squares fit gives rho with no knowledge of the truth and no assumption about the
gallery. The fit is only trusted when BOTH alpha, beta > 0 and R^2 > 0.9 (a negative component means
the exchangeable model is rejected, and rho is then reported as 0 with that fact recorded, not
silently clipped). The Spearman-Brown curve implied by the m=1 reliability is reported alongside as
a second, distribution-free check. K_eff(m) = m / (1 + (m-1) rho) is the number of INDEPENDENT
views the cloud is actually worth, which is the quantity the whole architecture is about.

Run (Slurm, per AGENTS.md 3.1; needs a GPU only for the embedding forward pass):
    sbatch slurm/probe_view_independence.sbatch
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from samclip import calibration, config, evaluate  # noqa: E402
from samclip.data import things_eeg  # noqa: E402
from samclip.data.targets import load_target_stack  # noqa: E402
from samclip.models import build_model  # noqa: E402


# --------------------------------------------------------------------------- geometry
def _norm(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-9, None)


def _mm_np(x: np.ndarray) -> np.ndarray:
    """Per-view moment match -- the same normalisation `scripts/probe_fusion.py` applies before
    comparing metrics across subjects whose encodings differ by an affine frame. Used ONLY in the
    bottleneck test, so its two arms are compared like-for-like in one frame."""
    return (x - x.mean(0, keepdims=True)) / (x.std(0, keepdims=True) + 1e-8)


def _metric(x: np.ndarray) -> np.ndarray:
    """Standardised squared chordal distance -- EXACTLY `calibration._sq_cos_dist`."""
    return calibration._sq_cos_dist(np.asarray(x, dtype=np.float64))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    iu = np.triu_indices(a.shape[0], 1)
    x, y = a[iu], b[iu]
    sx, sy = x.std(), y.std()
    if sx < 1e-12 or sy < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _effrank(m: np.ndarray) -> float:
    ev = np.linalg.svd(m, compute_uv=False)
    s = ev.sum()
    return float(s * s / max(float((ev ** 2).sum()), 1e-12))


def _reliability_fuse(views: list[np.ndarray]) -> tuple[np.ndarray, dict]:
    """v11's estimator, verbatim: LOO inverse-residual weights over the block metrics."""
    rel = []
    for b in range(len(views)):
        others = np.mean([views[j] for j in range(len(views)) if j != b], axis=0)
        rel.append(float(np.linalg.norm(views[b] - others) / max(float(np.linalg.norm(others)), 1e-12)))
    inv = 1.0 / (np.asarray(rel) + 1e-6)
    wgt = inv / inv.sum()
    fused = np.tensordot(wgt, np.asarray(views), axes=(0, 0))
    return fused, {
        "weights": [float(v) for v in wgt],
        "resid": rel,
        "weight_range": float(wgt.max() - wgt.min()),
        "is_flat": bool(wgt.max() - wgt.min() < 1e-3),
    }


def _shuffle_concepts(a: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Destroy the concept correspondence, keep the geometry (the M1 control)."""
    p = rng.permutation(a.shape[0])
    return a[p]


# --------------------------------------------------------------------------- per fold
def _bias_curve(zf: np.ndarray, dg: np.ndarray, Rs: list[int],
                rng: np.random.Generator) -> tuple[list[float], list[float]]:
    vals, shuf = [], []
    for rp in Rs:
        dq = _metric(_norm(zf[:, :rp].mean(axis=1)))
        vals.append(_corr(dq, dg))
        s = []
        for _ in range(5):
            s.append(_corr(_metric(_norm(_shuffle_concepts(zf[:, :rp].mean(axis=1), rng))), dg))
        shuf.append(float(np.mean(s)))
    return vals, shuf


def _gallery_reliability(stack: np.ndarray) -> dict:
    """Split-half reliability of the RAW image-layer ensemble, as a diagnostic.

    The cached stack is `(C, I, K, D)`; the only independent-ish "views" of one concept are the
    `K` ViT layers (and `I` images, which is 1 on the test split). Adjacent ViT layers are highly
    correlated, so this is an UPPER BOUND on what independent views would give, and it is labelled
    as such rather than used as the attenuation denominator. Reported for completeness; the
    decisive bottleneck test is `_source_vs_target`.
    """
    a = np.asarray(stack, dtype=np.float64)
    C, I, K, D = a.shape
    v = a.reshape(C, I * K, D)
    V = int(v.shape[1])
    out: dict = {"n_views": V}
    for m in range(1, V // 2 + 1):
        da = _metric(_norm(v[:, :m].mean(axis=1)))
        db = _metric(_norm(v[:, m:2 * m].mean(axis=1)))
        out[f"split_half_m{m}"] = _corr(da, db)
    return out


def _source_vs_target(z: np.ndarray, src_means: np.ndarray, dg_normed: np.ndarray,
                      seed: int) -> dict:
    """THE DECISIVE BOTTLENECK TEST.

    `corr(D_eeg, D_g) = 0.572` (whitened) / `0.640` (encoder frame) may already BE the cross-modal
    ceiling (M2 = 0.593 is the same object: one subject's full cloud vs the gallery metric). If so,
    no amount of EEG-side denoising moves the deployed number, and an order-2 training term aimed
    at the EEG side is futile regardless of how nice its loss curve looks.

    Test: build an EEG metric that is essentially noise-free -- the pool of the 9 SOURCE subjects'
    concept-metric estimates (9 x 80 = 720 repetitions' worth of averaging, and 9 independent
    encodings) -- and compare its agreement with the SAME gallery metric against the target's own
    80-repetition estimate. Both arms are moment-matched so the comparison is like-for-like and
    neither is favoured by a frame.

      * if corr(source9, D_g) ~= corr(target80, D_g), the EEG side is SATURATED -> the residual is
        cross-modal, and the only movable side is the GALLERY;
      * if corr(source9, D_g) >> corr(target80, D_g), the per-subject EEG estimate is the
        bottleneck and EEG-side training has headroom.

    WHAT THIS ARM IS AND IS NOT. It is an **estimability reference** for how much of the residual is
    subject-specific bias: it is NOT a deployable operator, because the source subjects' test
    concept means are INDEX-ALIGNED to the same 200 concepts and using them directly is the M1
    leakage channel that turned +15.37pp into -14.5pp under a concept permutation. The permutation
    control below is therefore mandatory: a pool that keeps its 0.80 under a concept shuffle is
    leaking the index, not measuring the metric. The TRAINING term this probe authorises is
    unaffected -- it anchors on `D_g`, which comes from images and carries no EEG index.
    """
    rng = np.random.default_rng(seed)
    V = [_metric(_norm(_mm_np(np.asarray(src_means[i], dtype=np.float64))))
         for i in range(src_means.shape[0])]
    pooled = np.mean(V, axis=0)
    tgt = _metric(_norm(_mm_np(np.asarray(z, dtype=np.float64).mean(axis=1))))
    shuf = []
    for _ in range(5):
        sv = [_metric(_norm(_mm_np(_shuffle_concepts(
            np.asarray(src_means[i], dtype=np.float64), rng))))
            for i in range(src_means.shape[0])]
        shuf.append(_corr(np.mean(sv, axis=0), dg_normed))
    return {
        "corr_target80_with_gallery": float(_corr(tgt, dg_normed)),
        "corr_source9pool_with_gallery": float(_corr(pooled, dg_normed)),
        "corr_source9pool_shuffled_control": float(np.mean(shuf)),
        "source_minus_target": float(_corr(pooled, dg_normed) - _corr(tgt, dg_normed)),
        "n_source_subjects": int(src_means.shape[0]),
    }


def probe_fold(z: np.ndarray, g: np.ndarray, shrink: float, fuse_blocks: int,
               n_split: int, seed: int, stack: np.ndarray | None = None,
               src_means: np.ndarray | None = None) -> dict[str, Any]:
    """`z` is the target's (C, R, d) per-trial cloud; `g` the (C, d) gallery."""
    C, R, d = z.shape
    rng = np.random.default_rng(seed)

    # --- the deployment frame: query cloud whitened by its OWN SAW map, gallery normalised -----
    mu, W, wdiag = calibration._whiten_from_cloud(z.reshape(C * R, d), shrink=shrink)
    zw = (z.reshape(C * R, d) - mu) @ W
    zw = zw.reshape(C, R, d)
    zr = _norm(np.asarray(z, dtype=np.float64))              # ENCODER frame (no whitening)
    g_normed = _norm(np.asarray(g, dtype=np.float64))
    dg = _metric(g_normed)

    # ---- Q1: the bias curve, in BOTH frames ------------------------------------------------
    # The gate is run in the DEPLOYMENT frame (whitened query vs raw gallery, exactly
    # `calibration.rep_cloud_scores` lines 1355-1356). But a training term can only see the
    # ENCODER's output space, so the headroom must ALSO exist without the whitening, or the
    # training term would be anchoring a frame it has no access to. Both are reported.
    Rs = [r for r in (1, 2, 4, 8, 16, 32, 64, 80) if r <= R]
    bias, bias_shuf = _bias_curve(zw, dg, Rs, rng)
    bias_raw, bias_raw_shuf = _bias_curve(zr, dg, Rs, rng)

    # ---- the deployed fusion, as a second reading of the same quantity ----------------------
    fused_diag: dict = {}
    fuse_corr = float("nan")
    mean80_corr = float("nan")
    if fuse_blocks > 1:
        edges = np.linspace(0, R, int(min(fuse_blocks, R)) + 1).astype(int)
        views = [_metric(_norm(zw[:, e0:e1].mean(axis=1)))
                 for e0, e1 in zip(edges[:-1], edges[1:]) if e1 > e0]
        if len(views) > 1:
            fused, fused_diag = _reliability_fuse(views)
            fuse_corr = _corr(fused, dg)
    if R >= 1:
        mean80_corr = _corr(_metric(_norm(zw.mean(axis=1))), dg)

    # ---- Q2: split-half reliability -> rho, K_eff ---------------------------------------------
    half = R // 2
    ms = [m for m in (1, 2, 4, 8, 16, 32, 40) if m <= half]
    r_curve = []
    for m in ms:
        vals = []
        for _ in range(int(n_split)):
            p = rng.permutation(R)
            A, B = p[:m], p[m:2 * m]
            da = _metric(_norm(zw[:, A].mean(axis=1)))
            db = _metric(_norm(zw[:, B].mean(axis=1)))
            vals.append(_corr(da, db))
        r_curve.append(float(np.nanmean(vals)))
    # CORRECT LINEARISATION. With view error = c (shared across ALL views, var rho*sigma^2) plus
    # e_i (independent, var (1-rho)*sigma^2), the average of m views has
    #     corr(Dbar_m^A, Dbar_m^B) = V_s / ( V_s + rho*sigma^2 + (1-rho)*sigma^2 / m )
    # so
    #     1/corr(m) - 1 = (rho*sigma^2/V_s)  +  ((1-rho)*sigma^2/V_s) * (1/m),
    # LINEAR IN 1/m (not in m-1: the earlier parametrisation mixed the two error components and is
    # the wrong line). Intercept alpha and slope beta then give rho = alpha / (alpha + beta), and
    # the model is only valid if BOTH alpha, beta > 0 and the line actually fits.
    rho, rho_raw, rho_model_ok, rho_fit = 0.0, float("nan"), False, {}
    if len(ms) >= 3 and np.isfinite(np.asarray(r_curve)).all():
        r = np.clip(np.asarray(r_curve, dtype=float), 1e-3, 0.999)
        y = 1.0 / r - 1.0
        x = 1.0 / np.asarray(ms, dtype=float)
        beta, alpha = np.polyfit(x, y, 1)                 # y = alpha + beta*(1/m)
        pred = alpha + beta * x
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = float(1 - ss_res / max(ss_tot, 1e-12))
        denom = alpha + beta
        rho_raw = float(alpha / denom) if abs(denom) > 1e-9 else float("nan")
        rho_model_ok = bool(alpha > 0 and beta > 0 and r2 > 0.9)
        r1 = float(r[0])
        sb = [m * r1 / (1.0 + (m - 1) * r1) for m in ms]  # Spearman-Brown from the m=1 reliability
        rho_fit = {
            "alpha": float(alpha), "beta": float(beta), "r2": r2,
            "rho_raw": rho_raw, "model_ok": rho_model_ok,
            "spearman_brown_pred": [float(v) for v in sb],
            "observed_over_SB": [float(o / s) if s > 1e-9 else None for o, s in zip(r, sb)],
        }
        rho = float(np.clip(rho_raw, 0.0, 0.999)) if rho_model_ok else 0.0
    k_eff = float(R / (1.0 + (R - 1) * rho))

    # ---- rank guard baseline (what a non-degenerate training term must not go below) ----------
    rank1 = _effrank(_metric(_norm(zw[:, 0])))
    rankR = _effrank(_metric(_norm(zw.mean(axis=1))))

    return {
        "C": int(C), "R": int(R), "d": int(d),
        "whiten": {k: wdiag[k] for k in ("cond", "rank_deficient")},
        "bias_curve": {"R_for_average": Rs,
                       "corr_with_gallery": [float(v) for v in bias],
                       "corr_shuffled_control": [float(v) for v in bias_shuf],
                       "corr_with_gallery_raw_frame": [float(v) for v in bias_raw],
                       "corr_shuffled_control_raw_frame": [float(v) for v in bias_raw_shuf]},
        "bias_headroom_R80_minus_R1": float(bias[-1] - bias[0]) if len(bias) > 1 else float("nan"),
        "bias_headroom_R80_minus_R1_raw_frame": float(bias_raw[-1] - bias_raw[0])
        if len(bias_raw) > 1 else float("nan"),
        "fusion_corr_with_gallery": fuse_corr,
        "single_mean_corr_with_gallery": mean80_corr,
        "fusion_minus_mean": float(fuse_corr - mean80_corr)
        if np.isfinite(fuse_corr) and np.isfinite(mean80_corr) else float("nan"),
        "fusion_diag": fused_diag,
        "rho": rho, "rho_raw": rho_raw, "rho_model_ok": bool(rho_model_ok),
        "rho_fit": rho_fit, "k_eff_at_R": k_eff,
        "gallery_reliability_raw_layers": _gallery_reliability(stack) if stack is not None else {},
        "source_vs_target": (_source_vs_target(z, src_means, dg, seed + 7919)
                             if src_means is not None else {}),
        "split_half_curve": {"m": ms, "corr": r_curve},
        "effrank_single": rank1, "effrank_pooled": rankR,
    }


# --------------------------------------------------------------------------- main
def _reading(*, bias_go: bool, bias_go_raw: bool, eeg_saturated, bc, br, sc, sr,
             c_src, c_tgt, c_ctl, d_st, t_st, rho, rho_ok, rho_go, n_folds: int,
             n_sv: int) -> str:
    """The pre-registered verdict text, as one explicit branch (a nested ternary here was
    unreadable and shipped a syntax error once)."""
    m = np.nanmean
    if not bias_go:
        return ("KILL: no bias headroom (or the control does not separate) -> the order-2 training "
                "term has nothing to move; keep the shipped 54.83 and record the falsification.")
    if eeg_saturated is True:
        return ("REDIRECT: bias headroom is real (R1 %.3f -> R80 %.3f, control <= %.3f) but the "
                "EEG side is SATURATED -- a 9-subject pooled EEG metric, i.e. ~720 repetitions and "
                "9 independent encodings' worth of denoising, agrees with the gallery no better "
                "than the target's own 80 repetitions (%.3f vs %.3f, delta %+.4f, t=%+.1f, %d/%d "
                "folds). The residual is CROSS-MODAL, so an EEG-side training term cannot reach "
                "it; the only movable side of the FGW coupling is the GALLERY metric -- aim there."
                % (m(bc[:, 0]), m(bc[:, -1]), float(np.nanmax(np.abs(sc[:, -1]))),
                   m(c_src), m(c_tgt), m(d_st), t_st, int(np.nansum(d_st > 0)), n_sv))
    if eeg_saturated is None:
        return ("GO (bottleneck test did not run -- no source-metric cache): bias headroom exists "
                "in the DEPLOYMENT frame (R1 %.3f -> R80 %.3f) and in the ENCODER frame (R1 %.3f "
                "-> R80 %.3f); controls collapse (<= %.3f). Whether the EEG side is saturated is "
                "UNRESOLVED and must be settled before any training code."
                % (m(bc[:, 0]), m(bc[:, -1]), m(br[:, 0]), m(br[:, -1]),
                   float(max(np.nanmax(np.abs(sc[:, -1])), np.nanmax(np.abs(sr[:, -1]))))))
    if not bias_go_raw:
        return ("PARTIAL: bias headroom is present in the deployment frame but NOT in the encoder "
                "frame (%.3f -> %.3f) -> a training term cannot anchor it; the whitening, not the "
                "encoder, is where the gap lives." % (m(br[:, 0]), m(br[:, -1])))
    return ("GO: bias headroom exists in the DEPLOYMENT frame (R1 %.3f -> R80 %.3f) AND in the "
            "ENCODER frame the training loop actually sees (R1 %.3f -> R80 %.3f); both "
            "concept-shuffle controls collapse (<= %.3f). The EEG side is NOT saturated: a "
            "9-subject pooled metric beats the target's own 80 repetitions on the same gallery "
            "(%.3f vs %.3f, delta %+.4f, t=%+.1f, %d/%d folds, its own shuffle control %.3f), so "
            "the residual is SUBJECT-SPECIFIC BIAS -- exactly what an exogenous-anchored L_bias "
            "targets. Variance channel: rho_mean=%.3f (exchangeable model valid on %d/%d folds) "
            "-> %s."
            % (m(bc[:, 0]), m(bc[:, -1]), m(br[:, 0]), m(br[:, -1]),
               float(max(np.nanmax(np.abs(sc[:, -1])), np.nanmax(np.abs(sr[:, -1])))),
               m(c_src), m(c_tgt), m(d_st), t_st, int(np.nansum(d_st > 0)), n_sv,
               float(np.nanmax(np.abs(c_ctl))),
               m(rho), int(rho_ok.sum()), n_folds,
               "still live" if rho_go else "exhausted (independent view errors)"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1-root", default=str(ROOT / "outputs/stage1/v8"))
    ap.add_argument("--src-metric-root", default=str(ROOT / "outputs/src_metric/v8"),
                    help="cached per-fold source-subject concept means (the decisive "
                         "bottleneck test's noise-free EEG arm); pass '' to skip")
    ap.add_argument("--src-tag-fmt", default="sub{:02d}_seed{seed}",
                    help="file-name template for the source-metric cache; it does NOT carry the "
                         "`k20` stage1 suffix")
    ap.add_argument("--tag-fmt", default="sub{:02d}_k20_seed{seed}")
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(1, 11)))
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--outdir", default=str(ROOT / "outputs/probe/view_independence"))
    ap.add_argument("--rep-shrink", type=float, default=0.1)
    ap.add_argument("--fuse", type=int, default=16)
    ap.add_argument("--n-split", type=int, default=20)
    ap.add_argument("--embed-batch", type=int, default=64)
    ap.add_argument("--mvnn", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[vi] device={device} folds={args.folds} seed={args.seed} fuse={args.fuse}")

    per_fold: dict[str, Any] = {}
    for s in args.folds:
        tag = args.tag_fmt.format(s, seed=args.seed)
        ckpt_path = Path(args.stage1_root) / tag / "last.pt"
        if not ckpt_path.is_file():
            print(f"[vi] MISSING {ckpt_path} -- skipped")
            continue
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        cfg = ckpt["cfg"]
        channel_set = cfg.get("channel_set", "all63")
        channels = (config.CHANNELS_OCCIPITO_PARIETAL
                    if channel_set == "occipital17" else None)
        mvnn = args.mvnn if args.mvnn is not None else \
            ("test" if cfg.get("mvnn", "off") != "off" else "off")
        img = cfg.get("image", {}) or {}
        targets_te = load_target_stack(img.get("feature_set", "clip_h14_multilevel"),
                                       img.get("layers"), "test")
        model = build_model(cfg, targets_te.shape[2], targets_te.shape[-1]).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        te = things_eeg.load_test_reps(s, channels, mvnn=mvnn)
        z = np.asarray(evaluate.embed_reps(model, te, device, batch=args.embed_batch))
        with torch.no_grad():
            t = torch.as_tensor(np.asarray(targets_te)[:, 0], dtype=torch.float32, device=device)
            g = model.encode_target(t, training=False).float().cpu().numpy()
        src_means = None
        if args.src_metric_root:
            src_tag = args.src_tag_fmt.format(s, seed=args.seed)
            sp = Path(args.src_metric_root) / f"{src_tag}.npz"
            if sp.is_file():
                src_means = np.load(sp)["src_means"]
            else:
                print(f"[vi] note: no source-metric cache at {sp}; bottleneck test skipped for "
                      f"this fold")
        print(f"[vi] sub{s:02d} cloud={z.shape} gallery={g.shape} "
              f"stack={tuple(np.asarray(targets_te).shape)} src={None if src_means is None else src_means.shape}")

        r = probe_fold(z, g, shrink=args.rep_shrink, fuse_blocks=args.fuse,
                       n_split=args.n_split, seed=args.seed + s,
                       stack=targets_te, src_means=src_means)
        r["target_subject"] = int(s)
        per_fold[f"sub{s:02d}"] = r
        with open(outdir / f"sub{s:02d}_seed{args.seed}.json", "w") as fh:
            json.dump(r, fh, indent=2)
        _sv = r.get("source_vs_target") or {}
        print(f"[vi] sub{s:02d}  bias  R1={r['bias_curve']['corr_with_gallery'][0]:+.4f} "
              f"R80={r['bias_curve']['corr_with_gallery'][-1]:+.4f} "
              f"(shuf {r['bias_curve']['corr_shuffled_control'][-1]:+.4f})  "
              f"RAW R1={r['bias_curve']['corr_with_gallery_raw_frame'][0]:+.4f} "
              f"R80={r['bias_curve']['corr_with_gallery_raw_frame'][-1]:+.4f}  "
              f"rho={r['rho']:.4f} K_eff={r['k_eff_at_R']:.1f} "
              f"fuse-mean={r['fusion_minus_mean']:+.4f} "
              f"rank1={r['effrank_single']:.1f} rankR={r['effrank_pooled']:.1f}")
        if _sv:
            print(f"[vi]        BOTTLENECK  corr(target80,Dg)={_sv['corr_target80_with_gallery']:+.4f} "
                  f"corr(source9pool,Dg)={_sv['corr_source9pool_with_gallery']:+.4f} "
                  f"(shuf {_sv['corr_source9pool_shuffled_control']:+.4f}) "
                  f"-> {'EEG SIDE SATURATED' if abs(_sv['source_minus_target']) < 0.03 else 'EEG SIDE HAS HEADROOM'} "
                  f"(delta {_sv['source_minus_target']:+.4f})")

    if not per_fold:
        raise SystemExit("[vi] no folds ran")

    def col(key):
        return np.asarray([per_fold[k][key] for k in sorted(per_fold)], dtype=float)

    bc = np.asarray([per_fold[k]["bias_curve"]["corr_with_gallery"] for k in sorted(per_fold)])
    sc = np.asarray([per_fold[k]["bias_curve"]["corr_shuffled_control"] for k in sorted(per_fold)])
    br = np.asarray([per_fold[k]["bias_curve"]["corr_with_gallery_raw_frame"]
                     for k in sorted(per_fold)])
    sr = np.asarray([per_fold[k]["bias_curve"]["corr_shuffled_control_raw_frame"]
                     for k in sorted(per_fold)])
    head = col("bias_headroom_R80_minus_R1")
    rho = col("rho")
    fm = col("fusion_minus_mean")

    # ---- PRE-REGISTERED VERDICT -------------------------------------------------------------
    # The training term only sees the ENCODER frame, so the pass condition requires headroom in
    # the RAW frame too: a whitened-frame-only headroom would authorise a term anchored to a
    # frame the training loop cannot reach.
    shuf_ok = bool(np.nanmax(np.abs(sc[:, -1])) < 0.10)          # control must collapse
    shuf_ok_raw = bool(np.nanmax(np.abs(sr[:, -1])) < 0.10)
    bias_go = bool(np.nanmean(head) > 0.03 and shuf_ok)
    bias_go_raw = bool(np.nanmean(br[:, -1] - br[:, 0]) > 0.03 and shuf_ok_raw)
    rho_ok = np.asarray([bool(per_fold[k]["rho_model_ok"]) for k in sorted(per_fold)])
    rho_go = bool(np.nanmean(rho) > 0.05 and rho_ok.mean() > 0.5)

    # ---- the bottleneck test: which SIDE of the coupling is actually movable -----------------
    sv = [per_fold[k]["source_vs_target"] for k in sorted(per_fold)
          if per_fold[k].get("source_vs_target")]
    if sv:
        c_tgt = np.asarray([d["corr_target80_with_gallery"] for d in sv], dtype=float)
        c_src = np.asarray([d["corr_source9pool_with_gallery"] for d in sv], dtype=float)
        c_ctl = np.asarray([d["corr_source9pool_shuffled_control"] for d in sv], dtype=float)
        d_st = c_src - c_tgt
        t_st = float(d_st.mean() / (d_st.std(ddof=1) / np.sqrt(len(d_st)))) if d_st.std(ddof=1) > 0 \
            else float("nan")
        # the reference arm is only interpretable if its concept-shuffle control is dead
        eeg_saturated = None if float(np.nanmax(np.abs(c_ctl))) >= 0.10 else \
            bool(abs(float(d_st.mean())) < 0.03)
    else:
        c_tgt = c_src = c_ctl = d_st = np.asarray([float("nan")])
        t_st, eeg_saturated = float("nan"), None

    verdict = {
        "n_folds": len(per_fold),
        "bias_R1_mean": float(np.nanmean(bc[:, 0])),
        "bias_R80_mean": float(np.nanmean(bc[:, -1])),
        "bias_headroom_mean": float(np.nanmean(head)),
        "bias_headroom_folds_positive": int(np.nansum(head > 0)),
        "shuffled_control_max_abs": float(np.nanmax(np.abs(sc[:, -1]))),
        "bias_R1_mean_raw_frame": float(np.nanmean(br[:, 0])),
        "bias_R80_mean_raw_frame": float(np.nanmean(br[:, -1])),
        "bias_headroom_mean_raw_frame": float(np.nanmean(br[:, -1] - br[:, 0])),
        "bias_headroom_folds_positive_raw_frame": int(np.nansum((br[:, -1] - br[:, 0]) > 0)),
        "shuffled_control_max_abs_raw_frame": float(np.nanmax(np.abs(sr[:, -1]))),
        "rho_mean": float(np.nanmean(rho)),
        "rho_raw_mean": float(np.nanmean(col("rho_raw"))),
        "rho_exchangeable_model_folds_ok": int(rho_ok.sum()),
        "k_eff_at_R_mean": float(np.nanmean(col("k_eff_at_R"))),
        "fusion_minus_mean_mean": float(np.nanmean(fm)),
        "fusion_minus_mean_folds_positive": int(np.nansum(fm > 0)),
        "effrank_single_mean": float(np.nanmean(col("effrank_single"))),
        "effrank_pooled_mean": float(np.nanmean(col("effrank_pooled"))),
        "control_separates": shuf_ok,
        "GO_bias_channel": bias_go,
        "GO_bias_channel_raw_frame": bias_go_raw,
        "GO_variance_channel": rho_go,
        "bottleneck_corr_target80": float(np.nanmean(c_tgt)),
        "bottleneck_corr_source9pool": float(np.nanmean(c_src)),
        "bottleneck_source9pool_shuffled_control": float(np.nanmean(c_ctl)),
        "bottleneck_source_minus_target": float(np.nanmean(d_st)),
        "bottleneck_paired_t": t_st,
        "bottleneck_folds_positive": int(np.nansum(d_st > 0)),
        "EEG_side_saturated": eeg_saturated,
    }
    verdict["reading"] = _reading(
        bias_go=bias_go, bias_go_raw=bias_go_raw, eeg_saturated=eeg_saturated,
        bc=bc, br=br, sc=sc, sr=sr, c_src=c_src, c_tgt=c_tgt, c_ctl=c_ctl, d_st=d_st, t_st=t_st,
        rho=rho, rho_ok=rho_ok, rho_go=rho_go, n_folds=len(per_fold), n_sv=len(sv),
    )
    summary = {"config": vars(args), "per_fold": per_fold, "verdict": verdict}
    with open(outdir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n[vi] ================================ VERDICT ================================")
    for k, v in verdict.items():
        if k != "reading":
            print(f"[vi]   {k:>34s}: {v}")
    print(f"[vi] {verdict['reading']}")
    print(f"[vi] wrote {outdir}/summary.json")


if __name__ == "__main__":
    main()
