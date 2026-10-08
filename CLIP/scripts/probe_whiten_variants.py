#!/usr/bin/env python
"""probe_whiten_variants -- which LABEL-FREE geometry is well-conditioned across all 10 folds?

WHY THIS EXISTS. G3 measured the deployment ladder over 10 folds x 3 seeds and localised the
entire remaining problem in one operator. Written from `outputs/g3_summary.json` and the
per-run reports:

    raw cosine                  26.82 +- 5.21      (tied with the SAMGA encoder's 26.22)
    + CSLS                      32.10 +- 6.51
    + CSLS + recovery           35.72 +- 6.39      recovery adds a FLAT +3.62 on every fold
    + T2 reps (rep cloud)       45.53 +- 9.26      the biggest lever, and the ONLY unstable one

Per fold, the T2 gain over `+ CSLS` ranges **+0.17 (fold 8) to +21.33 (fold 10)**, and it
correlates with ITS OWN fit diagnostics -- `weight_entropy` +0.49, `weight_share_top1` -0.49 --
and NOT with raw Top-1 (-0.04) or the recovery gain (+0.31). So the lever's variance is a
property of the LABEL-FREE GEOMETRY ESTIMATE, not of the encoder, and fold 8 is where that
estimate degenerates (mean weight-entropy 4.37 vs 4.55 on fold 10).

The estimator that degenerates is a FULL-RANK 64x64 covariance with a fixed shrinkage of 0.1
(`calibration._whiten_from_cloud`). But this project has measured the concept manifold at
**~16 dimensions** and shipped `spec_r0: 16` on that basis (axiom A3). Estimating 2080
covariance parameters to model a 16-dimensional signal, from 200 concepts, is the definition
of an ill-conditioned estimator -- and `shrink: 0.1` toward an isotropic target is a blunt
regulariser that treats signal and noise directions alike.

THE HYPOTHESIS THIS TESTS. The operator should be built from the STRUCTURE WE KNOW instead of
a generic full covariance: a rank-`r` factor model (signal subspace) plus an isotropic noise
floor. That is (a) far better conditioned, and (b) the version of the same idea the literature
uses for exactly this regime.

AND THE ONE THING SCORE CANNOT DO. SCORE AVERAGES the repetitions, so its covariance estimate
cannot separate the concept covariance from the trial-noise covariance -- averaging destroys
the within-concept scatter. We KEEP the repetitions, so `Sigma_noise` (the pooled within-concept
covariance) is directly estimable. Under a query = concept + noise, gallery = concept model the
right linear geometry is set by the NOISE covariance, not the total one, and `Sigma_noise` is
an object only the un-averaged repetitions can produce. That is the cross-trial half of our
subject-as-modality framing, and it is the mechanism this probe is built to check.

Runs on cached features when given `--cache`, otherwise embeds the fold's rep cloud (the only
expensive step). Usage:

    python scripts/probe_whiten_variants.py --target-subject 8 --ckpt outputs/stage1/g3/sub08_k20_seed2025/last.pt
    python scripts/probe_whiten_variants.py --subjects 1 2 3 4 5 6 7 8 9 10 --seed 2025
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from samclip import calibration, config, evaluate  # noqa: E402
from samclip.data import things_eeg  # noqa: E402
from samclip.data.targets import load_target_stack  # noqa: E402
from samclip.models import build_model  # noqa: E402
from samclip.models.multiroute import resolve_routes  # noqa: E402

from run_eval import _load_fold_arrays  # noqa: E402


# ---------------------------------------------------------------- geometry builders
# A builder returns `(q, g_m, info)`: the query and the gallery in the COMPARISON SPACE.
# See `g_full_asym` for why the pair is returned instead of a single `W`.
def _eigh(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vals, vecs = np.linalg.eigh(cov)
    return np.maximum(vals, 1e-12), vecs


def _factor(z_reps: np.ndarray, rank: int, shrink: float):
    """Rank-`rank` factor model: whiten inside the signal subspace, DROP the rest.

    The drop is the point, not a side effect. A near-zero eigenvalue direction is pure noise
    in this sample, and `_whiten_from_cloud` divides by `sqrt(lambda)` on it (clamped only at
    cond 1e3), which AMPLIFIES it. That is the plausible fold-8 failure: one such direction
    dominates the whitened geometry and the fitted weight vector collapses onto it
    (`weight_share_top1` up, `weight_entropy` down -- exactly the measured signature).

    Shrinking toward isotropic does not fix that, because it treats the 16 signal directions
    and the 48 noise directions identically. Selecting the subspace does.
    """
    cloud = z_reps.reshape(-1, z_reps.shape[-1]).astype(np.float64)
    mu = cloud.mean(axis=0, keepdims=True)
    xc = cloud - mu
    cov = (xc.T @ xc) / max(1, xc.shape[0] - 1)
    vals, vecs = _eigh(cov)
    top = slice(-rank, None)
    lam, vec = vals[top], vecs[:, top]
    lam = (1.0 - shrink) * lam + shrink * float(lam.mean())
    w = vec @ np.diag(1.0 / np.sqrt(lam))
    return w, mu, {"kind": f"factor{rank}", "lam_min": float(lam.min()),
                   "lam_max": float(lam.max()), "cond_top_r": float(lam.max() / lam.min())}


def _lw_shrinkage(xc: np.ndarray) -> float:
    """Ledoit-Wolf intensity, implemented here rather than imported.

    `sklearn` is in the venv but this probe's whole claim is about ONE number, and a
    version-dependent `LedoitWolf` change would silently move it. The closed form is three
    lines; the estimator it regularises is what matters.
    """
    n, d = xc.shape
    cov = (xc.T @ xc) / (n - 1)
    mu_i = np.trace(cov) / d
    # squared Frobenius norm of the deviation from the isotropic target, and the
    # per-sample variance of that deviation -- the standard LW numerator/denominator.
    dev = cov - mu_i * np.eye(d)
    num = float((dev ** 2).sum())
    x2 = (xc ** 2).T @ (xc ** 2) / n - ((xc.T @ xc) / n) ** 2
    den = float((x2 ** 2).sum())
    return float(min(1.0, max(0.0, num / den))) if den > 0 else 0.0


# A builder returns `(q, g_m, info)`: the query and the gallery in the COMPARISON SPACE.
#
# They are returned as a pair rather than as one `W` because the current operator is
# ASYMMETRIC and that asymmetry is load-bearing: `calibration.rep_cloud_scores` whitens the
# QUERY cloud and then compares against the RAW gallery, letting `coordinate_recovery` do its
# own per-dimension moment matching against the untouched gallery. A probe that returned a
# single `W` and applied it to both sides produced 28.50 on fold 8/seed 2025 where the banked
# G3 report says 34.50 -- a 6-point disagreement with our own record, which is how the
# asymmetry was found. `full_asym` is kept as the anchoring variant: it MUST reproduce the G3
# number, and the probe refuses to be believed if it does not.
def g_full_asym(z_reps, g, args) -> tuple[np.ndarray, np.ndarray, dict]:
    """THE CURRENT OPERATOR, verbatim: whiten the query cloud, leave the gallery raw.

    Reproduces the banked G3 number exactly (fold 8 / seed 2025 = 34.50, fold 10 = 62.50).
    """
    C, R, d = z_reps.shape
    cloud = z_reps.reshape(C * R, d).astype(np.float64)
    mu, w, diag = calibration._whiten_from_cloud(cloud, shrink=args.shrink)
    q = (z_reps.mean(axis=1).astype(np.float64) - mu) @ w
    return q, np.asarray(g, dtype=np.float64), {"kind": "full_asym", **diag}


# ---- same-dimension asymmetric variants -------------------------------------------------
# The baseline keeps the gallery RAW, and `g_full` above showed why that matters: applying the
# map to both sides costs 6-16 points. So a variant that wants to change the geometry must
# keep 64 dimensions (to stay comparable with the raw gallery) and touch only the query.
# That rules out projecting to a rank-16 subspace and points at the actual suspect: the
# WHITENING SCALE, not the rank.
def _asym_whiten(z_reps, g, cov: np.ndarray, max_cond: float, info: dict):
    """Whiten the query by `cov`, clamped so no direction is amplified beyond `max_cond`.

    `lam_max / max_cond` is the floor `_whiten_from_cloud` already applies, and the whole
    hypothesis is that 1e3 is far too loose: every direction below `lam_max/1e3` is divided by
    its own tiny eigenvalue, i.e. blown up to compete with the signal. The clamped inverse
    square root keeps 64 dimensions -- so the raw gallery stays a legal partner -- while
    declining to amplify directions the sample cannot support.
    """
    C, R, d = z_reps.shape
    cloud = z_reps.reshape(C * R, d).astype(np.float64)
    mu = cloud.mean(axis=0, keepdims=True)
    vals, vecs = _eigh(cov)
    lo = max(float(vals.max()) / max_cond, 1e-12)
    vals = np.maximum(vals, lo)
    w = vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T
    q = (z_reps.mean(axis=1).astype(np.float64) - mu) @ w
    n_clamped = int((np.linalg.eigvalsh(cov) < lo).sum())
    return q, np.asarray(g, dtype=np.float64), {
        **info, "max_cond": max_cond, "n_clamped_dirs": n_clamped,
        "cond_before": float(vals.max() / max(vals.min(), 1e-12))}


def _total_cov(z_reps):
    cloud = z_reps.reshape(-1, z_reps.shape[-1]).astype(np.float64)
    xc = cloud - cloud.mean(axis=0, keepdims=True)
    return (xc.T @ xc) / max(1, xc.shape[0] - 1)


def g_cond(max_cond: float):
    """The current geometry with a TIGHTER conditioning clamp -- isolate the scale."""
    def build(z_reps, g, args):
        return _asym_whiten(z_reps, g, _total_cov(z_reps), max_cond,
                            {"kind": f"cond{max_cond:g}_asym"})
    return build


def _pooled_within(z_reps: np.ndarray, shrink: float) -> np.ndarray:
    """Pooled within-concept covariance -- the object averaging destroys.

    `z_reps` is (C, R, d); the within-concept scatter is `sum_i sum_r (z_ir - zbar_i)^2`
    divided by `C*(R-1)` degrees of freedom. From averaged queries this is NOT estimable:
    averaging is exactly the operation that removes it. That is why every arm built on it is
    a genuine cross-trial advantage rather than a re-tuning of SCORE's moments.
    """
    C, R, d = z_reps.shape
    xc = z_reps - z_reps.mean(axis=1, keepdims=True)
    flat = xc.reshape(C * R, d)
    cov = (flat.T @ flat) / max(1, C * (R - 1))
    return (1.0 - shrink) * cov + shrink * (np.trace(cov) / d) * np.eye(d)


def g_noise(z_reps, g, args):
    """Whiten the query by the WITHIN-concept (trial-noise) covariance -- noisy side only.
    Query = concept + noise, gallery = concept. The discriminative directions are those where
    the concept cloud is wide relative to the trial noise, so `Sigma_noise` is what sets the
    right geometry for the NOISY query, and the CLEAN gallery must not be whitened by it. That
    is the same asymmetry the baseline uses, which is why this arm is legal where `noise`
    (symmetric) was not.

    `Sigma_noise` is not estimable from averaged queries: averaging is exactly the operation
    that removes the within-concept scatter. So this is the cross-trial half of the
    subject-as-modality framing expressed as one linear operator.
    """
    cov = _pooled_within(z_reps, args.shrink)
    return _asym_whiten(z_reps, g, cov, args.max_cond, {"kind": "noise_asym"})


def g_noise_cond(z_reps, g, args):
    cov = _pooled_within(z_reps, args.shrink)
    return _asym_whiten(z_reps, g, cov, args.max_cond,
                        {"kind": f"noise_cond{args.max_cond:g}_asym"})


def g_full(z_reps, g, args):
    """Symmetric full whitening -- kept ONLY as the measured cost of touching the gallery."""
    C, R, d = z_reps.shape
    cloud = z_reps.reshape(C * R, d).astype(np.float64)
    mu, w, diag = calibration._whiten_from_cloud(cloud, shrink=args.shrink)
    return _sym(z_reps, g, args, w, mu, {"kind": "full_sym", **diag})


def _sym(z_reps, g, args, w, mu, info):
    q = (z_reps.mean(axis=1).astype(np.float64) - mu) @ w
    gm = (np.asarray(g, dtype=np.float64) - mu) @ w
    return q, gm, info


def _factor(z_reps: np.ndarray, rank: int, shrink: float):
    cloud = z_reps.reshape(-1, z_reps.shape[-1]).astype(np.float64)
    mu = cloud.mean(axis=0, keepdims=True)
    xc = cloud - mu
    cov = (xc.T @ xc) / max(1, xc.shape[0] - 1)
    vals, vecs = _eigh(cov)
    top = slice(-rank, None)
    lam, vec = vals[top], vecs[:, top]
    lam = (1.0 - shrink) * lam + shrink * float(lam.mean())
    w = vec @ np.diag(1.0 / np.sqrt(lam))
    return w, mu, {"kind": f"factor{rank}_sym", "lam_min": float(lam.min()),
                   "lam_max": float(lam.max()), "cond_top_r": float(lam.max() / lam.min())}


def _factor_builder(rank: int):
    def build(z_reps, g, args):
        return _sym(z_reps, g, args, *_factor(z_reps, rank, args.shrink))
    return build


def _lw_shrinkage(xc: np.ndarray) -> float:
    """Ledoit-Wolf intensity, implemented here rather than imported.

    `sklearn` is in the venv but this probe's whole claim is about ONE number, and a
    version-dependent `LedoitWolf` change would silently move it. The closed form is three
    lines; the estimator it regularises is what matters.
    """
    n, d = xc.shape
    cov = (xc.T @ xc) / (n - 1)
    mu_i = np.trace(cov) / d
    dev = cov - mu_i * np.eye(d)
    num = float((dev ** 2).sum())
    x2 = (xc ** 2).T @ (xc ** 2) / n - ((xc.T @ xc) / n) ** 2
    den = float((x2 ** 2).sum())
    return float(min(1.0, max(0.0, num / den))) if den > 0 else 0.0


def g_lw(z_reps, g, args):
    """Same-dimension, query-only, but the shrinkage intensity is Ledoit-Wolf instead of 0.1.

    The 'just tune the shrinkage' arm. If this tracks the clamp arms the problem was the
    regularisation strength; if it does not, the problem is the scale floor.
    """
    cloud = z_reps.reshape(-1, z_reps.shape[-1]).astype(np.float64)
    mu = cloud.mean(axis=0, keepdims=True)
    xc = cloud - mu
    d = xc.shape[1]
    cov = (xc.T @ xc) / max(1, xc.shape[0] - 1)
    s = _lw_shrinkage(xc)
    cov = (1.0 - s) * cov + s * (np.trace(cov) / d) * np.eye(d)
    return _asym_whiten(z_reps, g, cov, args.max_cond, {"kind": "lw_asym",
                                                         "lw_intensity": s})


BUILDERS = {
    "full_asym": g_full_asym,          # the current operator; anchors the probe to G3
    "cond3_asym": g_cond(3.0),
    "cond10_asym": g_cond(10.0),
    "cond30_asym": g_cond(30.0),
    "cond100_asym": g_cond(100.0),
    "lw_asym": g_lw,
    "noise_asym": g_noise,
    "noise_clamped_asym": g_noise_cond,
    # symmetric arms -- the measured cost of whitening the clean gallery as well
    "full_sym": g_full,
    "factor16_sym": _factor_builder(16),
}


def score_with(z_reps: np.ndarray, g: np.ndarray, args, builder) -> dict:
    """One variant's Top-1: build the comparison space from the rep cloud, then recovery + CSLS.

    `coordinate_recovery` and `csls_scores` are held FIXED across variants with the same `k`
    and `rho` `run_eval` uses, so the only thing that moves is the geometry and a difference
    is attributable to the geometry alone.
    """
    C, R, d = z_reps.shape
    try:
        q, gm, info = builder(z_reps, g, args)
    except Exception as exc:                      # noqa: BLE001 - reported, never swallowed
        return {"top1": float("nan"), "top5": float("nan"), "mean_rank": float("nan"),
                "n": int(C), "info": {"kind": "ERROR", "error": repr(exc)},
                "landmark_rate": None}
    q_rec, rdiag = calibration.coordinate_recovery(q, gm, k=args.csls_k, rho=args.rho)
    s = calibration.csls_scores(q_rec, gm, k=args.csls_k)
    rep = calibration.report_with_scores(s)
    return {**rep, "info": info, "landmark_rate": rdiag.get("landmark_rate")}


def fold_features(ckpt: Path, target_subject: int, args, device):
    """`(z_reps, gallery)` for one fold.

    Cached under `outputs/probe/whiten_feats_*.npz` keyed by the fold run, because embedding
    the R=80 repetition cloud is the only expensive step and every variant below is a few
    matrix products on its output. Without the cache, asking a second question about the same
    fold costs the same as the first.
    """
    cache = Path("outputs/probe") / f"whiten_feats_{ckpt.parent.name}.npz"
    if cache.is_file():
        z = np.load(cache)
        return z["z_reps"], z["g"]
    import torch
    from torch.utils.data import DataLoader
    ckpt_d = torch.load(ckpt, map_location="cpu", weights_only=False)
    mcfg = ckpt_d["cfg"]
    channel_set = mcfg.get("channel_set", "all63")
    channels = (config.CHANNELS_OCCIPITO_PARIETAL
                if channel_set == "occipital17" else None)
    mvnn = args.mvnn or ("test" if mcfg.get("mvnn", "off") != "off" else "off")
    routes = resolve_routes(mcfg)
    feat = str(routes[0]["feature_set"]); layers = list(routes[0]["layers"])
    g = load_target_stack(feat, layers, "test")
    _, test = _load_fold_arrays(target_subject, channels, mvnn)
    model = build_model(mcfg, g.shape[2], g.shape[-1]).to(device)
    model.load_state_dict(ckpt_d["model"]); model.eval()
    loader = DataLoader(things_eeg.TestDataset(test, g), batch_size=200, shuffle=False,
                        collate_fn=things_eeg.collate)
    feats = evaluate.extract_features(model, loader, device)
    reps = things_eeg.load_test_reps(target_subject, channels, mvnn=mvnn)
    z = evaluate.embed_reps(model, reps, device)
    if isinstance(z, dict):
        z = z[next(iter(z))]
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, z_reps=z, g=feats["img"])
    return z, feats["img"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, nargs="*", default=[8])
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--stage1-root", default="outputs/stage1/g3")
    ap.add_argument("--out", default="outputs/probe/whiten_variants.json")
    ap.add_argument("--shrink", type=float, default=0.1)
    ap.add_argument("--max-cond", type=float, default=30.0,
                    help="conditioning clamp for the query-side whitening. The baseline's "
                         "1e3 is intentionally NOT the default here: this probe exists to "
                         "test whether that floor is the fold-8 degeneracy, and "
                         "`full_asym` carries 1e3 explicitly.")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--csls-k", type=int, default=10)
    ap.add_argument("--rho", type=float, default=0.1)
    ap.add_argument("--mvnn", default=None)
    ap.add_argument("--variants", nargs="*", default=list(BUILDERS))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[whiten] device={device} variants={args.variants} shrink={args.shrink} "
          f"rank={args.rank}")

    out: dict = {"variants": args.variants, "shrink": args.shrink, "per_fold": {}}
    for s in args.subjects:
        tag = f"sub{s:02d}_k20"
        ckpt = Path(args.stage1_root) / f"{tag}_seed{args.seed}" / "last.pt"
        if not ckpt.is_file():
            print(f"[whiten] fold {s}: missing {ckpt}")
            continue
        z, g = fold_features(ckpt, s, args, device)
        res = {}
        for name in args.variants:
            res[name] = score_with(z, g, args, BUILDERS[name])
        out["per_fold"][s] = res
        line = "  ".join(f"{n}={res[n]['top1']:5.1f}/{res[n]['top5']:5.1f}"
                         for n in args.variants)
        print(f"[whiten] fold {s:>2} ({z.shape[0]}x{z.shape[1]}x{z.shape[2]}): {line}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    print(f"[whiten] wrote {args.out}")


if __name__ == "__main__":
    main()
