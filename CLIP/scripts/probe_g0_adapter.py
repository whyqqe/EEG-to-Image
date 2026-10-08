#!/usr/bin/env python
"""G0 -- does a linear METRIC ADAPTER fitted on SOURCE subjects improve the deployed
operator on a held-out subject?  A probe, not a training run.

THE COMPONENT THIS GATES
------------------------
COMMET's Stage B contains one TRAINED object (`docs/eeg2image_commet_architecture.md` I5): a
projection `phi` fitted on source subjects to make the FROZEN encoder's per-trial metric agree
better with its own consensus metric. G0 asks whether that is even possible, at the cheapest
price that can answer it: the BEST-CASE LINEAR adapter, in closed form, with no training loop.

If the optimal linear map does not move the deployed number, no learned `phi` will, and I5 is
dead for the cost of one job. If it does move, I5 has legs and the nonlinear version is worth
building. That is the entire purpose of a gate.

WHAT THE ADAPTER IS, AND WHY IT IS THE RIGHT CLOSED FORM
--------------------------------------------------------
The deployed T2 operator consumes the R=80 per-trial cloud. Its per-trial estimate is limited by
WITHIN-STIMULUS scatter: two repetitions of the same image differ by neural noise, and that noise
is what makes a single-trial metric disagree with the R=80 consensus (measured +0.13 / -0.07, v12
smoke job 645753). The linear map that removes exactly that noise is the WHITENING BY THE
WITHIN-STIMULUS COVARIANCE:

    Sw = E[ (z - mean_k z)^2 ],     W = (Sw + gamma I)^{-1/2},

i.e. a Mahalanobis rescaling in which directions that carry only repetition noise are down-weighted
and directions that are stable across repetitions are kept. It is estimated from SOURCE subjects'
UNAVERAGED repetitions, where the concept/stimulus identity is a LABEL we are allowed to use --
which is the whole difference between this and the label-free SAW whitening already in the
operator (that one can only see the total scatter, signal and noise together).

THE THREE ARMS, AND WHY EACH ONE IS THERE
-----------------------------------------
  * ``identity``   -- the deployed row, reproduced. Every delta is paired against it.
  * ``within``     -- the adapter. Fitted from source within-stimulus scatter (uses source labels).
  * ``total``      -- an UNSUPERVISED whitening by the total scatter. This is the control that
                      separates "the adapter's second-order denoising helps" from "any whitening
                      helps": SAW whitening already lives inside the operator, so a gain that
                      `total` reproduces is not the adapter's mechanism.
  * ``lda_k``      -- the same within-scatter whitening followed by projection onto the top-k
                      between/within directions (the measured concept manifold is ~16 of 64).
  * ``shuffled``   -- the within-scatter estimator computed with the stimulus labels PERMUTED.
                      The adapter's only source of information is the label-to-repetition
                      grouping, so a permuted grouping is the correspondence-destroying control:
                      if `shuffled` matches `within`, the gain was never the grouping.

PROTOCOL NOTES (why this is label-free with respect to the target)
------------------------------------------------------------------
Only SOURCE subjects' EEG is used to fit the adapter; the target subject contributes neither a
label nor a gradient. The target's 200 test concepts happen to be the SAME concepts the sources
are tested on (THINGS-EEG2 shares one test set), but the adapter is applied ELEMENTWISE per
embedding and never sees an index, so the concept-index correspondence that made M1 leak is
absent by construction -- the same argument that licenses v11's block fusion.

Run (submitted via slurm; the login node has no GPU):
    python scripts/probe_g0_adapter.py --ckpts <fold ckpt> --target-subject 8 \
        --source-subjects 1 2 3 4 5 6 7 9 10 --out outputs/probe/g0/sub08.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from samclip import calibration, config, evaluate  # noqa: E402
from samclip.data import things_eeg  # noqa: E402
from samclip.data.targets import load_target_stack  # noqa: E402
from samclip.models import build_model  # noqa: E402


# --------------------------------------------------------------- adapter fits
def _within_scatter(clouds: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Per-stimulus residual scatter and stimulus-mean scatter, pooled over subjects.

    `clouds` is a list of `(C, R, d)` arrays -- one per source subject, ALL on the same C
    stimuli (the shared THINGS-EEG2 test set). Returns `(Sw, St)` as `(d, d)`:

      * `Sw = mean over stimuli and reps of (z - mean_k z)^2` -- repetition noise;
      * `St = covariance of all embeddings` -- signal + noise, the unsupervised control.

    Pooled over subjects rather than fitted per subject because the adapter must transfer to a
    HELD-OUT subject that contributes no data: a per-subject fit would not be a source-trained
    map, and the whole claim is that source labels carry the grouping information.
    """
    sw = None
    n_tot = 0
    parts = []
    for z in clouds:
        C, R, d = z.shape
        zc = z - z.mean(axis=1, keepdims=True)          # (C, R, d) residuals
        m = z.reshape(-1, d)
        if sw is None:
            sw = np.zeros((d, d), dtype=np.float64)
        sw += (zc.reshape(-1, d).T @ zc.reshape(-1, d))
        n_tot += zc.shape[0] * zc.shape[1]
        parts.append(m)
    sw = sw / max(n_tot, 1)
    allz = np.concatenate(parts, axis=0)
    ac = allz - allz.mean(axis=0, keepdims=True)
    st = (ac.T @ ac) / max(allz.shape[0] - 1, 1)
    return sw, st


def _inv_sqrt(m: np.ndarray, gamma: float) -> np.ndarray:
    """`(M + gamma*tr(M)/d * I)^{-1/2}` -- a regularised whitening that cannot amplify noise.

    The ridge is RELATIVE to the trace so `gamma` is a fraction of the average eigenvalue and
    the choice transfers across folds of different scale (an absolute floor, as the module
    docstring in `calibration.py` records for SAW, inverts near-zero eigenvalues to ~1/sqrt(eps)
    and amplifies pure noise by two orders of magnitude).
    """
    d = m.shape[0]
    m = 0.5 * (m + m.T)
    reg = m + float(gamma) * (np.trace(m) / d) * np.eye(d)
    vals, vecs = np.linalg.eigh(reg)
    lo = max(float(vals.max()) / 1e3, 1e-12)
    vals = np.maximum(vals, lo)
    return vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T


def _lda_projection(sw: np.ndarray, st: np.ndarray, k: int, gamma: float) -> np.ndarray:
    """Top-`k` between/within generalized directions, `(k, d)`, within-whitened then rotated.

    Solves `Sb v = lambda (Sw + gamma I) v` with `Sb = St - Sw` and returns the top-`k` rows,
    each normalised so `v^T (Sw + gamma I) v = 1`. Projecting through this keeps only the
    directions where stimulus structure dominates repetition noise -- the low-rank version of the
    same idea, and the one the measured concept-manifold dimension (~16 of 64) predicts.
    """
    d = sw.shape[0]
    reg = sw + float(gamma) * (np.trace(sw) / max(d, 1)) * np.eye(d)
    sb = st - sw
    vals, vecs = np.linalg.eigh(np.linalg.solve(reg, sb))
    order = np.argsort(vals)[::-1][: int(k)]
    v = vecs[:, order]                                  # (d, k)
    v = v / np.sqrt(np.einsum("dk,dk->k", v, reg @ v))  # unit within-metric norm
    return v.T                                          # (k, d)


def _shuffle_within(clouds: list[np.ndarray], seed: int) -> np.ndarray:
    """The within-scatter estimator with the stimulus grouping DESTROYED.

    THE OBVIOUS VERSION IS A NO-OP, WHICH IS WHY THIS ONE IS DIFFERENT. Permuting the
    repetition axis does not change a per-stimulus mean, so the residuals -- and therefore the
    within-scatter -- are bit-identical: the first version of this control returned `within`
    exactly and would have looked like a *passing* control. The grouping has to be broken at
    the level the estimator reads it: rows are REASSIGNED to random pseudo-stimuli, so the
    residual of a trial is taken against the mean of a random subset instead of against the
    mean of its own image's repetitions. Geometry is untouched; only the label grouping is
    destroyed.
    """
    rng = np.random.default_rng(seed)
    sh = []
    for z in clouds:
        C, R, d = z.shape
        flat = z.reshape(C * R, d)
        perm = rng.permutation(C * R).reshape(C, R)
        sh.append(flat[perm].reshape(C, R, d))
    return _within_scatter(sh)[0]


# --------------------------------------------------------------------- metric
def _agreement(z: np.ndarray) -> float:
    """corr(metric(single rep), metric(R-rep consensus)) -- the LABEL-FREE proxy G0 is about.

    This is the quantity the adapter is supposed to raise, and it needs no label: both metrics
    are functions of the target subject's own EEG. It is reported before any accuracy so the
    mechanism can be read independently of the retrieval score.
    """
    cons = _sq(np.asarray(z).mean(axis=1))
    single = _sq(np.asarray(z)[:, 0])
    iu = np.triu_indices(cons.shape[0], 1)
    return float(np.corrcoef(cons[iu], single[iu])[0, 1])


def _sq(x: np.ndarray) -> np.ndarray:
    return calibration._sq_cos_dist(np.asarray(x, dtype=np.float64))


@torch.no_grad()
def _fit_gallery(model, targets_te, device) -> np.ndarray:
    # `(C, I=1, K, D)` -> `(C, K, D)`: the layer axis is what the router blends over. Passing
    # the 4-D stack makes `LayerRouter` read `shape[1] == 1` and drop it as the image axis,
    # returning one embedding PER LAYER instead of the fused one.
    t = torch.as_tensor(np.asarray(targets_te)[:, 0], dtype=torch.float32, device=device)
    return model.encode_target(t, training=False).float().cpu().numpy()


# ---------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--target-subject", type=int, required=True)
    ap.add_argument("--source-subjects", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fit-reps", type=int, default=20,
                    help="repetitions per source stimulus used to FIT the adapter (the target "
                         "eval always uses all 80). Subsampling only lowers the estimator's "
                         "variance; 80 is affordable if needed.")
    ap.add_argument("--gamma", type=float, default=1e-2)
    ap.add_argument("--lda-k", type=int, default=16)
    ap.add_argument("--shuffle-seed", type=int, default=2025)
    ap.add_argument("--embed-batch", type=int, default=64)
    ap.add_argument("--csls-k", type=int, default=10)
    ap.add_argument("--rho", type=float, default=0.1)
    ap.add_argument("--rep-shrink", type=float, default=0.1)
    ap.add_argument("--fuse", type=int, default=16)
    ap.add_argument("--mvnn", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpts[0], map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    channel_set = cfg.get("channel_set", "all63")
    channels = (config.CHANNELS_OCCIPITO_PARIETAL
                if channel_set == "occipital17" else None)
    mvnn = args.mvnn if args.mvnn is not None else \
        ("test" if cfg.get("mvnn", "off") != "off" else "off")
    img = cfg.get("image", {}) or {}
    feature_set = img.get("feature_set", "clip_h14_multilevel")
    layers = img.get("layers")
    targets_te = load_target_stack(feature_set, layers, "test")
    model = build_model(cfg, targets_te.shape[2], targets_te.shape[-1]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"[g0] target sub-{args.target_subject:02d} seed-ckpt={args.ckpts[0]} "
          f"channels={channel_set} mvnn={mvnn} device={device}")

    # ---- target cloud (the deployed object) ----
    te = things_eeg.load_test_reps(args.target_subject, channels, mvnn=mvnn)
    z_t = np.asarray(evaluate.embed_reps(model, te, device, batch=args.embed_batch))
    g = _fit_gallery(model, targets_te, device)
    print(f"[g0] target cloud {z_t.shape}, gallery {g.shape}")

    # ---- source clouds (fit data) ----
    src_clouds = []
    for s in args.source_subjects:
        r = things_eeg.load_test_reps(s, channels, mvnn=mvnn)
        r = np.asarray(r)[:, : int(args.fit_reps)]
        zs = np.asarray(evaluate.embed_reps(model, r, device, batch=args.embed_batch))
        src_clouds.append(zs)
    print(f"[g0] {len(src_clouds)} source clouds, fit reps={args.fit_reps}, "
          f"each {src_clouds[0].shape}")

    # ---- fit the adapters ----
    d = src_clouds[0].shape[-1]
    sw, st = _within_scatter(src_clouds)
    sw_sh = _shuffle_within(src_clouds, args.shuffle_seed)
    # The control is only a control if it CHANGED something: a permutation that leaves `Sw`
    # invariant would make `shuffled` bit-identical to `within` and read as a passing control.
    rel_shift = float(np.linalg.norm(sw - sw_sh) / max(float(np.linalg.norm(sw)), 1e-12))
    print(f"[g0] control separation ||Sw - Sw_shuffled|| / ||Sw|| = {rel_shift:.4f} "
          f"(must be >> 0)")
    adapters: dict[str, np.ndarray | None] = {
        "identity": None,
        "within": _inv_sqrt(sw, args.gamma),
        "total": _inv_sqrt(st, args.gamma),
        "shuffled": _inv_sqrt(sw_sh, args.gamma),
        "lda_k": _lda_projection(sw, st, min(int(args.lda_k), d), args.gamma),
    }

    def rec_fn():
        def _wrap(q, gg, k=10, rho=0.1, min_landmark_rate=0.0, **kw):
            return calibration.subspace_soft_recovery(
                q, gg, k=k, rho=rho, rank=None, tau=0.03, iters=50,
                hard_landmarks=False, min_landmarks=8,
                alpha=float(kw.pop("alpha", 0.75)),
                fgw_de_ref=kw.pop("fgw_de_ref", None),
                fgw_di_ref=kw.pop("fgw_di_ref", None),
                fgw_de_mix=float(kw.pop("fgw_de_mix", 0.0)),
                fgw_spec_rank=kw.pop("fgw_spec_rank", None),
                fgw_topo_eps=kw.pop("fgw_topo_eps", None))
        return _wrap

    rows: dict = {}
    for name, w in adapters.items():
        # `w` is (out, in) for every arm -- (d,d) for the whitening arms and (k,d) for the
        # LDA projection -- so the map on an embedding matrix is `z @ w.T`.
        #
        # TWO SEMANTICS, AND THEY ARE NOT INTERCHANGEABLE. A SQUARE map is a re-weighting of
        # the EEG side only: it is fitted from EEG noise, so applying its inverse to the IMAGE
        # gallery would warp the image geometry by a statistic that has nothing to do with
        # images (`within` applied to both sides scored 20.00 against its own 64.50 twin on
        # sub-01 -- a collapse, and an artefact of the convention, not a result). A
        # DIM-CHANGING projection (`lda_k`) defines a shared subspace, so the gallery has to be
        # projected by the same map or the two sides are not even the same dimension; those
        # rows are flagged `gallery_mapped`.
        zw = z_t if w is None else np.einsum("crd,kd->crk", z_t, np.asarray(w))
        gallery_mapped = False
        if w is None or np.asarray(w).shape[0] == np.asarray(w).shape[1]:
            gw = g
        else:
            gw = np.einsum("cd,kd->ck", g, np.asarray(w))
            gallery_mapped = True
        agree = _agreement(zw)
        sc, diag = calibration.rep_cloud_scores(
            zw, gw, k=args.csls_k, rho=args.rho, shrink=args.rep_shrink,
            recovery_fn=rec_fn(), rep_blocks=int(args.fuse))
        rep = calibration.report_with_scores(sc)
        rows[name] = {
            "top1": rep["top1"], "top5": rep["top5"], "mean_rank": rep["mean_rank"],
            "metric_agreement": agree,
            "out_dim": int(zw.shape[-1]),
            "gallery_mapped": bool(gallery_mapped),
            "n_landmarks": diag.get("n_landmarks"),
            "plan_acc": diag.get("plan_acc"),
        }
        print(f"[g0] {name:<9s} top1={rep['top1']:6.2f} top5={rep['top5']:6.2f} "
              f"agree={agree:+.4f} dim={zw.shape[-1]}"
              f"{' (gallery mapped)' if gallery_mapped else ''}")

    base = rows["identity"]
    payload = {
        "target_subject": int(args.target_subject),
        "ckpt": str(args.ckpts[0]),
        "n_sources": len(args.source_subjects),
        "fit_reps": int(args.fit_reps),
        "gamma": float(args.gamma), "lda_k": int(args.lda_k), "fuse": int(args.fuse),
        "rows": rows,
        "delta_top1": {k: float(v["top1"] - base["top1"]) for k, v in rows.items()},
        "delta_agree": {k: float(v["metric_agreement"] - base["metric_agreement"])
                        for k, v in rows.items()},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"[g0] wrote {args.out}")

    # ---- pre-registered verdict (read here, not after seeing the grid mean) ----
    dw = rows["within"]["top1"] - base["top1"]
    dt = rows["total"]["top1"] - base["top1"]
    ds = rows["shuffled"]["top1"] - base["top1"]
    aw = rows["within"]["metric_agreement"] - base["metric_agreement"]
    print(f"\n[g0] within - identity : {dw:+.2f}pp   (agreement {aw:+.4f})")
    print(f"[g0] total  - identity : {dt:+.2f}pp   [unsupervised control]")
    print(f"[g0] shuffl - identity : {ds:+.2f}pp   [label-destroying control]")
    print("[g0] G0 passes on THIS fold iff within > 0 AND within > total AND within > shuffled.")


if __name__ == "__main__":
    main()
