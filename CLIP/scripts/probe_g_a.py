#!/usr/bin/env python
"""probe_g_a -- the pre-registered G-a gate: does NOISE-CORRECTED LOW-RANK BEAT T2?

WHAT G-a IS FOR
---------------
G3 localised the entire remaining problem in the deployment geometry: `+ CSLS + recovery`
gives a flat +8.90 over raw cosine, against SCORE's +23.90 on the same encoder class, and the
fold-to-fold spread (SD 9.26 vs SCORE's 1.62) comes entirely from the repetition-cloud step.
`probe_whiten_variants.py` then FALSIFIED the natural first explanation -- conditioning, rank,
shrinkage and the noise covariance, in their obvious forms, all failed to beat the current
full-covariance operator on 10 folds.

What survived is the mechanism in `samclip.concept_frame`: the concept cloud's covariance is
`A Sigma_c A^T + Sigma_n / R`, and because we KEEP the repetitions, `Sigma_n` is estimable and
the concept term can be recovered by subtraction. SCORE cannot do this, because it averages
the repetitions away. This is the one leverage point that is (a) structural rather than a
hyper-parameter and (b) unique to keeping the repetitions -- i.e. the cross-trial half of the
subject-as-modality framing.

THE GATE (pre-registered, no post-hoc relaxation)
-------------------------------------------------
    G-a passes  iff  the best arm beats `+ T2 reps` (45.53 +- 9.26, G3, 10 folds x 3 seeds)
                     on the MEAN, WITHOUT a worse fold SD.

Both halves matter and the second is not decoration: a mean win bought with more fold variance
would trade the 7.70pp deficit for a worse version of the 5.7x stability deficit.

The baseline is computed here rather than imported from the G3 reports, and it MUST reproduce
45.53 +- 9.26. If it does not, the probe is measuring a different pipeline and its verdict is
void -- that check runs before anything is concluded.

WHAT G-a CANNOT SETTLE (stated up front, not discovered afterwards)
------------------------------------------------------------------
The cross-subject PRIOR over the per-subject frames is NOT tested here. Each fold's model is
trained separately, so its 64-d space is its own and two folds' frames are not comparable
without a shared head; making them comparable is a training-time change. What IS tested is the
population-level stand-in: a rank fixed once across folds instead of re-estimated per fold.

Usage (all three seeds by default, so the gate matches G3's 30-run口径):

    python scripts/probe_g_a.py --seeds 2025 2026 2027
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from samclip import calibration, config, evaluate, concept_frame  # noqa: E402
from samclip.data import things_eeg  # noqa: E402
from samclip.data.targets import load_target_stack  # noqa: E402
from samclip.models import build_model  # noqa: E402
from samclip.models.multiroute import resolve_routes  # noqa: E402

from run_eval import _load_fold_arrays  # noqa: E402

G3_T2_MEAN, G3_T2_SD = 45.53, 9.26
SCORE_MEAN, SCORE_SD = 53.23, 1.62


# ------------------------------------------------------------------ feature loading
def fold_features(ckpt: Path, target_subject: int, args, device):
    """`(z_reps, gallery)` for one fold, cached by fold-run name.

    Embedding the R=80 repetition cloud is the only expensive step; every arm is then a few
    matrix products on its output, so the cache is what makes sweeping ~10 arms across 30
    runs cheap enough to be worth doing in one job.
    """
    cache = Path(args.feat_dir) / f"whiten_feats_{ckpt.parent.name}.npz"
    if cache.is_file():
        z = np.load(cache)
        return z["z_reps"], z["g"]
    import torch
    from torch.utils.data import DataLoader
    ckpt_d = torch.load(ckpt, map_location="cpu", weights_only=False)
    mcfg = ckpt_d["cfg"]
    channels = (config.CHANNELS_OCCIPITO_PARIETAL
                if mcfg.get("channel_set", "all63") == "occipital17" else None)
    mvnn = args.mvnn or ("test" if mcfg.get("mvnn", "off") != "off" else "off")
    routes = resolve_routes(mcfg)
    g = load_target_stack(str(routes[0]["feature_set"]), list(routes[0]["layers"]), "test")
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


# ------------------------------------------------------------------ the arms
def _full_total_cov(cloud: np.ndarray, shrink: float, max_cond: float):
    """The CURRENT operator: full-rank whitening of the TOTAL covariance, clamp 1e3."""
    mu, w, _ = calibration._whiten_from_cloud(cloud, shrink=shrink)
    return w, mu


def arm_specs(args) -> dict:
    """Every arm is `(kind, kwargs)`, applied through the same `score_arm` path.

    `baseline` reproduces the G3 `+ T2 reps` operator, AND only whitens the query side. The
    asymmetric form is not an oversight in the baseline -- `probe_whiten_variants.py` measured
    that applying the map to the raw gallery as well costs 6-16pp, because the gallery is the
    CLEAN target and whitening it distorts the geometry recovery is trying to reach.
    """
    specs = {"baseline_totalcov_full": ("total_full", {})}
    specs["nc_full"] = ("nc", {"rank": None})
    specs["nc_pos"] = ("nc", {"rank": None, "min_eig": 0.0})
    for r in (4, 6, 8, 10, 12, 16):
        specs[f"nc_r{r}"] = ("nc", {"rank": r})
    specs["nc_r8_nowhiten"] = ("nc", {"rank": 8, "whiten": False})
    specs["nc_r8_cond30"] = ("nc", {"rank": 8, "max_cond": 30.0})
    return specs


def arm_specs(args) -> dict:
    """Every arm is `(kind, kwargs)`, applied through the same `score_arm` path.

    `baseline_totalcov_full` reproduces the G3 `+ T2 reps` operator: it whitens the TOTAL
    covariance at full rank and only touches the QUERY side. The asymmetry is not an oversight
    -- `probe_whiten_variants.py` measured that applying the map to the raw gallery as well
    costs 6-16pp, because the gallery is the CLEAN target and whitening it distorts the
    geometry recovery is trying to reach. The symmetric arms are kept precisely to re-measure
    that cost under noise correction, since killing noise directions is exactly the change
    that could plausibly pay for it.
    """
    specs = {"baseline_totalcov_full": ("total_full", {})}
    specs["sym_totalcov_full"] = ("total_full", {"symmetric": True})
    specs["nc_asym_pos"] = ("nc", {"rank": None})
    for r in (8, 12, 16, 20):
        specs[f"nc_asym_r{r}"] = ("nc", {"rank": r})
    specs["nc_asym_r16_nowhiten"] = ("nc", {"rank": 16, "whiten": False})
    specs["nc_asym_r16_unit"] = ("nc", {"rank": 16, "fill": "unit"})
    specs["nc_asym_pos_unit"] = ("nc", {"rank": None, "fill": "unit"})
    for r in (8, 12, 16, 20):
        specs[f"nc_sym_r{r}"] = ("nc", {"rank": r, "symmetric": True})
    # CONTROLS for the one confound in the low-rank arms: a killed direction is a zero column,
    # and `moment_match` puts every column of the query on the gallery's per-dimension mean --
    # so a killed axis becomes the constant `mean(g[:, j])` rather than staying at zero. That
    # is only harmless if the gallery is centered on that axis. Centering the gallery for ALL
    # arms (including the baseline) makes the killed axes map to ~0 and separates "truncation
    # loses signal" from "truncation plus a non-centered gallery adds a constant".
    specs["baseline_centerg"] = ("total_full", {"center_gallery": True})
    specs["nc_asym_r16_centerg"] = ("nc", {"rank": 16, "center_gallery": True})
    specs["nc_asym_pos_centerg"] = ("nc", {"rank": None, "center_gallery": True})
    return specs


def score_arm(z_reps: np.ndarray, g: np.ndarray, spec, args):
    """Build the comparison space, then recovery + CSLS with G3's exact `k` and `rho`.

    Recovery and CSLS are held FIXED so the only thing that moves between arms is the geometry
    and a difference is attributable to the geometry alone. This is the same discipline the
    falsification probe used, deliberately, so a win here is comparable to that result.

    Returns `(report, S)`; `S` is the raw 200x200 score matrix and is NOT serialised -- it is
    far too large to keep per arm per run, and it exists only so the caller can fuse two arms
    at the score level (axiom A5) without re-running them.
    """
    C, R, d = z_reps.shape
    kind, kw = spec
    gi = np.asarray(g, dtype=np.float64)
    if kw.get("center_gallery"):
        gi = gi - gi.mean(axis=0, keepdims=True)
    try:
        if kind == "total_full":
            w, mu = _full_total_cov(z_reps.reshape(C * R, d).astype(np.float64),
                                    args.shrink, args.max_cond)
            info = {"kind": "total_full", "rank": d}
        else:
            frame = concept_frame.subject_frame(
                z_reps, rank=kw.get("rank"), shrink=args.shrink,
                whiten=kw.get("whiten", True),
                max_cond=kw.get("max_cond", args.max_cond),
                min_eig=kw.get("min_eig", 0.0), fill=kw.get("fill", "kill"))
            w, mu, info = frame.W, frame.mu, frame.diag
        q = (z_reps.mean(axis=1).astype(np.float64) - mu) @ w
        gm = (gi - mu) @ w if kw.get("symmetric") else gi
    except Exception as exc:                      # noqa: BLE001 - reported, never swallowed
        return ({"top1": float("nan"), "top5": float("nan"), "mean_rank": float("nan"),
                 "n": int(C), "info": {"kind": "ERROR", "error": repr(exc)}}, None)
    q_rec, rdiag = calibration.coordinate_recovery(q, gm, k=args.csls_k, rho=args.rho)
    s = calibration.csls_scores(q_rec, gm, k=args.csls_k)
    return ({**calibration.report_with_scores(s), "info": info,
             "landmark_rate": rdiag.get("landmark_rate")}, s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=[2025, 2026, 2027])
    ap.add_argument("--subjects", type=int, nargs="*", default=list(range(1, 11)))
    ap.add_argument("--stage1-root", default="outputs/stage1/g3")
    ap.add_argument("--feat-dir", default="outputs/probe")
    ap.add_argument("--out", default="outputs/probe/g_a.json")
    ap.add_argument("--shrink", type=float, default=0.1)
    ap.add_argument("--max-cond", type=float, default=1e3)
    ap.add_argument("--csls-k", type=int, default=10)
    ap.add_argument("--rho", type=float, default=0.1)
    ap.add_argument("--mvnn", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    specs = arm_specs(args)
    print(f"[g-a] device={device} seeds={args.seeds} folds={args.subjects}")
    print(f"[g-a] gate: beat mean {G3_T2_MEAN} +/- {G3_T2_SD} WITHOUT worse fold SD; "
          f"SCORE {SCORE_MEAN} +/- {SCORE_SD}")

    per_run: dict[str, list[float]] = {k: [] for k in specs}
    per_fold: dict[str, dict] = {}
    for seed in args.seeds:
        for s in args.subjects:
            ckpt = Path(args.stage1_root) / f"sub{s:02d}_k20_seed{seed}" / "last.pt"
            if not ckpt.is_file():
                print(f"[g-a] MISSING {ckpt}")
                continue
            z, g = fold_features(ckpt, s, args, device)
            res, mats = {}, {}
            for name, spec in specs.items():
                report, S = score_arm(z, g, spec, args)
                res[name] = report
                if S is not None:
                    mats[name] = S
                if np.isfinite(report["top1"]):
                    per_run[name].append(report["top1"])
            # Score-level fusion (axiom A5): average the CSLS score matrices of the baseline
            # and the best noise-corrected arm. Kept separate from `specs` because fusion is
            # not a geometry, and it must be run AFTER both members exist.
            for a, b in (("baseline_totalcov_full", "nc_asym_r16"),
                         ("baseline_totalcov_full", "nc_asym_r16_nowhiten"),
                         ("baseline_totalcov_full", "nc_asym_pos")):
                key = f"fuse[{a.split('_',1)[1]}+{b.split('_',1)[1]}]"
                if a in mats and b in mats:
                    fused = 0.5 * (mats[a] + mats[b])
                    rep = calibration.report_with_scores(fused)
                    res[key] = rep
                    per_run.setdefault(key, []).append(rep["top1"])
            per_fold[f"sub{s:02d}_seed{seed}"] = {
                k: {kk: vv for kk, vv in v.items() if kk != "S"} for k, v in res.items()}
            print(f"[g-a] sub{s:02d}/s{seed} " + " ".join(
                f"{n.split('nc_')[-1].replace('baseline_','base_')}={res[n]['top1']:.1f}"
                for n in specs))

    print("\n[g-a] ================ summary (Top-1, all runs) ================")
    print(f"{'arm':<36}{'mean':>7}{'sd':>7}{'n':>4}{'min':>7}{'max':>7}{'vs T2':>8}{'verdict':>9}")
    summary = {}
    for name, v in per_run.items():
        if not v:
            continue
        mean, sd = st.mean(v), (st.pstdev(v) if len(v) > 1 else float("nan"))
        passes = (mean > G3_T2_MEAN) and (sd <= G3_T2_SD)
        gate = "PASS" if passes else ("mean" if mean > G3_T2_MEAN else "-")
        summary[name] = {"mean": mean, "sd": sd, "n": len(v), "gate": gate}
        print(f"{name:<36}{mean:>7.2f}{sd:>7.2f}{len(v):>4}{min(v):>7.1f}{max(v):>7.1f}"
              f"{mean - G3_T2_MEAN:>+8.2f}{gate:>9}")

    base = summary.get("baseline_totalcov_full", {})
    print(f"\n[g-a] baseline = {base.get('mean', float('nan')):.2f} +/- "
          f"{base.get('sd', float('nan')):.2f}  (banked G3: {G3_T2_MEAN} +/- {G3_T2_SD})")
    if abs(base.get("mean", 0) - G3_T2_MEAN) > 1.0:
        print("[g-a] !! BASELINE DOES NOT REPRODUCE G3 -- verdict below is VOID")

    out = {"gate": {"t2_mean": G3_T2_MEAN, "t2_sd": G3_T2_SD,
                    "score_mean": SCORE_MEAN, "score_sd": SCORE_SD},
           "summary": summary, "per_fold": per_fold}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    print(f"[g-a] wrote {args.out}")


if __name__ == "__main__":
    main()
