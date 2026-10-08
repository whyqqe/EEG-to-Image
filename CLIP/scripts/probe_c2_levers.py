#!/usr/bin/env python
"""probe_c2_levers -- is the repetition lever real, and what gate lets it fire?

WHY THIS EXISTS. The v5 master plan's bottleneck table puts "single-trial EEG noise" first
and notes that it is UNEXPLOITED: on this fold, going from n=5 to n=10 repetitions of the
SAME subject/concept takes 500-way retrieval from 10.8% to **21.4%** -- it roughly doubles.
That gain is available at TEST TIME, with no training and no labels, which is why C2
(multi-repetition transductive refinement) is priced at +4/+8/+12pp in the v6 plan.

It measured ZERO: on the chained pipeline's checkpoints every route reported
`n_validated = 0` out of 103/91/54 mutual landmarks -- the `rep_agreement >= 0.5` gate
rejected every single landmark. "The lever is worthless" and "the gate is set at a value
no real encoder can reach" are very different conclusions, and one CPU pass over an
EXISTING checkpoint separates them. Nothing here trains anything.

WHAT IT MEASURES, per configuration:
  * the standard deployed rungs, so every row is read on the same ladder
    (raw cosine / +CSLS / +whiten+CSLS / +whiten+CSLS+recovery)
  * the repetition cloud used two ways:
      - `rep-whiten+CSLS`   better-conditioned whitening (16000 samples, not 200)
      - `rep-mean-scores`   average the PER-REPETITION CSLS score matrices, i.e. let the
                            repetitions vote in score space instead of in EEG space. This
                            is a genuinely different operator from averaging the trials
                            first, and it is the "n=10 halves the noise" effect expressed
                            as an ensemble rather than as a smoother input.
  * `C2(gate=g)` for a sweep of g, so the gate is DERIVED from the data instead of
    assumed, and the agreement distribution is dumped to justify the choice.

THE PROTOCOL LINE. Every `rep-*` row is T2/transductive: it uses the held-out subject's
own unlabelled test trials. It must never be quoted next to a T1 number as if they were the
same claim, and it may not be used to pick a checkpoint (the epoch is fixed before this runs).

Run (CPU, minutes):
    python scripts/probe_c2_levers.py \
        --ckpt outputs/stage1/v5-a1-k20/last.pt --target-subject 8 \
        --gates 0.0 0.05 0.1 0.15 0.2 0.3 0.4 \
        --out outputs/probe/c2_levers.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from samclip import calibration, config, evaluate  # noqa: E402
from samclip.data import things_eeg  # noqa: E402
from samclip.data.targets import load_target_stack  # noqa: E402
from samclip.models import build_model  # noqa: E402


def _channels(cfg: dict):
    return (config.CHANNELS_OCCIPITO_PARIETAL
            if cfg.get("channel_set", "all63") == "occipital17" else None)


def _top(scores: np.ndarray) -> dict:
    return calibration.report_with_scores(scores)


def _agreement_profile(z_reps: np.ndarray, g: np.ndarray, k: int) -> dict:
    """The distribution of per-repetition top-1 agreement with the averaged query.

    Computed with the SAME CSLS statistic the refinement uses, so the numbers are the ones
    the gate actually sees. Returning percentiles rather than a single mean is the point:
    a gate has to be chosen against the TOP of this distribution, and a mean hides whether
    there is any tail at all.
    """
    C, R, d = z_reps.shape
    qn = z_reps.mean(axis=1)
    qn = qn / np.maximum(np.linalg.norm(qn, axis=-1, keepdims=True), 1e-8)
    gn = g / np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-8)
    s = calibration.csls_scores(qn, gn, k=k)
    fwd, bwd = s.argmax(axis=1), s.argmax(axis=0)
    mutual = fwd[bwd] == np.arange(C)
    g_idx = np.nonzero(mutual)[0]
    if g_idx.size == 0:
        return {"n_mutual": 0}
    q_idx = bwd[g_idx]
    zf = z_reps.reshape(C * R, d)
    rn = zf / np.maximum(np.linalg.norm(zf, axis=-1, keepdims=True), 1e-8)
    rep_top = calibration.csls_scores(rn, gn, k=k).argmax(axis=1).reshape(C, R)
    agree = np.array([float((rep_top[i] == g_idx[t]).mean()) for t, i in enumerate(q_idx)])
    q = np.percentile(agree, [50, 75, 90, 95, 99, 100])
    return {
        "n_mutual": int(g_idx.size),
        "agree_p50_p75_p90_p95_p99_max": [round(float(x), 4) for x in q],
        "n_agree_ge_0.1": int((agree >= 0.1).sum()),
        "n_agree_ge_0.2": int((agree >= 0.2).sum()),
        "n_agree_ge_0.3": int((agree >= 0.3).sum()),
        "n_agree_ge_0.5": int((agree >= 0.5).sum()),
    }


def _rep_mean_scores(z_reps: np.ndarray, g: np.ndarray, k: int,
                     whiten: bool = True) -> np.ndarray:
    """Average the PER-REPETITION CSLS score matrices, after whitening the rep cloud.

    Different operator from `refine_with_reps`'s averaged query, and worth separating: it
    lets the repetitions disaggregate a single bad trial instead of letting that trial move
    the average. The whitening map and the CSLS row/column statistics are computed on the
    (C*R) cloud and on the averaged queries respectively, matching the deployment
    convention that the statistic belongs to the RETRIEVAL SET.
    """
    C, R, d = z_reps.shape
    zf = z_reps.reshape(C * R, d).astype(np.float64)
    if whiten:
        mu, w_map, _ = calibration._whiten_from_cloud(zf, shrink=0.1)
    else:
        mu, w_map = np.zeros((1, d)), np.eye(d)
    q_bar = (z_reps.mean(axis=1).astype(np.float64) - mu) @ w_map
    gn = g / np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-8)
    rn = (zf - mu) @ w_map
    rn = rn / np.maximum(np.linalg.norm(rn, axis=-1, keepdims=True), 1e-8)
    per_rep = calibration.csls_scores(rn, gn, k=k).reshape(C, R, g.shape[0])
    return per_rep.mean(axis=1), q_bar


def run(ckpt_path: Path, target_subject: int, args, device) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mcfg = ckpt["cfg"]
    channels = _channels(mcfg)
    mvnn = args.mvnn if args.mvnn is not None else \
        ("test" if mcfg.get("mvnn", "off") != "off" else "off")
    img = mcfg.get("image", {}) or {}
    fs = img.get("feature_set", "clip_h14_multilevel")
    layers = img.get("layers")

    _, test = things_eeg.load_subject_std(target_subject, channels, mvnn=mvnn)
    targets = load_target_stack(feature_set=fs, layers=layers, split="test")
    model = build_model(mcfg, targets.shape[2], targets.shape[-1]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    loader = DataLoader(things_eeg.TestDataset(np.asarray(test), targets),
                        batch_size=200, shuffle=False, collate_fn=things_eeg.collate)
    feats = evaluate.extract_features(model, loader, device)

    out: dict = {"ckpt": str(ckpt_path), "epoch": ckpt.get("epoch"),
                 "target_subject": int(target_subject), "mvnn": mvnn,
                 "feature_set": fs, "rows": {}}
    q, g = feats["eeg"], feats["img"]

    # ---- the deployed T1 ladder, recomputed here so every row shares one code path ----
    for name, kw in (
        ("T1 raw cosine", dict()),
        ("T1 + CSLS", dict(csls=True)),
        ("T1 + whiten + CSLS", dict(whiten=True, csls=True)),
        ("T1 + whiten + CSLS + recovery", dict(whiten=True, csls=True, recovery=True)),
    ):
        sc, _ = calibration.calibrate(q, g, k=args.csls_k, rho=args.rho, **kw)
        out["rows"][name] = _top(sc)

    # ---- the repetition cloud (T2) ---------------------------------------------------
    # Single-route only, deliberately: on a multi-route model `embed_reps` returns one
    # cloud per route and the question this probe answers ("can the repetition lever fire
    # at all on a checkpoint we can already score at 35.50?") would be confounded by the
    # fusion, which V1 has already been shown to lose on. Cheaper to answer it clean.
    if list(getattr(model, "route_names", [])):
        raise SystemExit("[c2] multi-route checkpoints are out of scope for this probe; "
                         "run per-route refinement through scripts/run_eval.py instead.")
    reps = things_eeg.load_test_reps(target_subject, channels, mvnn=mvnn)
    z = evaluate.embed_reps(model, np.asarray(reps), device, batch=args.rep_batch)
    out["n_reps"] = int(np.asarray(reps).shape[1])
    out["agreement_profile"] = _agreement_profile(z, g, args.csls_k)

    # (a) better-conditioned whitening, no fitted rotation
    sc, dg = calibration.refine_with_reps(z, g, k=args.csls_k, rho=args.rho,
                                          rep_agreement=0.0)  # gate open: pure rep-whiten
    out["rows"]["T2 rep-whiten + CSLS (no gate)"] = {**_top(sc), "diag": {
        k: v for k, v in dg.items() if not hasattr(v, "shape")}}

    # (b) the gate sweep -- the question this probe exists to answer
    out["gate_sweep"] = {}
    for gate in args.gates:
        sc, dg = calibration.refine_with_reps(z, g, k=args.csls_k, rho=args.rho,
                                              rep_agreement=gate)
        key = f"T2 C2(gate={gate:g})"
        out["rows"][key] = {**_top(sc),
                            "n_validated": int(dg.get("n_validated") or 0),
                            "n_mutual": int(dg.get("n_mutual") or 0),
                            "reason": dg.get("reason", "fitted")}
        out["gate_sweep"][f"{gate:g}"] = {
            "top1": _top(sc)["top1"], "n_validated": int(dg.get("n_validated") or 0)}

    # (c) repetitions voting in SCORE space rather than in EEG space
    sc_rep_mean, _ = _rep_mean_scores(z, g, args.csls_k, whiten=True)
    out["rows"]["T2 rep-mean-scores + CSLS"] = _top(sc_rep_mean)

    # (d) rep-cloud whitening, then the deployed recovery -- the T2 analogue of the T1
    # rung we are trying to beat, so the comparison is mechanism-for-mechanism. The
    # operator lives in `calibration.rep_cloud_scores` (shared with `probe_rep_dose` and
    # `run_eval`); it was inlined here first, and `probe_rep_dose` copied it.
    sc_b, rdiag = calibration.rep_cloud_scores(z, g, k=args.csls_k, rho=args.rho)
    out["rows"]["T2 rep-whiten + recovery + CSLS"] = {**_top(sc_b),
                                                      "recovery": rdiag.get("landmark_rate")}

    # (e) THE QUESTION THE PROBE EXISTS FOR, part two: does the repetition cloud ADD to the
    # best T1 rung, or does it only move the same error around? A T2 mechanism that beats
    # its own ablation but not T1 is worthless, so both directions are scored: each T2
    # matrix fused with the banked T1 best, and the two T2 mechanisms fused with each other.
    t1_best, _ = calibration.calibrate(q, g, k=args.csls_k, rho=args.rho,
                                       whiten=True, csls=True, recovery=True)
    for name, parts in (
        ("T1best + rep-mean-scores", (t1_best, sc_rep_mean)),
        ("T1best + rep-whiten+recovery", (t1_best, sc_b)),
        ("rep-mean-scores + rep-whiten+recovery", (sc_rep_mean, sc_b)),
    ):
        out["rows"][f"T2 fused: {name}"] = _top(calibration.fuse_scores(list(parts)))

    # cache the two clouds so every later question about this checkpoint is seconds, not
    # minutes -- the repetition pass is the only expensive part of this probe.
    if args.cache:
        np.savez(args.cache, q=q, g=g, z_reps=z)
        print(f"[c2] cached features -> {args.cache}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--target-subject", type=int, required=True)
    ap.add_argument("--mvnn", choices=["off", "train", "test"], default=None)
    ap.add_argument("--csls-k", type=int, default=10)
    ap.add_argument("--rho", type=float, default=0.1)
    ap.add_argument("--gates", type=float, nargs="+",
                    default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5])
    ap.add_argument("--rep-batch", type=int, default=256,
                    help="batch for the (C*R)-trial repetition pass; 8 is the library "
                         "default and is far too slow for 16000 trials on CPU")
    ap.add_argument("--cache", default=None,
                    help="npz path for (q, g, z_reps); the repetition pass is the only "
                         "expensive part, so caching makes later questions seconds")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[c2] device={device} ckpt={args.ckpt}")
    res = run(Path(args.ckpt), args.target_subject, args, device)

    print(f"\n{'row':<38} {'Top-1':>7} {'Top-5':>7} {'n_valid':>8} {'n_mutual':>9}")
    print("-" * 74)
    for name, m in res["rows"].items():
        print(f"{name:<38} {m['top1']:>7.2f} {m['top5']:>7.2f} "
              f"{m.get('n_validated', '-'):>8} {m.get('n_mutual', '-'):>9}")
    print("-" * 74)
    print("agreement profile:", json.dumps(res["agreement_profile"]))
    print(f"reps per concept: {res['n_reps']}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=2, default=str))
        print(f"[c2] wrote {args.out}")


if __name__ == "__main__":
    main()
