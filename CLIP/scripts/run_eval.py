#!/usr/bin/env python
"""Retrieval evaluation for one LOSO fold. Frozen features only, no generation.

THE REPORT HAS ONE AXIS NOW, AND THAT IS THE POINT
--------------------------------------------------
v1 reported two axes: ``shared`` vs ``conditioned`` (the trunk modulated by a subject
vector inferred from unlabelled support trials) crossed with geometry. The conditioning
axis is gone with the mechanism -- there is no ``z_s``, so there is nothing to compare
it against -- and the remaining axis is the one that actually moves the number:

    raw cosine -> + centre -> + SAW whiten -> + CSLS -> + whiten + CSLS -> + recovery

``+ centre`` is the rung that matters and it is reported FIRST on purpose. Measured on
this fold across three seeds (`probe_signal_weights.py`), subtracting the query cloud's
mean accounts for nearly all of the geometric gain (+6.5/+4.5/+2.5 points), while the
per-dimension signal-to-noise-optimal reweighting -- which needs the labels -- reaches
only 21.0/20.0/19.5. That bounds the entire coordinate-change family, so a report that
jumps straight from "raw cosine" to "+ whiten + CSLS" attributes the translation half of
the gain to the covariance half. See `samclip.calibration`'s module docstring.

   20|Nothing here needs a GPU: the encoder runs once over 200 test trials, and every stage
after that is numpy. The sbatch for this stage uses the CPU partition.

Protocol alignment (plan doc §8)
--------------------------------
  * 200-way retrieval, correct pairing on the diagonal, cosine similarity
  * final-epoch checkpoint (no test-based selection)
  * reference cell to beat: the SAMGA official code run on OUR data, this fold,
    final epoch = 22.00 / 50.00 (best epoch 25.00 / 51.00)

Run:
  python scripts/run_eval.py --ckpts outputs/stage1/sub-08/last.pt \
      --target-subject 8 --mvnn test --recovery
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

# The reference cell to beat, measured ON OUR DATA AND THIS FOLD with the SAMGA official
# code (`eeg-retrieval/scripts/run_samga_official_baseline.sh`, `inter.sh`'s full
# argument list, seed 2025): final epoch 22.00 / 50.00, best epoch 25.00 / 51.00.
#
# NOT 26.22 / 57.98. That number is SCORE's re-measurement of a SAMGA encoder and it is
# the right thing to quote when the model is a SAMGA encoder; it is the wrong thing to
# put in this column, because we are not scoring a SAMGA encoder and we cannot tell
# whether sub-08 is an easy or a hard fold of the ten the 34.4 published average is taken
# over (`compute_avg_results.py` averages `best top1 acc` over 10 folds). Quoting it here
# would make every row look 4 points worse than it is and send the project chasing a
# target that this fold's measurements do not support -- see the v4 doc §1.5.
REF_TOP1, REF_TOP5 = 22.00, 50.00

# ...and that number is the **sub-08 cell ONLY**. We have never run the SAMGA official
# baseline on the other nine folds, so on fold != 8 the default above is a FABRICATED
# comparison: a row printed as `-9.50` against a reference that was measured on a
# different subject reads as a real deficit and is not one. `--ref-top1 unknown` makes the
# report say "not measured" and print `n/a` instead of a delta (the G3 10-fold sweep uses
# this for folds != 8, and compares the 10-fold MEAN against the published 26.22 / 53.23
# final-epoch numbers, which IS an apples-to-apples claim). `--ref-top1 <f> --ref-top5 <f>`
# records a reference we did measure. `--ref-top1` alone is refused: half a reference is
# how a Top-1 delta ends up next to an unrelated Top-5.
REF_NOTE = ("SAMGA official code run on our data, THIS fold, final epoch (best epoch "
            "25.00/51.00). NOT the 26.22/57.98 that SCORE re-measured on a SAMGA encoder, "
            "and NOT the 34.4 10-fold average -- see the comment above REF_TOP1.")
REF_UNKNOWN_NOTE = ("NOT MEASURED for this fold -- the SAMGA official baseline was only "
                    "run on sub-08. No per-fold reference exists, so no delta is printed; "
                    "compare the 10-fold MEAN against 26.22 (SAMGA encoder, final epoch) "
                    "and 53.23 (SCORE).")

_UNKNOWN_REF = {"unknown", "none", "null", "n/a", "na"}


def _ref_delta(top1: float) -> str:
    """`ΔTop-1 vs ref` as text, or `n/a` when this fold has no measured reference.

    A wrong reference is worse than no reference: it turns "we have no baseline here"
    into "we are 9.5 points behind", and the second reads like a fact.
    """
    if REF_TOP1 is None:
        return "n/a"
    return f"{top1 - REF_TOP1:+.2f}"


def fuse_scores(score_list: list[np.ndarray], normalize: bool = True,
                weights: np.ndarray | None = None) -> np.ndarray:
    """Deployed sum-rule fusion -- see `calibration.fuse_scores`.

    Kept as a thin re-export rather than a second implementation: the training term
    (`losses.contrastive.score_fusion_loss`) and the two eval call sites must all be the
    same operator, and three copies of a formula is how two of them drift.
    """
    return calibration.fuse_scores(score_list, normalize=normalize, weights=weights)


def _feats_to_scores(feats: dict, args) -> dict[str, dict]:
    """The full calibration ladder on ONE route's features -> ``{row name: score matrix}``.

    Split out so the multi-route path can build the same ladder per route and then fuse
    it; duplicating the ladder for the fused case would let the two drift, and the fused
    row is meant to be comparable to the per-route rows rung for rung.
    """
    stages = {
        "+ CSLS": dict(center=False, whiten=False, csls=True, recovery=False),
        "+ centre": dict(center=True, whiten=False, csls=False, recovery=False),
        "+ centre + CSLS": dict(center=True, whiten=False, csls=True, recovery=False),
        "+ SAW whiten": dict(center=False, whiten=True, csls=False, recovery=False),
        "+ whiten + CSLS": dict(center=False, whiten=True, csls=True, recovery=False),
    }
    if args.recovery:
        stages["+ whiten + CSLS + recovery"] = dict(whiten=True, csls=True,
                                                    recovery=True)
        # SCORE's own composition, and the rung we were missing: recovery starts from
        # its internal per-DIMENSION moment matching, not from our full-covariance SAW
        # whitening. The two are different operators, and SAW whitening measurably loses
        # 2.5pp before any recovery is applied (34.00 -> 31.50), so feeding it to the
        # recovery may be handicapping the step that carries SCORE's largest test-phase
        # gain. Both rows are reported so the composition stays visible instead of being
        # assumed.
        stages["+ CSLS + recovery"] = dict(center=False, whiten=False, csls=True,
                                           recovery=True)
    out: dict[str, dict] = {}
    for name, kw in stages.items():
        scores, diag = calibration.calibrate(
            feats["eeg"], feats["img"], k=args.csls_k, rho=args.rho,
            min_landmark_rate=args.min_landmark_rate, recovery_fn=RECOVERY_FN, **kw)
        out[name] = {
            "scores": scores,
            "report": calibration.report_with_scores(scores),
            "stages": kw,
            "diag": {k: v for k, v in diag.items() if k != "recovery_diag"},
        }
        if "recovery_diag" in diag:
            out[name]["recovery_diag"] = diag["recovery_diag"]
    return out


def _ckpt_tag(path: Path) -> str:
    """A label that stays unique when one report holds several checkpoints.

    `path.parent.name` is NOT unique: two arms of an ablation both live under a
    directory called `sub-08`. The report is a dict keyed by tag, so the second would
    silently replace the first and a two-arm comparison would become a one-arm report.
    Two path components keep the arm name in the label.
    """
    p = Path(path)
    parent, grand = p.parent.name, p.parent.parent.name
    return f"{grand}/{parent}" if grand and grand not in (".", "/", "") else parent


def _load_fold_arrays(target_subject: int, channels, mvnn: str):
    """Target subject's (train, test) EEG on ONE shared normalisation.

    The z-score statistics come from the TRAIN split, never the test split, so no
    test-set scale leaks in. They are label-free (they use the EEG, not the concept
    identities), which is why applying them to a held-out subject is legitimate under a
    strict LOSO protocol rather than a hidden calibration.
    """
    tr, te = things_eeg.load_subject_std(target_subject, channels, mvnn=mvnn)
    return np.asarray(tr), np.asarray(te)


def _evaluate_v6(ckpt: dict, ckpt_path: Path, target_subject: int, args, device) -> dict:
    """The v6 report: per-route rows AND the deployed fused rows, on one calibration ladder.

    Both are reported on purpose. The fused row is the deployable number, but a fused
    number alone cannot tell "three routes each contributing" from "one route carrying the
    other two" -- and the second is a common and quiet outcome of late fusion. The
    per-route rows are what make a collapsed route visible.
    """
    from samclip.models.multiroute import MultiRouteSAMCLIP, resolve_routes

    model_cfg = ckpt["cfg"]
    channel_set = model_cfg.get("channel_set", "all63")
    channels = (config.CHANNELS_OCCIPITO_PARIETAL
                if channel_set == "occipital17" else None)
    mvnn = args.mvnn if args.mvnn is not None else \
        ("test" if model_cfg.get("mvnn", "off") != "off" else "off")

    routes = resolve_routes(model_cfg)
    names = [str(r["name"]) for r in routes]
    primary = names[0]
    stacks = {str(r["name"]): load_target_stack(str(r["feature_set"]),
                                                list(r["layers"]), "test")
              for r in routes}
    targets_te = stacks[primary]

    _, test = _load_fold_arrays(target_subject, channels, mvnn)
    model = MultiRouteSAMCLIP(model_cfg, routes, model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    loader = DataLoader(
        things_eeg.TestDataset(test, targets_te,
                               extra_targets={n: stacks[n] for n in names
                                              if n != primary}),
        batch_size=200, shuffle=False, collate_fn=things_eeg.collate)

    feats = evaluate.extract_route_features(model, loader, device)
    per_route = {n: _feats_to_scores(feats[n], args) for n in names}

    out: dict = {
        "ckpt": str(ckpt_path),
        "epoch": ckpt.get("epoch", ckpt.get("episode")),
        "channel_set": channel_set,
        "feature_set": f"v6[{'+'.join(names)}]",
        "target_layers": {n: [int(x) for x in routes[i]["layers"]]
                          for i, n in enumerate(names)},
        "target_fusion": model_cfg.get("target_fusion", "mean"),
        "arch": model_cfg.get("arch", "v3"),
        "objective": "v6",
        "routes": names,
        "primary": primary,
        "fusion_weight": float((model_cfg.get("fusion") or {}).get("weight", 1.0)),
        "fusion_normalize": bool((model_cfg.get("fusion") or {}).get("normalize", True)),
        "d_align": int(getattr(model, "d_align", model_cfg.get("d_embed", 0))),
        "smn_gate": model.smn_gate(),
        "target_layer_weights": model.target_layer_weights(),
        "mvnn": mvnn,
        "n_queries": int(test.shape[0]),
        "offset_ratio": model.subject_offset_ratio(
            torch.as_tensor(feats[primary]["eeg_raw"], dtype=torch.float32,
                            device=device)),
        "rows": {},
        "route_rows": {n: {k: v["report"] for k, v in per_route[n].items()}
                       for n in names},
    }

    # ---- the deployed fused ladder -------------------------------------------------
    # Rung for rung over the union of stage names the per-route ladders produced, so a
    # row that exists for a route is fused too. Fusing a row one route lacks would
    # silently fuse a shorter list.
    row_names = [k for k in per_route[primary]]
    for n in names:
        missing = [k for k in row_names if k not in per_route[n]]
        if missing:
            raise RuntimeError(f"route {n!r} is missing ladder rows {missing}; the "
                               f"routes must be scored on the SAME rungs or their fused "
                               f"row would combine different operators")
    for name in row_names:
        fused = fuse_scores([per_route[n][name]["scores"] for n in names],
                            normalize=out["fusion_normalize"])
        rep = calibration.report_with_scores(fused)
        out["rows"][name] = rep
        out["rows"][name]["diag"] = {"fused": True,
                                     "n_routes": len(names),
                                     "normalize": out["fusion_normalize"]}
        # The BEST single route on this rung, so "fusion beat the best route" is readable
        # directly from the report rather than by eye across three sub-tables.
        best = max(names, key=lambda n: per_route[n][name]["report"]["top1"])
        out["rows"][name]["best_route"] = best
        out["rows"][name]["best_route_top1"] = \
            per_route[best][name]["report"]["top1"]

    # ---- C2: transductive refinement on the UNAVERAGED test repetitions -------------
    if args.reps:
        reps = things_eeg.load_test_reps(target_subject, channels, mvnn=mvnn)
        z_reps = evaluate.embed_reps(model, reps, device)
        if isinstance(z_reps, dict):
            z_reps = {n: z_reps[n] for n in names}
        else:
            z_reps = {names[0]: z_reps}
        route_scores = []
        route_diags = {}
        for n in names:
            s, d = calibration.refine_with_reps(
                z_reps[n], feats[n]["img"], k=args.csls_k, rho=args.rho,
                rep_agreement=args.rep_agreement)
            route_scores.append(s)
            route_diags[n] = {k: d[k] for k in d if not hasattr(d[k], "shape")}
        fused = fuse_scores(route_scores, normalize=out["fusion_normalize"])
        out["rows"]["+ fused + C2 reps"] = {
            **calibration.report_with_scores(fused),
            "diag": {"fused": True, "n_routes": len(names),
                     "route_diags": route_diags},
        }
        # The best single route under C2, for the same attribution reason as above.
        c2 = [calibration.report_with_scores(s)["top1"] for s in route_scores]
        best_i = int(np.argmax(c2))
        out["rows"]["+ fused + C2 reps"]["best_route"] = names[best_i]
        out["rows"]["+ fused + C2 reps"]["best_route_top1"] = float(c2[best_i])
    return out


def _evaluate_checkpoint(ckpt_path: Path, target_subject: int, args, device) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_cfg = ckpt["cfg"]
    if str(model_cfg.get("objective", "v3")) == "v6":
        return _evaluate_v6(ckpt, ckpt_path, target_subject, args, device)

    # ---- derive the data geometry from the CHECKPOINT, not from CLI flags -------
    # A channel-set or feature-set mismatch between train and eval loads fine and
    # silently scores a different task, so the checkpoint is the single source of
    # truth for everything that has to match.
    channel_set = model_cfg.get("channel_set", "all63")
    channels = (config.CHANNELS_OCCIPITO_PARIETAL
                if channel_set == "occipital17" else None)
    img = model_cfg.get("image", {}) or {}
    feature_set = img.get("feature_set", "clip_h14_multilevel")
    layers = img.get("layers")

    # Training enables MVNN for source subjects on their train split; the held-out
    # subject's matching choice is its own test split. `--mvnn off` reproduces the
    # no-whitening ablation and must match a training run that used `mvnn: off`.
    mvnn = args.mvnn if args.mvnn is not None else \
        ("test" if model_cfg.get("mvnn", "off") != "off" else "off")

    _, test = _load_fold_arrays(target_subject, channels, mvnn)
    targets_te = load_target_stack(feature_set, layers, "test")

    model = build_model(model_cfg, targets_te.shape[2], targets_te.shape[-1]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # ---- M1: dump the SOURCE-metric template (docs/eeg2image_v10_pipeline.md §M1) ---------
    # Written from INSIDE the fold that owns this encoder, on purpose. The template must be the
    # same 9 subjects this checkpoint was trained on, embedded by THIS checkpoint, on the SAME
    # normalisation -- and the only way to guarantee that is to let the code that already does
    # all three produce it. A separate builder script would re-derive channels/mvnn/feature-set
    # from the config and would drift the first time one of those defaults changed.
    #
    # Concept indices line up across subjects by construction: THINGS-EEG2 tests every subject
    # on the same 200 concepts, so `src_means[s]` is the source subject's mean embedding of the
    # SAME concept as `z_reps[c]`. No alignment step is needed to make the template comparable.
    if args.dump_src_metric:
        src_ids = [s for s in config.all_subjects() if s != target_subject]
        means = []
        for s in src_ids:
            reps_s = things_eeg.load_test_reps(s, channels, mvnn=mvnn)
            z_s = evaluate.embed_reps(model, reps_s, device)
            if isinstance(z_s, dict):
                z_s = z_s[next(iter(z_s))]
            means.append(np.asarray(z_s).mean(axis=1))       # (C, d), repetition-averaged
        src_means = np.stack(means, axis=0)                   # (S, C, d), RAW (post-SMN) space
        np.savez(args.dump_src_metric, src_means=src_means,
                 src_ids=np.asarray(src_ids), subject=target_subject,
                 epoch=int(ckpt.get("epoch", ckpt.get("episode", -1))))
        print(f"[eval] M1 source template: {src_means.shape} from subjects {src_ids} "
              f"-> {args.dump_src_metric}")

    loader = DataLoader(things_eeg.TestDataset(test, targets_te), batch_size=200,
                        shuffle=False, collate_fn=things_eeg.collate)

    layer_w = model.target_layer_weights()
    out: dict = {
        "ckpt": str(ckpt_path),
        "epoch": ckpt.get("epoch", ckpt.get("episode")),
        "channel_set": channel_set,
        "feature_set": feature_set,
        "target_layers": [int(x) for x in (layers or [])],
        "target_fusion": model_cfg.get("target_fusion", "mean"),
        # The topology and the objective are recorded per row because the same tag can
        # now be evaluated under either, and "which objective produced this" is not
        # recoverable from a score.
        "arch": model_cfg.get("arch", "v3"),
        "objective": model_cfg.get("objective", "v3"),
        "d_align": int(getattr(model, "d_align", model_cfg.get("d_embed", 0))),
        "smn_gate": model.smn_gate(),
        # The learned layer blend, so a `routed` run can be read back: which of the five
        # CLIP layers the model chose to trust is the interpretable part of the target
        # design. None for `mean` (uniform by construction).
        "target_layer_weights": layer_w,
        "mvnn": mvnn,
        "n_queries": int(test.shape[0]),
        "rows": {},
    }

    feats = evaluate.extract_features(model, loader, device)
    out["rows"]["raw cosine"] = evaluate.retrieval_report(feats["eeg"], feats["img"])
    out["offset_ratio"] = model.subject_offset_ratio(
        torch.as_tensor(feats["eeg_raw"], dtype=torch.float32, device=device))

    # ---- geometric calibration ladder -------------------------------------------
    # Every rung is reported, in increasing order of intervention, so the report shows
    # which operation bought what. Reporting only the final number (as a single
    # "our model" row) is exactly how a coordinate change gets mis-attributed to an
    # architecture, and `recovery` is skipped in the label when the flag is off instead
    # of emitting a "+ recovery" row that silently equals "+ whiten + CSLS".
    #
    # `+ centre` and `+ centre + CSLS` are the new rows. They exist because the old set
    # could not distinguish "the mean was displaced" from "the covariance is informative"
    # -- `saw_whiten` does both, and the measurement says the first is worth
    # +6.5/+4.5/+2.5 points while the second is worth at most ~1 (`probe_signal_weights.py`).
    # Both old row names are kept unchanged so the reports already on disk stay
    # comparable row-for-row.
    stages = {
        "+ CSLS": dict(center=False, whiten=False, csls=True, recovery=False),
        "+ centre": dict(center=True, whiten=False, csls=False, recovery=False),
        "+ centre + CSLS": dict(center=True, whiten=False, csls=True, recovery=False),
        "+ SAW whiten": dict(center=False, whiten=True, csls=False, recovery=False),
        "+ whiten + CSLS": dict(center=False, whiten=True, csls=True, recovery=False),
    }
    if args.recovery:
        stages["+ whiten + CSLS + recovery"] = dict(whiten=True, csls=True,
                                                    recovery=True)
        # SCORE's own composition, and the rung this ladder was missing: SCORE applies
        # recovery to per-DIMENSION moment-matched features, not to our full-covariance
        # SAW whitening. `recover.py` does its own moment matching internally, so pairing
        # it with SAW whiten feeds it an input the paper never uses -- and SAW whiten
        # measurably costs 2.5pp before any recovery runs (34.00 -> 31.50). Reported
        # beside the old row so the composition is a measurement, not an assumption.
        stages["+ CSLS + recovery"] = dict(center=False, whiten=False, csls=True,
                                           recovery=True)
    row_scores: dict[str, np.ndarray] = {}
    fused_scores: dict[str, np.ndarray] = {}
    for name, kw in stages.items():
        scores, diag = calibration.calibrate(
            feats["eeg"], feats["img"], k=args.csls_k, rho=args.rho,
            min_landmark_rate=args.min_landmark_rate, recovery_fn=RECOVERY_FN, **kw)
        row_scores[name] = scores
        out["rows"][name] = {
            **calibration.report_with_scores(scores),
            "diag": {k: v for k, v in diag.items()
                     if k not in ("recovery_diag",)},
            **({"recovery_diag": diag["recovery_diag"]}
               if "recovery_diag" in diag else {}),
        }

    # ---- E2: SATTC's structural expert (the POINT-BASED baseline) -------------------
    # SATTC (CVPR 2026) is the field's strongest label-free test-time CALIBRATION method, and it
    # operates on the similarity matrix of the MEAN embedding -- an order-1 object. It is
    # therefore the baseline COMMET (order-2 metric from the repetition cloud) has to beat, and
    # the comparison must be PAIRED on the same encoder. `lam=0` is emitted as its own row so the
    # structural expert's contribution is isolated (lam=0 recovers pure geometry, since the PoE
    # reduces to a per-row rescaling, which cannot change a per-row ranking).
    if args.sim_calib == "satc":
        bases = [b.strip() for b in str(args.sim_calib_base).split(",") if b.strip()]
        missing = [b for b in bases if b not in row_scores]
        if missing:
            raise SystemExit(f"[e2] --sim-calib satc needs geometric rows {missing!r}, but the "
                             f"ladder produced {sorted(row_scores)}; pass --sim-calib-base.")
        for base_row in bases:
            for lam, nm in ((float(args.sim_calib_lam), f"+ SATTC({base_row[2:]})"),
                            (0.0, f"+ SATTC({base_row[2:]}) lam=0")):
                satc, sdg = calibration.structural_scores(
                    row_scores[base_row], k=args.csls_k, lam=lam)
                row_scores[nm] = satc
                out["rows"][nm] = {**calibration.report_with_scores(satc), "diag": sdg}

    # ---- T2: the repetition cloud (v6 pillar C2) ------------------------------------
    # Wired HERE because the single-route path is what every LOSO fold is, and `--reps`
    # used to be accepted and then produce NO row for those checkpoints: the block only
    # existed in `_evaluate_v6`, so a v4 fold trained and evaluated under `--reps` reported
    # a ladder identical to one run without it. A flag that is accepted and silently does
    # nothing is the failure mode this project keeps paying for, so a single-route model
    # now gets the row too. The operator is `calibration.rep_cloud_scores` -- shared with
    # both probes, which is the point of having promoted it out of them.
    if args.reps:
        reps = things_eeg.load_test_reps(target_subject, channels, mvnn=mvnn)
        z_reps = evaluate.embed_reps(model, reps, device)
        if isinstance(z_reps, dict):        # a route-keyed dict only for multi-route
            z_reps = z_reps[next(iter(z_reps))]
        full_R = int(z_reps.shape[1])
        # One value for the whole run: the stage-2 sweep varies it ACROSS runs into
        # separate directories, so a single run never mixes two estimators.
        rep_shrink = float(args.rep_shrink) if args.rep_shrink is not None else 0.1
        # M1 source-metric template, loaded once. `src_means` is (S, C, d) in the encoder's RAW
        # space; `rep_cloud_scores` whitens it with THIS fold's map so the template ends up in
        # the same frame as the target's own geometry.
        src_means = np.load(args.fgw_src_metric)["src_means"] if args.fgw_src_metric else None
        if src_means is not None and args.fgw_src_permute is not None:
            # Shuffle the CONCEPT axis only. Slots keep their position (so `de_src[.]` is still
            # a (C,C) matrix in a fixed frame) but no longer hold the concept the target's query
            # in that slot is about. Geometry is preserved up to relabelling; the correspondence
            # is destroyed. This is the cleanest available separation of the two mechanisms.
            _pr = np.random.default_rng(int(args.fgw_src_permute)).permutation(src_means.shape[1])
            src_means = src_means[:, _pr, :]
            print(f"[eval] CONTROL: source template concept axis permuted "
                  f"(seed={args.fgw_src_permute}); geometry intact, correspondence destroyed")
        # A LIST, not a scalar: the M1 readout is a CURVE in the mix, because the mix is a
        # bias-variance knob with a predicted interior optimum (see the operator's docstring in
        # `calibration._fgw_plan`). One scalar would answer "does it help" but not "where is the
        # peak", and a peak that is NOT interior is the signature of the wrong mechanism.
        src_mixes = [float(x) for x in str(args.fgw_src_mix).split(",") if x.strip() != ""]
        if not src_mixes:
            src_mixes = [0.0]
        if any(m > 0.0 for m in src_mixes) and src_means is None:
            raise SystemExit("--fgw-src-mix > 0 requires --fgw-src-metric: the mix is a blend "
                             "toward a source template that was never supplied.")
        src_mix = src_mixes[0]

        # ---- L2/L3/L4 rows: STRUCTURE ENSEMBLE + RELIABILITY-WEIGHTED FUSION (v11) ------------
        #
        # A SIBLING of the v10 sweep, not a cell inside it: the fusion is a different operator on
        # `de`, and hanging it off `--fgw-sweep` would make it silently absent unless the whole
        # (expensive, unrelated) validation grid was also requested. It ran and emitted nothing the
        # first time this was submitted (job 645706) for exactly that reason, which is the
        # silent-no-op failure this project keeps paying for.
        #
        # `fuse=B` splits the target's own R repetitions into B contiguous blocks, builds a metric
        # per block, weights each block by its agreement with the leave-one-out mean of the others,
        # and feeds the WEIGHTED FUSION to the FGW plan as the structural reference. `fuse=0` is the
        # shipped single-mean metric and is the paired twin of every fused cell, so "the fusion is
        # the only change" is a comparison between two rows that differ in exactly one argument.
        #
        # Licensed by `scripts/probe_fusion.py` (job 645697): pooling raises structure agreement
        # monotonically in the number of pooled views with a correspondence-destroying control at
        # ~0.000. This is NOT a rank reduction -- the four falsified families (P1/P2/A4) all reduced
        # the structural object and lost; fusion averages estimation noise away while keeping every
        # direction. `smoke_test.py` §16 pins the deployed operator (no-op at B<=1, noisy block
        # down-weighted, permutation-equivariant, self-disabling on a pure-noise cloud).
        if args.fgw_struct_fuse is not None:
            fuse_specs = [int(x) for x in str(args.fgw_struct_fuse).split(",")
                          if x.strip() != ""]
            if 0 not in fuse_specs:
                fuse_specs = [0] + fuse_specs              # the paired twin
            # ---- G1: the gallery-side view ensemble (see `_gallery_view_metric`) ----------
            # Computed ONCE per fold and reused by every (R, B) cell, because it depends only
            # on the frozen encoder and the gallery stack -- recomputing it inside the loop
            # would be K extra target passes per cell for a quantity that cannot move.
            gdi_ref, gdi_diag = (None, {"applied": False, "reason": "flag off"})
            if args.fgw_gallery_fuse:
                gdi_ref, gdi_diag = _gallery_view_metric(model, targets_te)
                out["gallery_view_diag"] = gdi_diag
                print(f"[g1] gallery-view metric: {gdi_diag}")
                if gdi_ref is None:
                    print("[g1] WARNING: gallery-fuse requested but not applicable; "
                          "no `,gv` row will be emitted (a row that equals its twin is "
                          "worse than no row).")
            variants = [("", None)] + ([(",gv", gdi_ref)] if gdi_ref is not None else [])
            # Self-contained R set: do NOT borrow the sweep's `r_list`, which only exists when
            # --fgw-sweep is on. The deployed level and the gate-limited level (R=20) are the two
            # the probe's K-curve maps onto.
            r_cells = [None, 20]
            for R_f in r_cells:
                r_f = None if (R_f is None or R_f >= full_R) else int(R_f)
                r_show = full_R if R_f is None else R_f
                for nb in fuse_specs:
                    for vsuffix, vref in variants:
                        ftag = f"R={r_show},a=0.75,t=0.03,fuse={int(nb)}{vsuffix}"
                        for suffix, fn_rec in (
                                ("", _recovery_fn_for(args, force_alpha=0.75, force_tau=0.03)),
                                (" (structural-off)",
                                 _recovery_fn_for(args, force_alpha=0.0, force_tau=0.03))):
                            sc, dg = calibration.rep_cloud_scores(
                                z_reps, feats["img"], k=args.csls_k, rho=args.rho,
                                shrink=rep_shrink,
                                min_landmark_rate=args.min_landmark_rate,
                                recovery_fn=fn_rec, rep_subsample=r_f,
                                src_means=src_means, src_mix=0.0,
                                rep_blocks=int(nb), gallery_di_ref=vref)
                            out["rows"][f"+ T2 {ftag}{suffix}"] = {
                                **calibration.report_with_scores(sc),
                                "diag": {k: v for k, v in dg.items()
                                         if not hasattr(v, "shape")},
                            }

        # ---- O2-MVE: MULTI-FAMILY METRIC VIEW POOLING (v13 §7) ------------------------------
        #
        # `--fgw-struct-fuse` pools ONE family (contiguous blocks) of the repetition cloud.
        # This pools SEVERAL partition families of the same cloud into a single structural
        # reference. The estimator's error is IMSE = bias^2 + sigma^2/K_eff with
        # K_eff = K/(1+(K-1)rho); rho <= 0 was measured on 10/10 folds, so K_eff grows with K
        # LINEARLY here, and each added family is an added independent view rather than a
        # duplicate. `--mve-fuse` is the control that holds the family fixed and varies only the
        # weighting, so "more views" and "better weights" cannot be confused.
        if args.mve_partitions or args.mve_fuse:
            _modes = ([m.strip() for m in str(args.mve_partitions or "").split(",") if m.strip()]
                      if args.mve_partitions else ["cont"])
            _B = int(args.mve_blocks)
            groups, qb_mve = [], None
            for _m in _modes:
                _v, _qb, _ = calibration.cloud_metric_views(
                    z_reps, shrink=rep_shrink, rep_blocks=_B, mode=_m)
                groups.append(_v)
                qb_mve = _qb
            _tag = f"MVE B={_B} views={'+'.join(_modes)}"
            for _sfx, _fn in (("", _recovery_fn_for(args, force_alpha=0.75, force_tau=0.03)),
                              (" (structural-off)",
                               _recovery_fn_for(args, force_alpha=0.0, force_tau=0.03))):
                _sc, _dg = calibration.mve_scores(
                    groups, qb_mve, feats["img"], k=args.csls_k, rho=args.rho,
                    min_landmark_rate=args.min_landmark_rate, recovery_fn=_fn)
                out["rows"][f"+ {_tag}{_sfx}"] = {
                    **calibration.report_with_scores(_sc),
                    "diag": {k: v for k, v in _dg.items() if not hasattr(v, "shape")},
                }
            # The single-family control, named so it is never read as the multi-family result. Its
            # `structural-off` twin is emitted too, because the pair (multi-family off, cont-only
            # off) MUST be bit-identical: with `alpha=0` the structural term is off, so an injected
            # reference cannot reach the output by any path. That equality is the in-file proof that
            # the fusion enters ONLY through the structural term -- a moving twin would mean the
            # reference leaked through the cross-modal cost instead, which is a different and
            # unlicensed mechanism. Checking it in-file avoids comparing across two eval runs whose
            # flags differ, which would not be a valid comparison.
            if args.mve_fuse and "cont" in _modes:
                _g1 = [groups[_modes.index("cont")]]
                for _sfx2, _fn2 in (("", _recovery_fn_for(args, force_alpha=0.75,
                                                          force_tau=0.03)),
                                    (" (structural-off)",
                                     _recovery_fn_for(args, force_alpha=0.0, force_tau=0.03))):
                    _sc, _dg = calibration.mve_scores(
                        _g1, qb_mve, feats["img"], k=args.csls_k, rho=args.rho,
                        min_landmark_rate=args.min_landmark_rate, recovery_fn=_fn2)
                    out["rows"][f"+ {_tag} (cont-only control){_sfx2}"] = {
                        **calibration.report_with_scores(_sc),
                        "diag": {k: v for k, v in _dg.items() if not hasattr(v, "shape")},
                    }

        # ---------------------------------------------------------------- v10 sweep
        # One embedding of the repetition cloud, and EVERY cell of the validation grid is a
        # slice of it, so the whole curve costs one extra operator call per cell rather than a
        # re-embedding. The grid is fixed here and not taken from the CLI on purpose: it is the
        # pre-registered instrument, and letting a failed curve be re-specified from the
        # command line is how a sweep stops being a test.
        #
        #   R  in {80, 40, 20, 10, 5, 1}  at BOTH tau=0.03 (the v10 headline) and tau=0.01
        #      -> the variance-gating curve. Does the structural gain survive fewer reps?
        #         Both temperatures, because the headline is at 0.03 while the v8_fgw075
        #         duplicate is at 0.01, and a curve at one tau cannot be checked against a
        #         reference at the other -- the reference is what says the sweep is the
        #         deployed operator.
        #   alpha in {0.25, 0.5, 0.75, 0.875}  at R=full, both taus
        #      -> the alpha grid for a nested-LOSO choice, ON THE v8 FAMILY that carries the
        #         headline (the banked alpha sweep was on g3, so it cannot select for v8).
        #   tau   in {0.01, 0.03, 0.05}  at R=full, alpha=0.75
        #      -> the temperature grid, same reason.
        #
        # Every cell also emits a STRUCTURAL-OFF twin (alpha forced to 0, same tau, same
        # subset), because the quantity the curve is about is the DIFFERENCE between the two
        # on the IDENTICAL repetition subset. `alpha=0` is bit-identical to the shipped
        # sinkhorn operator (verified), so the twin is the deployment without the term.
        if args.fgw_sweep:
            if getattr(args, "recovery_operator", "hard") != "fgw":
                raise SystemExit("--fgw-sweep needs --recovery-operator fgw (the structural "
                                 "term is what is being swept).")
            r_list = list(args.rep_subsample) if args.rep_subsample else [80, 40, 20, 10, 5, 1]
            # P1 is emitted as its OWN block just below, not as another sweep axis: adding it to
            # the grid would multiply every cell by the rank list, and the rank is not a setting
            # of the same operator -- it is a different operator on `de`. Keeping it off the grid
            # also guarantees the stage-1/2.5 cells stay bit-identical when P1 is off.
            sr_cell = None
            alpha_list = (0.25, 0.5, 0.75, 0.875)
            tau_list = (0.01, 0.03, 0.05)
            # The M1 dimension is appended, not substituted: with `--fgw-src-mix 0` every cell is
            # bit-identical to the stage-1 grid, so a re-run reproduces it and the mix is read as
            # a paired difference against its own 0-twin rather than against an older job.
            mix_list = ([0.0] + [m for m in src_mixes if m > 0.0]) if src_means is not None \
                else [0.0]
            grid = [(r, 0.75, 0.03, m) for r in r_list for m in mix_list]
            grid += [(r, 0.75, 0.01, 0.0) for r in r_list]
            grid += [(None, a, 0.03, 0.0) for a in alpha_list]
            grid += [(None, a, 0.01, 0.0) for a in alpha_list]
            grid += [(None, 0.75, t, 0.0) for t in tau_list]
            # the M1 readout at the two repetition counts the gate cares about: R=80 is the
            # deployed level, R=20 is where the gate bites hardest (gain 3.08 -> 1.12).
            if src_means is not None:
                grid += [(R, 0.75, 0.03, m) for R in (80, 20)
                         for m in src_mixes if m > 0.0]
            done = set()
            print(f"[eval] fgw-sweep: {len(grid)} cells on target sub-{target_subject:02d} "
                  f"(R_full={full_R}, src_mixes={mix_list})")
            for (r_cell, a_cell, t_cell, m_cell) in grid:
                key = (r_cell, a_cell, t_cell, m_cell)
                if key in done:
                    continue
                done.add(key)
                r_sub = None if (r_cell is None or r_cell >= full_R) else int(r_cell)
                r_show = full_R if r_cell is None else r_cell
                mode_sfx = "" if args.fgw_src_ref_mode == "eeg" else f",ref={args.fgw_src_ref_mode}"
                tag = f"R={r_show},a={a_cell:g},t={t_cell:g},m={m_cell:g}{mode_sfx}"
                for suffix, fn_rec in (("", _recovery_fn_for(args, force_alpha=a_cell,
                                                             force_tau=t_cell)),
                                       (" (structural-off)",
                                        _recovery_fn_for(args, force_alpha=0.0,
                                                         force_tau=t_cell))):
                    sc, dg = calibration.rep_cloud_scores(
                        z_reps, feats["img"], k=args.csls_k, rho=args.rho,
                        shrink=rep_shrink, min_landmark_rate=args.min_landmark_rate,
                        recovery_fn=fn_rec, rep_subsample=r_sub,
                        src_means=src_means, src_mix=float(m_cell),
                        src_ref_mode=args.fgw_src_ref_mode,
                        src_calib=bool(args.fgw_src_calib), spec_rank=sr_cell)
                    name = f"+ T2 {tag}{suffix}"
                    out["rows"][name] = {
                        **calibration.report_with_scores(sc),
                        "diag": {k: v for k, v in dg.items() if not hasattr(v, "shape")},
                    }
                    row_scores[name] = sc

            # ---- C1 rows: source-CALIBRATED metric transfer, pre-registered in
            # docs/eeg2image_v10_m1_theory.md §9. Tagged `calib` rather than reusing `m` because
            # it is a different operator on `de`, not a setting of the M1 blend: keeping them on
            # separate tags means the M1 mix curve cannot silently absorb a calibrated cell.
            if args.fgw_src_calib and src_means is not None:
                for R_c in (max(r_list), min(r_list)):
                    r_c = None if (R_c is None or R_c >= full_R) else int(R_c)
                    r_show = full_R if R_c is None else R_c
                    ctag = f"R={r_show},a=0.75,t=0.03,calib"
                    for suffix, fn_rec in (("", _recovery_fn_for(args, force_alpha=0.75,
                                                                 force_tau=0.03)),
                                           (" (structural-off)",
                                            _recovery_fn_for(args, force_alpha=0.0,
                                                             force_tau=0.03))):
                        sc, dg = calibration.rep_cloud_scores(
                            z_reps, feats["img"], k=args.csls_k, rho=args.rho,
                            shrink=rep_shrink, min_landmark_rate=args.min_landmark_rate,
                            recovery_fn=fn_rec, rep_subsample=r_c,
                            src_means=src_means, src_mix=0.0, src_calib=True,
                            spec_rank=sr_cell)
                        out["rows"][f"+ T2 {ctag}{suffix}"] = {
                            **calibration.report_with_scores(sc),
                            "diag": {k: v for k, v in dg.items() if not hasattr(v, "shape")},
                        }
            # ---- P1 rows: SPECTRAL RANK PRIOR (docs/eeg2image_v10_m1_theory.md §10). Separate
            # tags (`spec=`) from M1 (`m=`) and C1 (`calib`) because each is a different operator
            # on `de`, and keeping them apart means no readout can silently absorb another's cell.
            # `spec=full` is included deliberately: it is the paired twin of every truncated cell
            # and must be bit-identical to `m=0`, which is what proves the rank is the only change.
            if args.fgw_spec_from_src or args.fgw_spec_rank is not None:
                rank_specs: list = []
                if args.fgw_spec_rank is not None:
                    for tok in str(args.fgw_spec_rank).split(","):
                        tok = tok.strip()
                        if tok == "":
                            continue
                        rank_specs.append("src" if tok.lower() == "src" else int(tok))
                if args.fgw_spec_from_src and "src" not in rank_specs:
                    rank_specs.append("src")
                if 0 not in rank_specs and None not in rank_specs:
                    rank_specs.append(0)          # the paired full-rank twin
                for R_p in (max(r_list), min(r_list)):
                    r_p = None if (R_p is None or R_p >= full_R) else int(R_p)
                    r_show = full_R if R_p is None else R_p
                    for rs in rank_specs:
                        is_src = (rs == "src")
                        stag = f"R={r_show},a=0.75,t=0.03,spec={'src' if is_src else int(rs)}"
                        for suffix, fn_rec in (
                                ("", _recovery_fn_for(args, force_alpha=0.75,
                                                      force_tau=0.03)),
                                (" (structural-off)",
                                 _recovery_fn_for(args, force_alpha=0.0, force_tau=0.03))):
                            sc, dg = calibration.rep_cloud_scores(
                                z_reps, feats["img"], k=args.csls_k, rho=args.rho,
                                shrink=rep_shrink, min_landmark_rate=args.min_landmark_rate,
                                recovery_fn=fn_rec, rep_subsample=r_p,
                                src_means=src_means, src_mix=0.0,
                                spec_rank=(None if (is_src or int(rs) == 0) else int(rs)),
                                spec_from_src=is_src)
                            out["rows"][f"+ T2 {stag}{suffix}"] = {
                                **calibration.report_with_scores(sc),
                                "diag": {k: v for k, v in dg.items()
                                         if not hasattr(v, "shape")},
                            }
            # ---- P2 / P3 rows: PRIORS MOVED OUT OF THE WHITENED SPACE (docs §9.3-§9.4) ------
            #
            # `rawspec=` is P2 (unwhitened metric, optionally spectral-truncated) and `topo=` is
            # P3 (threshold-graph reference). Same tags-separate-from-everything-else discipline
            # as M1/C1/P1: no readout can silently absorb another cell's operator.
            #
            # Every `rawspec=0` cell is the PAIRED TWIN of each truncated cell -- it is the raw
            # metric with no truncation -- so "the rank is the only change" is a comparison
            # between two rows that differ in exactly one argument.
            if args.fgw_raw_metric or args.fgw_topo or (args.fgw_topo_eps is not None):
                raw_specs: list = [0]
                if args.fgw_spec_rank is not None:
                    for tok in str(args.fgw_spec_rank).split(","):
                        tok = tok.strip()
                        if tok == "":
                            continue
                        v = "src" if tok.lower() == "src" else int(tok)
                        if v not in raw_specs:
                            raw_specs.append(v)
                if args.fgw_spec_from_src and "src" not in raw_specs:
                    raw_specs.append("src")
                for R_p in (max(r_list), min(r_list)):
                    r_p = None if (R_p is None or R_p >= full_R) else int(R_p)
                    r_show = full_R if R_p is None else R_p
                    if args.fgw_raw_metric:
                        for rs in raw_specs:
                            is_src = (rs == "src")
                            rsv = "src" if is_src else int(rs)
                            stag = f"R={r_show},a=0.75,t=0.03,rawspec={rsv}"
                            for suffix, fn_rec in (
                                    ("", _recovery_fn_for(args, force_alpha=0.75,
                                                          force_tau=0.03)),
                                    (" (structural-off)",
                                     _recovery_fn_for(args, force_alpha=0.0,
                                                      force_tau=0.03))):
                                sc, dg = calibration.rep_cloud_scores(
                                    z_reps, feats["img"], k=args.csls_k, rho=args.rho,
                                    shrink=rep_shrink,
                                    min_landmark_rate=args.min_landmark_rate,
                                    recovery_fn=fn_rec, rep_subsample=r_p,
                                    src_means=src_means, src_mix=0.0,
                                    spec_rank=(None if (is_src or int(rs) == 0) else int(rs)),
                                    spec_from_src=is_src, raw_metric=True)
                                out["rows"][f"+ T2 {stag}{suffix}"] = {
                                    **calibration.report_with_scores(sc),
                                    "diag": {k: v for k, v in dg.items()
                                             if not hasattr(v, "shape")},
                                }
                    if args.fgw_topo or (args.fgw_topo_eps is not None):
                        ep = args.fgw_topo_eps
                        q_list = ([] if ep is not None else
                                  [float(x) for x in str(args.fgw_topo_q).split(",")
                                   if x.strip() != ""])
                        for q_tok in (q_list if q_list else [None]):
                            ttag = (f"topo=eps{float(ep):g}" if ep is not None
                                    else f"topo=q{q_tok:g}")
                            stag = f"R={r_show},a=0.75,t=0.03,{ttag}"
                            for suffix, fn_rec in (
                                    ("", _recovery_fn_for(args, force_alpha=0.75,
                                                          force_tau=0.03)),
                                    (" (structural-off)",
                                     _recovery_fn_for(args, force_alpha=0.0,
                                                      force_tau=0.03))):
                                sc, dg = calibration.rep_cloud_scores(
                                    z_reps, feats["img"], k=args.csls_k, rho=args.rho,
                                    shrink=rep_shrink,
                                    min_landmark_rate=args.min_landmark_rate,
                                    recovery_fn=fn_rec, rep_subsample=r_p,
                                    src_means=src_means, src_mix=0.0,
                                    raw_metric=True, topo_eps=ep,
                                    topo_auto=(ep is None),
                                    topo_q=(0.10 if q_tok is None else float(q_tok)))
                                out["rows"][f"+ T2 {stag}{suffix}"] = {
                                    **calibration.report_with_scores(sc),
                                    "diag": {k: v for k, v in dg.items()
                                             if not hasattr(v, "shape")},
                                }
            t2 = None

        # --------------------------------------------------- shipped single/duplicated row
        else:
            sr_cell = None
            r_list = args.rep_subsample if args.rep_subsample else [None]
            structural_off = (_recovery_fn_for(args, force_alpha=0.0)
                              if len(r_list) > 1 else None)
            for r_sub in r_list:
                tag = "T2 reps" if r_sub is None else f"T2 reps (R={r_sub})"
                for m in src_mixes:
                    # m=0 keeps the CANONICAL row name, because the fused rows downstream look
                    # `+ T2 reps` up by name and renaming the shipped row would silently drop
                    # the fusion. The M1 rows are additive rows, not a replacement.
                    mtag = "" if m == 0.0 else f" (src_mix={m:g})"
                    t2, t2_diag = calibration.rep_cloud_scores(
                        z_reps, feats["img"], k=args.csls_k, rho=args.rho,
                        shrink=rep_shrink, min_landmark_rate=args.min_landmark_rate,
                    recovery_fn=RECOVERY_FN, rep_subsample=r_sub,
                    src_means=src_means, src_mix=m, spec_rank=sr_cell)
                    out["rows"][f"+ {tag}{mtag}"] = {
                        **calibration.report_with_scores(t2),
                        "diag": {k: v for k, v in t2_diag.items() if not hasattr(v, "shape")},
                    }
                    row_scores[f"+ {tag}{mtag}"] = t2
                    if m != 0.0:
                        continue
                    # Keep the matrix so `--save-scores` can persist it too: the T2 row is the
                    # single strongest row on the v8 encoder, so a score dump that omits it cannot
                    # ensemble the very row the headline comes from.
                    if structural_off is not None:
                        t2_off, off_diag = calibration.rep_cloud_scores(
                            z_reps, feats["img"], k=args.csls_k, rho=args.rho,
                            shrink=rep_shrink, min_landmark_rate=args.min_landmark_rate,
                            recovery_fn=structural_off, rep_subsample=r_sub,
                            src_means=src_means, src_mix=m, spec_rank=sr_cell)
                        out["rows"][f"+ {tag} (structural-off)"] = {
                            **calibration.report_with_scores(t2_off),
                            "diag": {k: v for k, v in off_diag.items()
                                     if not hasattr(v, "shape")},
                        }
                        row_scores[f"+ {tag} (structural-off)"] = t2_off
        # The fused rows below use the FULL-cloud matrix: `args.rep_subsample` is a sweep
        # instrument, and silently fusing a subsampled cloud would change the shipped row.
        t2 = row_scores["+ T2 reps"] if "+ T2 reps" in row_scores else row_scores.get(
            f"+ T2 reps (R={r_list[0]})")
        # Fused with BOTH deployed T1 rungs, because which one to fuse is a measurement
        # and not an assumption. On sub-08 they tie on Top-1 (35.50) and the SCORE
        # composition is +7.5pp on Top-5 (73.50 vs 66.00), so fusing with only the SAW
        # rung -- which is what the probes did -- would report the weaker of the two and
        # make the choice look settled.
        for t1_name in ("+ whiten + CSLS + recovery", "+ CSLS + recovery"):
            if t1_name not in row_scores:
                continue
            # In sweep mode there is no single `t2` row to fuse, and in a multi-R run each R has
            # its own row -- emitting a fused row off whichever cell happened to come first
            # would put a subsampled number under the shipped row's name. The fused rows are
            # not what this sweep is for; they stay on the shipped path.
            if args.fgw_sweep or (args.rep_subsample is not None and len(args.rep_subsample) > 1):
                break
            fused = fuse_scores([row_scores[t1_name], t2])
            fused_scores[f"+ T1({t1_name[2:]}) + T2 reps"] = fused
            out["rows"][f"+ T1({t1_name[2:]}) + T2 reps"] = {
                **calibration.report_with_scores(fused),
                "diag": {"fused": True, "t1_rung": t1_name},
            }

    # Optional: persist the actual score matrices. The reports only carry scalar summaries, and
    # a summary cannot be re-pooled -- so without this, SEED ENSEMBLING is impossible: averaging
    # three seeds' top-1 scores is not the top-1 of their averaged score matrix, and the latter
    # is what a deployed system would do. Saved as npz (row name -> (n_query, n_gallery) float)
    # rather than json because these are 200x200 float matrices per row.
    if args.save_scores:
        Path(args.save_scores).parent.mkdir(parents=True, exist_ok=True)
        payload = {f"row::{k}": v for k, v in row_scores.items()}
        payload.update({f"fused::{k}": v for k, v in fused_scores.items()})
        payload["_truth"] = np.arange(len(feats["eeg"]))   # queries are concept-aligned
        np.savez_compressed(args.save_scores, **payload)
        print(f"[eval] saved {len(payload)} score matrices -> {args.save_scores}")

    # Per-concept ranks for the strongest row, so paired arm comparisons can be run on
    # the SAME rung. Dumping them for a fixed rung instead (say raw cosine) would
    # compare arms at a point the report itself shows is not the deployed one.
    if args.dump_ranks:
        best = max(out["rows"], key=lambda n: out["rows"][n]["top1"])
        print(f"[eval] dumping per-concept ranks for the strongest row: {best}")
        scores, _ = calibration.calibrate(feats["eeg"], feats["img"], k=args.csls_k,
                                          rho=args.rho,
                                          min_landmark_rate=args.min_landmark_rate,
                                          recovery_fn=RECOVERY_FN, **stages[best])
        order = np.argsort(-scores, axis=1)
        out["per_concept_rank"] = [int(r) for r in
                                   np.diag(np.argsort(order, axis=1)) + 1]
        out["rank_row"] = best
    return out


def _print_table(report: dict) -> None:
    print()
    print("=" * 92)
    ref = report.get("reference") or {}
    if ref.get("top1") is None:
        ref_str = f"NOT MEASURED for this fold ({REF_UNKNOWN_NOTE})"
    else:
        ref_str = f"{ref['top1']:.2f}/{ref['top5']:.2f}  ({ref.get('note', '')})"
    print(f"LOSO retrieval | target sub-{report['target_subject']:02d} | "
          f"{report['n_queries']}-way | reference: {ref_str}")
    print("=" * 92)
    for tag, res in report["checkpoints"].items():
        w = res.get("target_layer_weights")
        gate = res.get("smn_gate")
        print(f"\n[{tag}]  {res['ckpt']}  (epoch {res['epoch']})")
        print(f"  arch={res.get('arch')}  objective={res.get('objective')}  "
              f"d_align={res.get('d_align')}  "
              f"smn_gate={'n/a' if gate is None else round(float(gate), 4)}  "
              f"offset_ratio={res.get('offset_ratio', float('nan')):.3f}")
        if res.get("objective") == "v6":
            print(f"  routes={res.get('routes')}  primary={res.get('primary')}  "
                  f"fusion_weight={res.get('fusion_weight')}  "
                  f"normalize={res.get('fusion_normalize')}")
            print(f"  per-route layers={res.get('target_layers')}")
        if w is not None:
            print(f"  fusion={res['target_fusion']}  layer weights="
                  f"{[round(float(x), 3) for x in w]}")
        print(f"  {'row':<28} {'Top-1':>8} {'Top-5':>8} {'mean rank':>10} "
              f"{'ΔTop-1 vs ref':>14}")
        for name, m in res["rows"].items():
            t1, t5 = m["top1"], m["top5"]
            extra = ""
            if m.get("best_route"):
                extra = f"   [fused; best route {m['best_route']} @ " \
                        f"{m.get('best_route_top1', float('nan')):.2f}]"
            print(f"  {name:<28} {t1:>8.2f} {t5:>8.2f} {m['mean_rank']:>10.1f} "
                  f"{_ref_delta(t1):>14}{extra}")
        if res.get("route_rows"):
            print(f"  {'--- per route (raw cosine rung) ---':<28}")
            for n, rows in res["route_rows"].items():
                r = rows.get("+ whiten + CSLS", rows.get("+ CSLS"))
                if r is None:
                    continue
                print(f"    {n:<26} {r['top1']:>8.2f} {r['top5']:>8.2f} "
                      f"{r['mean_rank']:>10.1f} {_ref_delta(r['top1']):>14}   "
                      f"(+whiten+CSLS)")
    print("\n" + "=" * 92)


def _recovery_fn_for(args, force_alpha: float | None = None,
                     force_tau: float | None = None):
    """The recovery operator selected by `--recovery-operator`, or `None` for the default.

    ``hard`` is SCORE's operator (`calibration.coordinate_recovery`: mutual-NN landmarks, an
    orthogonal Procrustes fit in the full 64 dimensions, `rho` shrinkage) and it stays the
    default so every report already on disk keeps its meaning.

    ``sinkhorn`` selects `calibration.subspace_soft_recovery` with soft matching ON and the
    subspace truncation OFF. That arm -- and not the version with `rank=16` -- is the one the
    30-run paired probe selected: soft matching is worth +2.97pp and +3.20pp on the two
    headline rows (28/30 runs positive, t = 8.72), while the subspace truncation on its own
    measured -0.50pp and added nothing. Effective matched pairs go ~42 -> ~465.

    ``fgw`` is the same operator with the plan solved by Fused Gromov-Wasserstein instead of
    pure Sinkhorn on the cross-modal similarity (`alpha > 0`), adding the intra-domain
    structural term. `alpha = 0` is bit-identical to ``sinkhorn`` (verified), so the two are
    nested and the structural contribution is measured against a reproduction.

    ``force_alpha`` overrides `--recovery-alpha` (and permits 0, which the `fgw` guard below
    would otherwise reject). It exists so the R-sweep can build a STRUCTURAL-OFF TWIN of the
    configured operator that differs in nothing but the structural term -- same tau, same
    iters, same wrapper -- because a twin built any other way would let the wrapper change
    alongside the term and the difference would stop meaning what it says.
    """
    name = getattr(args, "recovery_operator", "hard")
    if name == "hard":
        return None
    tau = float(getattr(args, "recovery_tau", 0.05))
    iters = int(getattr(args, "recovery_iters", 50))
    alpha = float(getattr(args, "recovery_alpha", 0.0))
    if force_alpha is not None:
        alpha = float(force_alpha)
    if force_tau is not None:
        tau = float(force_tau)
    if force_alpha is None and name == "fgw" and alpha <= 0.0:
        raise SystemExit(
            "--recovery-operator fgw needs --recovery-alpha > 0; alpha=0 IS the `sinkhorn` "
            "operator, so asking for fgw at 0 is a request whose answer is already on disk.")

    def sinkhorn_recovery(q, g, k=10, rho=0.1, min_landmark_rate=0.0, **kw):
        # `**kw` exists so `rep_cloud_scores` can pass the v10 knobs (`--rep-subsample`,
        # `--fgw-src-metric`) down through the same wrapper the T1 rung uses.
        return calibration.subspace_soft_recovery(
            q, g, k=k, rho=rho, rank=None, tau=tau, iters=iters,
            hard_landmarks=False, min_landmarks=8,
            alpha=float(kw.pop("alpha", alpha)),
            fgw_outer=int(getattr(args, "recovery_fgw_outer", 10)),
            fgw_de_ref=kw.pop("fgw_de_ref", None),
            fgw_de_mix=float(kw.pop("fgw_de_mix", 0.0)),
            fgw_spec_rank=kw.pop("fgw_spec_rank", None),
            # G1 knob, forwarded for exactly the reason the A4 knob below needed a comment:
            # an argument that dies in `**kw` leaves the cell bit-identical to its twin and
            # reads as a result. `di_ref=None` is the shipped operator.
            fgw_di_ref=kw.pop("fgw_di_ref", None),
            # The A4 knob has to be forwarded here or it dies in `**kw` and the topological
            # reference silently never reaches `_fgw_plan` -- which is exactly what happened on
            # the first smoke, where `topo=auto` was bit-identical to `rawspec=0` (plan_acc
            # matched to 16 digits) and looked like a *result* rather than a dropped argument.
            fgw_topo_eps=kw.pop("fgw_topo_eps", None))

    sinkhorn_recovery.__name__ = f"{name}_recovery(tau={tau:g},alpha={alpha:g})"
    return sinkhorn_recovery


# Selected in `main()` from `--recovery-operator`; `None` means the deployed SCORE operator.
# A module global rather than a parameter because it has to reach four call sites that are
# otherwise driven by the `stages` dict, and threading it through every one would let a call
# site be forgotten -- which is exactly the silent-no-op failure this project keeps hitting.
RECOVERY_FN = None


@torch.no_grad()
def _gallery_view_metric(model, targets_te) -> tuple[np.ndarray | None, dict]:
    """G1: the gallery-side metric as a reliability-weighted fusion of its OWN layer views.

    THE MISSING HALF OF THE BILATERAL COUPLING. FGW compares two estimates of the shared
    concept metric, and the query side already has a multi-view estimator: v11's `rep_blocks`
    pools the target's own repetition blocks. The gallery side has always been the metric of
    ONE fused embedding. But the target stack is `(C, 1, K, D)` -- K frozen vision layers that
    are K INDEPENDENT views of the same 200 images -- so the same denoising operation the query
    side uses can be applied to the gallery side. This returns that fused metric.

    WHY THIS IS NOT THE M1 LEAKAGE CHANNEL, WHICH IS THE OBVIOUS OBJECTION. M1 blended the
    metric of a DIFFERENT subject (source EEG) into the query's, and the permutation control
    showed the gain came from the concept-index correspondence between that template and the
    gallery (corr 0.78). Here every view is a function of the SAME image features and is
    indexed by the SAME image slot, so pooling them introduces no cross-domain correspondence
    at all -- it is the gallery-side twin of v11's blocks, which the same control clears. The
    structural-off twin emitted beside every G1 row is the guard: `gallery_di_ref` can only
    reach the output through the structural term, so the twin must be bit-flat.

    `di_ref=None` (K < 2) disables it, and the caller then emits no G1 row rather than a row
    that silently equals its twin.
    """
    a = np.asarray(targets_te)
    if a.ndim != 4:
        return None, {"applied": False, "reason": f"stack rank {a.ndim} != 4"}
    C, _, K, _ = a.shape
    if K < 2:
        return None, {"applied": False, "reason": f"K={K} < 2 views"}
    di = []
    for k in range(K):
        # (C, I=1, K=1, D) -> (C, 1, D): `encode_target` expects (B, n_layers, D), and the
        # gallery path in `TestDataset` drops the singleton image axis the same way.
        t = torch.as_tensor(a[:, 0, k:k + 1, :], dtype=torch.float32,
                            device=next(model.parameters()).device)
        g_k = model.encode_target(t, training=False)
        di.append(calibration._sq_cos_dist(g_k.float().cpu().numpy()))
    # leave-one-out agreement -> inverse-residual weights: the SAME estimator v11 uses for
    # the query's blocks, so "the gallery is denoised the way the query is" is one mechanism
    # rather than two, and the weight diagnostic (`is_flat`) is read on one ruler.
    rel = []
    for i in range(K):
        ref = np.mean([di[j] for j in range(K) if j != i], axis=0)
        rel.append(float(np.linalg.norm(di[i] - ref) / max(float(np.linalg.norm(ref)), 1e-12)))
    inv = 1.0 / (np.asarray(rel) + 1e-6)
    w = inv / inv.sum()
    fused = np.tensordot(w, np.asarray(di), axes=(0, 0))
    # Re-standardise so the injected reference lives on the same scale as the one `_fgw_plan`
    # would have computed; a scale mismatch would change the effective alpha and make the G1
    # cell a comparison of alphas rather than of references.
    fused = (fused - fused.mean()) / max(float(fused.std()), 1e-12)
    diag = {"applied": True, "K": int(K), "weights": [float(x) for x in w],
            "resid": [float(x) for x in rel], "weight_range": float(w.max() - w.min()),
            "is_flat": bool(w.max() - w.min() < 1e-3)}
    return fused, diag


def main() -> None:
    global RECOVERY_FN
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True,
                    help="one or more checkpoints; the tag is the parent dir name")
    ap.add_argument("--target-subject", type=int, required=True)
    ap.add_argument("--mvnn", choices=["off", "train", "test"], default=None,
                    help="default: mirror the checkpoint's training config "
                         "(source subjects 'train' -> target 'test')")
    ap.add_argument("--recovery", action="store_true", help="SCORE coordinate recovery")
    ap.add_argument("--recovery-operator", choices=["hard", "sinkhorn", "fgw"],
                    default="hard",
                    help="`hard` = SCORE's mutual-NN Procrustes (default; keeps every "
                         "existing report comparable). `sinkhorn` = soft doubly-stochastic "
                         "matching, measured +2.80pp on the fused headline over 30 runs "
                         "(28/30 positive, t=8.77). `fgw` = the same operator with the plan "
                         "solved by Fused Gromov-Wasserstein, adding the intra-domain "
                         "structural term (needs --recovery-alpha > 0).")
    ap.add_argument("--recovery-tau", type=float, default=0.05,
                    help="Sinkhorn temperature; the measured plateau is 0.01-0.05 and it "
                         "degrades above 0.1")
    ap.add_argument("--recovery-alpha", type=float, default=0.0,
                    help="FGW structure weight; 0 is bit-identical to --recovery-operator "
                         "sinkhorn. Measured: interior optimum, alpha=1 collapses.")
    ap.add_argument("--recovery-fgw-outer", type=int, default=10)
    ap.add_argument("--recovery-iters", type=int, default=50)
    # ---- v10 structural knobs (stage 1/3 of docs/eeg2image_v10_plan.md) ----
    ap.add_argument("--rep-subsample", type=int, nargs="+", default=None,
                    help="keep only the first R repetitions of the target's test cloud; may be "
                         "a list to emit one row per R. This is the variance-gating knob: the "
                         "FGW structural gain is measured at R=80 (the deployed row) and the "
                         "curve in R decides whether the gain is structural or bought with test "
                         "information. Each R also emits an alpha=0 twin so the gain is a PAIRED "
                         "difference on the identical repetition subset.")
    ap.add_argument("--rep-shrink", type=float, default=None,
                    help="shrinkage of the rep-cloud whitening estimator (`_whiten_from_cloud`). "
                         "Default 0.1 is the probes' leftover. v10 stage 1 showed the structural "
                         "gain is gated by this estimator's variance, so this is the regulariser "
                         "the stage-2 sweep tunes; recorded in every row's diag.")
    ap.add_argument("--dump-src-metric", default=None,
                    help="write the M1 source-metric template here (npz with `src_means` "
                         "(S,C,d) = the SOURCE subjects' repetition-averaged concept "
                         "embeddings under THIS fold's checkpoint). Produced inside the fold "
                         "so channels/mvnn/feature-set cannot drift from the eval path.")
    ap.add_argument("--fgw-spec-rank", default=None,
                    help="P1: comma list of spectral ranks for the EEG-side GW metric. Full/"
                         "absent is bit-identical to the shipped cell. The rank prior is a "
                         "permutation-invariant scalar read off the eigenvalue spectrum, so it "
                         "carries no concept correspondence and cannot leak the way M1 did.")
    ap.add_argument("--fgw-spec-from-src", action="store_true",
                    help="P1: take the rank from the SOURCE subjects' effective rank "
                         "(participation ratio of the metric spectrum).")
    ap.add_argument("--fgw-raw-metric", action="store_true",
                    help="P2: evaluate the structural metric in the UNWHITENED space. P1 "
                         "demanded a low-rank EEG metric, but `_whiten_from_cloud` flattens the "
                         "spectrum by construction, so P1 truncated a broadband object (0/30 "
                         "folds). `de_raw` is built from the pre-whitening concept mean, where "
                         "the low-rank premise was actually measured. With --fgw-spec-rank the "
                         "truncation now acts on THAT object; pairing against a `rawspec` "
                         "full-rank twin is bit-identical to the un-truncated raw metric.")
    ap.add_argument("--fgw-topo", action="store_true",
                    help="P3: replace the structural reference with the eps-THRESHOLD GRAPH "
                         "(0/1 adjacency) instead of the metric. Connectivity survives a "
                         "whitener that flattens distances, and a 0/1 matrix has no spectrum to "
                         "truncate. `eps` is one permutation-invariant scalar read off the "
                         "GALLERY's H0 merge-gap -- no query, no label, no index correspondence.")
    ap.add_argument("--fgw-topo-eps", type=float, default=None,
                    help="P3: fix the threshold graph scale explicitly instead of deriving it "
                         "from the gallery's H0 persistence gap (which --fgw-topo does).")
    ap.add_argument("--fgw-topo-q", default="0.10",
                    help="P3: comma list of graph DENSITIES (edge fraction) for the eps-graph "
                         "reference. The density is A4's only hyperparameter, so it is swept "
                         "rather than fixed by assertion: each q gives exactly q*C(C-1)/2 edges, "
                         "computed per domain from that domain's own distance quantile.")
    ap.add_argument("--sim-calib", default=None, choices=[None, "satc"],
                    help="E2 BASELINE: SATTC's structural expert (CVPR 2026, "
                         "`calibration.structural_scores`). Fuses the geometric score with "
                         "mutual-NN agreement, bidirectional top-k and hubness via a weighted "
                         "product-of-experts. This is the POINT-BASED baseline COMMET must beat: "
                         "it is a similarity-MATRIX calibration on the mean embedding, whereas "
                         "COMMET calibrates the order-2 metric from the repetition cloud. Emits "
                         "`+ SATTC(<base>)` plus a `lam=0` twin that recovers pure geometry, and "
                         "reports `mutual_topk_enrichment` -- the diagnostic that says whether the "
                         "structural signal exists at all on this fold.")
    ap.add_argument("--sim-calib-lam", type=float, default=0.2,
                    help="E2: structural-expert weight in the PoE fusion (SATTC's own lam). "
                         "Kept explicit because the honest state is 'the structural signal is "
                         "weak and its weight must be measured'; lam=0 is the geometry-only twin.")
    ap.add_argument("--sim-calib-base", default="+ whiten + CSLS",
                    help="E2: the geometric row(s) the structural expert is fused onto. A COMMA "
                         "LIST, because the fair 'cloud vs point' test fuses SATTC on the SAME "
                         "recovery rung COMMET uses -- applying it only to the weak `+ whiten + "
                         "CSLS` rung while COMMET carries the recovery operator would attribute "
                         "the recovery gain to the cloud. Defaults to the pre-recovery rung so "
                         "the paper's own composition is emitted when the caller wants it.")
    ap.add_argument("--fgw-struct-fuse", default=None,
                    help="v11 L2/L3/L4: comma list of BLOCK counts B. Split the target's own R "
                         "repetitions into B contiguous blocks, build a structural metric per "
                         "block, weight each block by its agreement with the leave-one-out mean of "
                         "the others (measured, never tuned), and feed the WEIGHTED FUSION to the "
                         "FGW plan as the structural reference. `fuse=0` is the shipped "
                         "single-mean metric and is emitted as the paired twin. Licensed by "
                         "scripts/probe_fusion.py (job 645697): pooling raises structure agreement "
                         "monotonically in the number of pooled views with a "
                         "correspondence-destroying control pinned at ~0.000.")
    ap.add_argument("--mve-partitions", default=None,
                    help="O2-MVE: comma list of repetition PARTITION families to pool as metric "
                         "views, from {cont,stride,rand}. `cont` is the contiguous blocks "
                         "`--fgw-struct-fuse` already fuses; `stride` takes repetitions b::B so "
                         "slow equipment drift is spread across every block instead of being "
                         "given a different offset per block, which makes it a genuinely "
                         "different view family rather than a relabelling; `rand` is the seeded "
                         "control between them. All views are pooled by leave-one-out reliability "
                         "weights into ONE structural reference, so this RAISES the view count "
                         "K_eff whose inverse sets the estimator's variance (rho<=0 on 10/10 "
                         "folds means the gain in K is not saturated). Every row gets a "
                         "structural-off twin that must be bit-flat.")
    ap.add_argument("--mve-blocks", type=int, default=16,
                    help="O2-MVE: repetitions per partition family (default 16).")
    ap.add_argument("--mve-fuse", action="store_true",
                    help="O2-MVE: pool the `cont` view family ALONE. This is the control that "
                         "separates 'the estimator is better' from 'there are more views': its "
                         "views are the same objects `--fgw-struct-fuse` already fuses, so any "
                         "gain here is the leave-one-out weighting, not the new families.")
    ap.add_argument("--fgw-gallery-fuse", action="store_true",                    help="G1: BILATERAL structural reference. v11 pools the QUERY's own "
                         "repetition blocks (`--fgw-struct-fuse`); this does the same denoising "
                         "on the GALLERY side, fusing the metric of each of the K target layers "
                         "(K independent views of the same images, so no cross-domain "
                         "correspondence and not the M1 channel) by the same leave-one-out "
                         "reliability weights. Emits a `,gv` row per fused cell, each with its "
                         "structural-off twin -- which MUST be bit-flat, since the injected "
                         "reference can only reach the output through the structural term.")
    ap.add_argument("--fgw-src-calib", action="store_true",
                    help="C1: SOURCE-CALIBRATED METRIC TRANSFER. Fit a monotone elementwise map "
                         "phi on SOURCE subjects (phi(de_s) ~ di, where using the source<->gallery "
                         "correspondence is legitimate inductive transfer), then apply phi to the "
                         "TARGET'S OWN metric. phi never sees an index, so the M1 leakage channel "
                         "is absent by construction. Emits an extra `... ,calib` row carrying "
                         "corr(phi(de_target), di) against corr(de_target, di) -- the "
                         "pre-registered transfer criterion.")
    ap.add_argument("--fgw-src-permute", type=int, default=None, metavar="SEED",
                    help="CONTROL, not a feature. Permute the concept axis of the source "
                         "template with this RNG seed. The source template's index IS the "
                         "concept identity, and so is the gallery's, so `de_src` is a "
                         "label-frame object; a permuted template is the same matrix with the "
                         "correspondence destroyed. If the gain depends on that correspondence "
                         "the permuted run collapses, which says the mix was smuggling in the "
                         "target's labels by index rather than supplying geometry.")
    ap.add_argument("--fgw-src-metric", default=None,
                    help="path to an .npz holding `src_means` (S, C, d): S SOURCE subjects' "
                         "per-concept mean embeddings. Used as the M1 source-metric template, "
                         "blended into the EEG-side structural reference by --fgw-src-mix.")
    ap.add_argument("--fgw-src-ref-mode", default="eeg", choices=["eeg", "gallery", "self", "rand"],
                    help="which metric the M1 blend substitutes for the query-side `de`. "
                         "`eeg` (default) = the source subjects' EEG metric -- the arm under "
                         "adjudication. `gallery` = the metric of the PUBLIC image gallery (zero "
                         "EEG, available at test): if it reproduces the gain, the effect needs no "
                         "source EEG. `self` = the target's own metric (a no-op). `rand` = a "
                         "fixed-seed random symmetric metric (the floor). A single-argument change "
                         "from `eeg`, so every contrast is paired by construction.")
    ap.add_argument("--fgw-src-mix", default="0.0",
                    help="comma list of blend weights toward the source template: "
                         "de <- (1-mix)*de_target + mix*de_src. `0` is bit-identical to the "
                         "shipped operator. A list, not a scalar, because the mix is a "
                         "bias-variance knob whose PREdicted optimum is interior; a peak at an "
                         "endpoint means the mechanism is not the one claimed. A per-row diag "
                         "records corr(de_src, de_target), which is what predicts whether "
                         "mixing can pay off at all.")
    ap.add_argument("--fgw-sweep", action="store_true",
                    help="v10 validation grid on the repetition cloud: R in {80,40,20,10,5,1} "
                         "at the deployed (alpha=0.75,tau=0.01), alpha in {0.25,0.5,0.75,0.875} "
                         "at R=full, tau in {0.01,0.03,0.05} at R=full. Every cell also emits a "
                         "structural-off twin (alpha=0) so the gain is a paired difference. "
                         "Requires --reps and --recovery-operator fgw.")
    ap.add_argument("--reps", action="store_true",
                    help="v6 pillar C2: add the label-free transductive refinement row, "
                         "fitted on the UNAVERAGED test repetitions. Requires the "
                         "repetition cache (`load_test_reps`), which is built on demand.")
    ap.add_argument("--rep-agreement", type=float, default=0.5,
                    help="minimum fraction of a concept's repetitions whose top-1 CSLS "
                         "match must agree before a mutual-NN pseudo-pair is trusted as a "
                         "Procrustes landmark (see `calibration.refine_with_reps`).")
    ap.add_argument("--rho", type=float, default=0.1)
    ap.add_argument("--csls-k", type=int, default=10)
    ap.add_argument("--min-landmark-rate", type=float, default=0.0,
                    help="abstain from recovery below this mutual-NN landmark rate. "
                         "A map fitted from untrustworthy pseudo-pairs scores BELOW "
                         "the unrecovered baseline, so reporting 'recovery ran' "
                         "without the gate can report a loss as a gain.")
    ap.add_argument("--dump-ranks", action="store_true",
                    help="also write per-concept ranks for paired arm comparisons")
    ap.add_argument("--save-scores", default=None,
                    help="write the raw (n_query, n_gallery) score matrices to this .npz. "
                         "Needed for SEED ENSEMBLING: pooling three seeds means averaging "
                         "score matrices, which scalar reports cannot reconstruct.")
    ap.add_argument("--ref-top1", default=None,
                    help="the SAMGA-official Top-1 for THIS fold, as measured by us, or "
                         "'unknown'. REQUIRES --ref-top5. Left unset, the default 22.00 "
                         "is used, which is the sub-08 cell only -- pass 'unknown' on any "
                         "other fold or the report prints a delta against a reference "
                         "measured on a different subject.")
    ap.add_argument("--ref-top5", default=None,
                    help="the Top-5 that goes with --ref-top1 (refused on its own).")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    RECOVERY_FN = _recovery_fn_for(args)
    if RECOVERY_FN is not None:
        print(f"[eval] recovery operator: {RECOVERY_FN.__name__}")
    global REF_TOP1, REF_TOP5, REF_NOTE
    if args.ref_top1 is not None:
        if str(args.ref_top1).strip().lower() in _UNKNOWN_REF:
            if args.ref_top5 is not None:
                raise SystemExit("--ref-top1 unknown and --ref-top5 given: pick one.")
            REF_TOP1 = REF_TOP5 = None
            REF_NOTE = REF_UNKNOWN_NOTE
            print(f"[eval] reference: NOT MEASURED for sub-{args.target_subject:02d}; "
                  f"deltas will print as n/a. Compare the 10-fold MEAN to 26.22/53.23.")
        else:
            if args.ref_top5 is None:
                raise SystemExit("--ref-top1 needs --ref-top5: half a reference is how a "
                                 "Top-1 delta ends up next to an unrelated Top-5.")
            REF_TOP1, REF_TOP5 = float(args.ref_top1), float(args.ref_top5)
            if REF_TOP1 <= 0 or REF_TOP5 <= 0:
                raise SystemExit(f"reference {REF_TOP1}/{REF_TOP5} is not a percentage.")
            REF_NOTE = (f"caller-supplied reference for this fold "
                        f"({REF_TOP1:.2f}/{REF_TOP5:.2f})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] device={device} target=sub-{args.target_subject:02d}")

    report: dict = {
        "target_subject": args.target_subject,
        "reference": {"top1": REF_TOP1, "top5": REF_TOP5, "note": REF_NOTE},
        "checkpoints": {},
        "n_queries": config.N_TEST_CONCEPTS,
    }
    for c in args.ckpts:
        p = Path(c)
        tag = _ckpt_tag(p)
        if tag in report["checkpoints"]:
            raise SystemExit(
                f"two checkpoints map to the same report tag {tag!r}: {c}. The report "
                "is keyed by tag, so the second would silently replace the first. Pass "
                "checkpoints whose parent directories differ.")
        print(f"[eval] {tag}: {p}")
        report["checkpoints"][tag] = _evaluate_checkpoint(p, args.target_subject,
                                                          args, device)

    _print_table(report)
    text = json.dumps(report, indent=2, default=str)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
        print(f"[eval] wrote {args.out}")


if __name__ == "__main__":
    main()
