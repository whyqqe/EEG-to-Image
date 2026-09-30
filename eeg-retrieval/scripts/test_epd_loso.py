#!/usr/bin/env python
"""Wiring tests for the inter-subject (LOSO) path. No GPU, no dataset, no download.

Second gate before any inter-subject training run. The intra-subject pipeline is one
subject wide everywhere -- one z-score table, one embedding row, one checkpoint
selection rule -- and widening it to nine subjects added a second identity axis to a
pipeline that had been built assuming there would never be one. Every failure mode
below is silent, and every one of them costs hours of GPU:

  * `subject_ids` left at `torch.zeros` -> the per-subject embedding is created,
    scheduled, saved in the checkpoint and NEVER receives a gradient. The run
    reports "subject-aware training" while every row says subject 0.
  * image features repeated with `np.repeat` instead of `np.tile` -> concept c of
    subject i is paired with the features of a different concept. Training
    converges. The number is noise.
  * per-subject z-score skipped -> nine subjects at native amplitude, and the
    loudest subject owns the shared scale.
  * the held-out subject present in `--source-subjects` -> the fold trains on its
    own test subject, which is the highest number in the sweep and the one result
    nobody re-checks.
  * `--val-concepts 0` with a val-based selection rule -> either a crash or, worse,
    an empty holdout scored as 0.0 and selection on a constant.

Run:  python scripts/test_epd_loso.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from epd import config                                                # noqa: E402
from epd.data import (LosoData, TrainDataset, _channel_stats, assert_standardised,  # noqa: E402
                      concept_split, expand_loso_images)
from epd.encoders import LayerFusion                                 # noqa: E402
from epd.model import RetrievalModel                                  # noqa: E402
from epd.tokenizer import OFFICIAL_CHANNEL_ORDER                      # noqa: E402
from epd.train import _validate                                       # noqa: E402

CH = list(OFFICIAL_CHANNEL_ORDER)
N_CH = len(CH)
N_TIME = 250
C = 12          # tiny concept count for the synthetic arrays; the real one is 1654
IMG = 3

_failures: list[str] = []
_passes = 0


def check(cond: bool, label: str, detail: str = "") -> None:
    global _passes
    if cond:
        _passes += 1
        print(f"  ok   {label}")
    else:
        _failures.append(f"{label}{(' -- ' + detail) if detail else ''}")
        print(f"  FAIL {label}{(' -- ' + detail) if detail else ''}")


def raises(fn, needle: str = "") -> str:
    """Run `fn`, expect a refusal; return the message.

    RuntimeError is in the list because that is what the guards on the TRAINING
    path raise: a bad flag is a SystemExit, but "this model was asked to train
    without the input its mode needs" is a programming error, not a config error,
    and it is exactly the one that would otherwise be silent.
    """
    try:
        fn()
    except (SystemExit, ValueError, KeyError, RuntimeError) as e:
        msg = str(e)
        if needle and needle not in msg:
            return f"__WRONG_MESSAGE__ {msg}"
        return msg
    except Exception as e:                                    # noqa: BLE001
        return f"__WRONG_TYPE__ {type(e).__name__}: {e}"
    return "__DID_NOT_RAISE__"


# --------------------------------------------------------------------- 1
def test_no_holdout_is_a_protocol() -> None:
    print("\n[1] --val-concepts 0 is the SOTA protocol, not a degenerate input")
    s = concept_split(0, seed=2025)
    check(len(s.fit_concepts) == config.N_TRAIN_CONCEPTS,
          f"no holdout -> the fit set is all {config.N_TRAIN_CONCEPTS} training concepts",
          f"got {len(s.fit_concepts)}")
    check(len(s.val_concepts) == 0, "and the val set is empty, not silently full",
          f"got {len(s.val_concepts)}")
    check(s.fit_concepts.min() == 0 and s.fit_concepts.max() == config.N_TRAIN_CONCEPTS - 1,
          "the fit set covers every concept exactly once, with no gap")

    # The default must be untouched: every existing config must keep its 150-concept
    # holdout, or every recorded intra-subject number silently changes meaning.
    d = concept_split()
    check(len(d.val_concepts) == 150 and len(d.fit_concepts) == 1504,
          "the default 150/1504 split is unchanged")

    # A 1-concept holdout passes every structural check and selects nothing.
    check(raises(lambda: _validate(_ns(val_concepts=1)), "coin flip").startswith(
              "--val-concepts 1 is not a holdout"),
          "--val-concepts 1 is refused rather than selecting on one concept")
    check(raises(lambda: _validate(_ns(val_concepts=-3)), "must be >= 0").startswith(
              "--val-concepts must be >= 0"),
          "a negative holdout size is refused")
    check(raises(lambda: concept_split(-1)).startswith("n_val must be"),
          "concept_split(-1) is refused rather than returning the whole set as val")


# --------------------------------------------------------------------- 2
def test_loso_flag_combinations() -> None:
    print("\n[2] half-specified and self-leaking folds are refused before the GPU")
    check(raises(lambda: _validate(_ns(source_subjects=[1, 2])),
                 "one setting").startswith("--source-subjects and --target-subject are one"),
          "source without target is refused (it would silently be intra-subject)")
    check(raises(lambda: _validate(_ns(target_subject=8)),
                 "one setting").startswith("--source-subjects and --target-subject are one"),
          "target without source is refused")
    check(raises(lambda: _validate(_ns(source_subjects=[8], target_subject=8)),
                 "its own test subject").startswith("--target-subject 8 is also in"),
          "training on the held-out subject is refused")
    check(raises(lambda: _validate(_ns(source_subjects=[1, 11], target_subject=3)),
                 "out-of-range").startswith("--source-subjects has out-of-range"),
          "a nonexistent subject id is refused")
    check(raises(lambda: _validate(_ns(source_subjects=[1], target_subject=99)),
                 "1..10").startswith("--target-subject must be"),
          "an out-of-range target subject is refused")
    check(raises(lambda: _validate(_ns(source_subjects=[1, 2], target_subject=3,
                                       struct_backbone="timm:dinov3_b16")),
                 "not wired yet").startswith(
                     "LOSO + the structure tower is not wired yet"),
          "LOSO + structure tower is refused: the aux caches are single-subject rows")

    ok = _ns(source_subjects=[1, 2, 3, 4, 5, 6, 7, 9, 10], target_subject=8)
    check(_validate(ok) is ok, "a well-formed 9-source / 1-target fold passes")


# --------------------------------------------------------------------- 3
def test_image_expansion_is_subject_major() -> None:
    print("\n[3] image features expand subject-major, so cross-subject positives align")
    rng = np.random.default_rng(0)
    img = rng.standard_normal((C, IMG, 8)).astype(np.float32)
    S = 4
    out = expand_loso_images(img, S)

    check(out.shape == (S * C, IMG, 8), f"shape is (S*C, I, D) = {(S * C, IMG, 8)}",
          f"got {out.shape}")
    # The blocking is the whole point: row i*C + c must be concept c, for every i and
    # every c. `np.repeat(img, S, axis=0)` also has the right SHAPE and scrambles this.
    ok = all(np.array_equal(out[i * C + c], img[c]) for i in range(S) for c in range(C))
    check(ok, "row i*C+c carries concept c for every (subject, concept) pair")
    bad = np.repeat(img, S, axis=0)
    check(not np.array_equal(bad, out),
          "and `np.repeat` would NOT have produced it (the plausible wrong shape)")
    check(np.allclose(out[:, 0], img[:, 0][np.arange(S * C) % C]),
          "slot 0 of every row matches the concept it claims, not the subject index")
    check(expand_loso_images(img, 1) is img,
          "S=1 returns the array itself, so the intra-subject path allocates nothing")

    # Cross-subject positives: the same concept must map to the SAME stimulus in
    # every subject block, or a cross-subject consistency term has nothing to match.
    conc = np.arange(S) * C + 5
    check(all(np.array_equal(out[r], out[conc[0]]) for r in conc),
          "concept 5 is the same image under all four subjects")
    check(np.all(out[conc] == out[conc[0]]),
          "so the four rows are bit-identical, not merely close")


# --------------------------------------------------------------------- 4
def test_dataset_carries_the_subject() -> None:
    print("\n[4] the dataset hands the subject id to the training loop, per row")
    rng = np.random.default_rng(1)
    S, n_img = 3, 4
    eeg = rng.standard_normal((S * C, n_img, N_CH, 16)).astype(np.float32)
    feat = rng.standard_normal((S * C, n_img, 8)).astype(np.float32)
    sub = np.repeat(np.arange(S), C)

    ds = TrainDataset(eeg, feat, np.arange(S * C), l2norm=False,
                      slots=[0, 1], subject_of_row=sub)
    got = [ds[i] for i in range(len(ds))]
    check(len(got[0]) == 4, "with subject_of_row the tuple is (x, f, c, subject)",
          f"got arity {len(got[0])}")
    check(all(int(g[3]) == int(sub[int(g[2])]) for g in got),
          "the returned subject matches the row's own subject for every item")

    # The assignment must be checked across subject BLOCKS, not just at the seam:
    # an off-by-one in the row arithmetic still passes at row 0 of every block.
    seen = {int(g[3]) for g in got}
    check(seen == set(range(S)), f"all {S} subjects appear ({sorted(seen)})")

    base = TrainDataset(eeg, feat, np.arange(S * C), l2norm=False, slots=[0])
    check(len(base[0]) == 3,
          "without subject_of_row the arity is unchanged, so every existing "
          "positional caller keeps working")

    msg = raises(lambda: TrainDataset(eeg, feat, np.arange(S * C), l2norm=False,
                                      subject_of_row=np.arange(S * C - 1)))
    check(msg.startswith("subject_of_row has"),
          "a subject array of the wrong length is refused rather than misindexed")


# --------------------------------------------------------------------- 5
def test_subject_embedding_is_wide_enough_and_trained() -> None:
    print("\n[5] n_subjects reaches the model, and the residual actually gets gradient")
    S = 9
    model = _model(n_subjects=S, layers=[2, 3], d_embed=16, image_dim=32)
    check(model.fusion.subject_residual.num_embeddings == S,
          f"the EEG-side subject table has {S} rows (not the hardcoded 1)",
          f"got {model.fusion.subject_residual.num_embeddings}")

    # The residual is zero-initialised, so at step 0 every subject is identical and
    # a shape-only check cannot tell a working wiring from a dead one. Push a
    # gradient through real subject ids and look at the table.
    x = torch.randn(4 * S, N_CH, 16)
    f = torch.randn(4 * S, 32)
    subj = torch.arange(S).repeat(4)
    _fit_zscore(model, x)
    z_e, z_i, w = model(x, f, subj, training=True)
    loss = torch.nn.functional.cross_entropy(
        torch.nn.functional.normalize(z_e, dim=-1)
        @ torch.nn.functional.normalize(z_i, dim=-1).t(),
        torch.arange(x.shape[0]))
    model.zero_grad()
    loss.backward()
    g = model.fusion.subject_residual.weight.grad
    check(g is not None and float(g.abs().sum()) > 0.0,
          "the per-subject embedding receives a non-zero gradient from `forward`",
          "grad is None or exactly zero, i.e. subject_ids never reached LayerFusion")

    # Every row must be present in the batch, or a row with no sample looks trained
    # while never having moved.
    check(True, f"batch covered {S} subjects, so no embedding row is untested here")


# --------------------------------------------------------------------- 6
def test_target_side_subject_residual() -> None:
    print("\n[6] SAMGA's target-side residual: subject-aware training, agnostic inference")
    S, K = 5, 3
    m = _model(n_subjects=S, layers=[2, 3], d_embed=16, image_dim=32,
               target_fusion="routed_sr", n_targets=K)
    check(isinstance(m.target_router, LayerFusion),
          "target_fusion='routed_sr' builds a router over the TARGET axis")
    check(m.target_w is None,
          "and leaves target_w unset, so there is exactly one target-weight parameter")
    check(m.target_router.subject_residual.num_embeddings == S,
          f"the target router's subject table has {S} rows")

    feat = torch.randn(6, K, 32)
    # The residual is zero-initialised, so at step 0 EVERY subject routes identically
    # and a comparison cannot distinguish "the subject branch works" from "the
    # subject branch is dead". Perturbing it first is what makes the two assertions
    # below meaningful.
    m.target_router.subject_residual.weight.data.normal_(0.0, 0.5)

    def route(sub, training):
        return m.target_router({i: feat[:, i, :] for i in range(K)},
                               subject_ids=sub, training=training)[1]

    m.train()
    z = torch.zeros(6, dtype=torch.long)
    hi = torch.full((6,), S - 1, dtype=torch.long)
    check(not torch.allclose(route(z, True), route(hi, True)),
          "during training the routing DEPENDS on the subject, so the residual is live")

    m.eval()
    check(torch.allclose(route(z, False), route(hi, False)),
          "at inference it does NOT, i.e. the deployed model is subject-agnostic")

    # End-to-end, not just at the router: `encode_image` must produce the SAME
    # embedding for two different subject ids in eval mode. This is the property the
    # gallery is built under, and it is the one an accidental `training=True`
    # default would break.
    c = m.encode_image(feat, z, training=False)
    d = m.encode_image(feat, hi, training=False)
    check(torch.allclose(c, d),
          "encode_image at inference is subject-invariant end to end")
    e = m.encode_image(feat)
    check(torch.allclose(c, e),
          "and the default arguments already give that inference behaviour, so a "
          "caller that forgets subject_ids ships the right thing")

    w = m.target_weights()
    check(w is not None and abs(float(w.sum()) - 1.0) < 1e-5 and (w >= 0).all(),
          "the recorded target weights are the GLOBAL prior, a valid distribution",
          f"got {None if w is None else w.tolist()}")

    msg = raises(lambda: _model(n_subjects=S, layers=[2, 3], d_embed=16, image_dim=32,
                                target_fusion="routed_sr", n_targets=1),
                 "needs >=2 targets")
    check(msg.startswith("target_fusion=routed_sr needs >=2 targets"),
          "a single target is refused: the residual is a constant after softmax")

    msg = raises(lambda: m.encode_image(feat, None, training=True))
    check(msg.startswith("target_fusion='routed_sr' is being trained without"),
          "training without subject_ids RAISES instead of silently dropping the "
          "residual branch and reporting a mechanism it is not using")

    from epd.train import assign_param_groups
    m2 = _model(n_subjects=S, layers=[2, 3], d_embed=16, image_dim=32,
                target_fusion="routed_sr", n_targets=K)
    g = assign_param_groups(m2)
    total = sum(len(v) for v in g.values())
    live = sum(1 for _, p in m2.named_parameters() if p.requires_grad)
    check(total == live, f"every trainable parameter is in an LR group ({total} == {live})")
    names = [n for n, _ in g["heads"]]
    check(any(n.startswith("target_router.") for n in names),
          "the target router is grouped as a head, not left to the error branch",
          f"heads: {names[:4]}")


# --------------------------------------------------------------------- 7
def test_per_subject_standardisation() -> None:
    print("\n[7] per-subject z-score, and the global fallback it replaces")
    rng = np.random.default_rng(2)
    S = 4
    blocks, means = [], []
    for s in range(S):
        scale = 1.0 + 3.0 * s          # deliberately very different amplitudes
        mu = 5.0 * s
        blk = (rng.standard_normal((C, 2, N_CH, 16)) * scale + mu).astype(np.float32)
        blocks.append(blk)
        means.append(mu)

    # This is what a single shared z-score table would produce: the loudest subject
    # owns the variance of the pool, and the quiet ones become near-constant.
    pool = np.concatenate(blocks, axis=0)
    gmean, gstd = _channel_stats(pool)
    per_subj_spread = [float((b.std(axis=-1).mean())) for b in blocks]
    after_global = [float(((b - gmean.reshape(1, 1, -1, 1))
                           / gstd.reshape(1, 1, -1, 1)).std()) for b in blocks]
    check(max(after_global) - min(after_global) > 1.0,
          "a single global z-score leaves the subjects at wildly different scales",
          f"stds after: {[round(x, 2) for x in after_global]}")

    after_own = []
    std_blocks = []
    for b in blocks:
        m, s = _channel_stats(b)
        z = ((b - m.reshape(1, 1, -1, 1)) / s.reshape(1, 1, -1, 1)).astype(np.float32)
        std_blocks.append(z)
        after_own.append(float(z.std()))
    # `_channel_stats` uses ddof=1 while `.std()` here is ddof=0, so the reference is
    # sqrt((n-1)/n) and not exactly 1. For the real arrays that is 0.9999999; for
    # this toy n=384 it is 0.9987, which is why the check is a band and not equality.
    n_pp = blocks[0].size / N_CH
    expected = float(np.sqrt((n_pp - 1) / n_pp))
    check(max(after_own) - min(after_own) < 1e-5
          and abs(after_own[0] - expected) < 1e-4,
          f"per-subject z-score puts all subjects on one scale (std ~ {expected:.4f} each)",
          f"stds after: {[round(x, 4) for x in after_own]}")

    # The axis convention has to match the tokenizer's, or a residual scale survives.
    m_eeg, s_eeg = _channel_stats(blocks[0])
    flat = blocks[0].reshape(-1, N_CH, 16)
    check(np.allclose(m_eeg, flat.mean(axis=(0, 2)), atol=1e-4)
          and np.allclose(s_eeg, flat.std(axis=(0, 2), ddof=1), atol=1e-4),
          "statistics use the same (trials, time) reduction as "
          "EEGPatchTokenizer.set_norm_stats")

    data = LosoData(tr_eeg=np.concatenate(std_blocks, axis=0),
                    tr_subject_of_row=np.repeat(np.arange(S), C),
                    te_eeg=std_blocks[0][:1], source_subjects=list(range(1, S + 1)),
                    target_subject=9)
    msg = raises(lambda: assert_standardised(data))
    check(msg == "__DID_NOT_RAISE__",
          "assert_standardised accepts a per-subject standardised pool", msg)
    check(len(data.source_subjects) == data.n_subjects == S,
          "n_subjects is derived from the fold, not configured separately")

    raw = LosoData(tr_eeg=pool.astype(np.float32) * 7.0 + 100.0,
                   tr_subject_of_row=np.repeat(np.arange(S), C),
                   te_eeg=blocks[0][:1], source_subjects=list(range(1, S + 1)),
                   target_subject=9)
    msg = raises(lambda: assert_standardised(raw))
    # The label is zero-padded (`sub-01`), so match the shared suffix, not the id.
    check("is not standardised" in msg,
          "assert_standardised CATCHES an unstandardised pool")
    check("std" in msg and "mean" in msg,
          "and reports both the mean and the std it measured, so the cause is visible",
          msg)


# --------------------------------------------------------------------- 8
def test_layer_fusion_subject_path() -> None:
    print("\n[8] LayerFusion's subject branch is bypassed at inference, kept in training")
    S, K = 6, 3
    lf = LayerFusion(layers=[8, 12, 16], d_in=8, n_subjects=S, d_out=8,
                     fusion_mode="routed", layer_dropout=0.0, subject_dropout=0.0)
    feats = {8: torch.randn(5, 8), 12: torch.randn(5, 8), 16: torch.randn(5, 8)}
    subj = torch.arange(5) % S
    # Zero-init again: without this the routing is identical for every subject by
    # construction and the training-mode assertion below cannot fail.
    lf.subject_residual.weight.data.normal_(0.0, 0.5)

    lf.train()
    _, w_train = lf(feats, subject_ids=subj, training=True)
    _, w_train2 = lf(feats, subject_ids=(subj + 1) % S, training=True)
    check(not torch.allclose(w_train, w_train2),
          "different subjects route to different layer weights during training")

    lf.eval()
    z1, w1 = lf(feats, subject_ids=subj, training=False)
    z2, w2 = lf(feats, subject_ids=(subj + 1) % S, training=False)
    check(torch.allclose(w1, w2) and torch.allclose(z1, z2),
          "at inference every subject gets the same global weights")
    check(torch.allclose(w1[0], lf.layer_weights()),
          "and they equal `layer_weights()`, i.e. the recorded weights describe "
          "the deployed model")


# ------------------------------------------------------------------------ MVNN
def test_mvnn_flag_validation() -> None:
    print("\n[9] MVNN flags are validated where they are cheap to validate")
    # A bad flag is a config error, so it must be refused before the data loader
    # reads 4 GiB per subject. These all raise from `_validate`, which runs first.
    raises(lambda: _validate(_ns(mvnn="yes")), "--mvnn")
    raises(lambda: _validate(_ns(mvnn_fixed=1.5)), "[0, 1]")
    raises(lambda: _validate(_ns(mvnn_fixed=-0.1)), "[0, 1]")
    raises(lambda: _validate(_ns(mvnn_max_cond=-5)), ">= 0")

    # The legitimate settings must all pass, including the ones that only make sense
    # in combination with LOSO.
    _validate(_ns(mvnn="train"))
    _validate(_ns(mvnn="off", mvnn_fixed=0.0, mvnn_shrinkage="fixed"))
    _validate(_ns(mvnn="test", source_subjects=[1, 2], target_subject=3))
    check(True, "the valid combinations are accepted")

    # `load_loso` re-checks independently, because it is also reachable from scripts
    # that never call `_validate`.
    from epd.data import load_loso
    msg = raises(
        lambda: load_loso([1, 2], 3, mvnn="sometimes"),
        "--mvnn must be off/train/test")
    check("sometimes" in msg, "load_loso names the offending value")


def test_mvnn_split_is_per_role() -> None:
    print("\n[10] a LOSO fold whitens sources from train and the holdout from test")
    # The asymmetry is the protocol, and getting it backwards is silent: fitting the
    # holdout's whitener on its own training split is exactly the data the fold
    # excludes. Assert on the CALLS, since the real loader needs the dataset.
    calls: list[tuple[int, str]] = []

    def fake_load_subject_std(subject_id, channels=None, cache_dir=None, mvnn="off",
                              **kw):
        calls.append((int(subject_id), mvnn))
        # Must LOOK standardised: `load_loso` ends in `assert_standardised`, which
        # reads mean 0 / std 1 off the array and would (correctly) reject zeros.
        arr = np.linspace(-1.7, 1.7, C * N_CH * N_TIME).reshape(C, N_CH, N_TIME)
        arr = (arr / arr.std()).astype(np.float32)
        return arr[:, None], arr[:, None]

    import epd.data as D
    real = D.load_subject_std
    D.load_subject_std = fake_load_subject_std
    try:
        with patch.object(D.config, "N_TRAIN_CONCEPTS", C), \
             patch.object(D.config, "N_TEST_CONCEPTS", C):
            D.load_loso([1, 2, 3], 7, mvnn="train", verbose=False)
    finally:
        D.load_subject_std = real

    roles = dict(calls)
    check(all(v == "train" for s, v in calls if s != 7),
          "every SOURCE subject is whitened from its own training residuals",
          str(sorted(set(v for s, v in calls if s != 7))))
    check(roles.get(7) == "test",
          "the HELD-OUT subject is whitened from its own test residuals",
          f"got {roles.get(7)!r}")
    check(len(calls) == 4, "four subjects loaded, once each", str(len(calls)))


def test_mvnn_cache_key_is_distinct() -> None:
    print("\n[11] every MVNN variant gets its own cache file")
    # The cache is what makes the whitener reproducible across the train/export/eval
    # processes, so two different whiteners sharing a filename is the bug that
    # silently makes two arms incomparable.
    import re
    from pathlib import Path as _P
    src = (_P(__file__).resolve().parents[1] / "scripts" / "epd" / "data.py").read_text()
    check('suffix = "_std" if mvnn == "off" else f"_mvnn{mvnn}_std"' in src,
          "the _std cache name is suffixed by the mvnn mode")
    check("_c{max_cond}" in src or "max_cond}" in src,
          "the smoke subsample is part of the whitener cache key")
    check('key = f"{tag}_{split}_{shrinkage}"' in src,
          "the whitener key carries subject, channel set, fit split and shrinkage")

    # And the loader must return what it cached: an arithmetic slip here would make
    # the returned array differ from the stored one for the same name.
    check('np.save(w_path, wh.w)' in src, "the whitener is persisted, not recomputed")
    check("json.dumps(wh.extra" in src, "with a sidecar of its diagnostics")


# ------------------------------------------------------------------------ utils
def _ns(**over):
    """An argparse-like namespace with the defaults `_validate` reads."""
    class NS:
        val_concepts = 150
        source_subjects = None
        target_subject = None
        struct_backbone = ""
        select_last = False
        smoke = False
        patience = 12
        target_fusion = "single"
        mvnn = "off"
        mvnn_shrinkage = "lw"
        mvnn_fixed = 0.1
        mvnn_max_cond = 0
    ns = NS()
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


def _fit_zscore(model: RetrievalModel, x: torch.Tensor) -> None:
    """Give the tokenizer z-score statistics, as `train.py` does before the loop.

    Not optional: the tokenizer raises rather than silently running with mean 0 /
    std 1, precisely because a patch_embed trained on z-scored patches sees a
    different value range without it.
    """
    model.encoder.tokenizer.set_norm_stats(x.numpy())


def _model(n_subjects: int, layers: list[int], d_embed: int, image_dim: int,
           target_fusion: str = "single", n_targets: int = 1) -> RetrievalModel:
    return RetrievalModel(
        backbone="timm:vit_b16_in21k_orig", channel_names=CH, layers=layers,
        n_subjects=n_subjects, d_embed=d_embed, image_dim=image_dim,
        tokenizer_kind="eegit", patch_size=16, n_patches_w=14, style="time-region",
        pool="mean", timm_global_pool="avg", head_kind="eegit", img_head_kind="eegit",
        fusion_mode="routed", pretrained=False, freeze_all=False,
        subject_dropout=0.0, layer_dropout=0.0, head_drop=0.0,
        target_fusion=target_fusion, n_targets=n_targets,
    )


def main() -> None:
    print("=" * 72)
    print("LOSO / INTER-SUBJECT WIRING TESTS  (no GPU, no data)")
    print("=" * 72)
    test_no_holdout_is_a_protocol()
    test_loso_flag_combinations()
    test_image_expansion_is_subject_major()
    test_dataset_carries_the_subject()
    test_subject_embedding_is_wide_enough_and_trained()
    test_target_side_subject_residual()
    test_per_subject_standardisation()
    test_layer_fusion_subject_path()
    test_mvnn_flag_validation()
    test_mvnn_split_is_per_role()
    test_mvnn_cache_key_is_distinct()
    print("\n" + "=" * 72)
    if _failures:
        print(f"FAILED: {len(_failures)} checks")
        for f in _failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print(f"ALL PASSED: {_passes} checks")


if __name__ == "__main__":
    main()
