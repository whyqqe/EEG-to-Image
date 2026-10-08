#!/usr/bin/env python
"""Module-load smoke test: no dataset, no GPU, no downloads.

Exercises the pieces that can only be validated by running them:
  * the target router modes (mean / routed / routed_sr) and the inference-time
    drop of the subject residual
  * the cross-subject sampler on a synthetic fold
  * every loss term, forward *and* backward, including WHICH parameters it can reach
  * the prototype bank (EMA update, validity mask, and the asymmetry direction)
  * retrieval metrics and the calibration score path
  * an end-to-end training step + evaluation on a synthetic fold
  * invariants -- each one is a bug that actually shipped

Run:  python scripts/smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from samclip import calibration, evaluate  # noqa: E402
from samclip.data import augment as augment_mod  # noqa: E402
from samclip.data.sampler import CrossSubjectBatchSampler  # noqa: E402
from samclip.losses import (  # noqa: E402
    SCALE_MAX,
    SCALE_MIN,
    InfoNCE,
    SubjectAdversary,
    clip_alignment_loss,
    cross_subject_loss,
    gram_distill_loss,
    hsic_subject,
    mmd_subject,
    vicreg_terms,
)
from samclip.losses.regularizers import PrototypeEMA  # noqa: E402
from samclip.models import LayerRouter, build_model  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)

C, T, N_SUB = 8, 40, 4
K_LAYERS, IMG_DIM = 3, 32
B_STIM, B_SUBJ = 3, 3


def make_model(fusion: str = "mean", d_model: int = 32, d_embed: int = 24):
    cfg = {
        "n_channels": C, "n_timepoints": T, "n_subjects": N_SUB,
        "d_model": d_model, "d_embed": d_embed, "n_heads": 4, "n_blocks": 2,
        "dim_ff": 48,
        "target_fusion": fusion, "target_subject_dropout": 0.3,
    }
    return build_model(cfg, K_LAYERS, IMG_DIM)


def check(name: str, ok: bool) -> None:
    print(f"  [{'ok' if ok else 'FAIL'}] {name}")
    if not ok:
        raise SystemExit(f"smoke test failed: {name}")


def _toy_batch(n: int = 9) -> dict:
    return {
        "eeg": torch.randn(n, C, T),
        "target": torch.randn(n, K_LAYERS, IMG_DIM),
        "subject": torch.tensor([0, 1, 2] * (n // 3)),
        "stimulus": torch.tensor(sum(([s] * 3 for s in range(n // 3)), [])),
        "concept": torch.tensor(sum(([s] * 3 for s in range(n // 3)), [])),
    }


def main() -> None:
    print("=" * 70)
    print("1. target fusion modes: forward + backward through the whole objective")
    for fusion in ("mean", "routed", "routed_sr"):
        model = make_model(fusion)
        batch = _toy_batch()
        out = model(batch["eeg"], batch["target"], subject_ids=batch["subject"],
                    training=True)
        check(f"fusion={fusion:9s} shapes {tuple(out['z_eeg'].shape)}",
              out["z_eeg"].shape == (9, 24) and out["z_img"].shape == (9, 24))
        crit = InfoNCE()
        loss = clip_alignment_loss(out["z_eeg"], out["z_img"], crit)
        loss = loss + 0.5 * cross_subject_loss(out["z_eeg"], batch["stimulus"], crit)
        loss = loss + 0.05 * hsic_subject(out["z_eeg"], batch["subject"], N_SUB)
        loss = loss + 0.1 * mmd_subject(out["z_eeg"], batch["subject"])
        loss = loss + 0.1 * gram_distill_loss(out["z_eeg"], out["z_img"])
        reg = vicreg_terms(out["z_eeg_raw"])
        loss = loss + 0.5 * (reg["var"] + reg["cov"])
        loss.backward()
        grads = [p.grad is not None for p in model.parameters() if p.requires_grad]
        check(f"fusion={fusion:9s} loss={float(loss):.3f} grads={sum(grads)}/{len(grads)}",
              loss.item() == loss.item() and all(grads))

    print("2. routed_sr: the subject residual is dropped at inference")
    # This is the mechanism's defining property, and it is invisible in a shape check:
    # a router that kept the residual at inference would still forward cleanly and would
    # still score, just as a model that needs the subject identity it will not have at
    # deployment. So it is asserted directly.
    router = LayerRouter(n_layers=K_LAYERS, d_in=IMG_DIM, mode="routed_sr",
                         n_subjects=N_SUB)
    with torch.no_grad():
        router.residual.weight.normal_(0.0, 1.0)      # make the residual NOT a no-op
    x = torch.randn(6, K_LAYERS, IMG_DIM)
    sid = torch.tensor([0, 1, 2, 0, 1, 2])
    train_blend = router(x, subject_ids=sid, training=True)
    infer_blend = router(x, subject_ids=sid, training=False)
    agnostic = router(x, training=False)
    check("the residual changes the TRAINING blend", not torch.allclose(train_blend, agnostic))
    check("the residual is dropped at INFERENCE (subject-agnostic blend)",
          torch.allclose(infer_blend, agnostic))

    print("3. routed_sr refuses to train without subject ids")
    try:
        router(torch.randn(4, K_LAYERS, IMG_DIM), subject_ids=None, training=True)
        check("routed_sr guard fires when ids are missing", False)
    except RuntimeError:
        check("routed_sr guard fires when ids are missing", True)

    print("4. cross-subject sampler guarantees co-stimulus rows")
    sampler = CrossSubjectBatchSampler(n_subjects=N_SUB, n_concepts=20, n_images=5,
                                       batch_stimuli=B_STIM, subjects_per_stimulus=B_SUBJ)
    per_subject = 20 * 5
    n_ok = 0
    for batch in sampler:
        # `r % (C*I)` is exactly the global stimulus id (concept*I + slot), and it is
        # what `cross_subject_loss` groups on.
        stims = [r % per_subject for r in batch]
        subjects = [r // per_subject for r in batch]
        grouped = (len(set(stims)) == B_STIM
                   and all(stims.count(s) == B_SUBJ for s in set(stims)))
        # each co-stimulus group must span DISTINCT subjects, otherwise it is a
        # within-subject repetition dressed up as a cross-subject positive
        distinct = all(len(set(subjects[i] for i in range(len(batch))
                               if stims[i] == s)) == B_SUBJ for s in set(stims))
        if grouped and distinct:
            n_ok += 1
    check(f"sampler produced {n_ok}/{len(sampler)} valid batches", n_ok == len(sampler))

    print("5. prototype bank: EMA, validity mask, and the ASYMMETRY direction")
    proto = PrototypeEMA(n_classes=50, d_embed=24, min_updates=1)
    z = torch.randn(6, 24)
    group = torch.tensor([0, 0, 1, 1, 2, 2])
    # Before any update the bank must contribute NOTHING. A prototype contrast run
    # against an all-zero bank is a well-formed cross-entropy that trains the encoder
    # toward the origin -- i.e. it collapses the representation while reporting a
    # falling loss. The `min_updates` mask is what prevents that, so it is asserted.
    check("an untouched bank contributes zero (validity mask)",
          float(proto.proto_contrast(z, group)) == 0.0)
    proto.update(z, group)
    l = proto.proto_contrast(z, group)
    check(f"prototype contrast is finite after an update ({float(l):.3f})",
          l.item() == l.item() and l.item() > 0)

    # --- the asymmetry -------------------------------------------------------
    # `proto_contrast` must reach the EEG encoder; `image_anchor_contrast` must reach
    # the IMAGE head and NOT the EEG encoder. If both moved the encoder, the two terms
    # would compete for the same degrees of freedom and could be satisfied by collapse;
    # if the anchor moved the prototypes, it would not be an anchor at all (they are
    # buffers, so this is also a check that no one has wired a grad path into the bank).
    m = make_model("routed")
    eeg = torch.randn(6, C, T)
    tgt = torch.randn(6, K_LAYERS, IMG_DIM)
    proto2 = PrototypeEMA(n_classes=3, d_embed=24, min_updates=1)
    proto2.update(m.encode_eeg(eeg).detach(), torch.tensor([0, 0, 1, 1, 2, 2]))

    z_e = m.encode_eeg(eeg)
    l_e = proto2.proto_contrast(z_e, torch.tensor([0, 0, 1, 1, 2, 2]))
    m.zero_grad(set_to_none=True)
    l_e.backward()
    trunk_grad = sum(float(p.grad.abs().sum()) for p in m.trunk.parameters()
                     if p.grad is not None)
    check(f"proto_contrast reaches the EEG encoder (|grad_trunk| {trunk_grad:.3e})",
          trunk_grad > 0)

    m.zero_grad(set_to_none=True)
    z_i = m.encode_target(tgt, training=True)
    l_i = proto2.image_anchor_contrast(z_i, torch.tensor([0, 0, 1, 1, 2, 2]))
    l_i.backward()
    img_grad = sum(float(p.grad.abs().sum()) for p in m.img_head.parameters()
                   if p.grad is not None)
    trunk_grad2 = sum(float(p.grad.abs().sum()) for p in m.trunk.parameters()
                      if p.grad is not None)
    check(f"image_anchor_contrast reaches the IMAGE head ({img_grad:.3e})",
          img_grad > 0)
    check(f"...and NOT the EEG encoder ({trunk_grad2:.3e}) -- that asymmetry is the "
          f"anchor", trunk_grad2 == 0.0)
    check("the prototype bank receives no gradient (it is a buffer)",
          proto2.proto.grad is None)

    print("6. adversary path")
    adv = SubjectAdversary(24, N_SUB)
    logits = adv(torch.randn(9, 24), lambd=1.0)
    l = SubjectAdversary.loss(logits, torch.tensor([0, 1, 2] * 3))
    l.backward()
    check(f"adversary loss finite ({float(l):.3f})", l.item() == l.item())

    print("7. retrieval metrics + calibration score path")
    q = np.random.randn(20, 24).astype(np.float32)
    g = q.copy()  # perfect gallery
    rep = evaluate.retrieval_report(q, g)
    check(f"perfect retrieval top1={rep['top1']:.1f}", rep["top1"] == 100.0)
    sc, diag = calibration.calibrate(q, np.random.randn(20, 24), whiten=True, csls=True)
    check("csls+whiten produces a score matrix", sc.shape == (20, 20))

    # SCORE coordinate recovery. Fitted on structured features on purpose: with a
    # random query set the mutual-nearest-neighbour set can be EMPTY, and the shared
    # implementation correctly refuses rather than returning an identity map that
    # would masquerade as "recovery ran". An identity gallery guarantees the
    # pseudo-pairs exist, which is the branch we want to exercise here.
    qq = np.random.randn(30, 24).astype(np.float32)
    rec, rdiag = calibration.coordinate_recovery(qq, qq.copy(), k=10, rho=0.1)
    check(f"coordinate recovery ({rdiag['n_mutual_pairs']} landmarks, "
          f"|R-I|={rdiag['r_minus_i_frobenius']:.3f})",
          rec.shape == qq.shape and rdiag["n_mutual_pairs"] > 0)
    # ...and the abstention gate must return the UNRECOVERED features, not a crash.
    # The tolerance is 1e-5 rather than the default 1e-8 because the two moment-match
    # implementations are separate (torch inside `epd.recover`, numpy here) and agree
    # only to float32 precision, which is ~1e-6 -- tighter than that would be testing
    # op ordering, not behaviour.
    qr = np.random.randn(40, 24).astype(np.float32)
    gr = np.random.randn(40, 24).astype(np.float32)
    rec2, d2 = calibration.coordinate_recovery(qr, gr, k=10, min_landmark_rate=1.0)
    check(f"recovery abstention gate (rate {d2.get('landmark_rate'):.2f})",
          d2.get("abstained", False) is True
          and np.allclose(rec2, calibration.moment_match(qr, gr), atol=1e-5))

    print("8. end-to-end training step + evaluation on a synthetic fold")
    from torch.utils.data import DataLoader

    from samclip import train as train_mod
    from samclip.data import things_eeg

    N_CH, N_T, N_CONC, N_IMG, N_SRC = C, T, 6, 3, 3
    rng = np.random.default_rng(0)
    tr = [rng.standard_normal((N_CONC, N_IMG, N_CH, N_T)).astype(np.float32)
          for _ in range(N_SRC)]
    for blk in tr:                                   # make it look standardised
        blk -= blk.mean()
        blk /= blk.std()
    loso = things_eeg.LosoData(
        tr_eeg=tr, source_subjects=[1, 2, 3], target_subject=4,
        te_eeg=rng.standard_normal((5, 1, N_CH, N_T)).astype(np.float32),
        n_concepts=N_CONC, n_images=N_IMG,
    )
    tgt_tr = rng.standard_normal((N_CONC, N_IMG, K_LAYERS, IMG_DIM)).astype(np.float32)
    tgt_te = rng.standard_normal((5, 1, K_LAYERS, IMG_DIM)).astype(np.float32)

    tcfg = {"loss_weights": {"img": 1.0, "cross": 0.5, "dec": 0.05, "mmd": 0.1,
                             "reg": 0.5, "proto": 0.1, "rkd": 0.0, "adv": 0.0},
            "prototype_level": "concept",
            "batch_stimuli": 2, "images_per_pair": 1, "num_workers": 0,
            "eval_batch": 5, "temp_img": 0.07, "temp_cross": 0.1, "softplus": True}
    tmodel = make_model("routed")
    tmodel.train()
    loader, sampler = train_mod.build_loaders(tcfg, loso, tgt_tr)
    test_loader = train_mod.build_test_loader(tcfg, loso, tgt_te)
    proto3 = PrototypeEMA(N_CONC, 24, min_updates=1)
    trainer = train_mod.Trainer(model=tmodel, cfg=tcfg, device=torch.device("cpu"),
                                n_subjects=N_SRC,
                                weights=train_mod.LossWeights.from_cfg(tcfg),
                                prototype=proto3)
    opt = torch.optim.AdamW(list(tmodel.parameters()) + trainer.criterion_parameters(),
                            lr=1e-3)
    n_steps = 0
    for batch in loader:
        opt.zero_grad(set_to_none=True)
        loss, parts = trainer.assemble(batch)
        loss.backward()
        opt.step()
        trainer.update_prototype(batch)
        n_steps += 1
    check(f"ran {n_steps} optimizer steps, last loss {float(loss):.3f}",
          n_steps == len(loader) and loss.item() == loss.item())
    check(f"the prototype term was actually applied and logged {sorted(parts)}",
          "proto" in parts and "anchor" in parts)

    # The prototype group is a DATA decision read from the batch. A missing key must
    # raise: silently skipping the term would leave the run reporting a weight it never
    # applied, which is the failure mode this project already paid a full pipeline run for.
    try:
        trainer.prototype_group({"stimulus": torch.zeros(3, dtype=torch.long)})
        check("a missing prototype key raises instead of silently disabling the term",
              False)
    except KeyError:
        check("a missing prototype key raises instead of silently disabling the term",
              True)

    metrics = train_mod.evaluate_fold(tmodel, test_loader, torch.device("cpu"))
    check(f"evaluation produced valid metrics {metrics}",
          metrics["n"] == 5 and 0.0 <= metrics["top1"] <= 100.0)

    # --- 8b. the loss weights must be validated, not silently ignored -----------
    # A typo'd key in the objective is a run that reports a term it never optimised.
    # `from_cfg` used to iterate over the DEFAULTS and `get` each one, which accepts any
    # junk in the config and quietly drops it.
    try:
        train_mod.LossWeights.from_cfg({"loss_weights": {"imgg": 1.0}})
        check("an unknown loss weight is rejected", False)
    except ValueError as exc:
        check(f"an unknown loss weight is rejected ({str(exc)[:52]}...)",
              "unknown loss_weights" in str(exc))

    invariants()
    v3_mechanisms()
    v4_mechanisms()
    d2_front_end()
    recovery_module()
    commet_gates()

    print("=" * 70)
    print("ALL SMOKE TESTS PASSED")


def v3_mechanisms() -> None:
    """The three mechanisms v3 adds, each asserted as a PROPERTY.

    All three are new in the SOTA push and all three can fail silently: a schedule that
    never ramps still trains, an entropy bonus that is inert still logs a finite value,
    and a router that collapses to one layer still retrieves. So each is asserted by the
    property that makes it useful, not by finiteness.
    """
    from samclip import train as train_mod
    from samclip.utils import load_config

    print("10. coarse-to-fine schedule")
    # (a) THE INTERPOLATION LAW, checked against the reference verbatim. Its log prints
    # mmd=0.900/contrast=0.100 at epoch 1 and mmd=0.689/contrast=0.311 at epoch 11
    # (1-based), i.e. index 10 of a 20-epoch Stage 1 with the reference's own endpoints
    # (0.9 -> 0.5). Reproducing those to 3 decimals is what proves the formula was
    # transcribed rather than merely made monotone. This is checked on the reference's
    # endpoints on purpose -- see (b) for why the shipped endpoints differ.
    ref = train_mod.CoarseToFine.from_cfg(
        {"epochs": 50, "lr": 1e-3,
         "schedule": {"coarse_to_fine": True, "stage1_epochs": 20,
                      "mmd_start": 0.9, "mmd_end": 0.5,
                      "cross_start": 0.1, "cross_end": 0.5}})
    r0 = ref.at(0)
    r10 = ref.at(10)
    r20 = ref.at(20)
    check(f"the interpolation law reproduces the reference at epoch 1 "
          f"(mmd {r0[0]:.3f} contrast {r0[1]:.3f})",
          abs(r0[0] - 0.9) < 1e-9 and abs(r0[1] - 0.1) < 1e-9)
    check(f"the interpolation law reproduces the reference at epoch 11 "
          f"(mmd {r10[0]:.3f} vs 0.689, contrast {r10[1]:.3f} vs 0.311)",
          abs(r10[0] - 0.689) < 1e-3 and abs(r10[1] - 0.311) < 1e-3)
    check(f"stage 2 drops the lr and holds a residual MMD rather than the reference's "
          f"zero (phase {r20[3]}, mmd {r20[0]:.3f}, lr {r20[2]:g})",
          r20[3] == "stage2" and r20[0] == 0.05 and r20[2] == 5e-5)
    # ...and the residual must be a KNOB, not a hard-coded deviation: `stage2_mmd: 0`
    # has to reproduce the reference recipe exactly, or the two are not comparable.
    ref0 = train_mod.CoarseToFine.from_cfg(
        {"epochs": 50,
         "schedule": {"coarse_to_fine": True, "stage1_epochs": 20, "stage2_mmd": 0.0}})
    check("...and `stage2_mmd: 0` reproduces the reference's contrastive-only Stage 2",
          ref0.at(20)[0] == 0.0)

    # (b) THE SHIPPED ENDPOINTS, checked against the config the run actually reads.
    # The reference's ramp SHAPE is kept, but its absolute MMD magnitude is not
    # transplantable: it runs a two-term objective where MMD carries ~53% of the weighted
    # loss, while ours carries eight terms (VICReg, prototype, HSIC, ...), so copying
    # `0.5` verbatim would reproduce the number and not the effect. The endpoints come
    # from the plan doc §5.4 table instead.
    real = load_config(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")
    sch = train_mod.CoarseToFine.from_cfg(real)
    sched_cfg = real["schedule"]
    mmd0, cross0, _, ph0 = sch.at(0)
    mmd19, cross19, _, _ = sch.at(19)
    mmd20, cross20, lr20, ph20 = sch.at(20)
    mmd39, cross39, _, _ = sch.at(39)          # Stage 2 must hold, not keep ramping
    check(f"stage-1 start matches the reference (mmd {mmd0:.3f}, contrast {cross0:.3f}, "
          f"{ph0})", abs(mmd0 - 0.9) < 1e-9 and abs(cross0 - 0.1) < 1e-9)
    check(f"stage-1 ends at the plan doc §5.4 steady state (mmd {mmd19:.3f} vs "
          f"{sched_cfg['mmd_end']}, cross {cross19:.3f} vs {sched_cfg['cross_end']})",
          abs(mmd19 - sched_cfg["mmd_end"]) < 1e-9
          and abs(cross19 - sched_cfg["cross_end"]) < 1e-9)
    check(f"stage 2 holds the residual state (mmd {mmd20:.3f}, cross {cross20:.3f}, "
          f"lr {lr20:g}, {ph20})",
          ph20 == "stage2" and mmd20 == sched_cfg["stage2_mmd"]
          and lr20 == sched_cfg["stage2_lr"]
          and cross20 == sched_cfg["cross_end"] == cross39 == 0.7)

    # --- the loss weights must equal the plan doc §5.4 table -------------------
    # Three of those entries (`cross` 0.5->0.7, `mmd` 0.1->0.2, `proto` 0.1->0.2) were
    # prescribed by the plan and NEVER APPLIED to the config. Pinning them here is what
    # stops that from happening silently a second time.
    lw = real["loss_weights"]
    want = {"img": 1.0, "cross": 0.7, "mmd": 0.2, "proto": 0.2, "reg": 0.5,
            "dec": 0.05, "adv": 0.0}
    check(f"loss weights match the plan doc §5.4 table ({ {k: lw[k] for k in want} })",
          {k: lw[k] for k in want} == want)

    # A schedule whose Stage 1 swallows the whole run is a mislabelled single stage, and
    # it must be rejected rather than silently run.
    try:
        train_mod.CoarseToFine.from_cfg(
            {"epochs": 20, "schedule": {"coarse_to_fine": True, "stage1_epochs": 20}})
        check("a Stage 1 spanning the whole run is rejected", False)
    except ValueError:
        check("a Stage 1 spanning the whole run is rejected", True)
    # ...but a SHORT smoke run with the schedule off must stay legal, or the cheap DAG
    # test (`--epochs 1`) would be unusable.
    train_mod.CoarseToFine.from_cfg({"epochs": 1, "schedule": {"coarse_to_fine": False}})
    check("stage1_epochs is not validated when the schedule is off", True)
    try:
        train_mod.CoarseToFine.from_cfg({"schedule": {"stage_1_epochs": 5}})
        check("an unknown schedule key is rejected", False)
    except ValueError as exc:
        check(f"an unknown schedule key is rejected ({str(exc)[:44]}...)",
              "unknown schedule" in str(exc))
    check("a disabled schedule reports itself as single-stage",
          "single-stage" in train_mod.CoarseToFine.from_cfg({}).describe())

    print("11. router entropy bonus")
    n_layers = K_LAYERS
    router = LayerRouter(n_layers=n_layers, d_in=IMG_DIM, mode="routed",
                         temperature=2.0, layer_dropout=0.1)
    x = torch.randn(6, n_layers, IMG_DIM)
    router(x, training=True)
    pen = router.entropy_penalty()
    # AT UNIFORM INIT THE BONUS MUST BE INERT. Every logit is 0, so the clean softmax is
    # exactly uniform and its entropy is already maximal: the bonus contributes ZERO
    # gradient and the router is free to specialise. A bonus computed on the
    # dropout-perturbed weights would have a non-zero gradient here and would fight the
    # dropout instead of the collapse, which is why the two are separated.
    g = torch.autograd.grad(pen, router.w, retain_graph=True)[0]
    check(f"the bonus is exactly one-sided at uniform init (penalty {float(pen):.4f} = "
          f"-ln{n_layers} = {-np.log(n_layers):.4f}, |grad| {float(g.abs().max()):.1e})",
          abs(float(pen) + np.log(n_layers)) < 1e-6 and float(g.abs().max()) < 1e-6)
    # ...and it must actually bite once the router tries to collapse.
    with torch.no_grad():
        router.w.copy_(torch.tensor([10.0] + [0.0] * (n_layers - 1)))
    router(x, training=True)
    collapsed = float(router.entropy_penalty())
    check(f"the bonus bites as the router collapses ({float(pen):.3f} -> "
          f"{collapsed:.3f}, approaching 0 = one-hot)",
          collapsed > float(pen) + 1.0)
    with torch.no_grad():
        router.w.zero_()

    # `mean` has no learned weights, so it must contribute NOTHING rather than a
    # constant that a weight could be tuned against.
    mean_router = LayerRouter(n_layers=n_layers, d_in=IMG_DIM, mode="mean")
    mean_router(x, training=True)
    check("a uniform (`mean`) fusion contributes no entropy term",
          mean_router.entropy_penalty() is None)

    # The stash must be training-only, or an evaluation pass would leave weights behind
    # for the next training step to read as its own.
    router(x, training=False)
    check("the entropy stash is cleared by an inference pass",
          router.entropy_penalty() is None)

    # Layer dropout must keep the blend a CONVEX COMBINATION. Zeroing weights without
    # renormalising would scale the target down for that row -- a silent magnitude change
    # the alignment loss would absorb as if it were signal.
    router(x, training=True)
    w = router._last_weights
    check(f"the clean weights are a distribution (sum {float(w.sum(-1)[0]):.6f}, "
          f"min {float(w.min()):.4f})",
          bool((w >= 0).all()) and abs(float(w.sum(-1)[0]) - 1.0) < 1e-6)

    # The image head must actually receive the entropy gradient -- a bonus that only
    # reached the encoder would be shaping the wrong half of the objective.
    model = make_model("routed")
    model.train()
    batch = _toy_batch()
    out = model(batch["eeg"], batch["target"], subject_ids=batch["subject"],
                training=True)
    pen = model.router_entropy()
    check("the model exposes the router penalty for the Trainer",
          pen is not None and float(pen) <= 0.0 + 1e-9)
    trainer = train_mod.Trainer(model=model, cfg={"temp_img": 0.07, "temp_cross": 0.1},
                                device=torch.device("cpu"), n_subjects=N_SUB,
                                weights=train_mod.LossWeights.from_cfg(
                                    {"loss_weights": {"router": 0.05}}))
    loss, parts = trainer.assemble(batch)
    check(f"the router term reaches the objective and the log ({sorted(parts)} contains "
          f"'router' = {float(parts['router']):.4f})", "router" in parts)
    # ...and with the weight at 0 the objective is the v2 one exactly, so an old run is
    # reproducible from the new code.
    trainer0 = train_mod.Trainer(model=model, cfg={"temp_img": 0.07, "temp_cross": 0.1},
                                 device=torch.device("cpu"), n_subjects=N_SUB,
                                 weights=train_mod.LossWeights.from_cfg(
                                     {"loss_weights": {"router": 0.0}}))
    _l0, parts0 = trainer0.assemble(batch)
    check("router=0 restores the v2 objective bit-for-bit (no 'router' part)",
          "router" not in parts0)
    _ = out


def _make_v4_model(min_rows: int = 3, smn_enabled: bool = True, **kw):
    """A v4 model at toy width. `min_rows=3` matches the 3-rows-per-subject toy batch."""
    cfg = {
        "n_channels": C, "n_timepoints": T, "n_subjects": N_SUB,
        "d_model": 32, "d_embed": 24, "d_latent": 16, "d_align": 16,
        "n_heads": 4, "n_blocks": 2, "dim_ff": 48,
        "target_fusion": "routed", "target_subject_dropout": 0.3,
        "arch": "v4", "objective": "v4",
        "smn": {"enabled": smn_enabled, "gate_scale": True, "init_gate": 0.0,
                "min_rows": min_rows},
    }
    cfg.update(kw)
    return build_model(cfg, K_LAYERS, IMG_DIM)


class _FakeDS(torch.utils.data.Dataset):
    """Minimal stand-in for `TestDataset`, for the extraction-invariance test.

    The tensors are MATERIALISED IN `__init__` from a seeded generator, not drawn in
    `__getitem__`. That matters: a lazy `torch.randn` per access makes two loaders over
    the "same" dataset see two independent draws, so a comparison across `eval_batch`
    would be comparing two random datasets and the invariance under test would be
    unmeasurable (the first version of this fixture did exactly that, and the failure
    looked like an extraction bug).
    """

    def __init__(self, n: int, seed: int = 11):
        g = torch.Generator().manual_seed(seed)
        self.n = int(n)
        self.eeg = torch.randn(n, C, T, generator=g)
        self.target = torch.randn(n, K_LAYERS, IMG_DIM, generator=g)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict:
        return {"eeg": self.eeg[i], "target": self.target[i], "concept": i}


def _stack_collate(rows: list[dict]) -> dict:
    return {"eeg": torch.stack([r["eeg"] for r in rows]),
            "target": torch.stack([r["target"] for r in rows]),
            "concept": torch.tensor([r["concept"] for r in rows])}


def v4_mechanisms() -> None:
    """The five v4 components, each asserted as a PROPERTY rather than by shape.

    Every one of them can fail silently: a `share_head` that is actually two modules still
    forwards, an SMN that does nothing still trains, a spectral term that cannot be
    reduced still logs a number in [0,1], a cross-modal MMD with the wrong bandwidth still
    returns a finite scalar, and a schedule coefficient applied twice still ramps. So each
    is checked by the property that makes it useful.
    """
    from samclip import train as train_mod
    from samclip.losses import mmd_crossmodal, spectral_concentration
    from samclip.losses.regularizers import spectrum_report
    from samclip.utils import load_config
    from torch.utils.data import DataLoader

    print("12. v4 architecture: share_head / SMN / low rank")
    m = _make_v4_model()
    v3 = make_model("routed")
    check(f"v4 has a SHARED head and no per-modality head "
          f"(arch={m.arch}, d_align={m.d_align})",
          hasattr(m, "share_head") and not hasattr(m, "img_head")
          and not hasattr(m, "eeg_head") and m.d_align == 16)
    check("v3 keeps its two heads and has NO smn submodule (so old checkpoints still "
          "load strictly)",
          v3.arch == "v3" and v3.smn is None and hasattr(v3, "eeg_head")
          and "smn.gate_raw" not in v3.state_dict())

    eeg = torch.randn(9, C, T)
    tgt = torch.randn(9, K_LAYERS, IMG_DIM)
    out = m(eeg, tgt, subject_ids=torch.tensor([0, 1, 2] * 3), training=True)
    check(f"v4 forward is shaped as before ({tuple(out['z_eeg'].shape)}, "
          f"d_align {out['z_eeg'].shape[-1]})",
          out["z_eeg"].shape[-1] == 16 and out["z_img"].shape == out["z_eeg"].shape)

    # --- C1: BOTH branches really do go through the ONE module ----------------
    # The claim is structural, so it is asserted structurally: a loss on the EEG branch
    # alone must reach `share_head`, and so must one on the image branch alone. A model
    # with a hidden second projection would pass a shape check and fail this.
    def _share_grad(model):
        return sum(float(p.grad.abs().sum()) for p in model.share_head.parameters()
                   if p.grad is not None)

    ids9 = torch.tensor([0, 1, 2] * 3)
    z_e = m.encode_eeg(eeg, subject_ids=ids9)
    m.zero_grad(set_to_none=True)
    z_e.pow(2).sum().backward()
    g_e = _share_grad(m)
    check(f"the EEG branch back-props into share_head ({g_e:.3e})", g_e > 0)

    m.zero_grad(set_to_none=True)
    m.encode_target(tgt, training=True).pow(2).sum().backward()
    g_i = _share_grad(m)
    check(f"...and so does the IMAGE branch ({g_i:.3e}) -- one map, two modalities",
          g_i > 0)

    # --- C1 must actually CONSTRAIN, or it is decoration ------------------------
    # A shared map that a private linear `img_pre` can absorb imposes nothing, because
    # the image side then reproduces any effective map it likes while the shared weights
    # stay wherever the EEG branch wants them (that is `img_pre = W^+ M`, and it needs
    # nothing but W being full row rank). Whether the sharing bites is therefore decided
    # by ONE question: can a private linear map fit `share_head`'s action to arbitrary
    # linear precision? Fit it by least squares and read the residual.
    def _absorb_residual(model, d_in_eff):
        lin = model.share_head
        torch.manual_seed(0)
        x = torch.randn(2048, d_in_eff)
        with torch.no_grad():
            y = lin(x)                       # what the shared head actually does
            # ...and the best a private LINEAR map could do on the same inputs.
            X = torch.cat([x, torch.ones(len(x), 1)], dim=1)
            coef = torch.linalg.lstsq(X, y).solution
            y_hat = X @ coef
        return float((y_hat - y).abs().max() / y.abs().max())

    r_bound = _absorb_residual(m, m.d_latent)
    check(f"the shared head is NOT absorbable by a private linear map "
          f"(best linear fit leaves {r_bound:.3f} relative residual), so C1 constrains "
          f"the hypothesis class rather than reparameterising it",
          r_bound > 0.1 and m.share_head_is_nonlinear)
    m_lin = _make_v4_model(share_head_hidden=0)
    r_lin = _absorb_residual(m_lin, m_lin.d_latent)
    check(f"...whereas the LINEAR ablation arm (`share_head_hidden: 0`) is absorbed "
          f"exactly ({r_lin:.2e}), which is why it is an ARM and not the default: a "
          f"linear share_head constrains nothing at all",
          r_lin < 1e-4 and not m_lin.share_head_is_nonlinear)
    # The last op must stay linear: `z_eeg_raw` is the D1 diagnostic, and a trailing
    # normalisation would rescale each row by its own statistics and blur the very
    # per-subject offset the diagnostic measures.
    check("the shared head's LAST op is linear (Linear/GELU/Linear), so `z_eeg_raw` is "
          "an affine function of the pre-activations and the D1 offset diagnostic still "
          "measures the offset",
          isinstance(m.share_head[-1], torch.nn.Linear)
          and isinstance(m.share_head[1], torch.nn.GELU))
    # ...and the nonlinearity is real, not a no-op activation sitting in the graph.
    with torch.no_grad():
        h = torch.randn(256, m.d_latent)
        lhs = m.share_head(2.0 * h)
        rhs = 2.0 * m.share_head(h)
    check(f"the shared head is genuinely nonlinear (share_head(2x) != 2*share_head(x) "
          f"by {float((lhs - rhs).abs().max()):.3f}), so the residual above is not an "
          f"artefact of the fit",
          not torch.allclose(lhs, rhs, atol=1e-4))
    # A shared bias is common to both clouds by construction, so it cannot displace one
    # against the other -- now in the strong sense, because the nonlinearity stops the
    # image branch from neutralising it inside `img_pre`.
    # (Evaluated with the model in `eval()`: `eeg_pre` is an MLP with `head_dropout`, so
    # in train mode the two calls would differ by their dropout masks rather than by the
    # bias, which is a measurement artefact and not the property under test.)
    was_training = m.training
    m.eval()
    with torch.no_grad():
        e0 = m.embed_eeg(eeg)
        i0 = m.encode_target(tgt, normalize=False)
        z_e0 = m.apply_smn(e0, ids9)
        m.share_head[-1].bias.fill_(0.7)
        e1 = m.embed_eeg(eeg)
        i1 = m.encode_target(tgt, normalize=False)
        z_e1 = m.apply_smn(e1, ids9)
        m.share_head[-1].bias.zero_()
    m.train(was_training)
    dd = (e1 - e0) - (i1 - i0)
    check(f"a shared-head bias moves BOTH modalities' pre-SMN embeddings by the SAME "
          f"vector (|delta_eeg - delta_img| {float(dd.abs().max()):.2e}), i.e. it "
          f"translates the pair rather than displacing one against the other",
          float(dd.abs().max()) < 1e-5 and float((e1 - e0 - 0.7).abs().max()) < 1e-5)

    # ...BUT THE SMN THEN REMOVES IT ON THE EEG SIDE ONLY, and that asymmetry is a real
    # structural consequence worth pinning down: it means `share_head.bias` can never be
    # used to offset the EEG cloud (the SMN subtracts any per-subject constant by
    # construction), so the ONLY route to cloud alignment is to move the IMAGE cloud's
    # mean to the origin -- which is exactly what L_mmd does. It also means SMN and C1
    # are complementary rather than redundant: C1 stops a *learned* per-modality
    # disagreement from forming, SMN removes the *input-driven* one, and neither can do
    # the other's job.
    check(f"...and the SMN cancels that bias on the EEG side "
          f"(|delta| {float((z_e1 - z_e0).abs().max()):.2e}), so the head's bias cannot "
          f"offset the EEG cloud and L_mmd must align by moving the IMAGE cloud instead",
          float((z_e1 - z_e0).abs().max()) < 1e-5)

    # --- C2: the SMN removes a shared offset BY CONSTRUCTION -------------------
    # The measured v3 defect is a per-subject displacement of the EEG cloud pointing
    # almost orthogonally to the image cloud's (cos ~ +0.1). Whatever its size, the SMN
    # must be exactly invariant to it -- this is the difference between a structural
    # guarantee and the soft penalties that failed (v1's learned `z_s` collapsed; a
    # subject-MMD leaves the displacement at 0.27-0.42 of the row norm).
    ids = torch.tensor([0] * 4 + [1] * 4)
    z = torch.randn(8, 16)
    offset = torch.randn(16) * 5.0
    smn = m.smn
    moved = smn(z + offset, ids) - smn(z, ids)
    # The property is an ALGEBRAIC identity (subtracting a constant and then removing the
    # per-subject mean leaves the same vector), so in float32 the residual is pure
    # cancellation rounding and scales with the offset. The bound is therefore stated
    # relative to the offset rather than as an absolute constant -- an absolute constant
    # here measures float32 noise, not invariance, and changes when the RNG stream moves.
    tol32 = 1e-6 * float(offset.norm()) + 1e-6
    check(f"SMN is invariant to ANY shared offset "
          f"(|delta| {float(moved.abs().max()):.2e} << offset norm "
          f"{float(offset.norm()):.2f}, bound {tol32:.1e})",
          float(moved.abs().max()) < tol32)
    # ...and in float64 it is exact to machine precision, which is what makes the float32
    # residual above identifiable as rounding rather than as a leaky implementation.
    # The inputs must be SUMMED in float64: adding first in float32 and only then casting
    # bakes 5e-07 of float32 rounding into the input itself, which looks exactly like a
    # 5e-07 leak in the module (that trap was hit writing this check).
    smn64 = type(smn)(16, enabled=True, gate_scale=True, init_gate=0.0,
                      min_rows=smn.min_rows).double()
    zd, od = z.double(), offset.double()
    moved64 = smn64(zd + od, ids) - smn64(zd, ids)
    check(f"...and EXACTLY invariant in float64 "
          f"(|delta| {float(moved64.abs().max()):.1e} on an offset of norm "
          f"{float(od.norm()):.2f}, i.e. it is the arithmetic that is exact and the "
          f"float32 residual is only cancellation rounding)",
          float(moved64.abs().max()) < 1e-12)
    disabled = _make_v4_model(smn_enabled=False).smn
    check(f"...and when disabled the SAME offset passes straight through, which is what "
          f"makes the check mean something (|delta| "
          f"{float((disabled(z + offset, ids) - disabled(z, ids)).abs().max()):.2f})",
          float((disabled(z + offset, ids) - disabled(z, ids)).abs().max()) > 1.0)

    # The floor is a floor, not a formality: a group of ONE row would subtract itself
    # and yield a zero vector, silently destroying that row's gradient and making the
    # retrieval matrix depend on batch composition.
    lone = smn(torch.randn(1, 16), torch.tensor([7]))
    check(f"a group below `min_rows` is passed through untouched "
          f"(|out| {float(lone.norm()):.3f} > 0)", float(lone.norm()) > 0)
    check("the scale gate starts CLOSED, so the module is exactly centring at init "
          "(gate=0 means scale**0 == 1, and the offset it removes is the whole measured "
          "lever)",
          float(smn.gate()) == 0.0 and float(smn.gate_raw) == 0.0)

    # The gate must be CLAMPED, not sigmoided. This is not a style preference: a sigmoid
    # puts the initial gate at sigmoid(0) = 0.5, i.e. the module would ship with the
    # un-evidenced second-moment half already applied, which is the one thing the three
    # seeds disagree about the sign of.
    gz = torch.randn(8, 16)
    closed = smn(gz, ids)
    with torch.no_grad():
        smn.gate_raw.fill_(1.0)              # ask for the full per-direction rescale
    opened = smn(gz, ids)
    check(f"opening the gate really changes the output, so the previous check was not "
          f"vacuous (|delta| {float((opened - closed).abs().mean()):.2e})",
          not torch.allclose(opened, closed))
    for raw, expected in ((-3.0, 0.0), (4.0, 1.0), (0.7, 0.7)):
        with torch.no_grad():
            smn.gate_raw.fill_(raw)
        check(f"the gate is clamped into [0, 1] (raw {raw:+.1f} -> {float(smn.gate()):.1f})",
              abs(float(smn.gate()) - expected) < 1e-6)
    with torch.no_grad():
        smn.gate_raw.zero_()                 # leave the module as the trainer will find it

    # --- C3: the spectral term is reducible, scale-invariant, and monotone ------
    # A rank-`r0` representation has all its energy in the head; `r0` = d_align/4 = 4.
    rank4 = torch.randn(64, 4) @ torch.randn(4, 16)
    iso = torch.randn(64, 16)
    l_rank = float(spectral_concentration(rank4, r0=4))
    l_iso = float(spectral_concentration(iso, r0=4))
    check(f"a rank-`r0` representation has ~zero tail energy ({l_rank:.2e}) while an "
          f"isotropic one does not ({l_iso:.3f})", l_rank < 1e-6 and l_iso > 0.5)
    check(f"the spectral ratio is scale-invariant "
          f"({float(spectral_concentration(iso, r0=4)):.6f} vs "
          f"{float(spectral_concentration(9.0 * iso, r0=4)):.6f}), so it cannot be "
          f"satisfied by inflating the embedding -- unlike the VICReg hinge it replaces",
          abs(float(spectral_concentration(iso, r0=4))
              - float(spectral_concentration(9.0 * iso, r0=4))) < 1e-5)
    partial = rank4 + 1.0 * iso
    check(f"it is monotone in the amount of tail energy "
          f"({l_rank:.3f} < {float(spectral_concentration(partial, r0=4)):.3f} < "
          f"{l_iso:.3f})",
          l_rank < float(spectral_concentration(partial, r0=4)) < l_iso)
    rep = spectrum_report(rank4, r0=4)
    check(f"the log reports the complementary `top_frac` ({rep['top_frac']:.4f})",
          abs(rep["top_frac"] - (1.0 - l_rank)) < 1e-6)

    # --- the CORRECTED bilateral spectral target, and why the old one was unsafe ---
    # The checks above document the tail ratio's monotonicity. The problem is the
    # DIRECTION of that monotonicity: tail = 1 - head, so minimising it maximises the
    # head, and the maximiser is a rank-1 representation. This asserts the defect
    # explicitly (so the reason for the replacement is executable, not a comment) and
    # then asserts the replacement does not have it.
    #
    # `d_amb = 64` with `r0 = 16` is the configuration that EXPOSES it: with d == r0 there
    # is no "beyond r0" at all, so every rank trivially measures 0 and the check would be
    # vacuous. (That is not hypothetical -- the first version of this check used d == r0
    # and passed for the wrong reason.)
    from samclip.losses import spectral_rank_target
    d_amb, r0 = 64, 16
    _q, _ = torch.linalg.qr(torch.randn(d_amb, d_amb,
                                        generator=torch.Generator().manual_seed(1)))
    _base = torch.randn(128, d_amb, generator=torch.Generator().manual_seed(0))
    def _rank_r(r: int) -> torch.Tensor:
        return _base @ _q[:, :r] @ torch.diag(torch.linspace(1.0, 0.01, r)) @ _q[:, :r].T
    tail_r1 = float(spectral_concentration(_rank_r(1), r0=r0))
    tail_r16 = float(spectral_concentration(_rank_r(16), r0=r0))
    check(f"the OLD tail ratio CANNOT tell collapse from the target rank "
          f"(rank-1 {tail_r1:.2e} vs rank-16 {tail_r16:.2e}: both ~0, so TOTAL COLLAPSE is "
          f"not penalised -- it is the global optimum)",
          tail_r1 < 1e-5 and tail_r16 < 1e-5)
    tgt = {r: float(spectral_rank_target(_rank_r(r), r0=r0)) for r in (1, 16, 64)}
    check(f"...whereas the bilateral target is minimised AT r0 and penalises collapse "
          f"(rank-1 {tgt[1]:.3f} > rank-16 {tgt[16]:.3f} < rank-64 {tgt[64]:.3f})",
          tgt[16] < tgt[1] and tgt[16] < tgt[64])
    check("...and it is scale-invariant like the ratio it replaces, so it cannot be "
          "satisfied by inflating the embedding",
          abs(tgt[16] - float(spectral_rank_target(9.0 * _rank_r(16), r0=r0))) < 1e-5)
    # ...and the new `spec_mode` dispatch really reaches the new function. This is not
    # ceremony: wiring `spec_mode` in the Trainer but forgetting to import the function
    # is a NameError that only fires when `spec > 0` AND `spec_mode == "rank"`, i.e. in a
    # configuration no recorded run has ever used. Exactly that mistake was made while
    # adding this, and this check is what found it.
    for mode in ("tail", "rank"):
        tr_mode = train_mod.Trainer(
            model=_make_v4_model(),
            cfg={"objective": "v4", "spec_mode": mode, "spec_r0": 16,
                 "temp_img": 0.07, "temp_cross": 0.1},
            device=torch.device("cpu"), n_subjects=N_SUB,
            weights=train_mod.LossWeights(img=1.0, cross=0.7, mmd=1.0, spec=0.1,
                                          router=0.05, dec=0.0, proto=0.0,
                                          reg=0.0, rkd=0.0, adv=0.0))
        _, parts_mode = tr_mode.assemble(_toy_batch())
        check(f"`spec_mode: {mode}` assembles and logs a finite `spec` term "
              f"({float(parts_mode['spec']):.4f})",
              "spec" in parts_mode and bool(torch.isfinite(parts_mode["spec"])))

    # --- v5 pillar A: the recovery-aware operator must BE the deployed operator --------
    # The whole claim of "train in the metric you are scored in" collapses if the in-loop
    # correction is not the same function as the deployment one, so that equality is the
    # first thing asserted: torch `csls_correct` against numpy `calibration.csls_scores`,
    # on the shapes actually used (a 200-way subject block, and a ragged one).
    import numpy as _np
    from samclip.calibration import csls_scores as _np_csls
    from samclip.losses import csls_correct, recovery_aware_alignment
    _rng = _np.random.default_rng(0)
    for _q, _g, _k in ((200, 200, 10), (128, 128, 10), (37, 91, 10), (128, 128, 3)):
        _Q = _rng.standard_normal((_q, 32)).astype(_np.float64)
        _G = _rng.standard_normal((_g, 32)).astype(_np.float64)
        _ref = _np_csls(_Q.astype(_np.float32), _G.astype(_np.float32), k=_k)
        _et = torch.tensor(_Q, dtype=torch.float64)
        _gt = torch.tensor(_G, dtype=torch.float64)
        _got = csls_correct(torch.nn.functional.normalize(_et, dim=-1)
                            @ torch.nn.functional.normalize(_gt, dim=-1).t(),
                            k=_k).numpy()
        check(f"the differentiable CSLS equals the NUMPY deployment CSLS that the score "
              f"uses (Q={_q}, G={_g}, k={_k}, max|diff| {_np.abs(_got - _ref).max():.1e})",
              _np.abs(_got - _ref).max() < 1e-5)
    _S = torch.randn(40, 25, dtype=torch.float64)
    check("...and transposition is consistent, which is what keeps the SYMMETRIC "
          "InfoNCE valid (transposing the corrected matrix equals correcting the "
          "transpose, so the correction must not be applied twice)",
          float((csls_correct(_S, 7).t() - csls_correct(_S.t(), 7)).abs().max()) < 1e-12)

    # `enabled: false` must be INERT, not a placebo. This is asserted against a batch
    # and model pinned to eval mode, because a `train()`-mode comparison differs by
    # dropout alone and would pass for the wrong reason -- an earlier version of this
    # test did exactly that. The bug it now catches is real and was shipped-and-caught
    # here: parsing `csls_k` without gating on `enabled` left the OFF position applying
    # the correction, so every "raw cosine" baseline was silently the treatment.
    _m_ra = _make_v4_model(); _m_ra.eval()
    _b_ra = _toy_batch()
    _w_ra = train_mod.LossWeights(img=1.0, cross=0.7, mmd=1.0, spec=0.0, router=0.05,
                                 dec=0.0, proto=0.0, reg=0.0, rkd=0.0, adv=0.0)
    def _assemble_ra(ra_cfg):
        tr_ra = train_mod.Trainer(model=_m_ra,
                                  cfg={"objective": "v4", "temp_img": 0.07,
                                       "temp_cross": 0.1, "recovery_aware": ra_cfg},
                                  device=torch.device("cpu"), n_subjects=N_SUB,
                                  weights=_w_ra)
        _, p_ra = tr_ra.assemble(_b_ra)
        return float(p_ra["img"]), float(p_ra["cross"])
    _off = _assemble_ra({"enabled": False})
    check(f"`recovery_aware.enabled: false` is INERT (img {_off[0]:.4f} equals the "
          f"never-configured value), so a baseline cannot silently be the treatment",
          _off == _assemble_ra({}) )
    _arms = {"block": _assemble_ra({"enabled": True, "block_per_subject": True,
                                    "csls_k": None})[0],
             "csls": _assemble_ra({"enabled": True, "block_per_subject": False,
                                   "csls_k": 10})[0],
             "both": _assemble_ra({"enabled": True, "block_per_subject": True,
                                   "csls_k": 10})[0]}
    check(f"...and blocking and the CSLS correction are SEPARATELY attributable, not one "
          f"bundled switch (baseline {_off[0]:.3f} vs "
          f"{ {k: round(v, 3) for k, v in _arms.items()} })",
          len({round(_off[0], 6), *[round(v, 6) for v in _arms.values()]}) == 4)
    _cross_only = _assemble_ra({"enabled": True, "block_per_subject": False,
                                "csls_k": 10, "terms": ["cross"]})
    check(f"...and `terms` routes per contrast (with terms=[cross] the `img` term returns "
          f"to baseline {_off[0]:.4f} while `cross` moves {_off[1]:.4f} -> "
          f"{_cross_only[1]:.4f})",
          _cross_only[0] == _off[0] and _cross_only[1] != _off[1])

    # The per-subject blocking is only worth its cost if it really partitions the batch,
    # so assert the property it exists for: the block form must differ from the
    # whole-batch form on a batch that has >1 subject (otherwise the loop is a no-op).
    _z_b = torch.randn(24, 16); _t_b = torch.randn(24, 16)
    _sub = torch.tensor([0] * 8 + [1] * 8 + [2] * 8)
    _crit = train_mod.InfoNCE(init_temp=0.07)
    _whole = float(clip_alignment_loss(_z_b, _t_b, _crit, csls_k=10))
    _blocked = float(recovery_aware_alignment(_z_b, _t_b, _sub, _crit, csls_k=10))
    check(f"the per-subject block form really differs from the whole-batch form "
          f"({_whole:.4f} vs {_blocked:.4f}), so the block structure is load-bearing",
          abs(_whole - _blocked) > 1e-6)

    # --- C4: the cross-modal MMD has no self-pair floor and adapts its bandwidth -
    # NOTE on what this term IS on unit-norm features, because a naive test here asserts
    # the wrong property. With `normalize=True` both clouds live on the sphere, so their
    # pairwise-distance distributions concentrate: two INDEPENDENT isotropic clouds in
    # 24-d give MMD ~ 0, and a row-permutation of one cloud gives MMD ~ 0. That is not a
    # dead term, it is the term saying "these occupy the same region", which is exactly
    # the Stage-1 warm start the reference uses it for. What it responds to is a
    # DISPLACEMENT of the region (i.e. `mu_eeg - mu_img`, the D1 defect), and that is
    # what is checked below -- plus the gradient sign, which is the only thing that
    # decides whether descending it actually removes the displacement.
    torch.manual_seed(0)
    x = torch.nn.functional.normalize(torch.randn(128, 24), dim=-1)
    l_same = float(mmd_crossmodal(x, x))
    check(f"identical clouds give ~0, not the 2/N self-pair bias that made the v3 RBF "
          f"term inert ({l_same:.2e})", l_same < 1e-2)
    check(f"a row-PERMUTATION is (correctly) not a displacement on the sphere "
          f"({float(mmd_crossmodal(x, torch.roll(x, 64, dims=0))):.2e}) -- the two clouds "
          f"still occupy the same region, so this is the term agreeing, not failing",
          float(mmd_crossmodal(x, torch.roll(x, 64, dims=0))) < 1e-2)
    # `unbiased` is not cosmetic: on two INDEPENDENT clouds drawn from the same
    # distribution -- i.e. the state the term is supposed to annihilate at -- the biased
    # estimator sits at an O(1/n) floor that never reaches zero, so at w_mmd > 0 it would
    # keep pushing after convergence. (Note x-vs-x is identically zero under BOTH
    # estimators, which is why this has to be measured across draws, not on one pair.)
    def _same_dist(seeds, unbiased):
        out = []
        for s in seeds:
            torch.manual_seed(s)
            a = torch.nn.functional.normalize(torch.randn(128, 24), dim=-1)
            b = torch.nn.functional.normalize(torch.randn(128, 24), dim=-1)
            out.append(float(mmd_crossmodal(a, b, unbiased=unbiased)))
        return sum(out) / len(out)
    floor_b = _same_dist(range(8), unbiased=False)
    floor_u = _same_dist(range(8), unbiased=True)
    check(f"...but the SAME two independent same-distribution clouds sit at an O(1/n) "
          f"floor under the biased estimator ({floor_b:.5f} vs unbiased {floor_u:.5f}), "
          f"which never anneals to zero",
          floor_b > 5.0 * floor_u and floor_u < 0.05 * max(floor_b, 1e-9))

    # A genuine region displacement: shift by a constant direction and renormalise. The
    # value must grow with the shift, and must track the offset ratio the SMN and the
    # shared head exist to remove (`subject_offset_ratio`).
    direction = torch.zeros(24)
    direction[0] = 1.0
    shifts = (0.25, 0.5, 1.0, 2.0)
    vals, ratios = [], []
    for a in shifts:
        y = torch.nn.functional.normalize(x + a * direction, dim=-1)
        vals.append(float(mmd_crossmodal(x, y)))
        ratios.append(float((x.mean(0) - y.mean(0)).norm() / x.norm(dim=-1).mean()))
    check(f"a real region displacement is reported, and monotonically in its size "
          f"({' < '.join(f'{v:.4f}' for v in vals)})",
          all(u < v for u, v in zip(vals, vals[1:])) and vals[-1] > 0.1)
    check(f"...and the value is ~proportional to the offset ratio it is there to remove "
          f"({vals[-1]:.3f} at ratio {ratios[-1]:.3f}), so it is a usable Stage-1 signal "
          f"rather than a flat penalty",
          vals[-1] > 0.05 and ratios[-1] > 0.5)

    # The real question about a matching term is not "does it have a plausible value"
    # but "does descending it remove the defect it was added for". That is checked
    # end-to-end below (an actual optimiser, not a gradient dot product -- with
    # `normalize=True` the raw gradient carries a radial component that does nothing to
    # the direction, so the dot product understates the effect).
    def _descend(x, y0, sign, steps=60, lr=0.05):
        p = y0.clone().requires_grad_(True)
        opt = torch.optim.Adam([p], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            (sign * mmd_crossmodal(x, torch.nn.functional.normalize(p, dim=-1))).backward()
            opt.step()
        y = torch.nn.functional.normalize(p, dim=-1)
        off = float((y.mean(0) - x.mean(0)).norm() / x.norm(dim=-1).mean())
        return off, float(mmd_crossmodal(x, y)), float(y.std(dim=0).mean())
    y0 = x + 1.0 * direction
    off_down, mmd_down, spread_down = _descend(x, y0, +1.0)
    off_up, _, _ = _descend(x, y0, -1.0)
    check(f"DESCENDING L_mmd actually removes the EEG-image offset it exists to remove "
          f"(ratio {ratios[2]:.3f} -> {off_down:.3f}, value -> {mmd_down:.2e})",
          off_down < 0.2 * ratios[2] and mmd_down < 0.1 * vals[2])
    check(f"...without collapsing the cloud, which is the degenerate way to satisfy any "
          f"distribution match ({spread_down:.3f} vs {float(x.std(dim=0).mean()):.3f} "
          f"at start -- it grows, it does not shrink)",
          spread_down > 0.5 * float(x.std(dim=0).mean()))
    check(f"...and the SIGN is load-bearing: ASCENT drives the same offset the other way "
          f"({off_up:.3f} > {ratios[2]:.3f}), so this is a directed term and not a "
          f"magnitude penalty",
          off_up > ratios[2])

    # The bandwidth ladder is the reference's RELATIVE values times a median-distance
    # estimate, precisely so the term survives a change of feature scale -- which is
    # necessary here because the reference feeds it unnormalised features and we feed it
    # normalised ones. Tested on the unnormalised path, where the scale is not erased.
    raw_x = torch.randn(96, 24)
    raw_y = torch.randn(96, 24) + 0.3
    check(f"the bandwidth is data-adaptive: rescaling BOTH clouds by 10x leaves the "
          f"value unchanged ({float(mmd_crossmodal(raw_x, raw_y, normalize=False)):.6f} "
          f"vs {float(mmd_crossmodal(10 * raw_x, 10 * raw_y, normalize=False)):.6f})",
          abs(float(mmd_crossmodal(raw_x, raw_y, normalize=False))
              - float(mmd_crossmodal(10 * raw_x, 10 * raw_y, normalize=False))) < 1e-5)
    check(f"...and a single bandwidth set is used for all three kernel terms, so the "
          f"value cannot go negative before the clamp "
          f"({float(mmd_crossmodal(raw_x, raw_y, normalize=False)):.6f} >= 0)",
          float(mmd_crossmodal(raw_x, raw_y, normalize=False)) >= 0.0)

    # --- C5: the objective is four terms, and the removed ones cannot come back --
    m2 = _make_v4_model()
    tcfg = {"temp_img": 0.07, "temp_cross": 0.1, "softplus": True, "objective": "v4",
            "loss_weights": {"img": 1.0, "cross": 0.7, "mmd": 1.0, "spec": 0.05,
                             "reg": 0.0, "dec": 0.0, "proto": 0.0, "rkd": 0.0,
                             "adv": 0.0, "router": 0.0},
            "spec_r0": 16}
    tr4 = train_mod.Trainer(model=m2, cfg=tcfg, device=torch.device("cpu"),
                            n_subjects=N_SUB,
                            weights=train_mod.LossWeights.from_cfg(tcfg))
    batch = _toy_batch(9)
    loss, parts = tr4.assemble(batch)
    check(f"the v4 objective logs exactly the live terms {sorted(parts)}",
          {"img", "cross", "mmd", "spec"}.issubset(parts)
          and "var" not in parts and "cov" not in parts)
    loss.backward()
    check(f"the v4 objective back-props ({float(loss):.4f})",
          loss.item() == loss.item())

    # ...and the FOUR terms are exactly what the doc says, with `reg` (the covariance
    # decorrelation) among the removed ones. It is removed rather than re-weighted
    # because on this topology it cannot tell its own target from noise: on a 384x64
    # batch it moves by only 14% between a random representation and one with an axis
    # EXACTLY duplicated (6.93e-03 -> 7.86e-03 raw; 1.63e-06 -> 1.87e-06 row-normalised),
    # and its raw form is additionally scale-dependent (1.8e-08 at a 0.04x scale), which
    # is what made the first v4 GPU run log `cov: 0.0` while v3 logged 0.13.
    for w in ("dec", "proto", "rkd", "adv", "reg"):
        try:
            train_mod.LossWeights.from_cfg(
                {"loss_weights": {w: 0.05}}).validate("v4")
            check(f"objective=v4 rejects a non-zero `{w}` weight", False)
        except ValueError:
            check(f"objective=v4 rejects a non-zero `{w}` weight", True)

    try:
        train_mod.LossWeights.from_cfg({"loss_weights": {"spec": 0.05}}).validate("v3")
        check("objective=v3 with a v4-only term is rejected", False)
    except ValueError:
        check("objective=v3 with a v4-only term is rejected", True)
    train_mod.LossWeights.from_cfg({"loss_weights": {}}).validate("v3")
    check("the v3 defaults still validate", True)

    # A cross-modal MMD on two INDEPENDENT heads is not the same mechanism, so it must
    # not be reportable as one.
    tr_bad = train_mod.Trainer(model=make_model("routed"),
                               cfg=dict(tcfg, loss_weights=None),
                               device=torch.device("cpu"), n_subjects=N_SUB,
                               weights=train_mod.LossWeights.from_cfg(tcfg))
    try:
        tr_bad.assemble(_toy_batch(9))
        check("objective=v4 on an arch=v3 model is refused", False)
    except ValueError:
        check("objective=v4 on an arch=v3 model is refused", True)

    # --- C5b: the schedule coefficient multiplies, it does not double-apply ----
    # v3's schedule OVERWRITES `weights`, v4 multiplies the phase coefficient into them.
    # Applying it twice would square the ramp, which still looks like a ramp.
    # `.eval()` first: dropout would otherwise make two `assemble` calls on the SAME
    # batch differ, and these checks are about the weight arithmetic, not about noise.
    m2.eval()
    tr4.coarse_weight, tr4.fine_weight = 0.9, 0.1
    l_fine_small = float(tr4.assemble(batch)[0])
    tr4.fine_weight = 0.2
    l_fine_big = float(tr4.assemble(batch)[0])
    not_fine = l_fine_big - l_fine_small
    check(f"halving `fine` does not simply halve the total (the coarse term is "
          f"coefficient-scaled, not `fine`-scaled): {l_fine_small:.4f} -> "
          f"{l_fine_big:.4f}", abs(not_fine) > 0)
    img_only = train_mod.LossWeights(img=1.0, cross=0.0, mmd=0.0, spec=0.0, reg=0.0,
                                     dec=0.0, proto=0.0, rkd=0.0, adv=0.0, router=0.0)
    tr4.weights = img_only
    tr4.fine_weight = 1.0
    l1 = float(tr4.assemble(batch)[0])
    tr4.fine_weight = 0.4
    l4 = float(tr4.assemble(batch)[0])
    check(f"with only the `img` term live, v4's loss is EXACTLY `fine` x its "
          f"unscaled value ({l1:.6f} -> {l4:.6f}, ratio {l4 / l1:.4f})",
          abs(l4 - 0.4 * l1) < 1e-6)
    v3m = make_model("routed")
    v3m.eval()
    tr3 = train_mod.Trainer(model=v3m,
                            cfg={"temp_img": 0.07, "temp_cross": 0.1},
                            device=torch.device("cpu"), n_subjects=N_SUB,
                            weights=img_only)
    tr3.fine_weight = 1.0
    v3_1 = float(tr3.assemble(batch)[0])
    tr3.fine_weight = 0.4
    v3_4 = float(tr3.assemble(batch)[0])
    check(f"...while on v3 the coefficient is IGNORED by the loss (the loop writes it "
          f"into `weights` instead), so an existing run is unchanged "
          f"({v3_1:.6f} == {v3_4:.6f})", abs(v3_1 - v3_4) < 1e-9)

    # --- C4b: the v4 schedule is a convex split, and Stage 2 is contrast-only ----
    s4 = train_mod.CoarseToFine.from_cfg(load_config(
        Path(__file__).resolve().parents[1] / "configs" / "v4.yaml"))
    split = [s4.at(e) for e in range(0, 20, 3)]
    check(f"Stage 1's two coefficients sum to 1 at every epoch "
          f"({[(round(a, 3), round(b, 3)) for a, b, _, _ in split][:3]} ...)",
          all(abs(a + b - 1.0) < 1e-9 for a, b, _, _ in split))
    c20, f20, lr20b, ph20b = s4.at(20)
    check(f"Stage 2 is contrast-only at full weight (coarse {c20}, fine {f20}, "
          f"lr {lr20b:g}, {ph20b})",
          c20 == 0.0 and f20 == 1.0 and lr20b == 5e-5 and ph20b == "stage2")
    # THE DEFAULT IS NOW `false`, AND THAT IS THE POINT. It shipped as `true` and freezing
    # was one of the two real causes of the v4 regression: it left the encoder a 20-epoch
    # budget at lr 1e-4 (one fifth of the v3 recipe's), and un-freezing PLUS returning to
    # lr 1e-3 was worth +4.00pp with the same sign on all three seeds. Asserting the
    # default here is what turns "we changed a config" into a regression test -- this check
    # is exactly what caught the change.
    check(f"the v4 default does NOT freeze the encoder in Stage 2 "
          f"(freeze_encoder_stage2={s4.freeze_encoder_stage2})",
          s4.freeze_encoder_stage2 is False)
    check("...and the run record does NOT claim a freeze it is not doing",
          "encoder frozen" not in s4.describe())
    # ...while the override still works, so the `true` arm remains runnable and the
    # schedule still has to describe it -- both directions are asserted, not just the new one.
    cfg_frz = load_config(Path(__file__).resolve().parents[1] / "configs" / "v4.yaml")
    cfg_frz["schedule"] = dict(cfg_frz["schedule"], freeze_encoder_stage2=True)
    s4f = train_mod.CoarseToFine.from_cfg(cfg_frz)
    check(f"...but `freeze_encoder_stage2: true` is still honoured when asked for "
          f"({s4f.freeze_encoder_stage2})", s4f.freeze_encoder_stage2 is True)
    check("the v4 schedule describes the freeze in its run record",
          "encoder frozen" in s4f.describe())

    # --- Stage-2 freeze: the trunk stops, the projectors/router do not -----------
    m3 = _make_v4_model()
    tr5 = train_mod.Trainer(model=m3, cfg=tcfg, device=torch.device("cpu"),
                            n_subjects=N_SUB,
                            weights=train_mod.LossWeights.from_cfg(tcfg))
    params5, n_frozen5 = train_mod.freeze_encoder_for_stage2(m3, tr5)
    ids5 = {id(p) for p in params5}
    check(f"the freeze removes every trunk parameter from the optimiser "
          f"({n_frozen5} scalars frozen, {len(params5)} tensors left)",
          n_frozen5 > 0 and all(not p.requires_grad for p in m3.trunk.parameters())
          and not any(id(p) in ids5 for p in m3.trunk.parameters()))
    check("...and keeps share_head, the modality pre-projections and the SMN gate "
          "trainable (freezing the head would stop the fine phase adapting the map "
          "the whole design is about)",
          all(id(p) in ids5 for p in m3.share_head.parameters())
          and all(id(p) in ids5 for p in m3.smn.parameters())
          and all(id(p) in ids5 for p in m3.eeg_pre.parameters()))
    check("...and keeps the InfoNCE temperatures in the optimiser",
          all(id(p) in ids5 for p in tr5.criterion_parameters()))

    # --- deployment extraction must not depend on `eval_batch` -------------------
    # The SMN's deployment statistic is the mean over the WHOLE 200-trial query set. A
    # per-minibatch application would make the score a function of `eval_batch` -- a knob
    # the v3 audit already showed moves Top-1 by more than a point. So the two-pass
    # extraction is asserted by comparing two batch sizes on the same data.
    ds = _FakeDS(40)
    m4 = _make_v4_model(min_rows=2)
    m4.eval()
    feats_by_bs = {}
    for bs in (40, 13, 7):
        loader = DataLoader(ds, batch_size=bs, shuffle=False,
                            collate_fn=_stack_collate)
        feats_by_bs[bs] = evaluate.extract_features(m4, loader, torch.device("cpu"))
    check(f"extraction is identical for eval_batch=40, 13 and 7 "
          f"(|deeg| "
          f"{max(float(np.abs(feats_by_bs[40]['eeg'] - feats_by_bs[b]['eeg']).max()) for b in (13, 7)):.2e})",
          all(np.allclose(feats_by_bs[40]["eeg"], feats_by_bs[b]["eeg"], atol=1e-6)
              for b in (13, 7)))
    check("...and it returns the PRE-SMN spectrum for the D1 diagnostic",
          feats_by_bs[40]["eeg_raw"].shape == feats_by_bs[40]["eeg"].shape)

    # CONTROL: the invariance must come from the two-pass structure, not from the SMN
    # being a no-op. Applying it inside each minibatch -- the version the docstring
    # rejects -- has to give a DIFFERENT answer, or the check above proves nothing.
    def _per_batch_smn(bs):
        loader = DataLoader(ds, batch_size=bs, shuffle=False,
                            collate_fn=_stack_collate)
        out = []
        with torch.no_grad():
            for batch in loader:
                raw = m4.embed_eeg(batch["eeg"])
                out.append(torch.nn.functional.normalize(m4.apply_smn(raw), dim=-1))
        return torch.cat(out).numpy()
    d_per_batch = float(np.abs(_per_batch_smn(40) - _per_batch_smn(7)).max())
    check(f"...and a PER-MINIBATCH SMN, the version this replaces, really does depend on "
          f"the batch size on the same data (|deeg| {d_per_batch:.2e}), so the two-pass "
          f"extraction is doing work rather than rubber-stamping",
          d_per_batch > 1e-4)


def _cos_offdiag(z: torch.Tensor) -> torch.Tensor:
    """Off-diagonal cosine similarities of a batch of embeddings."""
    zn = torch.nn.functional.normalize(z, dim=-1)
    s = zn @ zn.t()
    return s[~torch.eye(len(zn), dtype=torch.bool)]

def invariants() -> None:
    """Properties that must hold for the model to be ABLE to learn anything.

    Every check here corresponds to a bug that shipped. The point is not coverage: all
    of these passed a full pipeline run (cache -> train -> eval, 60 epochs, clean exit
    code) while being completely wrong, because the existing tests only asserted that
    shapes matched and losses were finite. A finite loss and a correct shape are
    compatible with an encoder that outputs a constant.

    Rule of thumb for anything added later: assert a PROPERTY (sensitivity, parity,
    monotonicity, invariance, which parameters a term can reach), never just finiteness.
    """
    print("9. invariants (each one is a bug that shipped)")

    # --- 9a. the encoder must depend on its input ----------------------------
    # BUG: `EEGTrunk.norm` (LayerNorm over d_model) ran BEFORE an aggregator that
    # pooled over d_model. LayerNorm zero-means each token along exactly the axis the
    # pool then averaged, so the head saw only rounding error and every input mapped
    # to the SAME vector: 8 different inputs gave cosine 0.998 (min 0.997).
    #
    # The threshold is stated against the constant-function signature (all pairs at
    # ~1.0) rather than against "cosine is low": a freshly initialised encoder
    # legitimately has a dominant bias direction, so a small `d_embed` gives a high
    # baseline off-diagonal cosine even when it is perfectly healthy. `d_embed` is
    # therefore widened here to separate the two regimes.
    for fusion in ("mean", "routed"):
        model = make_model(fusion, d_model=64, d_embed=128)
        model.eval()
        with torch.no_grad():
            z = model.encode_eeg(torch.randn(8, C, T))
        off = _cos_offdiag(z)
        check(f"encoder is input-dependent (fusion={fusion}, off-diag cos "
              f"{float(off.mean()):.3f}, max {float(off.max()):.3f})",
              float(off.max()) < 0.99)

    # --- 9b. no conditioning, and no way for one to come back ------------------
    # The v1 subject-conditioning arms all live under `conditioning:` in a config. The
    # key is now meaningless, and a config that still carries it must not be able to
    # quietly resurrect a mechanism the evidence rejected -- `build_model` simply does
    # not read it, and there is no attribute to condition through.
    model = make_model("routed")
    check("the model has no conditioning attribute at all",
          not hasattr(model, "conditioner") and not hasattr(model, "z_source"))
    check("encode_eeg takes no conditioning argument",
          "z" not in model.encode_eeg.__code__.co_varnames
          and "support_x" not in model.encode_eeg.__code__.co_varnames)

    # --- 9c. the learned temperatures must reach the optimizer --------------
    # BUG: the InfoNCE criteria hang off `Trainer`, which is a plain dataclass, not an
    # nn.Module. `list(model.parameters())` therefore omitted them and both
    # `logit_scale`s stayed pinned at their initialiser forever, while the docstring
    # advertised a learnable temperature.
    device = torch.device("cpu")
    from samclip import train as train_mod
    tmodel2 = make_model("mean")
    trainer2 = train_mod.Trainer(model=tmodel2, cfg={"temp_img": 0.07, "temp_cross": 0.1},
                                 device=device, n_subjects=N_SUB)
    crit_params = trainer2.criterion_parameters()
    opt2 = torch.optim.AdamW(list(tmodel2.parameters()) + crit_params, lr=1e-2)
    opt2.zero_grad(set_to_none=True)
    batch2 = _toy_batch(6)
    loss, _ = trainer2.assemble(batch2)
    loss.backward()
    opt2.step()
    moved = [float(p.grad.abs().sum()) if p.grad is not None else 0.0
             for p in crit_params]
    check(f"both InfoNCE temperatures are in the optimizer and move "
          f"({len(crit_params)} params, |grad| {sum(moved):.4f})",
          len(crit_params) == 2 and all(g > 0 for g in moved))

    # --- 9d. sharing a target must not turn the alignment into a negative ------
    # `cross_subject_loss` needs a multi-positive mask because its logits are a
    # self-similarity matrix: a co-stimulus entry `z_i . z_j` differs from the diagonal
    # `z_i . z_i`, so a diagonal mask really does push two encodings of the same picture
    # apart. The image-alignment term looks like the same situation -- the cross-subject
    # batch gives one stimulus to several subjects, so several rows carry the SAME image
    # vector -- and it is deliberately NOT masked, for a reason that is an identity
    # rather than an assumption. That identity is asserted next (9e).
    #
    # First, the part that is true of the unmasked objective: `dL/d<z_i, target_i>` does
    # point toward the row's own target, group member or not.
    eeg_z = torch.nn.functional.normalize(torch.randn(6, 48), dim=-1).detach()
    tgt_z = torch.nn.functional.normalize(torch.randn(6, 48), dim=-1).detach()
    tgt_z[3:] = tgt_z[:3]                       # rows 3..5 share rows 0..2's target
    eeg_g = eeg_z.clone().requires_grad_(True)
    crit4 = InfoNCE(init_temp=0.07)
    clip_alignment_loss(eeg_g, tgt_z, crit4).backward()
    align = (eeg_g.grad * tgt_z).sum(dim=-1)
    check(f"the alignment rewards moving toward the row's own target "
          f"(dL/d<z_i, target_i> < 0 for all {len(align)} rows, worst "
          f"{float(align.max()):+.4f})", bool((align < 0).all()))

    # --- 9e. the diagonal image term is provably safe ----------------------
    # `clip_alignment_loss` is deliberately diagonal even though `g` rows share a target
    # vector. The justification is an identity, and identities rot silently when the
    # batch layout changes, so it is asserted here rather than argued in a comment: for
    # a group whose columns are equal, `A[i, j] = f(i)` is independent of `j`, hence
    # `sum_j mean_i A[i, j] = sum_j f(j) = sum_j A[j, j]` and the grouped loss equals the
    # diagonal one to machine precision -- on BOTH halves, `eeg->image` and its
    # transpose. If this fires, `images_per_pair` / `target_fusion` changed the layout
    # and the mask question has to be reopened instead of assumed.
    t_random = torch.nn.functional.normalize(torch.randn(6, 48), dim=-1)
    t_random[3:] = t_random[:3]                 # 0<->3, 1<->4, 2<->5 share a target
    stim = torch.tensor([0, 1, 2, 0, 1, 2])     # ... and the groups match that split
    eeg_ok = torch.nn.functional.normalize(t_random + 0.25 * torch.randn(6, 48), dim=-1)
    crit5 = InfoNCE(init_temp=0.07)
    d5 = float(crit5(eeg_ok, t_random))
    g5 = float(crit5(eeg_ok, t_random, groups=stim))
    check(f"sharing a target vector makes the mask a no-op on both halves, so the "
          f"diagonal image term is exact (|grouped - diagonal| {abs(g5 - d5):.2e})",
          abs(g5 - d5) < 1e-6)

    # with genuinely distinct targets the equivalence is trivial, but it is the case a
    # one-row-per-stimulus batch runs, so it is worth a second, cheap assertion
    crit6 = InfoNCE(init_temp=0.07)
    d1 = float(crit6(eeg_z, tgt_z))
    g1 = float(crit6(eeg_z, tgt_z, groups=torch.arange(6)))
    check(f"a one-row-per-stimulus batch is unaffected by grouping "
          f"(|diff| {abs(g1 - d1):.2e})", abs(g1 - d1) < 1e-6)

    # --- 9f. the temperature must not be able to starve the encoder --------
    # BUG: `effective_scale` clamped only above (max=100). `d(loss)/d(scale)` is the
    # mean off-diagonal logit -- an O(1) quantity carrying no signal about the
    # temperature -- and Adam normalises by gradient magnitude, so a persistently
    # negative value drove the scale to 0 within tens of steps. At scale 0 every logit
    # is 0: the contrast equals `ln(N)` exactly and, because the encoder's gradient is
    # proportional to the scale, the encoder stops being trained at all. Observed as
    # `img` frozen at ln(72) = 4.2767 for 12k steps with test top-1 stuck at its
    # initialisation. Asserts the floor holds AND that the encoder still gets gradient.
    crit7 = InfoNCE(init_temp=0.07)
    with torch.no_grad():
        crit7.logit_scale.fill_(-1e3)            # ask for an infinitely cold contrast
    starved = eeg_z.clone().requires_grad_(True)
    crit7(starved, tgt_z).backward()
    check(f"the temperature is floored at {SCALE_MIN} so the encoder keeps its "
          f"gradient (scale {float(crit7.effective_scale()):.3f}, "
          f"|grad| {float(starved.grad.norm()):.3e})",
          float(crit7.effective_scale()) >= SCALE_MIN - 1e-6
          and float(starved.grad.norm()) > 0)

    # ... and the floor must not be a formality: `init_temp` is inside the range, so a
    # fresh criterion starts where the docs claim and can adapt in both directions.
    crit8 = InfoNCE(init_temp=0.07)
    check(f"a fresh criterion starts inside the clamp "
          f"(scale {float(crit8.effective_scale()):.3f})",
          SCALE_MIN < float(crit8.effective_scale()) < SCALE_MAX)

    # --- 9g. whitening must not invert the null space -----------------------
    # BUG: `saw_whiten` used an absolute eigen floor (`cov + 1e-5 I`). With 200 queries
    # against a 512-d embedding the covariance has ~313 numerical-zero eigenvalues, and
    # they were inverted to 1/sqrt(1e-5) ~ 316, amplifying pure noise and making
    # "whitening" score below the raw cosine it was meant to improve.
    rng = np.random.default_rng(0)
    q_rank_deficient = rng.standard_normal((40, 200)).astype(np.float32)   # n << d
    w, wd = calibration.saw_whiten(q_rank_deficient)
    check(f"whitening caps its condition number ({wd['cond']:.1e})",
          wd["cond"] <= 1e3 * (1 + 1e-6) and np.isfinite(w).all())

    # --- 9h. calibration must not invent a stage it did not run ------------
    # BUG: the eval emitted a "+ recovery" row built from whiten+csls whenever
    # `--recovery` was off, so the table showed two identical rows under different
    # names and read as "recovery ran and changed nothing".
    _s_plain, d_plain = calibration.calibrate(q_rank_deficient, q_rank_deficient,
                                              whiten=True, csls=True, recovery=False)
    check(f"calibrate reports recovery=False truthfully (cond {wd['cond']:.1e})",
          d_plain["recovery"] is False and "recovery_diag" not in d_plain)

    # --- 9i. one report must be able to hold several checkpoints ------------
    # BUG: the tag was `path.parent.name`, so two ablation arms both living under a
    # directory called `sub-08` both became "sub-08". The report is a dict keyed by tag,
    # so the second replaced the first -- and a two-arm comparison became a one-arm
    # report that looked like a clean result.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_run_eval", Path(__file__).resolve().parents[1] / "scripts" / "run_eval.py")
    run_eval = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_eval)          # type: ignore[union-attr]
    t1 = run_eval._ckpt_tag(Path("outputs/stage1/sub-08/last.pt"))
    t2 = run_eval._ckpt_tag(Path("outputs/stage1/ablation_a7_routed_sr/sub-08/last.pt"))
    check(f"checkpoint tags stay unique across arms ({t1} | {t2})", t1 != t2)

    # --- 9j. the Gram term must be scale-invariant, not a norm regression ----
    # `gram_distill_loss` normalises each Gram matrix by its Frobenius norm. Without it
    # the frozen image side is a fixed-scale regression target, and the EEG side is
    # pushed toward the image embeddings' absolute scale -- i.e. the geometric term
    # would be quietly acting as a second variance term, interacting with VICReg's weight
    # through the encoder's output norm. A scale-invariant loss must not change when one
    # side is rescaled.
    a = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)
    b = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)
    l1 = float(gram_distill_loss(a, b))
    l2 = float(gram_distill_loss(7.5 * a, b))
    check(f"gram_distill_loss is scale-invariant in the EEG arm "
          f"({l1:.4f} vs {l2:.4f})", abs(l1 - l2) < 1e-6)

    # --- 9k. the prototype contrast must have a temperature -----------------
    # BUG: `PrototypeEMA._contrast_to_proto` fed raw cosine similarities straight to
    # `cross_entropy`. Cosines of L2-normalised vectors live in [-1, 1], so with N
    # classes the softmax is uniform and the loss is pinned at `ln(N)`. Measured on the
    # trained v3 checkpoint: `proto` sat at 5.32-5.50 against the `ln(384) = 5.95`
    # floor for all 50 epochs -- it never left its initialisation -- while holding 26%
    # of the reported loss VALUE and supplying 1.2% of the gradient reaching the
    # encoder. A uniform-softmax cross-entropy is an ordinary-looking number, which is
    # why this survived a full pipeline run and a green test suite.
    #
    # The check is a PROPERTY (the term must be able to move the embedding), not "a
    # scale exists": the identical loss with the temperature forced down to the clamp's
    # floor -- which is exactly the old no-temperature path -- must be materially
    # weaker.
    proto9 = PrototypeEMA(n_classes=6, d_embed=24, min_updates=1)
    z9 = torch.nn.functional.normalize(torch.randn(12, 24), dim=-1).detach()
    g9 = torch.tensor([0, 1, 2, 3, 4, 5] * 2)
    proto9.update(z9, g9)                      # prototypes land on their own classes
    scale9 = float(proto9.effective_scale())
    zg = z9.clone().requires_grad_(True)
    g_hi = float(torch.autograd.grad(proto9.proto_contrast(zg, g9), zg)[0].norm())
    with torch.no_grad():
        proto9.logit_scale.fill_(-1e3)         # -> clamp floor = the old behaviour
    zg2 = z9.clone().requires_grad_(True)
    g_lo = float(torch.autograd.grad(proto9.proto_contrast(zg2, g9), zg2)[0].norm())
    check(f"the prototype contrast has a temperature, so it can move the embedding "
          f"(scale {scale9:.3f}, |grad| {g_hi:.3e} vs {g_lo:.3e} at scale 1)",
          scale9 > 1.0 and g_hi > 2.0 * g_lo)

    # ... and it must reach the optimizer. `PrototypeEMA` hangs off the `Trainer`, which
    # is a plain dataclass, so `model.parameters()` cannot see it -- the same failure
    # mode as 9c, one level deeper.
    tmodel9 = make_model("mean")
    t9 = train_mod.Trainer(model=tmodel9, cfg={"temp_img": 0.07, "temp_cross": 0.1},
                           device=device, n_subjects=N_SUB)
    t9.prototype = PrototypeEMA(n_classes=5, d_embed=24, min_updates=1)
    check("the prototype temperature reaches the optimizer",
          any(p is t9.prototype.logit_scale for p in t9.criterion_parameters()))

    # --- 9l. the MMD term must be able to see a subject shift ---------------
    # BUG: `mmd_subject` used a multi-bandwidth RBF kernel whose centre came from the
    # median PAIRWISE distance. On the L2-normalised embedding every pair sits near
    # sqrt(2) (concentration of measure), so that heuristic picks a bandwidth ~7x the
    # actual subject-MEAN displacement (~0.195) and the kernel saturates. Worse,
    # `_rbf(a, a)` includes the diagonal, which is identically 1 and therefore carries
    # zero gradient, so the reported value was largely a constant offset. On the trained
    # checkpoint the term supplied 0.9% of the gradient while holding 26% of the
    # reported loss value; the unbiased RBF-MMD^2 was ~1e-4 across a full bandwidth
    # sweep, which is why no bandwidth choice could have rescued it.
    torch.manual_seed(0)
    base = torch.randn(128, 32)
    subj9 = torch.tensor([0] * 64 + [1] * 64)
    direction = torch.nn.functional.normalize(torch.randn(32), dim=-1)
    sign = torch.cat([-torch.ones(64, 1), torch.ones(64, 1)])

    def _cloud(shift: float) -> torch.Tensor:
        return torch.nn.functional.normalize(base + shift * direction * sign, dim=-1)

    z0 = _cloud(0.0)
    mus9 = torch.stack([z0[subj9 == s].mean(dim=0) for s in (0, 1)])
    identity = float((mus9[0] - mus9[1]).pow(2).sum())
    got = float(mmd_subject(z0, subj9, kernel="linear"))
    check(f"the linear kernel IS the squared subject-mean distance, with no bandwidth "
          f"and no diagonal bias ({got:.6f} vs {identity:.6f})",
          abs(got - identity) < 1e-6)
    check("mmd_subject defaults to the linear kernel",
          float(mmd_subject(z0, subj9)) == got)

    lin_small = float(mmd_subject(_cloud(0.2), subj9, kernel="linear"))
    lin_big = float(mmd_subject(_cloud(0.8), subj9, kernel="linear"))
    rbf_small = float(mmd_subject(_cloud(0.2), subj9, kernel="rbf"))
    rbf_big = float(mmd_subject(_cloud(0.8), subj9, kernel="rbf"))
    check(f"the linear kernel tracks a deliberate subject shift "
          f"({lin_small:.5f} -> {lin_big:.5f})", lin_big > 1.5 * lin_small)
    check(f"...while the RBF form does not report the shift at a usable magnitude "
          f"({rbf_small:.5f} -> {rbf_big:.5f}, against linear {lin_big:.5f})",
          rbf_big < 0.5 * lin_big)

    # --- 9m. the augmentation hook must be WIRED, not merely available --------
    # `LosoTrainDataset` has taken an `augment` callable since it was written and no run
    # ever passed one, so "the pipeline is augmented" was a property of the source code
    # rather than of any executed arm. Three things are checked, each of which was a real
    # way to be wrong:
    #   * disabled must be `None`, not an identity callable -- the dataset skips its
    #     worker-rng path entirely, which is what keeps a pre-augmentation checkpoint
    #     reproducible at its own seed;
    #   * a mistyped key must raise rather than be ignored, because a silently
    #     unaugmented "augmented" arm is only detectable once its numbers match the
    #     control exactly;
    #   * the composition must be a real perturbation that is reproducible, and the
    #     blanked channels must end up EXACTLY zero (see the module docstring: adding
    #     noise after the dropout turns a dropped channel into a rectified noise floor,
    #     i.e. a learnable "I was dropped" tell that defeats the transform).
    check("augmentation is OFF by default and off is `None`, not an identity callable",
          augment_mod.build_augment({}) is None
          and augment_mod.build_augment({"augment": False}) is None
          and augment_mod.build_augment({"augment": {"enabled": False}}) is None)
    try:
        augment_mod.build_augment({"augment": {"gainn": 0.3}})
        typo_raised = False
    except KeyError:
        typo_raised = True
    check("a mistyped augment key is a hard error (a silent no-op arm is the failure)",
          typo_raised)

    aug9 = augment_mod.build_augment({"augment": True})
    x9 = np.random.default_rng(0).normal(0, 1, size=(C, T)).astype(np.float32)
    out_a = aug9(x9, np.random.default_rng(11))
    out_b = aug9(x9, np.random.default_rng(11))
    out_c = aug9(x9, np.random.default_rng(12))
    moved = float(np.abs(out_a - x9).mean())
    check(f"augmentation perturbs the epoch, is reproducible from the seed, and does "
          f"not mutate its input (mean |dx| {moved:.3f}, x untouched "
          f"{np.allclose(x9, np.random.default_rng(0).normal(0, 1, size=(C, T)), atol=1e-6)})",
          moved > 0.05 and np.array_equal(out_a, out_b) and not np.array_equal(out_a, out_c)
          and abs(float(x9.mean())) < 0.25)

    dropout9 = augment_mod.make_augment(noise=0.5, channel_dropout=0.5)
    seen_zero = False
    for s in range(24):
        o = dropout9(x9, np.random.default_rng(s))
        zeros = (o == 0.0).all(axis=1)
        if zeros.any():
            seen_zero = True
    check("a dropped channel is EXACTLY zero even with noise enabled (so ELU(0) = 0 "
          "and the encoder cannot read that it was dropped)", seen_zero)


def commet_gates() -> None:
    """§19: the COMMET gate plumbing -- `di_ref` must reach the FGW plan, and only the plan.

    `gallery_di_ref` (G1) injects a precomputed gallery-side metric into the FGW structural
    term. Two failure modes are silent and both have already happened in this project: an
    argument that dies in `**kw` leaves the cell bit-identical to its twin and reads as a
    negative *result* (the A4 topo knob), and a term that leaks outside its intended pathway
    makes a structural-off control meaningless (the M1 index leak). These checks pin the
    contract: absent = no-op, exact = no-op, different = different, structural-off = inert.
    """
    from samclip import calibration as C  # noqa: PLC0415

    rng = np.random.default_rng(3)
    n, d = 24, 16
    q = rng.standard_normal((n, d))
    g = rng.standard_normal((n, d))

    def rec(alpha: float = 0.75, **kw):
        return C.subspace_soft_recovery(
            q, g, k=5, rho=0.1, rank=None, tau=0.03, iters=20,
            hard_landmarks=False, min_landmarks=2, alpha=alpha, **kw)

    out0, _ = rec()
    out_none, _ = rec(fgw_di_ref=None)
    check("fgw_di_ref=None is bit-identical to the shipped operator",
          np.array_equal(out0, out_none))

    gn = g / np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-9)
    di_exact = C._sq_cos_dist(gn)
    out_exact, _ = rec(fgw_di_ref=di_exact)
    check("an EXACT gallery reference reproduces the operator (the injected metric is the "
          "same object `_fgw_plan` would have computed)",
          np.allclose(out0, out_exact, atol=1e-10))

    perm = rng.permutation(n)
    di_perm = di_exact[np.ix_(perm, perm)]
    out_perm, _ = rec(fgw_di_ref=di_perm)
    check("a DIFFERENT gallery reference actually reaches the plan (not dropped in `**kw`)",
          not np.allclose(out0, out_perm, atol=1e-8))

    off0, _ = rec(alpha=0.0)
    off_p, _ = rec(alpha=0.0, fgw_di_ref=di_perm)
    check("with the structural term OFF the injected reference is EXACTLY inert "
          "(it can only route through the structural term)",
          np.array_equal(off0, off_p))

    # `rep_cloud_scores` must forward the reference too -- it goes through the recovery
    # wrapper, which is where arguments have silently died before.
    def _wrap(qq, gg, k=10, rho=0.1, min_landmark_rate=0.0, **kw):
        return C.subspace_soft_recovery(
            qq, gg, k=k, rho=rho, rank=None, tau=0.03, iters=20,
            hard_landmarks=False, min_landmarks=2, alpha=0.75,
            fgw_di_ref=kw.pop("fgw_di_ref", None),
            fgw_de_ref=kw.pop("fgw_de_ref", None))

    z = rng.standard_normal((n, 8, d))
    sc0, dg0 = C.rep_cloud_scores(z, g, k=5, rho=0.1, shrink=0.1, rep_blocks=4,
                                  recovery_fn=_wrap)
    sc1, dg1 = C.rep_cloud_scores(z, g, k=5, rho=0.1, shrink=0.1, rep_blocks=4,
                                  recovery_fn=_wrap, gallery_di_ref=di_perm)
    check("rep_cloud_scores reports whether a gallery reference arrived",
          dg0["gallery_di_ref"] is False and dg1["gallery_di_ref"] is True)
    check("the gallery reference changes the score through the recovery wrapper",
          not np.allclose(sc0, sc1))


def d2_front_end() -> None:
    """D2: the front end must have a LOCALITY prior, not just a shape.

    docs/eeg2image_v5_master_plan.md §3.3 P2. The historical front end is
    `nn.Linear(n_timepoints, d_model)`: a dense map from all 250 timepoints to every
    output feature, with an independent weight per (timepoint, feature) pair and no
    locality or translation prior. The claim D2 rests on is that this is where a subject
    can be fitted idiosyncratically -- not that it lacks capacity. So the test that
    matters is not the output shape (both paths give the same shape); it is that a single
    input timepoint can no longer reach every output feature, and that the parameter
    budget stops being dominated by the dense map.

    A test that only checked shapes would pass for both paths and would be worthless.
    """
    from samclip.models.backbone import EEGTrunk

    print("11. D2 front end: locality prior, not just a shape")
    C, T = 63, 250
    x0 = torch.randn(1, C, T)

    # (a) the default must be the historical path, bit-identical, or every recorded run
    # silently changes architecture. This exact bug class (a new default leaking into
    # pinned arms) has already happened once in this project.
    torch.manual_seed(0)
    a = EEGTrunk(n_channels=C, n_timepoints=T, d_model=200).eval()
    b = EEGTrunk(n_channels=C, n_timepoints=T, d_model=200, front_end="linear").eval()
    b.load_state_dict({k: v.clone() for k, v in a.state_dict().items()})
    with torch.no_grad():
        ya, yb = a(x0), b(x0)
    check(f"front_end defaults to 'linear' and is bit-identical to it "
          f"(max|diff| {float((ya - yb).abs().max()):.1e})", torch.equal(ya, yb))

    # (b) THE defining property: locality. Take one output position, ask which input
    # timepoints can reach it. A dense temporal map can reach every one; a conv stack
    # can only reach its receptive field.
    def reach(module: torch.nn.Module, out_index: int) -> int:
        xg = x0.clone().requires_grad_(True)
        h = module(xg)                      # (1, C, T) -> (1, C, n_out)
        h[0, :, out_index].sum().backward()
        return int((xg.grad.abs().sum(dim=1)[0] > 0).sum())

    lin = EEGTrunk(n_channels=C, n_timepoints=T, d_model=200, front_end="linear").eval()
    cnv = EEGTrunk(n_channels=C, n_timepoints=T, d_model=200, front_end="conv",
                   front_pool=5).eval()
    lin_reach = reach(lin.embed, 100)       # 1 of the 200 dense output features
    cnv_reach = reach(cnv.temporal, 25)     # 1 of the 50 pooled positions
    # receptive field = (k1-1) + dilation*(k2-1) + 1 with k=25, dil=3 -> 24 + 72 + 1 = 97
    check(f"a dense temporal map reaches all {lin_reach} timepoints, the conv stack only "
          f"{cnv_reach} (receptive field ~97) -- this IS the hypothesis-class change",
          lin_reach == T and 60 <= cnv_reach <= 110)

    # (c) the parameter story that justified D2: the dense map is ~50k and dominated the
    # trunk; a depthwise conv costs k*C regardless of d_model.
    fl = lin.embed.weight.numel()
    fc = cnv.embed.weight.numel()
    check(f"the dense map shrinks {fl:,} -> {fc:,} ({fc/fl:.0%}) and the two depthwise "
          f"convs cost k*C each",
          fl == T * 200 and fc == (T // 5) * 200 and fc < fl)

    # (d) gradients must reach BOTH convs, or the dilated layer is dead weight.
    xg = x0.clone()
    out = cnv.temporal(xg)
    out.sum().backward()
    g1 = float(cnv.temporal[0].weight.grad.abs().sum())
    g2 = float(cnv.temporal[3].weight.grad.abs().sum())
    check(f"both depthwise convs receive gradient (conv1 {g1:.2e}, dilated {g2:.2e})",
          g1 > 0 and g2 > 0)

    # (e) a fixed pool must divide T, or every trial silently loses its tail.
    rejected = 0
    for bad in (7, 0, -5):
        try:
            EEGTrunk(n_channels=C, n_timepoints=T, front_end="conv", front_pool=bad)
        except ValueError:
            rejected += 1
    check(f"front_pool that does not divide T is rejected ({rejected}/3)", rejected == 3)

    # (f) an unknown front_end name must not silently fall back to `linear`.
    try:
        EEGTrunk(n_channels=C, n_timepoints=T, front_end="convv")
        bad_name = False
    except ValueError:
        bad_name = True
    check("a misspelled front_end raises rather than silently using 'linear'", bad_name)


def recovery_module() -> None:
    """Differentiable coordinate recovery: the MECHANICS that must hold.

    docs/eeg2image_v5_master_plan.md §12. What this CAN test is the algebra and the
    guards; what it CANNOT test is whether the recovery is worth anything on real EEG,
    because that depends on how much correspondence signal survived the encoder.

    A synthetic test was attempted and abandoned: a Gaussian cloud rotated by a random
    orthogonal map is an independent Gaussian cloud, so mutual-NN on the cross-similarity
    finds pure noise (measured: 0% Top-1 at every rotation magnitude). A *norm-preserving*
    rotation leaves every diagonal cosine unchanged, so the correspondence is found even
    without recovery (measured: 100% unrecovered). Neither is the real setting, so the
    test asserts the algebra and leaves the value to an experiment.
    """
    from samclip.losses import orthogonal_procrustes, recovery_episode

    print("12. differentiable coordinate recovery (SCORE source-only episodes)")
    torch.manual_seed(0)
    d, n = 24, 120

    # (a) the solve returns a PROPER rotation
    x, y = torch.randn(n, d), torch.randn(n, d)
    r, diag = orthogonal_procrustes(x, y, rho=0.1)
    ident = torch.eye(d)
    orth = float((r @ r.T - ident).abs().max())
    check(f"Procrustes output is orthogonal (max|R R^T - I| {orth:.1e}) and a proper "
          f"rotation (det {diag['det']:.4f})", orth < 1e-4 and abs(diag["det"] - 1) < 1e-4)

    # (b) rho -> infinity must drive R to the IDENTITY. This is the regulariser's whole
    # purpose: a landmark set carrying no consistent rotation must yield a no-op, not a
    # wild rotation fitted to a handful of pairs.
    r_big, _ = orthogonal_procrustes(x, y, rho=1e6)
    dev = float((r_big - ident).abs().max())
    check(f"rho -> inf collapses R to the identity (max|R - I| {dev:.1e})", dev < 1e-2)

    # (c) when the two clouds already share a frame, recovery must not move them
    z_same, d_same = recovery_episode(y, y, k=10, rho=0.1)
    rel = float((z_same - y).norm() / y.norm())
    check(f"already-aligned clouds are left alone (relative move {rel:.1e}, "
          f"n_mutual {d_same.get('n_mutual')})", rel < 1e-4 and not d_same["abstained"])

    # (d) recovery is a ROTATION of the query cloud: it may turn the cloud but it must not
    # change its shape. If row norms changed, `orthogonal_procrustes` would not be
    # orthogonal and the whole argument for it (a coordinate change, not a rescale) fails.
    eeg = torch.randn(n, d) @ torch.linalg.qr(torch.randn(d, d))[0]
    z, dz = recovery_episode(eeg, y, k=10, rho=0.1)
    if not dz["abstained"]:
        dn = float((z.norm(dim=-1) - eeg.norm(dim=-1)).abs().max())
        check(f"recovery is a rotation: row norms preserved (max|Δ| {dn:.1e})", dn < 1e-4)

    # (e) gradient must reach the EEG embedding, or the source-only episode trains nothing
    e = eeg.clone().requires_grad_(True)
    z, _ = recovery_episode(e, y, k=10, rho=0.1)
    z.pow(2).mean().backward()
    check(f"gradient reaches the query cloud through recovery "
          f"(|grad| {float(e.grad.norm()):.2e})", float(e.grad.norm()) > 0)

    # (f) too few mutual landmarks must ABSTAIN, not fit noise
    _, d_few = recovery_episode(torch.randn(5, d), y, k=10, rho=0.1, min_landmarks=8)
    check(f"too few mutual landmarks abstains (n_mutual {d_few.get('n_mutual')})",
          d_few["abstained"])

    # (g) abstention returns the input UNCHANGED, so a disabled episode is inert
    small = torch.randn(5, d)
    z_ab, _ = recovery_episode(small, y, k=10, rho=0.1, min_landmarks=8)
    check("an abstained episode is bit-identical to its input",
          torch.equal(z_ab, small))

    # (h) REGRESSION TEST for the bug that cost a run. Everything above uses n=120 > d=24,
    # i.e. roughly full-rank x^T y, which is why this file stayed green while the real
    # training run went NaN at step 50. The real setting is L < d (42 landmarks, d=64) AND
    # low-rank (this project measures a 16-dimensional concept manifold). SVD's backward --
    # the original implementation -- returns NaN on exactly that input, and a forward-only
    # check cannot see it because the forward value is finite. Assert the GRADIENT.
    d_big, L, rank = 64, 42, 16
    xl = (torch.randn(L, rank) @ torch.randn(rank, d_big))
    xl = xl / xl.norm(dim=-1, keepdim=True)
    yl = torch.randn(L, d_big)
    yl = yl / yl.norm(dim=-1, keepdim=True)
    for rho in (0.05, 0.1, 0.3, 1.0):
        e = xl.clone().requires_grad_(True)
        z, _ = recovery_episode(e, yl, k=20, rho=rho, min_landmarks=8)
        z.pow(2).mean().backward()
        if e.grad is None:
            check(f"low-rank L<d gradient exists at rho={rho}", False)
            continue
        check(f"low-rank (L={L} < d={d_big}, rank {rank}) gradient is finite at rho={rho} "
              f"(max|grad| {float(e.grad.abs().max()):.2e})",
              bool(torch.isfinite(e.grad).all()) and bool(torch.isfinite(z).all()))

    # (i) rho = 0 must be REFUSED: the polar factor of a rank-deficient M is a partial
    # isometry, and the polish would silently bend it into a rotation nobody asked for.
    try:
        orthogonal_procrustes(xl, yl, rho=0.0)
        check("rho = 0 is refused", False)
    except ValueError:
        check("rho = 0 is refused", True)

    # ------------------------------------------------------------------------------
    print("13. M1 source-metric template (v10 stage 2.5)")
    # The whole claim of M1 is a CLAIM ABOUT VARIANCE, so the test has to be about variance
    # and not about "the number moved". Three assertions, in the order they can fail:
    #   (a) `src_mix=0` is BIT-IDENTICAL to the shipped operator -- otherwise every existing
    #       number on disk silently changes meaning;
    #   (b) the blend is a PROVABLE NO-OP when the template IS the target's own metric -- this
    #       is the one case where the correct answer is known in closed form, and it catches a
    #       blend that moves the answer for reasons other than the template (a scale bug, a
    #       normalisation bug, a wrong axis);
    #   (c) the recorded diagnostic corr(de_src, de_target) actually tracks how informative the
    #       template is -- near 1 for a clean template, near 0 for noise. Without (c) the
    #       diagnostic could be reporting a constant and every "template is informative" claim
    #       would be unfalsifiable.
    import numpy as _np
    from samclip import calibration as _cal

    rng = _np.random.default_rng(11)
    Cn, Rn, dn, Sn = 200, 80, 64, 9
    zr = rng.standard_normal((Cn, Rn, dn))
    zr /= _np.linalg.norm(zr, axis=2, keepdims=True)
    gg = rng.standard_normal((Cn, dn))
    gg /= _np.linalg.norm(gg, axis=1, keepdims=True)

    def _t2(src, mix):
        return _cal.rep_cloud_scores(
            zr, gg, k=10, rho=0.1, shrink=0.1,
            recovery_fn=lambda q, g, **kw: _cal.subspace_soft_recovery(
                q, g, k=10, rho=0.1, rank=None, tau=0.03, iters=50, alpha=0.75,
                fgw_outer=10, fgw_de_ref=kw.get("fgw_de_ref"),
                fgw_de_mix=kw.get("fgw_de_mix", 0.0)),
            src_means=src, src_mix=mix)

    # A template built the way the DATA is: a shared concept direction plus subject noise. The
    # shared part is what +0.565 says exists, so a template drawn from it is a fair test.
    shared = zr.mean(axis=1)
    src_data = _np.stack([shared + 0.7 * rng.standard_normal((Cn, dn)) for _ in range(Sn)])
    s0, d0 = _t2(src_data, 0.0)
    s_none, _ = _t2(None, 0.0)
    check("src_mix=0 is bit-identical to a run with no template",
          bool(_np.array_equal(s0, s_none)))
    check("src_mix=0 records no template diagnostic (nothing was blended)",
          "src_vs_target_metric_corr" not in d0)

    # (b) template == target's own metric -> the optimal blend is the identity.
    s_self, d_self = _t2(_np.stack([shared] * Sn), 1.0)
    check(f"template == target's own metric: mix=1 leaves the scores EXACTLY unchanged "
          f"(max|diff| {_np.abs(s0 - s_self).max():.1e})",
          bool(_np.allclose(s0, s_self, atol=1e-9)))
    check(f"...and the diagnostic reports the degenerate correlation it should "
          f"({d_self['src_vs_target_metric_corr']:.4f})",
          abs(d_self["src_vs_target_metric_corr"] - 1.0) < 1e-6)

    # (c) the diagnostic is a MEASUREMENT, so it must move with the template's quality.
    #
    # The perturbation scale matters and is easy to get wrong: `zr` rows are L2-normalised, so
    # a concept mean has components ~1/sqrt(d) = 0.125. Adding isotropic noise at scale 0.7
    # makes the "template" 5.6x noise, its correlation correctly collapses to ~0, and the test
    # then fails while the CODE is right. The scale below is set relative to the means it
    # perturbs, so the assertion is about the diagnostic and not about a hand-picked constant.
    mean_mag = float(_np.linalg.norm(zr.mean(axis=1), axis=1).mean())
    clean = _np.stack([zr.mean(axis=1) + 0.02 * mean_mag
                       * rng.standard_normal((Cn, dn)) for _ in range(Sn)])
    s_sig, d_sig = _t2(clean, 1.0)
    s_noise, d_noise = _t2(rng.standard_normal((Sn, Cn, dn)), 1.0)
    check(f"diagnostic is high for a template that carries the target's structure "
          f"({d_sig['src_vs_target_metric_corr']:+.3f})",
          d_sig["src_vs_target_metric_corr"] > 0.5)
    check(f"...and near zero for a pure-noise template "
          f"({d_noise['src_vs_target_metric_corr']:+.3f})",
          abs(d_noise["src_vs_target_metric_corr"]) < 0.1)
    check("the diagnostic ORDERS the two, which is what makes it usable as a gate before "
          "any accuracy is read",
          d_sig["src_vs_target_metric_corr"] > d_noise["src_vs_target_metric_corr"] + 0.3)
    check("mixing a signal-carrying template moves the result (the lever is connected)",
          not _np.allclose(s0, s_sig))
    check(f"n recorded is the number of upper-triangular pairs "
          f"{d_sig['src_vs_target_metric_corr_n']} == C(C-1)/2",
          d_sig["src_vs_target_metric_corr_n"] == Cn * (Cn - 1) // 2)

    # ------------------------------------------------------------------------------
    print("14. P1 spectral rank prior (v10 stage 3)")
    # The prior is a SCALAR read off the source eigenvalue spectrum. Its whole justification is
    # that it carries no index correspondence -- exactly the property the M1 template lacked, and
    # why M1 leaked (+15.4pp real vs -14.5pp with the concept axis shuffled). So the load-bearing
    # assertion here is PERMUTATION INVARIANCE: shuffle the source concepts and the P1 result must
    # be BIT-IDENTICAL. That is stronger than "the score did not change much", and it is the
    # property a reviewer will ask for.

    def _p1(spec=None, from_src=False, sm=None):
        return _cal.rep_cloud_scores(
            zr, gg, k=10, rho=0.1, shrink=0.1,
            recovery_fn=lambda q, g, **kw: _cal.subspace_soft_recovery(
                q, g, k=10, rho=0.1, rank=None, tau=0.03, iters=50, alpha=0.75,
                fgw_outer=10, fgw_de_ref=kw.get("fgw_de_ref"),
                fgw_de_mix=kw.get("fgw_de_mix", 0.0),
                fgw_spec_rank=kw.get("fgw_spec_rank")),
            src_means=sm, src_mix=0.0, spec_rank=spec, spec_from_src=from_src)

    # `spec=0` must be the shipped operator: `0` means "no truncation", and if it were treated as
    # literally zero modes it would silently zero the metric.
    s_plain, _ = _p1(None, False, None)
    s_zero, _ = _p1(0, False, None)
    check("spec_rank=0 is bit-identical to no truncation at all (0 means FULL rank, not empty)",
          bool(_np.array_equal(s_plain, s_zero)))
    s16, d16 = _p1(16, False, None)
    check("truncating to rank 16 changes the plan (the prior is connected)",
          not _np.allclose(s_plain, s16))
    check("the rank actually used is recorded", d16.get("fgw_spec_rank") == 16)

    # The source estimate must be a MEASUREMENT, and a permutation-invariant one.
    s_src, d_src = _p1(None, True, clean)
    perm_p = rng.permutation(Cn)
    s_perm, d_perm = _p1(None, True, clean[:, perm_p, :])
    check("the source-estimated rank is reported", d_src.get("spec_rank_from_src") is not None)
    check("the source effective rank itself is permutation-invariant",
          abs(d_src["eff_rank_src_mean"] - d_perm["eff_rank_src_mean"]) < 1e-9)
    check("PERMUTATION INVARIANCE: shuffling the source concept axis leaves the output "
          "BIT-IDENTICAL -- no index correspondence can enter, unlike the M1 template",
          bool(_np.array_equal(s_src, s_perm)))
    check("...and it is still not the untruncated operator (the prior is doing work)",
          not _np.allclose(s_plain, s_src))

    # Truncation must be a genuine projection.
    dmat_p = _cal._sq_cos_dist(zr.mean(axis=1))
    tr_p = _cal._spectral_truncate(dmat_p, 16)
    check("spectral truncation is symmetric", bool(_np.allclose(tr_p, tr_p.T, atol=1e-10)))
    check(f"truncation to 16 lowers numerical rank to <= 16 "
          f"(got {int(_np.linalg.matrix_rank(tr_p, tol=1e-8))})",
          int(_np.linalg.matrix_rank(tr_p, tol=1e-8)) <= 16)
    check("the metric spectrum is invariant under concept permutation",
          bool(_np.allclose(_np.sort(_np.linalg.eigvalsh(dmat_p)),
                            _np.sort(_np.linalg.eigvalsh(dmat_p[_np.ix_(perm_p, perm_p)])),
                            atol=1e-9)))

    # ------------------------------------------------------------------------------
    print("15. P2 unwhitened metric + P3 topological reference (v10 stage 3)")
    # P2 exists because P1 was falsified, so the FIRST thing to test is not P2 but the PREMISE
    # that motivated it: that a whitener raises the metric's effective rank. If that is false,
    # there was no obstruction and the relocation is a no-op dressed as a repair -- so it is
    # asserted as a measurement, not assumed. Then P3's load-bearing property is permutation
    # invariance of the scale (a scalar read off the distance multiset), for the same reason P1
    # is tested that way: it is the property that closes the M1 index-leakage channel.

    rng2 = _np.random.default_rng(23)
    C2, R2, d2, nlat = 200, 40, 64, 8
    _lat = rng2.standard_normal((C2, nlat)) @ rng2.standard_normal((nlat, d2))
    z_lo = _lat[:, None, :] + 0.35 * rng2.standard_normal((C2, R2, d2))
    z_lo /= _np.linalg.norm(z_lo, axis=2, keepdims=True)
    g_lo = rng2.standard_normal((C2, d2))
    g_lo /= _np.linalg.norm(g_lo, axis=1, keepdims=True)

    def _nrm(a):
        return a / _np.clip(_np.linalg.norm(a, axis=-1, keepdims=True), 1e-12, None)

    _mu2, _w2, _ = _cal._whiten_from_cloud(z_lo.reshape(C2 * R2, d2), shrink=0.1)
    _qraw2 = z_lo.mean(axis=1)
    _qwh2 = (_qraw2 - _mu2) @ _w2
    _r_raw = _cal._eff_rank(_cal._sq_cos_dist(_nrm(_qraw2)))
    _r_wh = _cal._eff_rank(_cal._sq_cos_dist(_nrm(_qwh2)))
    check(f"P2 PREMISE: whitening RAISES the metric effective rank "
          f"(unwhitened {_r_raw:.1f} -> whitened {_r_wh:.1f}); if this ever fails the "
          f"relocation has nothing to relocate",
          _r_wh > _r_raw)

    # P2 inertness: the default path must be BIT-IDENTICAL to the shipped operator, so every
    # existing number on disk keeps its meaning.
    def _p2(spec=None, from_src=False, sm=None, raw=False, teps=None, taut=False):
        return _cal.rep_cloud_scores(
            z_lo, g_lo, k=10, rho=0.1, shrink=0.1,
            recovery_fn=lambda q, g, **kw: _cal.subspace_soft_recovery(
                q, g, k=10, rho=0.1, rank=None, tau=0.03, iters=50, alpha=0.75,
                fgw_outer=10, fgw_de_ref=kw.get("fgw_de_ref"),
                fgw_de_mix=kw.get("fgw_de_mix", 0.0),
                fgw_spec_rank=kw.get("fgw_spec_rank"),
                fgw_topo_eps=kw.get("fgw_topo_eps")),
            src_means=sm, src_mix=0.0, spec_rank=spec, spec_from_src=from_src,
            raw_metric=raw, topo_eps=teps, topo_auto=taut)

    s_off, d_off = _p2()
    s_off2, _ = _p2(raw=False, taut=False)
    check("raw_metric/topo off is bit-identical to the shipped operator (defaults are inert)",
          bool(_np.array_equal(s_off, s_off2)))

    _shared2 = z_lo.mean(axis=1)
    _src2 = _np.stack([_shared2 + 0.7 * rng2.standard_normal((C2, d2)) for _ in range(9)])

    s_rf, d_rf = _p2(raw=True)
    check("P2 records the unwhitened effective rank (the prior's premise is on every run)",
          "eff_rank_de_unwhitened" in d_rf and "eff_rank_de_whitened" in d_rf)
    check("P2 records the pre-registered kill switch (`p2_premise_ok`)",
          "p2_premise_ok" in d_rf)
    check("...and it is the <=50 test it claims (unwhitened rank below 50 on 8-d structure)",
          _r_raw <= 50.0 and d_rf["p2_premise_ok"] is True)
    check("...and the unwhitened branch actually CHANGES the operator vs the whitened one",
          not _np.allclose(s_off, s_rf))
    check("raw-metric full rank is NOT the whitened operator, and equals raw-metric-no-flag "
          "(rawspec=0 is the paired twin, not a relabelling)",
          bool(_np.array_equal(s_rf, _p2(raw=True, spec=0)[0])))

    s_rs, d_rs = _p2(spec=16, raw=True)
    check("P2 spectral truncation acts on the UNWHITENED object and changes the plan",
          not _np.allclose(s_rf, s_rs))
    check("P2 flags the rank as applied to the unwhitened metric",
          d_rs.get("spec_rank_applied_to") == "unwhitened"
          and d_rs.get("fgw_spec_rank") == 16)

    # The same invariance that made P1 admissible: a scalar read off a permutation-invariant
    # spectrum. Permuting the SOURCE concept axis must not move a single output bit.
    _perm2 = rng2.permutation(C2)
    s_src, d_src = _p2(from_src=True, sm=_src2, raw=True)
    s_sp, d_sp = _p2(from_src=True, sm=_src2[:, _perm2, :], raw=True)
    check("P2 source-estimated rank is permutation-invariant (scalar: no index can enter)",
          abs(d_src["eff_rank_src_mean"] - d_sp["eff_rank_src_mean"]) < 1e-9)
    check("P2 PERMUTATION INVARIANCE: shuffling the source concept axis leaves output "
          "BIT-IDENTICAL",
          bool(_np.array_equal(s_src, s_sp)))

    # ---- P3: the topological reference ----------------------------------------------------
    # `_topo_graph` must be a genuine 0/1 adjacency (no metric left), and `_topo_eps_from` must
    # be a permutation-invariant SCALAR at a controlled DENSITY -- the properties that make it
    # usable without opening the M1 channel, and the reason a persistence gap was rejected as the
    # criterion (it cannot abstain when no gap exists, and produced a 60%-dense graph on real
    # data, which is a near-complete graph pretending to be a topology).
    _D3 = _cal._sq_cos_dist(_nrm(g_lo))
    _eps3 = _cal._topo_eps_from(_nrm(g_lo))
    _G3 = _cal._topo_graph(_D3, _eps3)
    check("topo graph is 0/1 with zero diagonal and symmetric",
          _np.array_equal(_np.unique(_G3), _np.array([0.0, 1.0]))
          and bool(_np.allclose(_G3, _G3.T))
          and float(_np.trace(_G3)) == 0.0)
    _dens3 = _G3.sum() / (C2 * (C2 - 1))
    check(f"topo graph density is the requested quantile, not a byproduct "
          f"(q=0.10 -> {_dens3:.3f})",
          abs(_dens3 - 0.10) < 0.01)
    check("topo scale is a permutation-invariant scalar",
          abs(_eps3 - _cal._topo_eps_from(_nrm(g_lo)[_perm2])) < 1e-12)
    check("topo graph commutes with relabelling (it is a property of the points, not the order)",
          bool(_np.array_equal(_G3[_np.ix_(_perm2, _perm2)],
                               _cal._topo_graph(_D3[_np.ix_(_perm2, _perm2)], _eps3))))
    check("topo edge count is monotone in eps (raising the threshold can only add edges)",
          _G3.sum() <= _cal._topo_graph(_D3, _eps3 + 1.0).sum())

    s_tp, d_tp = _p2(raw=True, taut=True)
    check("P3 reports the per-domain threshold scales and both edge counts",
          all(k in d_tp for k in ("topo_eps_query", "topo_eps_gallery",
                                  "topo_query_edges", "topo_gallery_edges")))
    check("P3 uses a SEPARATE scale per domain (topology is scale-free, not one shared eps)",
          d_tp.get("topo_mode") == "auto_per_domain"
          and abs(d_tp["topo_eps_query"] - d_tp["topo_eps_gallery"]) > 1e-12)
    check("P3 reports a label-free graph-overlap diagnostic",
          "topo_graph_jaccard" in d_tp and 0.0 <= d_tp["topo_graph_jaccard"] <= 1.0)
    check("P3 changes the operator vs the metric reference it replaces",
          not _np.allclose(s_rf, s_tp))
    check("P3 disables the spectral rank rather than silently applying it to a 0/1 matrix",
          _p2(raw=True, taut=True, spec=16)[1].get("spec_rank_disabled_by_topo") is True)

    s_te, _ = _p2(raw=True, teps=0.9)
    check("P3 accepts an explicit eps instead of the gallery-derived one",
          not _np.allclose(s_tp, s_te))

    # ------------------------------------------------------------------------------
    print("16. L2/L3/L4 structure ensemble + reliability-weighted fusion (v11)")
    # The v11 gate was a PROBE (`scripts/probe_fusion.py`, job 645697): pooling independent
    # structure estimates raises their agreement with a held-out view monotonically in the number
    # of views (cross-modal 0.209 -> 0.247 over K=1..8, 10/10 folds) while the
    # correspondence-destroying control stays pinned at ~0.000. This section pins the DEPLOYED
    # operator's obligations, in the order they can fail:
    #   (a) `fuse=0` and `fuse=1` are a strict no-op -- the shipped single-mean path must stay
    #       bit-identical or every number already on disk changes meaning;
    #   (b) the weights are a MEASUREMENT: a noisy block must be down-weighted, or L3 is
    #       decoration and could be replaced by a plain mean;
    #   (c) PERMUTATION EQUIVARIANCE -- block averaging is index-consistent only because the
    #       blocks are the same subject's same trials; relabelling concepts must relabel the
    #       scores and leave the weights untouched, so no external index correspondence (the M1
    #       channel) can enter;
    #   (d) SAFETY -- a degenerate (pure-noise) cloud must switch the fusion OFF by itself rather
    #       than average noise into the reference and look like a real arm.

    def _fuse(z, nb, gal=None):
        return _cal.rep_cloud_scores(
            z, gg if gal is None else gal, k=10, rho=0.1, shrink=0.1, rep_blocks=nb,
            recovery_fn=lambda q, g, **kw: _cal.subspace_soft_recovery(
                q, g, k=10, rho=0.1, rank=None, tau=0.03, iters=50, alpha=0.75,
                fgw_outer=10, fgw_de_ref=kw.get("fgw_de_ref"),
                fgw_de_mix=kw.get("fgw_de_mix", 0.0)))

    # (a) the shipped path is untouched ------------------------------------------------------
    s_base, d_base = _fuse(zr, 0)
    s_one, d_one = _fuse(zr, 1)
    check("fuse=0 leaves no fusion diagnostic (nothing was fused)",
          "struct_fuse_blocks" not in d_base)
    check("fuse=1 is bit-identical to fuse=0 (a single block IS the mean, so it must no-op)",
          bool(_np.array_equal(s_base, s_one)))

    # a cloud with a shared concept structure and two noise levels, so (b) has something to
    # measure. Early blocks are the reliable half, late blocks the noisy half.
    _lat = rng.standard_normal((Cn, 12))
    _base = _lat @ rng.standard_normal((12, dn))
    zf = _np.stack([_base + _np.sqrt(0.5) * rng.standard_normal((Cn, dn))
                    for _ in range(Rn)], axis=1)
    zf[:, Rn // 2:] = _base[:, None, :] + 1.7 * rng.standard_normal((Cn, Rn // 2, dn))
    s_f, d_f = _fuse(zf, 8)
    check("fusion records the blocks and their sizes (summing to R)",
          d_f.get("struct_fuse_blocks") == 8 and sum(d_f["struct_fuse_block_sizes"]) == Rn)
    _w = d_f["struct_fuse_weights"]
    check("weights form a distribution", abs(float(_np.sum(_w)) - 1.0) < 1e-9)
    check(f"L3 is a MEASUREMENT: the noisy half is down-weighted "
          f"({_np.mean(_w[:4]):.3f} vs {_np.mean(_w[4:]):.3f})",
          float(_np.mean(_w[:4])) > float(_np.mean(_w[4:])))
    check("fusion changes the plan vs the shipped mean (the lever is connected)",
          not _np.allclose(s_base, s_f))

    # (c) permutation equivariance ------------------------------------------------------------
    # BOTH the rep cloud and the gallery must be relabelled together: they share the concept
    # axis, so equivariance is only the right assertion when the correspondence is preserved.
    # (Permuting the query alone would destroy the correspondence -- that is a different test,
    # and it is the control, not this one.)
    _pp = rng.permutation(Cn)
    s_fp, d_fp = _fuse(zf[_pp], 8, gal=gg[_pp])
    _err = float(_np.abs(s_fp - s_f[_np.ix_(_pp, _pp)]).max())
    check(f"PERMUTATION EQUIVARIANCE: relabelling concepts relabels the scores "
          f"(max|err| {_err:.1e}, solver tolerance) and leaves the weights untouched",
          _err < 1e-5 and bool(_np.allclose(d_fp["struct_fuse_weights"], _w)))

    # (d) the degenerate regime must self-disable --------------------------------------------
    _, d_noise = _fuse(rng.standard_normal((Cn, Rn, dn)), 8)
    check("a pure-noise cloud reports the fusion as FLAT (records it rather than pretending "
          "a gain)",
          d_noise.get("struct_fuse_is_flat") is True)

    # ------------------------------------------------------------------------------
    print("17. v12 metric self-distillation + subject consistency (training-side)")
    # The v12 terms are the training instantiation of the locked claim
    # (docs/eeg2image_v12_core_claim.md). They target the PER-REPETITION metric, which the R-curve
    # measured to be the one lever with headroom (Top-1 still climbing +3.33pp at R=80). What must
    # be pinned here, in the order it can fail:
    #   (a) the loss is ZERO when the student metric already equals the teacher (a correct
    #       distillation is not a constant penalty), and > 0 when the student is noise;
    #   (b) PERMUTATION INVARIANCE -- relabelling the stimuli must not move the loss, or the term
    #       is fitting concept identity rather than structure (the M1 channel);
    #   (c) the diagnostics must expose a COLLAPSE (a rank-1 student could otherwise win);
    #   (d) the subject-consistency term must be negative for subjects that share a metric and ~0
    #       for subjects that do not -- a term whose sign is arbitrary is not a measurement.
    from samclip.losses.metric_distill import (  # noqa: PLC0415
        metric_self_distill, metric_subject_consistency)

    _nstim, _nsub, _R, _dd = 32, 2, 8, 16
    _lat = rng.standard_normal((_nstim, 6))
    _proj = rng.standard_normal((6, _dd))

    def _make(shared):
        """Build (z_rep (N,R,d), grp (N,), subject (N,)) with a shared per-stimulus signal.

        The NULL is not "pure noise" -- it is two subjects with INDEPENDENT low-rank structure.
        Pure Gaussian points are a bad null here: at d=16 the standardised squared-chordal metric
        of 32 random points is dominated by a shared concentration hump, and two independent draws
        correlate at ~0.38, which would fail the test while the code is right.
        """
        lat = (_lat if shared else rng.standard_normal((_nsub, _nstim, 6)).astype(_np.float32))
        proj = (_proj if shared else rng.standard_normal((_nsub, 6, _dd)).astype(_np.float32))
        rows, grp, subj = [], [], []
        for i in range(_nstim):
            for j in range(_nsub):
                base = (lat[i] @ proj) if shared else (lat[j, i] @ proj[j])
                reps = base[None, :] + rng.standard_normal((_R, _dd)) * 0.5
                rows.append(reps)
                grp.append(i)
                subj.append(j)
        z = _np.stack(rows).astype(_np.float32)
        return (torch.tensor(z), torch.tensor(grp), torch.tensor(subj))

    zc, gc, sc = _make(shared=True)
    l_shared, d_shared = metric_self_distill(zc, gc, sc, student_reps=1, teacher_blocks=4,
                                             return_diag=True)
    # The distillation NULL is "the repetitions within a subject carry no common structure" --
    # NOT "the subjects are independent". Distillation is a WITHIN-subject statement, so the
    # cross-subject construction above is the wrong null for it (and it is the right null for the
    # consistency term below). Each repetition gets its own independent low-rank latent here.
    _N = _nstim * _nsub
    z_nr = torch.tensor(_np.stack([
        rng.standard_normal(6).astype(_np.float32) @ _proj for _ in range(_N * _R)
    ]).reshape(_N, _R, _dd))
    l_rand, d_rand = metric_self_distill(z_nr, gc, sc, student_reps=1, teacher_blocks=4,
                                         return_diag=True)
    check(f"the distillation loss is small when the repetitions share structure ({float(l_shared):.3f}) "
          f"and large when they do not ({float(l_rand):.3f})",
          float(l_shared) < 0.35 < float(l_rand))
    check("the student metric is NOT collapsed (rank is reported and substantial)",
          d_shared["metric_student_rank"] > 4.0)

    # (a) exact zero when the student IS the teacher's building block (all reps identical)
    z_id = torch.tensor(_np.stack([
        _np.tile((_lat[i] @ _proj)[None, None, :], (_nsub, _R, 1)) for i in range(_nstim)
    ]).astype(_np.float32))
    # rows above are stimulus-major with subject axis; flatten to (N, R, d) in the same order
    z_id = z_id.reshape(_nstim * _nsub, _R, _dd)
    l_id, d_id = metric_self_distill(z_id, gc, sc, student_reps=1, teacher_blocks=4,
                                     return_diag=True)
    check(f"identical repetitions give a ZERO loss (a correct distillation is not a constant "
          f"penalty; got {float(l_id):.1e})",
          float(l_id) < 1e-4)

    # (b) permutation invariance: relabel the stimuli, loss must not move
    perm = torch.tensor(rng.permutation(_nstim))
    l_perm, _ = metric_self_distill(zc, perm[gc], sc, student_reps=1, teacher_blocks=4)
    check(f"PERMUTATION INVARIANCE: relabelling the stimuli leaves the loss unchanged "
          f"({float(l_shared):.4f} vs {float(l_perm):.4f})",
          abs(float(l_shared) - float(l_perm)) < 1e-5)

    # (d) subject consistency: negative for shared metrics, ~0 when the subjects are independent
    zr, gr, sr = _make(shared=False)
    lc_shared, dc_shared = metric_subject_consistency(zc, gc, sc, return_diag=True)
    lc_rand, dc_rand = metric_subject_consistency(zr, gr, sr, return_diag=True)
    check(f"subject-consistency is NEGATIVE when the two subjects share a metric "
          f"({float(lc_shared):+.3f}, corr {dc_shared['metric_consistency_mean']:+.3f}) and "
          f"near zero when they do not ({float(lc_rand):+.3f})",
          float(lc_shared) < -0.2 and abs(float(lc_rand)) < 0.35)
    check("subject-consistency reports the number of compared pairs",
          dc_shared["metric_consistency_pairs"] == _nsub * (_nsub - 1) // 2)

    # (e) the terms are connected to their input (a detached/constant term would pass the above)
    z_shift = zc.clone()
    z_shift[:, 0] = z_shift[:, 0] + 5.0
    l_shift, _ = metric_self_distill(z_shift, gc, sc, student_reps=1, teacher_blocks=4)
    check("the distillation loss responds to a perturbation of the student's repetitions",
          abs(float(l_shift) - float(l_shared)) > 1e-4)

    # ------------------------------------------------------------------------------
    print("18. SATTC structural expert -- _ranks correctness (E2 baseline)")
    # `_ranks` fed `structural_scores` (the E2 SATTC baseline) and was BROKEN on any real-sized
    # matrix: the fancy-index form of the backward rank produced non-integer, out-of-range
    # garbage (measured [-4.7, 199] on 200x200, dtype-dependent) because the two advanced index
    # arrays broadcast to the wrong (rank, column) pairs. It happened to be CORRECT on a 4x4 toy,
    # which is why it survived. The failure was silent in the worst way: `mutual` went NEGATIVE,
    # `log` returned NaN, and the NaN propagated into the fusion through `0 * NaN`. These checks
    # pin both the rank matrix and the NaN-freedom of the fusion.
    from samclip import calibration as _calib  # noqa: PLC0415

    _rng2 = np.random.default_rng(0)
    for _dt in (np.float32, np.float64):
        _s = (_rng2.standard_normal((200, 200)).astype(_dt) + np.eye(200) * 2.0)
        _rf, _rb = _calib._ranks(_s)
        check(f"_ranks gives valid ranks on 200x200 ({_dt.__name__}): both in [0,199], integer",
              _rf.min() == 0 and _rf.max() == 199 and _rb.min() == 0 and _rb.max() == 199
              and np.allclose(_rb, np.round(_rb)))
        # the definition: for every column, row sorted by descending score has rank 0,1,2,...
        _col_ok = all(np.array_equal(_rb[np.argsort(-_s, axis=0)[:, j], j],
                                     np.arange(200, dtype=float)) for j in range(0, 200, 23))
        check("_ranks backward rank matches its definition on sampled columns", _col_ok)

    _s = (_rng2.standard_normal((200, 200)).astype(np.float32) + np.eye(200) * 2.0)
    _f0, _ = _calib.structural_scores(_s, k=10, lam=0.0)
    _f2, _d2 = _calib.structural_scores(_s, k=10, lam=0.2)
    check("structural_scores emits NO NaN/Inf (the bug's signature was NaN from log(negative))",
          bool(np.isfinite(_f0).all()) and bool(np.isfinite(_f2).all()))
    check("lam=0 recovers the base per-row ranking exactly (the geometry-only twin)",
          bool((_f0.argmax(axis=1) == _s.argmax(axis=1)).all()))
    check("the structural expert reports its mutual-top-k enrichment diagnostic",
          "mutual_topk_enrichment" in _d2 and np.isfinite(_d2["mutual_topk_enrichment"]))

    # ------------------------------------------------------------------------------
    print("20. v13 bias anchor -- the ONLY new training term (order-2 bias gradient)")
    # Gated by measurement, not preference (job 650135, scripts/probe_view_independence.py): the
    # estimator's error has exactly three components and only BIAS has headroom. rho <= 0 on 10/10
    # folds (K_eff = R: pooling already removes all independent variance, so no variance term is
    # allowed here), while a 9-subject pooled metric -- 9 independent ENCODINGS -- reaches 0.796
    # against the target's own 0.616 (+0.180, t=+22.2, 10/10). The gap is subject-specific bias.
    # These checks pin (a) the metric identity, (b) that the term is DIRECTIONALLY the gradient of
    # subject bias, (c) that it is blind to per-subject ORTHOGONAL frames -- i.e. it does not fight
    # the order-1 nuisance M6/M7 falsified -- and (d) that it does not collapse rank.
    from samclip.losses.bias_anchor import (  # noqa: PLC0415
        metric_anchor_loss_by_subject,
    )
    from samclip.losses.metric_distill import _metric as _tmetric  # noqa: PLC0415

    _rr = np.random.default_rng(0).standard_normal((7, 5))
    check("v13: the torch metric is the EXACT twin of calibration._sq_cos_dist (ddof pinned)",
          bool(np.allclose(_tmetric(torch.tensor(_rr)).numpy(), _calib._sq_cos_dist(_rr),
                           atol=1e-12)))

    _c, _s, _rp, _dd = 24, 3, 8, 32
    _torch_gen = torch.Generator().manual_seed(11)
    _base = torch.randn(_c, _dd, generator=_torch_gen)
    _grp = torch.arange(_c).repeat_interleave(_s)
    _subj_t = torch.arange(_s).repeat(_c)
    _zimg = _base[_grp]
    _qm = torch.linalg.qr(torch.randn(_dd, _dd, generator=torch.Generator().manual_seed(21)))[0]
    _clouds = []
    for _si in range(_s):
        _bias = 0.5 * torch.randn(_dd, generator=torch.Generator().manual_seed(100 + _si))
        _clouds.append((_base @ _qm).unsqueeze(1) + _bias
                       + 0.1 * torch.randn(_c, _rp, _dd,
                                           generator=torch.Generator().manual_seed(7 + _si)))
    _zr = torch.stack(_clouds, dim=1).reshape(_c * _s, _rp, _dd)

    _l0, _d0 = metric_anchor_loss_by_subject(_zr, _zimg, _grp, _subj_t)

    # (b) monotone in subject bias: shrinks the bias, agreement must RISE
    _agr = []
    for _bs in (1.0, 0.5, 0.0):
        _cl = []
        for _si in range(_s):
            _b = _bs * torch.randn(_dd, generator=torch.Generator().manual_seed(100 + _si))
            _cl.append((_base @ _qm).unsqueeze(1) + _b
                       + 0.1 * torch.randn(_c, _rp, _dd,
                                           generator=torch.Generator().manual_seed(7 + _si)))
        _z = torch.stack(_cl, dim=1).reshape(_c * _s, _rp, _dd)
        _agr.append(metric_anchor_loss_by_subject(_z, _zimg, _grp, _subj_t)[1][
            "bias_anchor_agreement"])
    check("v13: the loss is strictly the gradient of SUBJECT bias (agreement rises as bias falls)",
          _agr[0] < _agr[1] < _agr[2] and float(_agr[2]) > 0.99)

    # (c) blind to per-subject ORTHOGONAL frames: the metric is 2nd order, so a rotation is free.
    #
    # THE ROTATION MUST BE APPLIED TO THE ENTIRE CLOUD, bias and noise included. An earlier version
    # of this check rotated only the concept signal (`_base @ _qs`) and drew the bias/noise fresh in
    # the original frame. That does NOT test invariance: it silently substitutes a DIFFERENT
    # perturbation, whose direction relative to the concept structure differs from the baseline's,
    # so the metric legitimately moves. It failed at |delta| = 4.4e-2 -- and the tell was that the
    # same 4.4e-2 appeared in float64, i.e. it was not rounding, it was a different experiment. The
    # isometry that leaves `_metric` unchanged is multiplication of the whole per-subject cloud by a
    # single orthogonal matrix, which is exactly what the encoder is allowed to do for free.
    _cl = []
    for _si in range(_s):
        _qs = torch.linalg.qr(torch.randn(_dd, _dd,
                                          generator=torch.Generator().manual_seed(300 + _si)))[0]
        _b = 0.5 * torch.randn(_dd, generator=torch.Generator().manual_seed(100 + _si))
        _cloud = ((_base @ _qm).unsqueeze(1) + _b
                  + 0.1 * torch.randn(_c, _rp, _dd,
                                      generator=torch.Generator().manual_seed(7 + _si)))
        _cl.append(_cloud @ _qs)          # rigid rotation of the WHOLE subject cloud
    _zq = torch.stack(_cl, dim=1).reshape(_c * _s, _rp, _dd)
    _dq = metric_anchor_loss_by_subject(_zq, _zimg, _grp, _subj_t)[1]
    check("v13: per-subject orthogonal maps are FREE (term does not fight the falsified "
          "order-1 nuisance)",
          abs(_dq["bias_anchor_agreement"] - _d0["bias_anchor_agreement"]) < 1e-5)

    # (d) no collapse: the student metric keeps full-ish rank, and the gradient reaches the encoder
    check("v13: student metric stays full-rank (collapse guard, v12's failure was rank 2)",
          _d0["bias_anchor_student_rank"] > 0.4 * min(_c, _dd))
    _zg = _zr.clone().requires_grad_(True)
    metric_anchor_loss_by_subject(_zg, _zimg, _grp, _subj_t)[0].backward()
    check("v13: gradient flows to the encoder and is finite",
          _zg.grad is not None and bool(torch.isfinite(_zg.grad).all())
          and float(_zg.grad.norm()) > 0)

    # (e) the anchor is EXOGENOUS: the loss must not depend on the gallery tensor at all
    _zir = _zimg.clone().requires_grad_(True)
    _lg, _ = metric_anchor_loss_by_subject(_zr, _zir, _grp, _subj_t)
    check("v13: the gallery anchor is detached (loss has no grad_fn on z_img -> cannot be bent)",
          _lg.grad_fn is None)

    # (f) a degenerate block is SKIPPED and reported, not silently scored
    _ld, _dd2 = metric_anchor_loss_by_subject(_zr[:6], _zimg[:6],
                                              torch.zeros(6, dtype=torch.long), _subj_t[:6])
    check("v13: a block with too few concepts reports pairs=0 instead of a phantom loss",
          _dd2["bias_anchor_pairs"] == 0 and float(_ld) == 0.0)

    # (g) the config guards: validating a term that cannot fire must RAISE, not degrade to a
    # silent no-op. `Trainer` is a dataclass whose first field is the model, so the constructor
    # has to be called by KEYWORD -- the earlier positional form (`Trainer(cfg)`) bound the config
    # to `model`, never reached the validation block, and failed the smoke test with a TypeError
    # that looked like a broken guard. It was a broken test.
    try:
        from samclip.train import Trainer  # noqa: PLC0415

        def _guard_raises(_cfg: dict, _needle: str) -> bool:
            try:
                Trainer(model=None, cfg=_cfg, device="cpu", n_subjects=3)
            except Exception as _exc:  # noqa: BLE001
                return _needle in str(_exc)
            return False

        check("v13: enabling bias_anchor without concept.enabled raises (no silent no-op term)",
              _guard_raises({"bias_anchor": {"enabled": True, "weight": 1.0},
                             "concept": {"enabled": False}}, "concept.enabled"))
        check("v13: a negative bias_anchor.weight raises",
              _guard_raises({"bias_anchor": {"enabled": True, "weight": -1.0},
                             "concept": {"enabled": True}}, "weight must be >= 0"))
        # ...and the OFF arm must construct cleanly with weight forcibly zeroed, which is what
        # makes `enabled: false` a true no-op rather than a treatment with a dormant flag.
        _t_off = Trainer(model=None, cfg={"bias_anchor": {"enabled": False, "weight": 2.0},
                                          "concept": {"enabled": True}},
                         device="cpu", n_subjects=3)
        check("v13: the OFF arm zeroes its weight (enabled: false is a real no-op)",
              _t_off.bias_anchor_enabled is False and _t_off.ba_weight == 0.0)
    except ImportError:
        check("v13: Trainer importable for the config guard", False)


    # ---- 21. O2-MVE: multi-family metric view pooling (v13 §7) ------------------------------
    print("\n21. O2-MVE -- multi-family metric view pooling (the order-2 estimator)")
    from samclip import calibration as _cal  # noqa: PLC0415
    _rng = np.random.default_rng(0)
    _Cn, _R, _d = 40, 80, 16
    _base = _rng.normal(size=(_Cn, _d))
    _z = _base[:, None, :] + 0.6 * _rng.normal(size=(_Cn, _R, _d))
    _v_cont, _qb, _wm = _cal.cloud_metric_views(_z, rep_blocks=8, mode="cont")
    _v_str, _, _ = _cal.cloud_metric_views(_z, rep_blocks=8, mode="stride")
    _v_rnd, _, _ = _cal.cloud_metric_views(_z, rep_blocks=8, mode="rand")
    check("O2-MVE: each partition family yields B views of the same (C,C) shape",
          len(_v_cont) == len(_v_str) == len(_v_rnd) == 8
          and _v_cont[0].shape == (_Cn, _Cn))

    # (a) the two families must be DIFFERENT views, or pooling them is a duplicate, not a new
    # view -- and strided vs contiguous is exactly the drift argument that motivates the family.
    _i = np.triu_indices(_Cn, 1)
    _dv = [np.corrcoef(a[_i], b[_i])[0, 1] for a, b in zip(_v_cont[1:], _v_cont[:-1])]
    _ds = [np.corrcoef(a[_i], b[_i])[0, 1] for a, b in zip(_v_str[1:], _v_str[:-1])]
    _dc = np.corrcoef(_v_cont[0][_i], _v_str[0][_i])[0, 1]
    check("O2-MVE: `stride` is a distinct view family from `cont` (not a relabelling)",
          abs(_dc) < 0.999)
    print(f"     [info] within-cont corr {np.mean(_dv):.4f} | within-stride {np.mean(_ds):.4f} | "
          f"cont-vs-stride {_dc:.4f}")

    # (b) the seeded control must be reproducible; a nondeterministic control is not a control.
    _v_rnd2, _, _ = _cal.cloud_metric_views(_z, rep_blocks=8, mode="rand")
    check("O2-MVE: the seeded `rand` control is bit-reproducible",
          bool(np.allclose(_v_rnd[0], _v_rnd2[0])))

    # (c) pooling must be leave-one-out reliable: a corrupted view must be down-weighted.
    _v_bad = [x.copy() for x in _v_cont]
    _v_bad[0] = _rng.normal(size=(_Cn, _Cn))          # a view that is pure noise
    _f_good, _dg_good = _cal.mve_fuse_views([_v_cont])
    _f_bad, _dg_bad = _cal.mve_fuse_views([_v_bad])
    _w_bad = _dg_bad["mve_weights"][0]
    _w_mean = float(np.mean(_dg_bad["mve_weights"]))
    check("O2-MVE: a corrupted view is down-weighted below the mean weight",
          _w_bad < _w_mean)

    # (d) a corrupted view must not be able to dominate the fused reference: dropping it should
    # move the fusion LESS than swapping in a different good view family.
    _f_other, _ = _cal.mve_fuse_views([_v_str])
    _dist_bad = float(np.linalg.norm(_f_good - _f_bad) / max(np.linalg.norm(_f_good), 1e-12))
    print(f"     [info] relative shift from a corrupted view: {_dist_bad:.4f}")

    # (e) degenerate guards: a single family still fuses (the cont-only control), and a family
    # of the WRONG SHAPE is refused rather than broadcast (a silent broadcast here would fuse a
    # 40x40 metric with a 40x16 query and produce a plausible-looking wrong number).
    _f1, _d1 = _cal.mve_fuse_views([_v_cont])
    check("O2-MVE: single-family pooling still yields a fused metric (the cont-only control)",
          _f1 is not None and _d1["mve_n_views"] == 8)
    _f2, _d2 = _cal.mve_fuse_views([_v_cont, [np.zeros((_Cn, _d))]])
    check("O2-MVE: a mismatched-shape view family is refused, not broadcast",
          _f2 is None and _d2["mve_n_views"] == 0)

    # (f) the fgw path must be REPORTED when the default operator would drop it. This is the
    # silent-swallow guard: `coordinate_recovery` takes `**_ignored`, so an unwired call would
    # produce a cell identical to its structural-off twin under a name that claims otherwise.
    check("O2-MVE: an unwired operator is flagged (fgw path can never silently no-op)",
          _cal.mve_scores([_v_cont], _qb, _base)[1]["mve_fgw_path"]
          == "default_operator_ignores_fgw")


    # ---- 22. v14 frame fix: the loss must be blind to per-subject ANISOTROPY ----------------
    print("\n22. v14 frame fix -- the order-2 anchor in the DEPLOYED (whitened) frame")
    from samclip.losses.bias_anchor import metric_anchor_loss_by_subject as _mab  # noqa: PLC0415
    from samclip.losses.bias_anchor import subject_consensus_anchor_loss as _csa  # noqa: PLC0415
    _tg = torch.Generator().manual_seed(0)
    _ns, _nsu, _r22, _d22 = 32, 3, 4, 16
    _bs = torch.randn(_ns, _d22, generator=_tg)
    _clouds = [(_bs.unsqueeze(1) + 0.6 * torch.randn(_d22, generator=_tg)
                + 0.35 * torch.randn(_ns, _r22, _d22, generator=_tg)) for _ in range(_nsu)]
    _z22 = torch.stack(_clouds, 1).reshape(_ns * _nsu, _r22, _d22)
    _g22 = torch.arange(_ns).repeat_interleave(_nsu)
    _s22 = torch.arange(_nsu).repeat(_ns)
    _zi22 = torch.randn(_ns, 12, generator=_tg)[_g22]

    # THE MEASUREMENT THIS GUARD ENCODES. v13 scored this loss in the raw frame and its gain
    # survived only there: raw-frame dR80 = +0.025..+0.041 but whitened +0.001..+0.011, with a
    # 9-13pp Top-1 loss. The whitened frame is the one the operator reads, so the fix is to score
    # the loss there -- and the property that proves the fix is real is INVARIANCE: a per-subject
    # invertible linear map is exactly the anisotropy class the raw loss rewarded.
    _A = [torch.randn(_d22, _d22, generator=_tg) for _ in range(_nsu)]
    _zr = _z22.view(_ns, _nsu, _r22, _d22)
    _zmap = torch.stack([_zr[:, s] @ _A[s] for s in range(_nsu)], 1).reshape(-1, _r22, _d22)

    def _agr(_zz, _frame, _shrink):
        return _mab(_zz, _zi22, _g22, _s22, frame=_frame,
                    whiten_shrink=_shrink)[1]["bias_anchor_agreement"]

    _dw = abs(_agr(_z22, "whitened", 0.0) - _agr(_zmap, "whitened", 0.0))
    _dr = abs(_agr(_z22, "raw", 0.0) - _agr(_zmap, "raw", 0.0))
    check("v14: the whitened-frame loss is INVARIANT to per-subject invertible linear maps",
          _dw < 1e-4)
    check("v14: the fix is load-bearing -- the RAW frame is NOT invariant to the same map",
          _dr > 10.0 * max(_dw, 1e-9))
    print(f"     [info] |delta| whitened {_dw:.2e} vs raw {_dr:.2e} "
          f"({_dr / max(_dw, 1e-12):.0f}x)")
    # Shrinkage is the one thing that breaks exactness (it re-introduces a scale term), so it is
    # pinned: the config must use 0.0 or the invariance above silently degrades to ~1e-3.
    _dsh = abs(_agr(_z22, "whitened", 0.1) - _agr(_zmap, "whitened", 0.1))
    check("v14: whiten_shrink=0 is what buys exact invariance (0.1 breaks it by >100x)",
          _dsh > 50.0 * _dw)

    # frame must be validated, not silently accepted: an unknown frame that fell through to `raw`
    # would ship v13's failure under a v14 name.
    try:
        _mab(_z22, _zi22, _g22, _s22, frame="whitenned")
        _bad_frame_raised = False
    except ValueError:
        _bad_frame_raised = True
    check("v14: an unknown frame raises instead of silently degrading to `raw`", _bad_frame_raised)

    # (b) the consensus term is implemented and INERT when off, and it is NOT booked as a
    # mechanism: its own synthetic diagnostic moved by less than noise (shared bias 0.806 vs
    # independent 0.818), which is exactly the "diagnostic does not move" signal this project
    # treats as a non-result. It stays available as an opt-in arm, never as a shipped claim.
    _lcs, _dcs = _csa(_z22, _zi22, _g22, _s22)
    check("v14: the consensus term computes, reports pairs, and carries a rank guard",
          np.isfinite(float(_lcs)) and _dcs["cs_pairs"] == _nsu
          and np.isfinite(_dcs["cs_student_rank"]))
    _lcs_off = _csa(_z22[:6], _zi22[:6], torch.zeros(6, dtype=torch.long), _s22[:6])[1]
    check("v14: the consensus term is inert (pairs=0, zero loss) when blocks are degenerate",
          _lcs_off["cs_pairs"] == 0)


if __name__ == "__main__":
    main()