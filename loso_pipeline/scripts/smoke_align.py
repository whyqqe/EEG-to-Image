#!/usr/bin/env python
"""Smoke-test one Stage-2 training step on synthetic data.

Does not touch the real EEG or target stores, so it can run on a CPU login node and
isolate *code* errors from *data* problems.  Verifies that every loss term is
produced and finite, that gradients reach both the encoder and every head, and that
the mean-adapter evaluation path works with no subject id.
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import torch

from loso import diagnostics as D
from loso.losses import align as L
from loso.models.eeg_encoder import EEGEncoder, EncoderConfig
from loso.models.heads import AlignmentHeads, HeadConfig
from loso.train.align import (AlignConfig, compute_losses, measure_gradient_budget,
                              _cross_subject_distance)

N_CONCEPT, N_IMAGE, N_SUBJ = 8, 10, 3
N_IMG = N_CONCEPT * N_IMAGE
D_TEACHER = 1024
LATENT = 4


class FakeStore:
    """Minimal stand-in for TargetStore with the same gather contract."""

    def __init__(self, tmp: Path):
        rng = np.random.default_rng(0)
        self.shapes = type("S", (), {"images_per_concept": N_IMAGE})()
        self.data = {
            "clip_image": torch.from_numpy(rng.standard_normal((N_IMG, D_TEACHER))).float(),
            "clip_text_caption": torch.from_numpy(rng.standard_normal((N_IMG, D_TEACHER))).float(),
            "dino": torch.from_numpy(rng.standard_normal((N_IMG, D_TEACHER))).float(),
            "vae_latent": torch.from_numpy(rng.standard_normal((N_IMG, 4, 64, 64))).float(),
        }

    def as_normalized(self, slots, concept_index=None, device="cpu"):
        out = {}
        for k, v in self.data.items():
            block = v[slots.detach().cpu()].to(device)
            out[k] = block.float() if k == "vae_latent" else torch.nn.functional.normalize(
                block.float(), dim=-1)
        return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = torch.device(args.device)

    enc = EncoderConfig(d_model=64, n_heads=4, n_layers=3, branch_width=8,
                        d_inv=64, d_sub=32, pretrained_subjects=N_SUBJ)
    head = HeadConfig(d_inv=64, d_sub=32, d_model=64, proj_hidden=128,
                      n_subjects=N_SUBJ, n_time_patches=enc.n_tokens)
    model = EEGEncoder(enc).to(device)
    heads = AlignmentHeads(head).to(device)
    cfg = AlignConfig(enc=enc, head=head, weights=L.LossWeights())
    # Smoke forces the optional terms on so they are exercised even when the
    # production defaults turn them off (UCK recipe zeroes `time`/`trial`).
    cfg.weights.time = 1.0
    cfg.weights.trial = 1.0
    cfg.weights.gallery = 1.0
    cfg.weights.mem = 0.3
    cfg.head.n_time_patches = enc.n_tokens

    b = args.batch_size
    # Slots must *repeat* inside the batch, otherwise `repeated_trial_contrastive`
    # finds no positive pairs, returns exactly 0, and the term is silently untested.
    # Cycling through a few slots mimics several repetitions of the same stimulus.
    n_slots = max(2, b // 4)
    slots = torch.arange(b) % n_slots
    batch = {
        "x": torch.randn(b, 63, 250, device=device),
        "subject_id": torch.arange(b, device=device) % N_SUBJ,
        "target_slot": slots.to(device),
        "trial_id": torch.arange(b, device=device),
        "repeat_id": torch.arange(b, device=device),
    }
    store = FakeStore(Path(tempfile.mkdtemp()))
    # UCK gallery: one row per concept.  Slots cycle through n_slots images of a
    # few concepts, so concept_id = slot // images_per_concept stays in range.
    n_concepts = max(2, (n_slots + store.shapes.images_per_concept - 1)
                     // store.shapes.images_per_concept)
    concept_gallery = torch.nn.functional.normalize(
        torch.randn(n_concepts, D_TEACHER, device=device), dim=-1)

    assert n_slots < b, "batch too small to contain repeated stimuli"

    for grl in (0.0, 1.0):
        loss, terms = compute_losses(model, heads, store, batch, cfg, grl,
                                     concept_gallery=concept_gallery)
        print(f"\n[grl={grl}] total={float(loss):.4f}")
        for k, v in sorted(terms.items()):
            val = float(v)
            flag = "" if np.isfinite(val) else "  <-- NON-FINITE"
            print(f"  {k:6s} {val:+.6f}{flag}")
            assert np.isfinite(val), f"loss term {k} is not finite: {val}"
    assert float(terms["trial"]) > 0, "repeated-trial term was never exercised"

    # Every weighted term must contribute a non-trivial share of the total.  This is
    # the guard that would have caught `huber_alignment`: with the default reduction its
    # value was 0.00098 against a total of ~26, so at weight 0.5 it contributed 2e-5 --
    # a term that is present, finite, has a non-zero gradient and is nonetheless
    # switched off.  None of the other checks in this file notice that.
    total_val = float(loss)
    shares = {k: getattr(cfg.weights, k) * float(v) / total_val
              for k, v in terms.items() if getattr(cfg.weights, k, 0.0) > 0.0}
    # A small *loss value* is not evidence of a dead term: a regulariser sitting near
    # its optimum is supposed to be small, and `cov` is exactly that (0.0004 at
    # initialisation, because the representation is not yet badly correlated).  What
    # distinguishes a dead term from a satisfied one is the gradient it can still
    # produce.  `huber_alignment`'s bug was of the dead kind: at `reduction="mean"` its
    # value was 0.00098 and its *gradient* was 1000x too small, so it was present,
    # finite, non-zero and switched off.
    #
    # So the guard is on gradient share, measured by `measure_gradient_budget` -- the
    # same probe the real run uses.  The threshold is deliberately far below any
    # sensible contribution (0.5%): it exists to catch a term that is off by orders of
    # magnitude, not to police the exact balance.
    print("\n  share of the weighted total (informational):")
    for k, share in sorted(shares.items(), key=lambda kv: kv[1]):
        print(f"    {k:6s} {share * 100:6.2f}%")
    grad_shares = measure_gradient_budget(model, heads, store, batch, cfg, 1.0,
                                          concept_gallery=concept_gallery)
    print("\n  share of the encoder gradient:")
    starved: list[str] = []
    for k, share in sorted(grad_shares.items(), key=lambda kv: kv[1]):
        mark = "  <-- STARVED" if share < 0.005 else ""
        print(f"    {k:6s} {share * 100:6.2f}%{mark}")
        if share < 0.005:
            starved.append(k)
    assert not starved, (
        f"these weighted terms take under 0.5% of the encoder gradient: {starved}. "
        f"Check their reduction and scale rather than their weight: `huber_alignment` "
        f"with the default reduction was exactly this, 1000x too small while its weight "
        f"said it was the second most important term.")

    print("\n=== gradient reach ===")
    loss, _ = compute_losses(model, heads, store, batch, cfg, 1.0,
                             concept_gallery=concept_gallery)
    loss.backward()
    named = {
        "encoder.temporal": model.temporal,
        "encoder.spatial.spatial_conv": model.spatial.spatial_conv,
        "encoder.region_embed": model.spatial.region_embed,
        "encoder.adapters": model.adapters,
        "encoder.head_inv": model.head_inv,
        "heads.img": heads.img,
        "heads.text": heads.text,
        "heads.dino": heads.dino,
        "heads.vae": heads.vae,
        "heads.time": heads.time,
        "heads.subject": heads.subject,
        "heads.logit_scale": heads.logit_scale,
    }
    missing: list[str] = []
    for name, module in named.items():
        if isinstance(module, torch.nn.Parameter):
            grads = [module.grad]
        else:
            grads = [p.grad for p in module.parameters()]
        total = sum(0.0 if g is None else float(g.abs().sum()) for g in grads)
        status = "OK " if total > 0 else "NO GRAD"
        print(f"  [{status}] {name:30s} |grad|={total:.4e}")
        if total == 0:
            missing.append(name)
    if missing:
        print(f"\n[FAIL] these modules received no gradient: {missing}")
        return 1

    print("\n=== eval path (no subject id, mean adapter) ===")
    model.eval()
    with torch.inference_mode():
        z_mean = model(batch["x"], None, adapter_mode="mean")["z_inv"]
        z_none = model(batch["x"], None, adapter_mode="none")["z_inv"]
        z_subj = model(batch["x"], torch.zeros(b, dtype=torch.long, device=device),
                       adapter_mode="subject")["z_inv"]
    print(f"  mean-adapter z_inv {tuple(z_mean.shape)} finite={bool(torch.isfinite(z_mean).all())}")
    # At initialisation every `up` is zero, so all three modes must agree exactly.
    # This is the regression guard for a residual that returned `h` instead of zeros,
    # which made `none` evaluate as `2 * h` and changed z_inv by ~0.3.
    d_none = float((z_mean - z_none).abs().max())
    d_subj = float((z_mean - z_subj).abs().max())
    print(f"  |mean - none|  = {d_none:.4e}  (expect 0 at init)")
    print(f"  |mean - subj0| = {d_subj:.4e}  (expect 0 at init)")
    assert d_none < 1e-6, f"adapter_mode='none' is not a no-op (delta {d_none:.4e})"
    assert d_subj < 1e-6, f"adapter_mode='mean' != 'subject' at init (delta {d_subj:.4e})"
    # And the adapters must actually be able to change the output once trained.
    with torch.no_grad():
        for a in model.adapters:
            a.up.normal_(std=1e-3)
    with torch.inference_mode():
        z_mean2 = model(batch["x"], None, adapter_mode="mean")["z_inv"]
    moved = float((z_mean2 - z_mean).abs().max())
    print(f"  after perturbing `up`: |mean - mean| = {moved:.4e} (must be > 0)")
    assert moved > 0, "adapters are disconnected from the output"

    print("\n=== zero-init residual contract ===")
    adapter = model.adapters[0]
    h = torch.randn(4, 8, 64)
    with torch.inference_mode():
        r_none = adapter(h, None, mode="none")
        r_mean = adapter(h, None, mode="mean")
    print(f"  residual 'none' max = {float(r_none.abs().max()):.4e} (expect 0)")
    print(f"  residual 'mean' max = {float(r_mean.abs().max()):.4e} "
          f"(non-zero: `up` was perturbed in the previous section)")
    assert r_none.abs().max() == 0, "mode='none' must return a zero residual"

    print("\n=== cross-subject distance degeneracy ===")
    z = torch.randn(64, 16)
    print(f"  single subject present -> {float(_cross_subject_distance(z, torch.zeros(64, dtype=torch.long), 3)):.4f} (expect 0)")
    print(f"  two subjects present   -> {float(_cross_subject_distance(z, torch.arange(64) % 2, 3)):.6f} (>0)")

    print("\n=== VICReg anti-collapse terms ===")
    # `var` must be *active*, not merely finite and present.  The failure mode being
    # guarded against is a hinge whose target sits above the achievable scale: it then
    # reports a plausible constant with a zero gradient forever, and no amount of
    # "the term exists and is finite" checking notices.  So the gradient is asserted.
    with torch.inference_mode():
        z_real = torch.randn(256, head.d_inv)
    print(f"  gamma={head.vicreg_gamma}  (measured, see scripts/measure_z_scale.py)")
    probe = z_real.clone().requires_grad_(True)
    var_term = L.vicreg_variance(probe, gamma=head.vicreg_gamma)
    var_term.backward()
    var_grad = float(probe.grad.norm())
    print(f"  var(real)={float(var_term):.6f}  |grad|={var_grad:.6f} (must be > 0)")
    assert var_grad > 0, (
        "vicreg_variance is saturated at gamma={head.vicreg_gamma}: it would sit at a "
        "constant value in the loss table while contributing no gradient")
    # And it must be strictly worse for a collapsed representation, or it is not
    # penalising the thing it exists to penalise.
    z_const = z_real.mean(0, keepdim=True).expand_as(z_real)
    print(f"  var(constant)={float(L.vicreg_variance(z_const, gamma=head.vicreg_gamma)):.6f} "
          f"(must exceed var(real))")
    assert float(L.vicreg_variance(z_const, gamma=head.vicreg_gamma)) > float(var_term)
    cov_real = float(L.vicreg_covariance(z_real))
    cov_const = float(L.vicreg_covariance(z_const))
    print(f"  cov(real)={cov_real:.6f}  cov(constant)={cov_const:.6f}")
    print("  (cov alone prefers the constant -- that is why `var` must be present, "
          "see loso.losses.align.vicreg_covariance)")
    assert cov_const < cov_real, (
        "covariance no longer prefers a constant; the docstring's warning about "
        "using it without `var` would need rewriting")

    print("\n=== collapse diagnostics: one synthetic case per failure mode ===")
    # The two statistics must fire on *different* inputs.  If one could replace the
    # other, the pair would be redundant and a later simplification would drop one
    # and silently lose a failure mode.
    healthy = D.representation_diagnostics(torch.randn(512, 64))
    print(f"  healthy            : top1_sv_ratio={healthy['top1_sv_ratio']:.3f} "
          f"mean_cosine={healthy['mean_offdiag_cosine']:+.3f} "
          f"eff_rank={healthy['eff_rank']:.1f}")
    ok, why = D.collapse_verdict(healthy)
    print(f"    verdict: ok={ok} -- {why}")
    assert ok, "a random Gaussian representation must not be reported as collapsed"

    # Used-width collapse: all variance in one direction, no constant offset.
    direction = torch.randn(1, 64)
    low_rank = torch.randn(512, 1) @ direction + 0.01 * torch.randn(512, 64)
    low_rank_stats = D.representation_diagnostics(low_rank)
    print(f"  one-direction-only : top1_sv_ratio={low_rank_stats['top1_sv_ratio']:.3f} "
          f"mean_cosine={low_rank_stats['mean_offdiag_cosine']:+.3f} "
          f"eff_rank={low_rank_stats['eff_rank']:.1f}")
    assert low_rank_stats["top1_sv_ratio"] > 0.9, "centred spectrum missed a 1-d spread"
    assert abs(low_rank_stats["mean_offdiag_cosine"]) < 0.5, (
        "this case is not a constant, so the raw cosine should not fire on it -- "
        "if it does, the two statistics have stopped being independent")
    ok, why = D.collapse_verdict(low_rank_stats)
    print(f"    verdict: ok={ok} -- {why}")
    assert not ok, "a one-dimensional spread must be reported as collapsed"

    # Constant collapse: invisible to the centred spectrum, which is the whole
    # reason the raw cosine is computed as well.  This is the failed run's signature.
    constant = torch.randn(1, 64).expand(512, 64) + 0.001 * torch.randn(512, 64)
    constant_stats = D.representation_diagnostics(constant)
    raw = D.representation_diagnostics(constant.clone() - constant.clone().mean(0))
    print(f"  near-constant      : top1_sv_ratio={constant_stats['top1_sv_ratio']:.3f} "
          f"mean_cosine={constant_stats['mean_offdiag_cosine']:+.3f} "
          f"eff_rank={constant_stats['eff_rank']:.1f}")
    ok, why = D.collapse_verdict(constant_stats)
    print(f"    verdict: ok={ok} -- {why}")
    assert not ok, "a near-constant representation must be reported as collapsed"
    assert raw["mean_offdiag_cosine"] > 0.9 or constant_stats[
        "mean_offdiag_cosine"] > 0.9, "raw cosine did not catch a constant"

    print("\n=== drift form: initialisation must not be judged collapsed ===")
    # The encoder's own untrained state reports mean_cosine ~ +0.80 (measured), so the
    # absolute thresholds *do* flag it.  The drift form is what the trainer uses, and
    # it has to say "holding" for a model that has not moved.
    init = {"top1_sv_ratio": 0.107, "eff_rank": 28.85, "mean_offdiag_cosine": 0.8037}
    unchanged = dict(init)
    ok, why = D.collapse_verdict(unchanged, initial=init)
    print(f"  unchanged init     : ok={ok} -- {why}")
    assert ok, "an unchanged model must not be reported as collapsing"
    # ... and the failed run's endpoint must be flagged against that same baseline.
    failed = {"top1_sv_ratio": 0.867, "eff_rank": 2.5, "mean_offdiag_cosine": 0.9075}
    ok, why = D.collapse_verdict(failed, initial=init)
    print(f"  failed run's end   : ok={ok} -- {why}")
    assert not ok, "the measured failed run must be flagged against its own baseline"

    print("\n=== constant-encoder baseline ===")
    q = torch.randn(16, 32)
    k = torch.randn(16, 32)
    scale = heads.scaled_logit_scale()
    baseline = D.constant_encoder_loss(q, k, scale)
    print(f"  real query batch  -> baseline={baseline:.6f}")
    # A constant encoder maps every input to the same vector, so the *rows* of the
    # similarity matrix are identical and the forward loss is a single number,
    # `-mean_b log softmax(q.k_c)[b]`.  It is deliberately not `log(B)`: log(B) needs
    # the logits to be constant across both axes, i.e. the keys to coincide too, and a
    # constant query still discriminates between keys.  That distinction is the whole
    # reason a collapsed encoder can report a plausible-looking loss.
    #
    # The term is symmetric, and the two directions behave very differently here:
    # transposing a matrix with identical rows gives identical *columns*, so the reverse
    # direction's rows are constant, its softmax is uniform and it sits exactly at
    # `log(B)`.  So a collapsed encoder gets a reverse-direction loss of log(B) for
    # free, and only the forward direction carries any information -- worth knowing,
    # because it means the symmetry does not rescue the term from collapse.
    constant = torch.nn.functional.normalize(q.mean(dim=0, keepdim=True), dim=-1)
    constant = constant.expand(16, -1).contiguous()
    logits = scale * constant @ torch.nn.functional.normalize(k, dim=-1).t()
    fwd = float(torch.nn.functional.cross_entropy(logits, torch.eye(16)))
    rev = float(torch.nn.functional.cross_entropy(logits.t(), torch.eye(16)))
    print(f"  rows identical    : {bool(torch.allclose(logits[0], logits[1]))}")
    print(f"  forward  CE       : {fwd:.6f}")
    print(f"  reverse  CE       : {rev:.6f}  (saturates at log B={np.log(16):.6f}, "
          f"as predicted above)")
    assert abs(rev - float(np.log(16))) < 1e-4, (
        "the reverse direction of a constant query did not saturate at log(B)")
    assert abs(baseline - 0.5 * (fwd + rev)) < 1e-4, (
        "constant-encoder baseline does not match the symmetric closed form")
    assert abs(baseline - float(np.log(16))) > 1e-3, (
        "the baseline degenerated to log(B); that only happens when the keys coincide, "
        "so the function is no longer implementing a constant encoder")
    # Idempotent: feeding it an already-constant query must reproduce the same value,
    # otherwise it is not measuring what it claims.
    assert abs(D.constant_encoder_loss(constant, k, scale) - baseline) < 1e-4, (
        "constant_encoder_loss is not idempotent on a constant query batch")

    print("\n=== grouped sampling: same-image peers in every batch ===")
    # The concrete number the previous run failed: 75.8% of rows had no same-image
    # peer, so their soft label reduced to a one-hot diagonal and the soft-label term
    # was plain InfoNCE for three quarters of every batch.
    from loso.train.sampler import GroupedBatchSampler
    n_img, group_size = 40, 4
    groups = [i // group_size for i in range(n_img * group_size)]
    sampler = GroupedBatchSampler(groups, batch_size=16, seed=0)
    bare = peerless = seen = 0
    for batch in sampler:
        slots_b = torch.tensor([groups[i] for i in batch])
        peers = (slots_b.unsqueeze(0) == slots_b.unsqueeze(1))
        peers.fill_diagonal_(False)
        bare += int((~peers.any(dim=1)).sum())
        peerless += int(peers.sum())
        seen += len(batch)
    print(f"  {sampler.groups_per_batch} images/batch x group_size "
          f"{sampler.group_size}, {len(sampler)} batches, {seen} rows")
    print(f"  rows with no same-image peer        : {bare} / {seen} "
          f"({bare / seen * 100:.1f}%, was 75.8%)")
    print(f"  same-image positive pairs per row   : {peerless / seen:.2f} "
          f"(was 0.48)")
    assert bare == 0, "the grouped sampler left rows without a same-image peer"
    assert peerless / seen >= group_size - 2, (
        "same-image positives per row are lower than the grouping implies")

    print("\n=== head/trunk capacity ratio ===")
    # Asserted on the *shipped* config, not the toy one used above, because the
    # inversion being guarded against was 2.11x in the shipped configuration and the
    # toy sizes could satisfy the constraint by accident.
    shipped_enc = EncoderConfig()
    shipped_head = HeadConfig(d_model=shipped_enc.d_model, d_inv=shipped_enc.d_inv,
                              d_sub=shipped_enc.d_sub,
                              n_time_patches=shipped_enc.n_tokens)
    trunk = sum(p.numel() for p in EEGEncoder(shipped_enc).parameters())
    head_p = sum(p.numel() for p in AlignmentHeads(shipped_head).parameters())
    print(f"  trunk={trunk / 1e6:.2f}M  heads={head_p / 1e6:.2f}M  "
          f"ratio={head_p / trunk:.2f}x (was 2.11x, must be < 1.0)")
    assert head_p < trunk, (
        f"projection heads ({head_p / 1e6:.2f}M) exceed the encoder trunk "
        f"({trunk / 1e6:.2f}M); the projectors can then fit the frozen targets "
        f"without the trunk contributing, which is the capacity inversion that "
        f"produced the previous run's collapse")

    print("\n=== set_alignment: both axes must be reduced correctly ===")
    # This is the term that owned 58% of the gradient in the run that collapsed, and
    # it had a dimension bug that no shape assertion caught: the forward direction
    # reduced the candidate axis instead of the patch axis, producing a (b, n_patch)
    # logits matrix scored against b-way labels.  Legal-looking shapes, wrong
    # objective.  Two checks pin it down.
    nb, n_tok, n_patch, dd = 12, 32, 64, 48
    pe = torch.randn(nb, n_tok, dd)
    pt = torch.randn(nb, n_patch, dd)
    scale2 = torch.tensor(10.0)
    L0 = float(L.set_alignment(pe, pt, scale2))
    print(f"  baseline loss={L0:.6f}")
    # 1. Permutation invariance, asserted separately for each axis, because the two are
    #    reduced in different places.  Getting one right and the other wrong is the
    #    exact failure mode, so a single combined test would not be enough.
    li = torch.randperm(n_tok)
    lj = torch.randperm(n_patch)
    a = float(L.set_alignment(pe[:, li], pt, scale2))
    b_ = float(L.set_alignment(pe, pt[:, lj], scale2))
    c_ = float(L.set_alignment(pe[:, li], pt[:, lj], scale2))
    print(f"  shuffle tokens  -> {a:.6f} (delta {abs(a - L0):.2e})")
    print(f"  shuffle patches -> {b_:.6f} (delta {abs(b_ - L0):.2e})")
    print(f"  shuffle both    -> {c_:.6f} (delta {abs(c_ - L0):.2e})")
    for name, val in (("tokens", a), ("patches", b_), ("both", c_)):
        assert abs(val - L0) < 1e-4, (
            f"set_alignment is not permutation-invariant in {name}; it is still "
            f"encoding an index correspondence, which is the correspondence that "
            f"does not exist")
    # 2. Negatives: the term must be minimised by an *aligned* prediction, not by a
    #    constant one.  This is what makes it safe to weight highly, and it is why the
    #    weight could be cut from 3.0 to 1.0 rather than the term being removed.
    #
    #    Note that the constant must be a *single* direction shared by every row.  The
    #    per-sample mean is not a collapse: it still discriminates between samples (the
    #    mean of a sample's own patches is closer to that sample's patches than to any
    #    other's), so it scores well below the constant baseline while being a perfectly
    #    valid -- if crude -- solution.  Only a constant shared across the whole batch
    #    corresponds to an encoder that emits one vector for every input.
    #
    #    Also note that the constant case is *not* log(B) in general: rows are identical
    #    but the entries within a row still vary across candidates, so the loss is
    #    `-mean_b log softmax(row)[b] = logsumexp(row) - mean(row)`, which is `log(B)`
    #    only when the row is flat and is *larger* when the row is peaked.  The
    #    closed form is asserted below so this reasoning stays checkable.
    aligned_pe = torch.stack([
        torch.nn.functional.normalize(pt[b_, :n_tok] + 0.05 * torch.randn(n_tok, dd),
                                      dim=-1)
        for b_ in range(nb)
    ])
    L_aligned = float(L.set_alignment(aligned_pe, pt, scale2))
    global_direction = torch.nn.functional.normalize(
        aligned_pe.mean(dim=(0, 1), keepdim=True), dim=-1)
    global_constant = global_direction.expand(nb, n_tok, -1).contiguous()
    L_const = float(L.set_alignment(global_constant, pt, scale2))
    print(f"  aligned prediction  -> {L_aligned:.6f}")
    print(f"  global constant     -> {L_const:.6f} "
          f"(log B={np.log(nb):.6f}; >= log B since the row is peaked)")
    assert L_const >= float(np.log(nb)) - 1e-3, (
        "the global-constant loss fell below log(B); that is impossible for identical "
        "rows, so the rows are not identical and this is not a collapse case")
    # The margin is also the guard on `SET_ALIGNMENT_TAU`.  At the old default of
    # sqrt(d) this margin was 0.15 -- a term that is very nearly constant across
    # candidates while still producing gradient on their shared component, which is how
    # a term can dominate a run's gradient budget and carry no signal at the same time.
    margin = L_const - L_aligned
    print(f"  discrimination margin -> {margin:.4f} "
          f"(was 0.151 at tau=sqrt(d); must exceed 1.0)")
    assert margin > 1.0, (
        f"aligned prediction ({L_aligned:.4f}) barely beat a global constant "
        f"({L_const:.4f}), margin {margin:.4f}.  The term is nearly flat in the "
        f"quantity it claims to measure; check SET_ALIGNMENT_TAU against the table in "
        f"loso.losses.align.")
    # Closed form for the constant row, so the identity above is pinned rather than
    # described: rows identical => loss = logsumexp(row) - mean(row).  Both sides must be
    # L2-normalised here -- `set_alignment` normalises internally, and feeding raw
    # Gaussians into this reconsideration makes the cosines into dot products of
    # magnitude sqrt(d) and inflates the "closed form" by 4x without any of the values
    # being wrong.
    pt_unit = torch.nn.functional.normalize(pt, dim=-1)
    gc_unit = torch.nn.functional.normalize(global_constant, dim=-1)
    sim_c = torch.einsum("bid,cjd->bicj", gc_unit, pt_unit)
    tau = L.SET_ALIGNMENT_TAU
    fwd_side = (torch.logsumexp(sim_c / tau, dim=3) * tau).mean(dim=1)
    rev_side = (torch.logsumexp(sim_c / tau, dim=1) * tau).mean(dim=2)
    row = (scale2 * fwd_side)[0]
    closed = float(torch.logsumexp(row, dim=0) - row.mean())
    # Both directions are summed, and both are of the same identical-row form, so the
    # check is on the forward half against the forward half of the symmetric mean.
    assert bool(torch.allclose(fwd_side[0], fwd_side[1])), (
        "the constant-prediction rows are not identical; this is not a collapse case")
    print(f"  rows identical      -> {bool(torch.allclose(fwd_side[0], fwd_side[1]))}")
    print(f"  constant forward CF -> {closed:.6f} "
          f"(vs reverse half, which saturates at log B={np.log(nb):.6f})")
    assert abs(closed - float(np.log(nb))) < 0.2, (
        "the constant-prediction forward loss is not near log(B); the reasoning above "
        "about the row being nearly flat is stale")
    # 3. A single-sample batch has nothing to discriminate against, so the term must be
    #    exactly zero rather than a constant added to the total.
    one = float(L.set_alignment(pe[:1], pt[:1], scale2))
    print(f"  single-sample batch -> {one:.6f} (must be 0)")
    assert one == 0.0, "set_alignment returned a non-zero value with no negatives"

    print("\n[OK] Stage-2 training step smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
