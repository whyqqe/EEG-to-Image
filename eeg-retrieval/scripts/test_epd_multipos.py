#!/usr/bin/env python
"""Unit tests for multi-positive alignment (SCORE Eq. 1). No GPU, no dataset.

Why this needs its own gate
---------------------------
The multi-positive loss is the difference between a LOSO fold that is set up
correctly and one that fights itself, and it fails *quietly in both directions*:

  * If the mask is built wrong, the loss still decreases. It just decreases toward a
    different objective -- pushing nine subjects who saw the same picture apart. There
    is no crash, no NaN, and the run looks healthy.
  * If the mask is built too generously (a tolerance instead of exact equality, or
    ids read off the wrong tensor), distinct stimuli become positives of each other
    and the objective is satisfied by collapsing instances together. Also no crash.

So the tests here are about the MASK, not about whether the loss goes down. The
load-bearing one is `test_degenerates_to_pairwise`: with every group a singleton the
multi-positive expression must equal the pairwise InfoNCE to floating-point, because
that is what makes `--multipos` a single-variable ablation. If that invariant breaks,
every comparison between the two arms silently acquires a second difference.

Run:  python scripts/test_epd_multipos.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from epd.losses import InfoNCE, stimulus_groups          # noqa: E402


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
    try:
        fn()
    except (SystemExit, ValueError, KeyError, RuntimeError) as e:
        msg = str(e)
        if needle and needle not in msg:
            check(False, f"refusal message mentions {needle!r}", f"got: {msg}")
        return msg
    check(False, "expected a refusal", "call returned normally")
    return ""


def _criterion(**kw) -> InfoNCE:
    """A criterion with a fixed scale, so the numbers are reproducible across runs.

    `learnable=False` and `softplus=False` pin the effective logit scale at
    exp(log(1/0.07)) = 14.29. With a learnable temperature the two paths would start
    from the same value but any test that steps an optimiser would drift, and these
    tests are about the loss VALUE, not about optimisation.
    """
    return InfoNCE(init_temp=0.07, softplus=False, learnable=False, **kw)


def _tiled_batch(n_stim: int, n_subj: int, d: int, seed: int = 0):
    """The shape a LOSO batch actually has: each stimulus repeated once per subject.

    Returns (eeg, image, groups, n_stim, n_subj). This is the layout
    `expand_loso_images` + subject-major stacking produces, which is why the tests use
    it rather than random groups -- the group structure is not arbitrary here, it is
    `n_subj` copies of each of `n_stim` stimuli, contiguous in blocks.
    """
    g = torch.Generator().manual_seed(seed)
    img = torch.randn(n_stim, d, generator=g)
    # Rows are subject-major in the real pipeline, but the mask only ever depends on
    # which rows are equal, so a stimulus-major tiling tests the same thing and is
    # easier to reason about. Interleaving is exercised in test_mask_ignores_row_order.
    image = img.repeat(n_subj, 1)
    eeg = torch.randn(n_stim * n_subj, d, generator=g)
    groups = stimulus_groups(image)
    return eeg, image, groups, n_stim, n_subj


def test_degenerates_to_pairwise() -> None:
    """The load-bearing invariant: singleton groups == the pairwise loss, exactly."""
    crit = _criterion()
    g = torch.Generator().manual_seed(1)
    for n in (2, 8, 64):
        a = torch.randn(n, 32, generator=g)
        b = torch.randn(n, 32, generator=g)
        # Distinct stimuli -> every group a singleton.
        groups = torch.arange(n)
        pair = crit(a, b)
        multi = crit(a, b, groups)
        check(torch.allclose(pair, multi, atol=1e-7),
              f"singleton groups == pairwise at B={n}",
              f"pairwise {float(pair):.8f} vs multi {float(multi):.8f}")
    check(torch.allclose(crit(a, b), crit(a, b, None), atol=0.0),
          "groups=None is bit-identical to the default pairwise path")


def test_mask_uses_exact_equality() -> None:
    """`stimulus_groups` must not merge near-identical-but-distinct stimuli."""
    g = torch.Generator().manual_seed(2)
    base = torch.randn(4, 16, generator=g)
    # Two rows one ulp apart are DIFFERENT stimuli. A tolerance-based grouping would
    # fuse them, and on real data that is how two different pictures end up as
    # positives of each other.
    nudged = base.clone()
    nudged[0, 0] = torch.nextafter(nudged[0, 0], torch.tensor(float("inf")))
    groups = stimulus_groups(nudged)
    check(int(torch.unique(groups).numel()) == 4,
          "a one-ulp perturbation creates a new group rather than merging",
          f"got {int(torch.unique(groups).numel())} groups, wanted 4")


def test_groups_come_from_tiling() -> None:
    """n_subj identical copies of n_stim stimuli produce exactly n_stim groups."""
    for n_stim, n_subj in ((3, 2), (5, 9), (17, 9)):
        _e, _i, groups, ns, nsub = _tiled_batch(n_stim, n_subj, 24)
        n_groups = int(torch.unique(groups).numel())
        check(n_groups == n_stim, f"{n_subj}x{n_stim} tiling gives {n_stim} groups",
              f"got {n_groups}")
        counts = torch.bincount(groups)
        check(bool((counts == n_subj).all()),
              f"every group has exactly {n_subj} members", f"counts {counts.tolist()}")


def test_mask_ignores_row_order() -> None:
    """The loss must not depend on how the batch is shuffled.

    A batch is drawn by a shuffled DataLoader, so subject-major and stimulus-major
    orderings of the SAME multiset of rows both occur. If the value moves when they
    are permuted, the grouping is reading position rather than content.
    """
    crit = _criterion()
    eeg, image, groups, _ns, _nsub = _tiled_batch(6, 3, 32, seed=3)
    g = torch.Generator().manual_seed(4)
    perm = torch.randperm(eeg.shape[0], generator=g)
    v_a = crit(eeg, image, groups)
    v_b = crit(eeg[perm], image[perm], groups[perm])
    check(torch.allclose(v_a, v_b, atol=1e-6),
          "the loss is invariant to a row permutation of the batch",
          f"{float(v_a):.8f} vs {float(v_b):.8f}")
    # And the groups are recovered identically, not merely permuted consistently.
    check(torch.equal(stimulus_groups(image[perm]),
                      groups[perm]),
          "stimulus_groups commutes with the permutation")


def test_inert_under_identical_targets() -> None:
    """Documented property: a tiled target makes multi-positive EQUAL to pairwise.

    Not a bug in the mask -- the mask is correct here, and the two losses still
    coincide exactly. Tiling makes every subject's target for a picture the same
    vector, so within a batch every duplicate row is identical to row i's own
    positive: equal logits, so `-mean` over the positives collapses to the single
    term the pairwise loss already had, in both directions.

    The consequence for the pipeline is the part worth pinning down: `--multipos`
    on a subject-INDEPENDENT target (`--target-fusion single`) cannot change the
    number, so any measured gap attributed to it there is noise. It becomes a real
    objective only together with SAMGA's subject-aware router.
    """
    crit = _criterion()
    n_stim, n_subj, d = 8, 3, 24
    eeg, image, groups, _ns, _nsub = _tiled_batch(n_stim, n_subj, d, seed=11)
    a = eeg.clone().requires_grad_(True)
    b = image.clone().requires_grad_(True)
    pairwise = crit(a, b, None)
    multi = crit(a, b, groups)
    check(torch.allclose(pairwise, multi, atol=1e-6),
          "tiled target: multi-positive equals pairwise in value",
          f"{float(pairwise):.8f} vs {float(multi):.8f}")
    gp = torch.autograd.grad(pairwise, a, retain_graph=True)[0]
    gm = torch.autograd.grad(multi, a)[0]
    check(float((gp - gm).abs().max()) < 1e-6,
          "tiled target: and in gradient, so it cannot move the parameters",
          f"max |dg| {float((gp - gm).abs().max()):.2e}")


def test_active_under_subject_dependent_targets() -> None:
    """The regime where SCORE Eq. 1 is a real objective, and what it does there.

    With per-subject targets the duplicate is no longer a copy of row i's positive:
    `logits[i, i] != logits[i, partner]`, so the positive set genuinely changes the
    loss, and pairwise is spending gradient denying the cross-subject alignment
    multi-positive is asking for. That is the +2.41 SCORE measures, and it requires
    BOTH the grouped mask AND a subject-dependent target.
    """
    crit = _criterion()
    n_stim, n_subj, d = 8, 3, 24
    g = torch.Generator().manual_seed(12)
    base = torch.randn(n_stim, d, generator=g)
    # Subject-dependent target: a different vector per subject for the same picture.
    image = torch.cat([base + 0.4 * torch.randn(n_stim, d, generator=g)
                       for _ in range(n_subj)], 0)
    eeg = torch.randn(n_stim * n_subj, d, generator=g)
    groups = torch.arange(n_stim).repeat(n_subj)      # the TRUE stimulus id

    a_p = eeg.clone().requires_grad_(True)
    lp = crit(a_p, image, None)
    a_m = eeg.clone().requires_grad_(True)
    lm = crit(a_m, image, groups)
    check(abs(float(lp) - float(lm)) > 1e-4,
          "subject-dependent target: the two objectives differ in value",
          f"pairwise {float(lp):.6f} vs multi {float(lm):.6f}")

    def cross_delta(loss, a):
        a.retain_grad()
        loss.backward()
        with torch.no_grad():
            before = float((a[0] @ image[n_stim]).item())
            after = float(((a - 0.5 * a.grad)[0] @ image[n_stim]).item())
        return after - before

    d_pair = cross_delta(lp, a_p)
    d_multi = cross_delta(lm, a_m)
    check(d_multi > d_pair,
          "multi-positive trains EEG toward ANOTHER subject's target harder than pairwise",
          f"pairwise {d_pair:+.5f} vs multi-positive {d_multi:+.5f}")


def test_feature_equality_cannot_find_groups_when_routed() -> None:
    """Why the stimulus id is carried by the dataset instead of inferred.

    Grouping by identical features is correct only while the target is
    subject-independent. Under a routed target it splits one picture into one group
    per subject, so the mask misses exactly the cross-subject pairs multi-positive
    exists to supply -- and the failure is silent, because the resulting mask is
    still a valid mask.
    """
    n_stim, n_subj, d = 6, 3, 16
    g = torch.Generator().manual_seed(13)
    base = torch.randn(n_stim, d, generator=g)
    tiled = base.repeat(n_subj, 1)
    routed = torch.cat([base + 0.4 * torch.randn(n_stim, d, generator=g)
                        for _ in range(n_subj)], 0)
    true_groups = torch.arange(n_stim).repeat(n_subj)

    # Compare the PARTITION, not the label values: `torch.unique` sorts, so the
    # inferred labels are a permutation of the true ids for the same grouping. Only
    # the equivalence classes matter, both here and in the mask.
    def same_partition(x, y):
        return torch.equal(x[:, None] == x[None, :], y[:, None] == y[None, :])

    check(same_partition(stimulus_groups(tiled), true_groups),
          "tiled target: identical-feature grouping recovers the true partition")
    inferred = stimulus_groups(routed)
    check(not same_partition(inferred, true_groups),
          "routed target: identical-feature grouping NO LONGER matches the partition")
    check(int(torch.unique(inferred).numel()) == n_stim * n_subj,
          "routed target: it degenerates to one group per row, i.e. the mask is the "
          "identity and multi-positive silently becomes the pairwise loss",
          f"{int(torch.unique(inferred).numel())} groups for {n_stim} pictures")


def test_average_not_sum() -> None:
    """Guards the documented factor-of-two choice against a later 'faithfulness' fix.

    SCORE writes `L_MP = L_{E->I} + L_{I->E}`. We return half of that, so that the
    singleton case lands on the pairwise value. If someone changes this to the literal
    sum, the ablation silently gains a 2x loss-scale change and these tests fail --
    which is the intended behaviour.
    """
    crit = _criterion()
    g = torch.Generator().manual_seed(6)
    n, d = 12, 16
    a = torch.randn(n, d, generator=g)
    b = torch.randn(n, d, generator=g)
    groups = torch.arange(n)                     # singleton -> closed form available
    pairwise = crit(a, b, None)
    la = torch.nn.functional.normalize(a, dim=-1)
    lb = torch.nn.functional.normalize(b, dim=-1)
    logits = crit.effective_scale().clamp(max=100.0) * (la @ lb.t())
    labels = torch.arange(n)
    e2i = torch.nn.functional.cross_entropy(logits, labels)
    i2e = torch.nn.functional.cross_entropy(logits.t(), labels)
    check(torch.allclose(pairwise, 0.5 * (e2i + i2e), atol=1e-6),
          "pairwise is the AVERAGE 0.5*(L_E->I + L_I->E), not the sum",
          f"{float(pairwise):.8f} vs sum {float(e2i + i2e):.8f}")
    check(torch.allclose(crit(a, b, groups), 0.5 * (e2i + i2e), atol=1e-6),
          "multi-positive keeps the same average convention as the pairwise path")


def test_empty_group_is_refused() -> None:
    """A zero group size would divide by zero into a silent NaN."""
    crit = _criterion()
    a = torch.randn(4, 8)
    b = torch.randn(4, 8)
    # Simulate a corrupt id tensor: torch.Tensor.scatter-style construction where a
    # row's id matches nothing including itself is impossible for a real mask, so the
    # guard is exercised by an id tensor shorter than the batch, which cannot be
    # broadcast into a square mask.
    raises(lambda: crit(a, b, torch.arange(3)), "")
    check(True, "a groups tensor that cannot form a square mask raises")


def test_scale_is_unchanged_by_the_flag() -> None:
    """The flag must not touch the temperature semantics it shares with the baseline."""
    for softplus in (False, True):
        for learnable in (False, True):
            c = InfoNCE(init_temp=0.07, softplus=softplus, learnable=learnable)
            g = torch.Generator().manual_seed(7)
            a = torch.randn(16, 32, generator=g)
            b = torch.randn(16, 32, generator=g)
            grp = torch.arange(16)
            with torch.no_grad():
                v0 = float(c(a, b))
                v1 = float(c(a, b, grp))
                s0 = float(c.effective_scale())
            check(abs(v0 - v1) < 1e-6 and s0 > 0,
                  f"scale untouched (softplus={softplus}, learnable={learnable})",
                  f"{v0:.6f} vs {v1:.6f}, scale {s0:.4f}")


def main() -> int:
    test_degenerates_to_pairwise()
    test_mask_uses_exact_equality()
    test_groups_come_from_tiling()
    test_mask_ignores_row_order()
    test_inert_under_identical_targets()
    test_active_under_subject_dependent_targets()
    test_feature_equality_cannot_find_groups_when_routed()
    test_average_not_sum()
    test_empty_group_is_refused()
    test_scale_is_unchanged_by_the_flag()

    print(f"\n{_passes} passed, {len(_failures)} failed")
    for f in _failures:
        print(f"  - {f}")
    return 1 if _failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
