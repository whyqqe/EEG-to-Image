"""Tests for POLARIS-on-CogCapPro.

The load-bearing ones are `test_upstream_intersection_*`. Design doc §2.4 claims CogCapPro's
`top_k=10` truncation destroys cross-subject positives in the inter-subject setting, with a
specific mechanism ("36 rows per image, 25 pushed away"). Those tests exist to establish what
is actually true, and they distinguish two regimes that the doc conflates:

  * distinct per-image conditioning targets, S=9 sources -> the S-1 siblings are the S-1 most
    similar rows, they all fit inside top_k=10, and NOTHING is truncated. The repair is a
    no-op, which `test_union_equals_upstream_when_nothing_is_truncated` pins down.
  * conditioning targets that are shared across stimuli (tied similarities) -> the tied block
    exceeds top_k, the intersection keeps an arbitrary slice, and positives ARE dropped.

So the defect is real but conditional, and its trigger is the tie structure of the targets
rather than the repetition count the doc named.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cogcap import recover as R                      # noqa: E402
from cogcap.losses import (                          # noqa: E402
    clip_loss_multi_positive,
    random_rotation,
    rotation_aug_loss,
    spectral_flatness_loss,
)
from cogcap.model import SEA, CogCapPro              # noqa: E402


# ------------------------------------------------------------------ SEA
def test_sea_is_exactly_identity_for_unseen_subject():
    sea = SEA(n_channels=8, n_subjects=3)
    with torch.no_grad():
        sea.A.normal_(0, 0.5)
        sea.t.normal_(0, 0.5)
    x = torch.randn(5, 8, 16)
    ids = torch.tensor([-1, -1, -1, -1, -1])
    assert torch.equal(sea(x, ids), x), "unknown subject must map to identity, not to some subject"


def test_sea_starts_at_identity_then_moves():
    sea = SEA(n_channels=6, n_subjects=2)
    x = torch.randn(3, 6, 10)
    ids = torch.tensor([0, 1, 0])
    assert torch.allclose(sea(x, ids), x, atol=1e-5), "Cayley(A=0) must be I"
    with torch.no_grad():
        sea.A.normal_(0, 0.3)
    assert not torch.allclose(sea(x, ids), x, atol=1e-3)


def test_sea_rotations_are_orthogonal():
    sea = SEA(n_channels=7, n_subjects=4)
    with torch.no_grad():
        sea.A.normal_(0, 1.0)
    rot = sea.rotations()
    eye = torch.eye(7).expand(4, 7, 7)
    assert torch.allclose(rot @ rot.transpose(-1, -2), eye, atol=1e-5)


# ------------------------------------------------------------------ model plumbing
def test_model_forward_shapes_and_subject_axis_effect():
    model = CogCapPro(["image", "depth", "edge"], n_subjects=9, use_sea=True)
    x = torch.randn(6, 63, 250)
    mods = [torch.randn(6, 1024) for _ in range(3)]
    ids = torch.tensor([0, 0, 1, 1, 2, -1])
    out = model(x, mods, subject_ids=ids)
    assert set(out["z"]) == {"image", "depth", "edge", "fusion"}
    assert out["z"]["image"].shape == (6, 1024)
    # the same EEG under two different subject ids must not give the same embedding,
    # otherwise the subject axis is decorative
    with torch.no_grad():
        a = model.brain(x[:1].repeat(2, 1, 1), torch.tensor([0, 1]))
    assert not torch.allclose(a[0][0], a[1][0], atol=1e-4)


def test_subject_wise_none_has_no_subject_parameters():
    m = CogCapPro(["image"], n_subjects=9, subject_wise="none", use_sea=False).eval()
    x = torch.randn(4, 63, 250)
    out_a = m(x, [torch.randn(4, 1024)], subject_ids=torch.tensor([0, 1, 2, 3]))
    out_b = m(x, [torch.randn(4, 1024)], subject_ids=torch.tensor([-1, -1, -1, -1]))
    assert torch.allclose(out_a["z"]["image"], out_b["z"]["image"], atol=1e-5)


# ------------------------------------------------------------------ multi-positive loss
def _tied_batch(n_concepts=2, n_imgs=3, n_subjects=9, tied=True, seed=0):
    """Conditioning targets that are either per-stimulus (distinct) or per-concept (tied)."""
    g = torch.Generator().manual_seed(seed)
    n_stim = n_concepts * n_imgs
    per_stim = F.normalize(torch.randn(n_stim, 32, generator=g), dim=1)
    concept_of_stim = torch.arange(n_stim) // n_imgs
    if tied:
        per_concept = F.normalize(torch.randn(n_concepts, 32, generator=g), dim=1)
        base = per_concept[concept_of_stim]
    else:
        base = per_stim
    targets = base.repeat(n_subjects, 1)                      # subject-major
    stim = torch.arange(n_stim).repeat(n_subjects)
    return targets, stim, n_stim


def test_upstream_intersection_truncates_when_targets_tie():
    """The defect is real in the tied regime: >top_k rows share the maximum similarity."""
    targets, stim, n_stim = _tied_batch(tied=True)
    eeg = targets.clone()
    _, _, diag_fixed = clip_loss_multi_positive(
        eeg, targets, torch.tensor(4.0), stim, top_k=10, repair_topk=True, want_diag=True)
    _, _, diag_up = clip_loss_multi_positive(
        eeg, targets, torch.tensor(4.0), stim, top_k=10, repair_topk=False, want_diag=True)
    print(f"  tied: pos/row={diag_fixed['n_pos_mean']:.2f} "
          f"union_kept={diag_fixed['n_pos_kept_mean']:.2f} "
          f"upstream_kept={diag_up['n_pos_kept_mean']:.2f} "
          f"dropped_by_upstream={diag_up['n_pos_dropped']:.2f}")
    assert diag_fixed["n_pos_kept_mean"] == pytest.approx(diag_fixed["n_pos_mean"]), \
        "the union must keep every sibling"
    assert diag_up["n_pos_kept_mean"] < diag_fixed["n_pos_kept_mean"], \
        "upstream's intersection is expected to drop positives when the tie block exceeds top_k"


def test_upstream_does_not_truncate_when_targets_are_distinct():
    """The benign regime, and the correction to design doc §2.4's numbers.

    With per-image targets and S=9 sources, each stimulus's 8 siblings are the 8 most similar
    rows and fit inside top_k=10, so upstream loses nothing. §2.4 asserted 36 rows per image
    and 25 dropped; with `train_avg: True` the row count is S, and at S=9 the truncation is
    absent. Recorded as a test so the claim cannot quietly come back.
    """
    targets, stim, _ = _tied_batch(tied=False)
    eeg = targets.clone()
    _, _, diag_up = clip_loss_multi_positive(
        eeg, targets, torch.tensor(4.0), stim, top_k=10, repair_topk=False, want_diag=True)
    print(f"  distinct: pos/row={diag_up['n_pos_mean']:.2f} "
          f"upstream_kept={diag_up['n_pos_kept_mean']:.2f} "
          f"dropped={diag_up['n_pos_dropped']:.2f}")
    assert diag_up["n_pos_dropped"] == pytest.approx(0.0, abs=1e-6)


def test_repair_equals_upstream_when_nothing_is_truncated():
    """The repair must be a no-op in the benign regime, or it is not a repair.

    In the distinct-target regime the diagonal plus the S-1 siblings are exactly the S rows
    with maximal target similarity, so they are all inside top_k=10 and the upstream
    intersection already equals the full positive set.
    """
    targets, stim, _ = _tied_batch(tied=False, seed=3)
    eeg = F.normalize(torch.randn(targets.shape, generator=torch.Generator().manual_seed(3)), dim=1)
    scale = torch.tensor(3.0)
    l_fix, _, _ = clip_loss_multi_positive(eeg, targets, scale, stim, top_k=10, repair_topk=True)
    l_up, _, _ = clip_loss_multi_positive(eeg, targets, scale, stim, top_k=10, repair_topk=False)
    assert float(l_fix) == pytest.approx(float(l_up), abs=1e-5)


def test_loss_ignores_diagonal_as_its_own_negative():
    """A single row with no sibling must still produce a finite loss."""
    eeg = F.normalize(torch.randn(4, 16), dim=1)
    tgt = F.normalize(torch.randn(4, 16), dim=1)
    stim = torch.arange(4)
    loss, _, _ = clip_loss_multi_positive(eeg, tgt, torch.tensor(2.0), stim, top_k=3)
    assert torch.isfinite(loss)


# ------------------------------------------------------------------ L_spec / L_aug
def test_spectral_flatness_is_lower_for_isotropic_mapping():
    """A rotation-like cross-subject map is flat; a stretched one is not."""
    torch.manual_seed(0)
    n_sub, n_stim, d = 4, 32, 16
    z0 = torch.randn(n_stim, d)
    rot = torch.linalg.qr(torch.randn(d, d))[0]

    iso = torch.stack([z0 @ rot] * n_sub, 0)                       # identical frames
    stretch = torch.stack([z0 * (torch.logspace(0, 1.5, d) * (1 + 0.05 * s)) for s in range(n_sub)], 0)

    def pack(mats):
        S, n, dd = mats.shape
        stim = torch.arange(n).repeat(S)
        subj = torch.arange(S).repeat_interleave(n)
        return mats.reshape(S * n, dd), stim, subj

    l_iso = spectral_flatness_loss(*pack(iso), rank=8)
    l_str = spectral_flatness_loss(*pack(stretch), rank=8)
    assert l_iso is not None and l_str is not None
    assert float(l_iso) < float(l_str), (float(l_iso), float(l_str))


def test_spectral_flatness_returns_none_without_enough_structure():
    z = torch.randn(8, 16)
    assert spectral_flatness_loss(z, torch.arange(8), torch.zeros(8, dtype=torch.long)) is None


def test_rotation_aug_loss_is_zero_for_a_rotation_invariant_encoder():
    # The T x T *temporal* Gram matrix (x^T x) is invariant under x -> R x on the channel
    # axis, since (Rx)^T (Rx) = x^T x. The C x C channel Gram is NOT invariant (it becomes
    # R x x^T R^T), which was this test's first mistake; so was using the leading slice of
    # the flattened input.
    def invariant(x, ids):
        return (x.transpose(1, 2) @ x).reshape(x.shape[0], -1)

    x = torch.randn(40, 8, 8)
    ids = torch.zeros(40, dtype=torch.long)
    stim = torch.arange(40)
    loss = rotation_aug_loss(invariant, x, ids, stim, ids, 8)
    assert loss is not None and float(loss) < 1e-6


def test_rotation_aug_loss_is_positive_for_a_rotation_sensitive_encoder():
    def sensitive(x, ids):
        return x.reshape(x.shape[0], -1)

    x = torch.randn(40, 8, 8)
    ids = torch.zeros(40, dtype=torch.long)
    stim = torch.arange(40)
    loss = rotation_aug_loss(sensitive, x, ids, stim, ids, 8, generator=torch.Generator().manual_seed(0))
    assert loss is not None and float(loss) > 1e-4


def test_random_rotation_is_orthogonal():
    r = random_rotation(6, 5, torch.device("cpu"))
    eye = torch.eye(6).expand(5, 6, 6)
    assert torch.allclose(r @ r.transpose(-1, -2), eye, atol=1e-5)


# ------------------------------------------------------------------ recovery
def test_saw_whitens_channel_covariance_towards_identity():
    torch.manual_seed(0)
    c, t, n = 12, 30, 200
    a = torch.randn(c, c)
    cov = a @ a.t() + torch.eye(c)
    chol = torch.linalg.cholesky(cov)    
    x = torch.einsum("cd,ndt->nct", chol, torch.randn(n, c, t))
    w, mu = R.fit_saw(x, lam=1e-4)
    y = R.apply_saw(x, w, mu)
    yc = y.permute(0, 2, 1).reshape(-1, c)
    got = (yc - yc.mean(0)).t() @ (yc - yc.mean(0)) / (yc.shape[0] - 1)
    off = (got - torch.eye(c)).abs().mean()
    assert off < 0.15, f"whitening left covariance structure: mean|Sigma - I| = {off:.3f}"


def test_moment_affine_is_a_no_op_when_distributions_match():
    torch.manual_seed(1)
    q = torch.randn(64, 16)
    scale, shift = __import__("cogcap.train", fromlist=["moment_affine"]).moment_affine(q, q.clone())
    assert torch.allclose(scale, torch.ones_like(scale), atol=1e-4)
    assert torch.allclose(shift, torch.zeros_like(shift), atol=1e-4)


def test_branch_defect_is_zero_for_identical_maps():
    r = torch.linalg.qr(torch.randn(10, 10))[0]
    maps = {m: (r, torch.zeros(10), torch.zeros(10)) for m in ("a", "b", "c")}
    assert R.branch_defect(maps) == pytest.approx(0.0, abs=1e-5)


def test_branch_defect_is_large_for_unrelated_maps():
    torch.manual_seed(0)
    maps = {m: (torch.linalg.qr(torch.randn(10, 10))[0], torch.zeros(10), torch.zeros(10))
            for m in ("a", "b", "c")}
    assert R.branch_defect(maps) > 0.5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="uses the model in a fit loop")
def test_h1_sensor_operator_recovers_a_planted_rotation():
    """If the input really is rotated by one sensor-space operator, H1's fit should find it."""
    from cogcap.model import CogCapPro
    torch.manual_seed(0)
    model = CogCapPro(["image"], n_subjects=1, use_sea=False, fusion=False).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    q_true = torch.linalg.qr(torch.randn(63, 63, device="cuda"))[0].float()
    x = torch.randn(96, 63, 250, device="cuda")
    with torch.no_grad():
        target = {m: F.normalize(v, dim=1) for m, v in zip(
            ["image"], model.brain(torch.einsum("cd,bdt->bct", q_true, x), None))}
    op, hist = R.fit_sensor_operator(model, x, target, torch.arange(96), torch.arange(96),
                                     ["image"], steps=150, lr=5e-2)
    assert hist[-1] < hist[0], (hist[0], hist[-1])


def test_loso_expansion_is_subject_major(tmp_path, monkeypatch):
    """`np.tile` vs `np.repeat` is the silent trap in this pipeline.

    Conditions must pair stimulus `i` with every source subject, so the expanded array's row
    `s*n_stim + i` must equal `feat[i]`. `np.repeat` would instead give `feat[s]`-style
    pairings and every inter-subject number would be wrong while still looking plausible.
    """
    from cogcap import config
    from cogcap import data as D

    n_concepts, n_imgs, n_sub = 2, 3, 4
    n_stim = n_concepts * n_imgs
    c, t = 4, 16
    eeg_dir = tmp_path / "eeg"
    out_dir = tmp_path / "out"
    feat_dir = tmp_path / "feat"
    (feat_dir).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(config, "EEG_DIR", eeg_dir)
    monkeypatch.setattr(config, "COGCAP_OUT", out_dir)
    monkeypatch.setattr(config, "IP_ADAPTER_FEATURE_DIR", feat_dir)
    monkeypatch.setattr(config, "IMAGE_FEATURE_DIR", feat_dir)
    monkeypatch.setattr(config, "N_IMAGES_PER_CONCEPT", n_imgs)

    rng = np.random.default_rng(0)
    for s in range(1, n_sub + 1):
        d = eeg_dir / f"sub-{s:02d}"
        d.mkdir(parents=True, exist_ok=True)
        # train [n_concepts, n_imgs, 1, C, T]; the single rep keeps the average exact
        tr = np.full((n_concepts, n_imgs, 1, c, t), float(s), dtype=np.float32)
        tr = tr + rng.normal(0, 1e-3, tr.shape).astype(np.float32)
        np.save(d / "train.npy", tr)
        te = np.full((n_concepts, 1, 2, c, t), float(s), dtype=np.float32)
        np.save(d / "test.npy", te)

    train_feat = rng.normal(size=(n_concepts, n_imgs, 1024)).astype(np.float32)
    test_feat = rng.normal(size=(n_concepts, 1, 1024)).astype(np.float32)
    np.save(feat_dir / "clip_h14_train.npy", train_feat)
    np.save(feat_dir / "clip_h14_test.npy", test_feat)

    sources = [1, 2, 3]
    data = D.LOSOData(sources, 4, ["image"], verbose=False)
    S = len(sources)

    flat = torch.from_numpy(train_feat.reshape(n_stim, -1))
    for s in range(S):
        for i in range(n_stim):
            row = s * n_stim + i
            assert torch.allclose(data.train_mod["image"][row], flat[i]), (s, i)

    # the EEG block for subject s must be exactly that subject's own cached array
    assert data.train_eeg.shape[0] == S * n_stim
    assert data.train_subj.tolist() == np.repeat(np.arange(S), n_stim).tolist()
    assert data.train_stim.tolist() == np.tile(np.arange(n_stim), S).tolist()
    for s, sid in enumerate(sources):
        want = torch.from_numpy(np.load(D.ensure_eeg_cache(sid, "train"))).float()
        assert torch.allclose(data.train_eeg[s * n_stim:(s + 1) * n_stim], want), sid

    # the held-out subject is not among the sources
    assert data.test_subj.unique().tolist() == [-1]
    assert data.test_eeg.shape[0] == n_concepts


def test_loso_rejects_target_in_sources():
    from cogcap import data as D
    with pytest.raises(AssertionError):
        D.LOSOData([1, 2, 8], 8, ["image"], verbose=False)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
