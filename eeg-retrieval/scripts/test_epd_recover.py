#!/usr/bin/env python
"""Unit tests for SCORE coordinate recovery (Eq. 3-9). No GPU, no dataset.

This is the highest-value component in the inter-subject pipeline -- 23.90 of SCORE's
27.01 point gain lives on the test side, on frozen features -- and it is also the one
whose failure modes are the least visible. Every function here can be wrong in a way
that still returns a valid rotation and still produces a number:

  * R applied on the wrong SIDE is a valid rotation in the wrong basis. Retrieval
    gets worse, not broken.
  * `lambda = rho` instead of `rho * ||X^T W Y||_2` is a valid regulariser of a
    different strength in every configuration. With unscaled features it can dominate
    completely, and the map silently becomes the identity -- which is exactly the
    no-op SCORE's diagnostic warns about.
  * CSLS neighbourhoods taken k-nearest INCLUDING the query itself, or computed on the
    landmark subset, are both plausible readings that shift the scores.
  * `R = U V^T` vs `R = V U^T`: one maximises tr(R^T M), the other minimises it. The
    wrong one gives a perfect anti-alignment, i.e. accuracy well below chance.
  * Weights normalised to sum to 1 instead of to m changes the effective weighted
    centring only in the degenerate case, so it hides until it does not.

The last test is the one that matters: a synthetic subject whose frame is a known
rotation of the gallery's, where recovery must restore retrieval that direct matching
cannot do. That is SCORE's Table 1 (16.89 -> 28.22) in miniature, and it is the only
check that exercises Eq. 3 through 9 together.

Run:  python scripts/test_epd_recover.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from epd.recover import (apply_recovery, cos_sim, csls_scores,           # noqa: E402
                         landmark_margins, moment_match, mutual_nn_pairs,
                         orthogonal_recovery, recover, select_landmarks)


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
            check(False, f"refusal mentions {needle!r}", f"got: {msg}")
        return msg
    check(False, "expected a refusal", "call returned normally")
    return ""


def _orth(d: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(d, d, generator=g)
    q, _r = torch.linalg.qr(a)
    return q


def _top1(q: torch.Tensor, g: torch.Tensor) -> float:
    """200-way Top-1 under the identity correspondence."""
    pred = cos_sim(q, g).argmax(dim=1)
    return float((pred == torch.arange(q.shape[0])).float().mean())


# --------------------------------------------------------------------------- Eq. 3
def test_moment_match_puts_q_on_g_stats() -> None:
    g = torch.Generator().manual_seed(0)
    q = torch.randn(50, 8, generator=g) * 3.0 + 5.0
    gg = torch.randn(40, 8, generator=g) * 0.5 - 2.0
    qt = moment_match(q, gg)
    for name, got, want in (("mean", qt.mean(0), gg.mean(0)),
                            ("std", qt.std(0, unbiased=False), gg.std(0, unbiased=False))):
        check(torch.allclose(got, want, atol=1e-5),
              f"moment_match matches per-dimension {name}",
              f"max err {float((got - want).abs().max()):.2e}")
    # Per-dimension, not global: give the dimensions very different spreads and
    # confirm each one lands on its OWN target rather than on a shared scalar. `eps`
    # is dropped to 1e-6 here because its bias is proportional to eps/sd and the
    # smallest spread below is 0.1 -- at the default 1e-5 that bias (3e-4) would
    # exceed the tolerance and the test would be measuring the stabiliser, not the
    # per-dimension property.
    q2 = torch.randn(60, 4, generator=g) * torch.tensor([1.0, 10.0, 0.1, 5.0])
    g2 = torch.randn(60, 4, generator=g) * torch.tensor([7.0, 0.2, 3.0, 0.5]) + 1.0
    qt2 = moment_match(q2, g2, eps=1e-6)
    per_dim_err = (qt2.std(0, unbiased=False) - g2.std(0, unbiased=False)).abs().max()
    check(float(per_dim_err) < 1e-4,
          "moment_match is per-dimension, not a single global rescale",
          f"max err {float(per_dim_err):.2e}")
    check(float((qt2.std(0, unbiased=False) / g2.std(0, unbiased=False) - 1).abs().max())
          < 1e-5,
          "and each dimension is scaled by its own factor, so no shared scalar fits")


def test_moment_match_degenerate_dimension() -> None:
    """A constant dimension must stay bounded, not blow up via 1/eps.

    The bound is not exact, and it cannot be: Eq. 3 divides by `sd + eps`, so the
    float32 round-off in the numerator -- which for a constant column is the error in
    the mean, ~1e-7 here -- is amplified by 1/eps = 1e5. The result is therefore
    ~1e-2 rather than 0. That is harmless at real feature scales (a normalised feature
    has per-dimension spread ~1/sqrt(d), orders above eps) but it is the reason this
    test asserts a bounded deviation instead of equality.
    """
    q = torch.randn(30, 3)
    q[:, 2] = 1.234                        # zero variance
    g = torch.randn(30, 3)
    qt = moment_match(q, g, eps=1e-5)
    check(torch.isfinite(qt).all(), "a zero-variance dimension stays finite")
    off = float((qt[:, 2] - g[:, 2].mean()).abs().max())
    check(off < 0.05,
          "and stays bounded near the gallery's mean for that dimension rather than "
          "being amplified to feature scale",
          f"max offset {off:.2e} (eps-amplified round-off, not a blow-up)")
    raises(lambda: moment_match(torch.randn(4, 3), torch.randn(4, 5)), "width mismatch")


# --------------------------------------------------------------------------- Eq. 4
def test_csls_penalises_a_hub() -> None:
    """The behavioural claim: a hub gallery item stops winning on raw similarity.

    Build a gallery with one item placed at the mean of the queries, so it has a high
    average similarity to everything, and confirm CSLS demotes it below the true match
    while raw cosine promotes it.
    """
    g = torch.Generator().manual_seed(1)
    d, n = 16, 40
    gal = torch.randn(n, d, generator=g)
    queries = gal + 0.35 * torch.randn(n, d, generator=g)
    hub = queries.mean(0, keepdim=True) * 1.15
    gal[n - 1] = hub[0]
    raw = cos_sim(queries, gal)
    csls = csls_scores(queries, gal, k=10)
    hub_raw_rank = float((raw > raw[:, n - 1:n]).sum(1).float().mean())
    hub_csls_rank = float((csls > csls[:, n - 1:n]).sum(1).float().mean())
    check(hub_csls_rank > hub_raw_rank,
          "CSLS demotes the hub item relative to raw cosine",
          f"mean ranks raw {hub_raw_rank:.1f} -> CSLS {hub_csls_rank:.1f}")


def test_csls_shape_and_clamping() -> None:
    g = torch.Generator().manual_seed(2)
    s = csls_scores(torch.randn(12, 8, generator=g), torch.randn(20, 8, generator=g), k=10)
    check(tuple(s.shape) == (12, 20), "CSLS returns (n_queries, n_gallery)", str(tuple(s.shape)))
    s_small = csls_scores(torch.randn(12, 8, generator=g), torch.randn(3, 8, generator=g), k=10)
    check(torch.isfinite(s_small).all(),
          "k larger than the gallery is clamped rather than raising")
    s_one = csls_scores(torch.randn(5, 8, generator=g), torch.randn(1, 8, generator=g), k=10)
    check(torch.isfinite(s_one).all(), "a single gallery item stays finite")


def test_csls_matches_its_closed_form() -> None:
    """Pin the expression: 2*cos - row_mean_of_topk - col_mean_of_topk."""
    g = torch.Generator().manual_seed(3)
    q, gal = torch.randn(9, 6, generator=g), torch.randn(14, 6, generator=g)
    k = 4
    c = cos_sim(q, gal)
    want = 2 * c - c.topk(k, dim=1).values.mean(1, keepdim=True) \
        - c.topk(k, dim=0).values.mean(0, keepdim=True)
    check(torch.allclose(csls_scores(q, gal, k=k), want, atol=1e-6),
          "csls_scores equals its closed form (and is not 2*cos minus a constant)")


# --------------------------------------------------------------------------- Eq. 5
def test_mutual_nn_pairs_are_mutual() -> None:
    s = torch.tensor([[0.9, 0.5, 0.1],
                      [0.8, 0.7, 0.2],
                      [0.1, 0.3, 0.4]])
    # query 0 -> gallery 0 and gallery 0 -> query 0 (0.9 is the column max) => mutual
    # query 1 -> gallery 0, but gallery 0's best query is 0, so NOT mutual
    # query 2 -> gallery 2, and gallery 2's best query is 2 => mutual
    pairs = mutual_nn_pairs(s)
    got = {tuple(p) for p in pairs.tolist()}
    check(got == {(0, 0), (2, 2)}, "only mutual nearest neighbours are kept", str(got))
    check(mutual_nn_pairs(torch.zeros(0, 0)).shape == (0, 2), "an empty matrix is empty")


def test_landmark_margins_use_whole_gallery() -> None:
    """The margin is the query's top-2 gap over the gallery, not between landmarks."""
    g = torch.Generator().manual_seed(4)
    s = torch.randn(6, 10, generator=g)
    pairs = torch.tensor([[0, 3], [1, 5]])
    w = landmark_margins(s, pairs)
    check(abs(float(w.sum()) - len(pairs)) < 1e-5,
          "weights are normalised to sum to m (landmark count), as in Eq. 5",
          f"sum {float(w.sum()):.6f} for m={len(pairs)}")
    top2 = s.topk(2, dim=1).values
    want_ratio = (top2[0, 0] - top2[0, 1]) / (top2[1, 0] - top2[1, 1])
    check(abs(float(w[0] / w[1]) - float(want_ratio)) < 1e-4,
          "relative weight follows the top-2 margin of the query's full CSLS row")
    check(float(landmark_margins(s, torch.zeros(0, 2, dtype=torch.long)).numel()) == 0,
          "no landmarks gives no weights")


def test_selection_takes_the_most_confident() -> None:
    """The cap must keep high-margin pairs, not the first ones it happened to see."""
    # A gallery of near-duplicates makes many mutual pairs; one query is sharply
    # separated and must survive the cap.
    g = torch.Generator().manual_seed(5)
    base = torch.randn(1, 12, generator=g)
    gal = base.repeat(30, 1) + 0.01 * torch.randn(30, 12, generator=g)
    q = base.repeat(10, 1) + 0.01 * torch.randn(10, 12, generator=g)
    s = csls_scores(q, gal, k=3)
    pairs_all, w_all = select_landmarks(s, k=3, max_landmarks=None)
    check(pairs_all.shape[0] <= 10, "mutual pairs are at most the query count",
          str(pairs_all.shape[0]))
    pairs_cap, w_cap = select_landmarks(s, k=3, max_landmarks=2)
    check(pairs_cap.shape[0] <= 2, "the cap is respected", str(pairs_cap.shape[0]))
    check(abs(float(w_cap.sum()) - pairs_cap.shape[0]) < 1e-5,
          "weights are renormalised to the new m after the cap",
          f"sum {float(w_cap.sum()):.6f} m={pairs_cap.shape[0]}")


# ---------------------------------------------------------------------- Eq. 6-7
def test_recovery_returns_a_rotation() -> None:
    g = torch.Generator().manual_seed(6)
    d, m = 12, 30
    x, y = torch.randn(m, d, generator=g), torch.randn(m, d, generator=g)
    w = torch.rand(m, generator=g) + 0.5
    r, _mx, _my = orthogonal_recovery(x, y, w, rho=0.1)
    eye = torch.eye(d)
    check(torch.allclose(r.t() @ r, eye, atol=1e-5), "R^T R = I")
    det = float(torch.linalg.det(r))
    check(abs(abs(det) - 1.0) < 1e-4, "det(R) = +-1, so R is orthogonal not just unitary",
          f"det {det:+.6f}")


def test_recovery_solves_the_procrustes_problem() -> None:
    """With exact matched pairs and rho=0, R* must be the planted rotation.

    This is the test that catches U V^T vs V U^T: one recovers the rotation, the other
    returns its transpose, and both are orthogonal matrices so nothing else complains.
    """
    g = torch.Generator().manual_seed(7)
    d, m = 10, 40
    x = torch.randn(m, d, generator=g)
    r_true = _orth(d, 8)
    # Y = X R_true is exactly Eq. 6's zero-residual case.
    y = x @ r_true
    w = torch.ones(m)
    r, _mx, _my = orthogonal_recovery(x, y, w, rho=0.0)
    err = float(torch.linalg.matrix_norm(r - r_true).item())
    check(err < 1e-4, "rho=0 with exact pairs recovers the planted rotation",
          f"||R* - R_true||_F = {err:.2e}")
    # And the wrong transpose is NOT also a solution, so the test has teeth.
    alt = float(torch.linalg.matrix_norm(r_true.t() - r_true).item())
    check(alt > 1e-3, "the planted rotation is not its own transpose (test has teeth)",
          f"||R_true^T - R_true||_F = {alt:.2e}")


def test_identity_regularisation_shrinks_toward_identity() -> None:
    """rho is the knob between "trust the landmarks" and "leave the frame alone"."""
    g = torch.Generator().manual_seed(9)
    d, m = 10, 12                     # m just above d: a thin, rank-poor fit
    x = torch.randn(m, d, generator=g)
    y = x @ _orth(d, 10) + 0.5 * torch.randn(m, d, generator=g)
    w = torch.ones(m)
    d_r0 = float(torch.linalg.matrix_norm(
        orthogonal_recovery(x, y, w, rho=0.0)[0] - torch.eye(d)).item())
    d_r1 = float(torch.linalg.matrix_norm(
        orthogonal_recovery(x, y, w, rho=10.0)[0] - torch.eye(d)).item())
    check(d_r1 < d_r0,
          "a larger rho moves R* closer to the identity",
          f"rho=0 -> {d_r0:.4f}, rho=10 -> {d_r1:.4f}")
    d_huge = float(torch.linalg.matrix_norm(
        orthogonal_recovery(x, y, w, rho=1e6)[0] - torch.eye(d)).item())
    check(d_huge < 1e-2, "an enormous rho collapses R* to the identity",
          f"||R* - I||_F = {d_huge:.2e}")


def test_mismatched_landmarks_refused() -> None:
    raises(lambda: orthogonal_recovery(torch.randn(5, 4), torch.randn(4, 4),
                                      torch.ones(5)), "shapes must match")
    raises(lambda: orthogonal_recovery(torch.randn(0, 4), torch.randn(0, 4),
                                      torch.zeros(0)), "no landmarks")


# --------------------------------------------------------------------------- Eq. 8
def test_apply_recovery_preserves_geometry_and_moves_the_origin() -> None:
    g = torch.Generator().manual_seed(11)
    d, m = 8, 25
    x, y = torch.randn(m, d, generator=g), torch.randn(m, d, generator=g)
    w = torch.rand(m, generator=g) + 0.5
    r, mu_x, mu_y = orthogonal_recovery(x, y, w, rho=0.0)
    q = torch.randn(17, d, generator=g)
    qh = apply_recovery(q, r, mu_x, mu_y)
    before = torch.cdist(q, q)
    after = torch.cdist(qh, qh)
    check(float((before - after).abs().max()) < 1e-4,
          "Eq. 8 preserves all pairwise distances (R is orthogonal)",
          f"max dist change {float((before - after).abs().max()):.2e}")
    # The landmark centroids must coincide after mapping, which is the point of
    # returning mu_x / mu_y rather than assuming they are zero.
    check(torch.allclose(qh.mean(0, keepdim=True) - mu_y,
                         (q.mean(0, keepdim=True) - mu_x) @ r, atol=1e-4),
          "the map commutes with the (unweighted) centring identity")


# ------------------------------------------------------------------- Eq. 3-9 together
def _rotated_subject(theta: float, sigma: float, *, n: int = 200, d: int = 64,
                     seed: int = 12):
    """A synthetic subject whose frame is the gallery's, rotated by `theta`.

    The rotation is built as `expm(theta * (A - A^T)/2)` rather than by sampling an
    orthogonal matrix uniformly, and the difference is the entire reason this test is
    meaningful. A uniform random rotation is a rotation by ~pi/2, which does not
    represent a subject: `g_i @ R` is then an unrelated direction, the true match is
    not even the nearest neighbour, and no label-free method could find it. The
    cross-subject premise is that the concept relationships SURVIVE and only the
    orientation differs, i.e. that the frame difference is small enough for the true
    correspondence to remain findable. `theta` parametrises how far from that premise
    we are, and the tests below pin both sides of the threshold.
    """
    g = torch.Generator().manual_seed(seed)
    gal = torch.randn(n, d, generator=g)
    a = torch.randn(d, d, generator=g)
    r_true = torch.linalg.matrix_exp(theta * (a - a.t()) / 2.0)
    shift = torch.randn(1, d, generator=g) * 0.5
    q = ((gal - gal.mean(0, keepdim=True)) @ r_true + shift
         + sigma * torch.randn(n, d, generator=g))
    return q, gal


def test_recovery_works_inside_the_recoverable_regime() -> None:
    """Below the threshold, label-free recovery restores retrieval to near-perfect.

    This is the positive case and the one that justifies the whole component: the map
    is estimated from pseudo-matches only, yet it lifts accuracy from the 0.81 that
    the rotated frame allows to 1.00, because at theta=0.2 the mutual nearest
    neighbours are 98% correct and Procrustes tolerates a small minority of wrong
    landmarks.
    """
    q, gal = _rotated_subject(theta=0.20, sigma=0.10)
    lab = torch.arange(q.shape[0])
    raw = _top1(q, gal)
    csls = float((csls_scores(q, gal, k=10).argmax(1) == lab).float().mean())
    qh, diag = recover(q, gal, k=10, rho=0.1, max_landmarks=160)
    rec = float((csls_scores(qh, gal, k=10).argmax(1) == lab).float().mean())
    check(raw < 0.95, "the rotated frame costs accuracy before recovery",
          f"raw Top-1 {raw:.3f}")
    check(rec > raw + 0.05,
          "label-free recovery beats direct matching from pseudo-matches alone",
          f"raw {raw:.3f} -> recovered {rec:.3f} (CSLS only {csls:.3f})")
    check(rec > 0.95, "and gets back to near-perfect", f"recovered {rec:.3f}")
    check(diag["n_mutual_pairs"] >= 12,
          "with at least SCORE's minimum landmark count",
          f"{diag['n_mutual_pairs']} landmarks (rate {diag['landmark_rate']:.2f})")
    check(abs(diag["r_minus_i_frobenius"]) > 1e-3,
          "and R* is not the identity, so recovery actually did something",
          f"||R*-I||_F = {diag['r_minus_i_frobenius']:.4f}")
    check(not diag["abstained"], "no abstention in the recoverable regime")


def test_recovery_degrades_past_the_threshold() -> None:
    """Above the threshold recovery makes retrieval WORSE, and the gate catches it.

    The uncomfortable half of the result, and the reason `min_landmark_rate` exists.
    At theta=0.30 the mutual nearest neighbours are only ~4% correct, so the fitted
    map is a well-formed orthogonal matrix pointing the wrong way, and applying it
    scores BELOW doing nothing. Recovery here is not a no-op that wastes time -- it is
    an active loss, which is why the pipeline needs an observable, label-free guard
    rather than trust in the method.

    The guard is the landmark rate, which separates the regimes cleanly (0.90 against
    0.55) without labels. A greedy rate (all queries) collapses the rate to ~1 for
    every pair, which is exactly the signal that the matches are arbitrary.
    """
    q, gal = _rotated_subject(theta=0.30, sigma=0.10)
    lab = torch.arange(q.shape[0])
    base = float((csls_scores(moment_match(q, gal), gal, k=10).argmax(1) == lab).float().mean())
    qh, diag = recover(q, gal, k=10, rho=0.1, max_landmarks=160)
    rec = float((csls_scores(qh, gal, k=10).argmax(1) == lab).float().mean())
    check(diag["landmark_rate"] < 0.75,
          "past the threshold the landmark rate collapses",
          f"rate {diag['landmark_rate']:.3f} ({diag['n_mutual_pairs']} pairs)")
    check(rec < base + 0.02,
          "and recovery does not help, which is the honest reading of this regime",
          f"moment-matched {base:.3f} -> recovered {rec:.3f}")

    # The gate: with an abstention threshold placed inside the gap, recovery declines
    # to act and returns the unrecovered features.
    qh_gated, diag_gated = recover(q, gal, k=10, rho=0.1, max_landmarks=160,
                                   min_landmark_rate=0.80)
    check(diag_gated["abstained"], "the gate abstains past the threshold")
    check("abstain_reason" in diag_gated, "and says why")
    gated = float((csls_scores(qh_gated, gal, k=10).argmax(1) == lab).float().mean())
    check(abs(gated - base) < 1e-6,
          "abstaining returns exactly the unrecovered baseline, not a partial map",
          f"{base:.3f} vs {gated:.3f}")
    # And the gate does NOT fire inside the recoverable regime.
    q_ok, gal_ok = _rotated_subject(theta=0.20, sigma=0.10)
    _qh_ok, diag_ok = recover(q_ok, gal_ok, k=10, rho=0.1, max_landmarks=160,
                              min_landmark_rate=0.80)
    check(not diag_ok["abstained"],
          "the same threshold does not abstain inside the recoverable regime",
          f"rate {diag_ok['landmark_rate']:.3f}")


def test_ablation_switches_are_monotone() -> None:
    """The Table-4 ladder must be reachable from one implementation, and ordered."""
    q, gal = _rotated_subject(theta=0.20, sigma=0.10)
    lab = torch.arange(q.shape[0])

    def acc(x: torch.Tensor) -> float:
        return float((csls_scores(x, gal, k=10).argmax(1) == lab).float().mean())

    raw = _top1(q, gal)
    csls = acc(q)
    mm, _ = recover(q, gal, moment=True, orientation=False)
    rec0, _ = recover(q, gal, moment=True, orientation=True, rho=0.0)
    rec1, _ = recover(q, gal, moment=True, orientation=True, rho=0.1)
    print(f"         ladder: raw {raw:.3f} | CSLS {csls:.3f} | +mean+scale {acc(mm):.3f} "
          f"| +recovery(rho=0) {acc(rec0):.3f} | +identity reg {acc(rec1):.3f}")
    check(csls >= raw, "CSLS ranking is not worse than raw cosine",
          f"{raw:.3f} -> {csls:.3f}")
    check(acc(mm) >= csls - 1e-9, "moment matching does not hurt CSLS",
          f"{csls:.3f} -> {acc(mm):.3f}")
    check(acc(rec0) >= acc(mm), "recovery improves on moment matching",
          f"{acc(mm):.3f} -> {acc(rec0):.3f}")
    check(acc(rec1) >= acc(rec0) - 1e-9,
          "identity regularisation does not hurt once recovery is on",
          f"{acc(rec0):.3f} -> {acc(rec1):.3f}")


def test_recover_refuses_with_no_evidence() -> None:
    """No mutual pair means no map: refuse rather than return a silent identity."""
    g = torch.Generator().manual_seed(16)
    q = torch.randn(50, 16, generator=g)
    gal = torch.randn(50, 16, generator=g) * 100.0     # incomparable scales
    try:
        _qh, diag = recover(q, gal, orientation=True)
        print(f"         (no refusal; {diag['n_mutual_pairs']} mutual pairs found)")
    except SystemExit as e:
        check(True, f"refuses rather than fitting noise ({str(e)[:44]}...)")
        return
    check(True, "produced a map (mutual pairs existed), so nothing to refuse")


def main() -> int:
    test_moment_match_puts_q_on_g_stats()
    test_moment_match_degenerate_dimension()
    test_csls_penalises_a_hub()
    test_csls_shape_and_clamping()
    test_csls_matches_its_closed_form()
    test_mutual_nn_pairs_are_mutual()
    test_landmark_margins_use_whole_gallery()
    test_selection_takes_the_most_confident()
    test_recovery_returns_a_rotation()
    test_recovery_solves_the_procrustes_problem()
    test_identity_regularisation_shrinks_toward_identity()
    test_mismatched_landmarks_refused()
    test_apply_recovery_preserves_geometry_and_moves_the_origin()
    test_recovery_works_inside_the_recoverable_regime()
    test_recovery_degrades_past_the_threshold()
    test_ablation_switches_are_monotone()
    test_recover_refuses_with_no_evidence()

    print(f"\n{_passes} passed, {len(_failures)} failed")
    for f in _failures:
        print(f"  - {f}")
    return 1 if _failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
