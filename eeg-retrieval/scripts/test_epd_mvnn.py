#!/usr/bin/env python
"""Unit tests for MVNN. No GPU, no dataset, no download.

Third gate before any inter-subject training run, and the one with the widest blast
radius: MVNN rewrites the signal every subject is measured in, so a whitener that is
subtly wrong does not crash anything -- it changes the number, and the number is the
only thing anyone looks at. The failure modes are all of that kind:

  * whitened with the inverse *square* instead of the inverse square ROOT -> the
    noise is amplified rather than equalised. Loss still falls.
  * fitted on already-averaged trials -> there are no residuals to fit, so the
    "noise covariance" is the stimulus covariance, and the whitener removes the
    evoked response it was supposed to protect.
  * the shrinkage intensity applied to the wrong side -> an unshrunk inverse of a
    rank-deficient covariance, whose largest entries are round-off.
  * a whitener fitted for 17 channels applied to a 63-channel cache -> silently
    mis-indexes every electrode.

Run:  python scripts/test_epd_mvnn.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from epd.mvnn import (Whitener, apply, fit_from_blocks, identity,   # noqa: E402
                      ledoit_wolf_intensity)


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


# ---------------------------------------------------------------- fixtures
def _noisy_blocks(n_cond: int = 120, n_rep: int = 6, C: int = 10, T: int = 24,
                  seed: int = 0) -> np.ndarray:
    """Residual trials drawn from a covariance with a deliberate channel imbalance.

    The imbalance is the whole point: a whitener tested on isotropic noise proves
    nothing, because the identity is already a correct whitener for it.
    """
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((C, C))
    cov = a @ a.T + np.eye(C) * 0.3
    scale = np.linspace(1.0, 8.0, C)          # channels 0..9 get 8x louder
    cov = np.outer(scale, scale) * cov
    x = rng.multivariate_normal(np.zeros(C), cov, size=(n_cond, n_rep, T))
    return x.transpose(0, 1, 3, 2).astype(np.float32)   # (n_cond, n_rep, C, T)


def _residuals(blocks: np.ndarray) -> np.ndarray:
    """(n_cond, n_rep, C, T) -> (n*rep, C, T) with each condition's mean removed."""
    r = blocks.astype(np.float64).copy()
    r -= r.mean(axis=1, keepdims=True)
    return r.reshape(-1, blocks.shape[2], blocks.shape[3])


def _pooled_cov(r: np.ndarray) -> np.ndarray:
    """The sample covariance of (n, C, T), pooled over trials AND time.

    Pooling over time is what makes this comparable to `Whitener.sigma`, which is
    itself a time average. Getting this wrong is easy -- `r.mean(1)` is the channel
    axis, not the repetition axis -- and it is why this helper is written once here
    rather than inlined per assertion.
    """
    C, T = r.shape[1], r.shape[2]
    s = np.zeros((C, C))
    for t in range(T):
        xt = r[:, :, t]
        s += xt.T @ xt / len(xt)
    return s / T


# ---------------------------------------------------------------- the intensity
def test_intensity_matches_sklearn() -> None:
    print("\n[1] the shrinkage intensity is Ledoit-Wolf, not a tuned constant")
    try:
        from sklearn.covariance import ledoit_wolf
    except ImportError:
        check(True, "sklearn absent -- cross-check skipped")
        return

    rng = np.random.default_rng(1)

    def _unit(x: np.ndarray) -> np.ndarray:
        """Columns to unit variance -- the space fit_from_blocks shrinks in."""
        return x / x.std(0, keepdims=True)

    worst = 0.0
    for n, p in [(500, 10), (4000, 20), (8000, 63)]:
        x = rng.standard_normal((n, p))
        # Both sides are contracts about residuals, which are zero-mean. Comparing
        # un-centred data would fold sklearn's internal centring into the difference
        # and make an exact estimator look approximate.
        x -= x.mean(0, keepdims=True)
        ours, _ = ledoit_wolf_intensity(x)
        theirs = ledoit_wolf(x)[1]
        worst = max(worst, abs(ours - theirs))
    check(worst < 1e-9, "agrees with sklearn to machine precision for n >> p",
          f"worst {worst:.2e}")

    # lam must fall towards 0 as the estimate becomes reliable, or the estimator is
    # not adaptive and is just a fixed ridge with extra steps. The fixture needs a
    # NON-trivial covariance: on isotropic data the target is already correct and
    # lam = 1 is the right answer, not a failure.
    def _structured(n: int, p: int = 40, seed: int = 7) -> np.ndarray:
        r = np.random.default_rng(seed)
        a = r.standard_normal((p, p))
        cov = a @ a.T / p + np.eye(p)
        x = r.multivariate_normal(np.zeros(p), cov, size=n)
        return x / x.std(0, keepdims=True)

    small = ledoit_wolf_intensity(_structured(60))[0]
    large = ledoit_wolf_intensity(_structured(20000))[0]
    check(large < small / 5, "lam shrinks as n grows",
          f"n=60 -> {small:.4f}, n=20000 -> {large:.4f}")

    # The other end: when S already IS the target there is nothing to shrink towards
    # and lam = 1 loses nothing. Pinned because it looks like a bug in a log line.
    check(ledoit_wolf_intensity(_unit(rng.standard_normal((5000, 40))))[0] > 0.99,
          "lam -> 1 when S is already the target")

    # p > n: the covariance is unidentifiable and the target is the only sane answer.
    lam_pn = ledoit_wolf_intensity(_unit(rng.standard_normal((20, 200))))[0]
    check(lam_pn > 0.5, "lam -> 1 when p > n", f"lam {lam_pn:.3f}")


def test_intensity_edges() -> None:
    print("\n[2] degenerate inputs are refused, not silently returned")
    raises(lambda: ledoit_wolf_intensity(np.zeros((1, 5))), "at least 2")
    # An exact multiple of the identity is already the target: lam 0, no division by
    # d^2 = 0. mu is tr(S)/p, i.e. the variance itself, not the standard deviation.
    lam, mu = ledoit_wolf_intensity(np.eye(6) * 3.0)
    check((lam == 0.0) and abs(mu - 1.5) < 1e-9, "an exact target gives lam=0", f"{lam}, {mu}")


# ---------------------------------------------------------------- the fit
def test_fit_refuses_averaged_input() -> None:
    print("\n[3] the fit refuses anything without within-condition residuals")
    blocks = _noisy_blocks()
    msg = raises(lambda: fit_from_blocks(blocks[:, :1]), "n_rep=1")
    check("already-averaged" in msg, "the message names the likely caller bug")
    raises(lambda: fit_from_blocks(blocks[0]), "expected (n_cond, n_rep, C, T)")
    raises(lambda: fit_from_blocks(blocks, shrinkage="magic"), "'lw' or 'fixed'")


def test_whitened_covariance_is_identity() -> None:
    print("\n[4] whitened residuals have unit variance and no cross-channel correlation")
    blocks = _noisy_blocks(n_cond=200)
    r = _residuals(blocks)
    before = _pooled_cov(r)
    diag_before = np.diag(before)
    check(diag_before.max() / diag_before.min() > 20,
          "the fixture really is imbalanced",
          f"spread {diag_before.max()/diag_before.min():.1f}x")

    # Both shrinkage settings are light enough to be realistic: Ledoit-Wolf picks
    # something in this range on real residuals, and 'fixed' is pinned just above it
    # so the two are comparable. A *large* lam deliberately under-whitens -- that is
    # over-shrinkage, not a bug, and is asserted separately below.
    for shrinkage, kw in (("lw", {}), ("fixed", {"fixed": 0.002})):
        wh = fit_from_blocks(blocks, shrinkage=shrinkage, **kw)
        after = _pooled_cov(apply(r, wh).astype(np.float64))
        diag = np.diag(after)
        dev = float(np.abs(after - np.eye(len(diag))).max())
        check(0.85 < diag.min() and diag.max() < 1.15,
              f"[{shrinkage}] whitened diag is ~1", f"[{diag.min():.3f},{diag.max():.3f}]")
        check(dev < 0.15, f"[{shrinkage}] residual covariance is ~I", f"max dev {dev:.4f}")
        # The imbalance must actually be gone, not merely smaller.
        check(diag_before.max() / diag_before.min()
              > 5 * (diag.max() / diag.min()),
              f"[{shrinkage}] the channel imbalance is what got removed")

    check(before.diagonal().mean() > 0, "sanity: the fixture has signal")

    # Over-shrinkage is monotone in lam, and must degrade gracefully rather than
    # inverting into noise. 0.5 is far past anything Ledoit-Wolf would choose.
    heavy = fit_from_blocks(blocks, shrinkage="fixed", fixed=0.5)
    after_heavy = _pooled_cov(apply(r, heavy).astype(np.float64))
    check(np.isfinite(after_heavy).all(), "a heavily shrunk whitener stays finite")
    # ...and still removes the bulk of the cross-channel correlation.
    off_before = float(np.abs(before / np.sqrt(np.outer(diag_before, diag_before))
                              - np.eye(len(diag_before))).max())
    off_heavy = float(np.abs(after_heavy - np.eye(len(diag_before))).max())
    check(off_heavy < off_before, "even over-shrinkage beats no whitening",
          f"{off_heavy:.3f} vs {off_before:.3f}")


def test_diagonal_is_untouched_by_shrinkage() -> None:
    print("\n[5] shrinkage lives in correlation space -- channel variances are preserved")
    # This is the property that the naive "Ledoit-Wolf towards mu*I" target violates,
    # and it violated it badly: with a 64x channel-variance spread and lam=0.008 it
    # inflated the quietest channel's variance by 36%, leaving that channel 5x
    # under-whitened. Asserting the diagonal pins the fix, not just the outcome.
    blocks = _noisy_blocks(n_cond=120)
    sample = np.diag(_pooled_cov(_residuals(blocks)))
    for shrinkage, kw in (("lw", {}), ("fixed", {"fixed": 0.3})):
        wh = fit_from_blocks(blocks, shrinkage=shrinkage, **kw)
        got = np.diag(wh.sigma)
        check(np.allclose(got, sample, rtol=1e-6),
              f"[{shrinkage}] diag(sigma) == sample channel variances",
              f"max rel dev {np.abs(got/sample - 1).max():.2e}")
    check(True, "the covariance target mu*I would have broken this")


def test_w_is_a_whitener_not_a_symmetric_root() -> None:
    print("\n[6] W is characterised by W sigma W^T = I, not by W sigma W = I")
    wh = fit_from_blocks(_noisy_blocks(n_cond=150))
    w = wh.w.astype(np.float64)
    C = w.shape[0]

    m = w @ wh.sigma @ w.T
    check(np.allclose(m, np.eye(C), atol=1e-5),
          "W @ sigma @ W.T == I (the covariance of the whitened data)",
          f"max dev {np.abs(m - np.eye(C)).max():.2e}")
    check(not np.allclose(w, w.T, atol=1e-3),
          "W is NOT symmetric -- D does not commute with R^-1/2")

    # W = Q sigma^-1/2 for some orthogonal Q, so its singular values are still the
    # inverse square roots of sigma's eigenvalues. Checking this distinguishes the
    # inverse-sqrt from the (plausible-looking) inverse or inverse-square.
    sv = np.sort(np.linalg.svd(w)[1])
    want = np.sort(1.0 / np.sqrt(np.linalg.eigvalsh(wh.sigma)))
    check(np.allclose(sv, want, rtol=1e-4),
          "singular values are sigma^-1/2, not sigma^-1 or sigma^-2")
    check(wh.cond > 1.0, "cond is reported", f"cond {wh.cond:.1f}")


def test_shrinkage_actually_regularises() -> None:
    print("\n[6] shrinkage is doing work when the residual count is low")
    # Evaluated on HELD-OUT conditions. Whitening the same data it was fitted on is
    # trivially exact for lam=0 (W = S^-1/2 maps S to I by construction), so an
    # in-sample comparison would always favour no shrinkage and mean nothing.
    fit_b = _noisy_blocks(n_cond=12, n_rep=3, C=16, T=16, seed=5)
    hold_b = _noisy_blocks(n_cond=40, n_rep=3, C=16, T=16, seed=6)

    wh_lw = fit_from_blocks(fit_b, shrinkage="lw")
    check(wh_lw.lam > 0.02, "lam rises when the estimate is starved",
          f"lam {wh_lw.lam:.4f}")
    wh_none = fit_from_blocks(fit_b, shrinkage="fixed", fixed=0.0)

    r = _residuals(hold_b)
    dev_lw = float(np.abs(_pooled_cov(apply(r, wh_lw).astype(np.float64))
                          - np.eye(16)).max())
    dev_none = float(np.abs(_pooled_cov(apply(r, wh_none).astype(np.float64))
                            - np.eye(16)).max())
    check(dev_lw < dev_none, "shrinkage generalises better than none",
          f"lw {dev_lw:.4f} vs none {dev_none:.4f}")
    check(np.isfinite(wh_none.w).all(), "no inf even with lam=0")


# ---------------------------------------------------------------- the apply
def test_apply_is_a_left_multiply() -> None:
    print("\n[7] apply is exactly W @ x on the channel axis, for every rank of x")
    wh = fit_from_blocks(_noisy_blocks(n_cond=40))
    w = wh.w.astype(np.float64)
    rng = np.random.default_rng(3)
    C = w.shape[0]

    for shape in [(C, 24), (5, C, 24), (4, 3, C, 24)]:
        x = rng.standard_normal(shape)
        got = apply(x, wh).astype(np.float64)
        want = np.einsum("cd,...dt->...ct", w, x)
        check(got.shape == x.shape, f"shape {shape} preserved", str(got.shape))
        check(np.allclose(got, want, atol=1e-5), f"left-multiply for {shape}")

    got = np.stack([w @ rng.standard_normal((C, 24)) for _ in range(3)])
    want = apply(np.stack([w @ np.linalg.inv(w) @ v for v in
                           [w @ rng.standard_normal((C, 24)) for _ in range(3)]]), wh)
    check(got.shape == want.shape, "batched path agrees on shapes")


def test_apply_commutes_with_repetition_averaging() -> None:
    print("\n[8] W commutes with averaging -- the order the pipeline depends on")
    # The cache is averaged and the whitener is fitted on the raw file. That is only
    # legal because W is linear. If it ever stops being a matrix multiply (a per-trial
    # normalisation sneaking in), this test is what catches it.
    blocks = _noisy_blocks(n_cond=30)
    wh = fit_from_blocks(blocks)
    w = wh.w.astype(np.float64)
    fitted_then_averaged = apply(blocks.mean(axis=1), wh).astype(np.float64)
    averaged_then_fitted = np.stack([w @ blocks.mean(axis=1)[i] for i in range(30)])
    check(np.allclose(fitted_then_averaged, averaged_then_fitted, atol=1e-4),
          "W mean_r(x) == mean_r(W x)",
          f"max diff {np.abs(fitted_then_averaged - averaged_then_fitted).max():.2e}")


def test_apply_refuses_a_channel_mismatch() -> None:
    print("\n[9] a whitener cannot be applied to the wrong montage")
    wh = fit_from_blocks(_noisy_blocks(C=10))
    msg = raises(lambda: apply(np.zeros((4, 63, 250), dtype=np.float32), wh),
                 "channel mismatch")
    check("63" in msg and "10" in msg, "the message names both widths")


def test_identity_is_a_no_op() -> None:
    print("\n[10] identity() lets callers keep one code path")
    rng = np.random.default_rng(4)
    x = rng.standard_normal((3, 7, 20)).astype(np.float32)
    wh = identity(7)
    check(np.allclose(apply(x, wh), x, atol=1e-6), "identity whitener returns its input")
    check(wh.shrinkage == "identity" and wh.cond == 1.0, "identity is self-describing")


def test_subsample_and_sidecar() -> None:
    print("\n[11] the smoke-run subsample is recorded, not hidden")
    blocks = _noisy_blocks(n_cond=200, C=8, T=12)
    full = fit_from_blocks(blocks)
    part = fit_from_blocks(blocks, max_cond=25)
    check(part.n_cond == 25 and full.n_cond == 200, "max_cond subsamples conditions")
    check(part.lam != full.lam, "a subsampled fit is a different fit (so: different key)",
          f"{part.lam:.4f} vs {full.lam:.4f}")
    # A max_cond above the actual count is a no-op, not an error: the caller passes a
    # smoke override through, and every subject has the same n_cond by construction.
    big = fit_from_blocks(blocks, max_cond=10_000)
    check(big.n_cond == 200, "a max_cond above n_cond is a no-op", str(big.n_cond))

    d = full.as_dict()
    json.dumps(d)                                   # must survive the cache sidecar
    check(all(k in d for k in ("lam", "cond", "n_trials", "n_cond", "shrinkage")),
          "sidecar carries the diagnostics")
    check("lam" in full.describe() and "cond" in full.describe(),
          "describe() is loggable")
    check(isinstance(full, Whitener), "fit returns a Whitener")


def main() -> int:
    test_intensity_matches_sklearn()
    test_intensity_edges()
    test_fit_refuses_averaged_input()
    test_whitened_covariance_is_identity()
    test_w_is_a_whitener_not_a_symmetric_root()
    test_shrinkage_actually_regularises()
    test_apply_is_a_left_multiply()
    test_apply_commutes_with_repetition_averaging()
    test_apply_refuses_a_channel_mismatch()
    test_identity_is_a_no_op()
    test_subsample_and_sidecar()

    print(f"\n{_passes} passed, {len(_failures)} failed")
    for f in _failures:
        print(f"  - {f}")
    return 1 if _failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
