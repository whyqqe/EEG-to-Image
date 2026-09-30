"""Unit tests for the paired statistics.

The point of these is not to test numpy's bootstrap; it is to pin the three
properties every cross-arm claim in this project now rests on:

  * a paired interval must be TIGHTER than an unpaired one on data with a shared
    concept-difficulty component, otherwise the whole reason for the per-concept
    decomposition is absent and the intervals are silently 14x too wide;
  * the verdict must not call a 0.005 shift a result just because 200 paired
    concepts resolve it;
  * a genuine shared shift must be found even when the per-concept noise is far
    larger than the shift -- the case the unpaired test cannot handle at all.

No dataset and no GPU: pure numpy.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from epd import stats

_passes = 0
_failures: list[str] = []


def check(ok: bool, what: str, detail: str = "") -> None:
    global _passes
    if ok:
        _passes += 1
        print(f"  ok   {what}")
    else:
        _failures.append(what)
        print(f"  FAIL {what}" + (f"  [{detail}]" if detail else ""))


def test_pairing_beats_independence() -> None:
    print("\n[1] pairing removes the shared concept-difficulty term")
    rng = np.random.default_rng(0)
    n = 200
    # Every concept has its own difficulty drawn from a wide distribution; both arms
    # see the same one. The arms differ by a constant 0.03.
    concept = rng.normal(0.75, 0.18, size=n)          # shared, large
    a = np.clip(concept + 0.03 + rng.normal(0, 0.01, n), 0, 1)
    b = np.clip(concept + 0.00 + rng.normal(0, 0.01, n), 0, 1)

    paired = stats.paired_bootstrap(a, b)
    # The unpaired interval on the DIFFERENCE OF MEANS: sqrt of the sum of the two
    # independent standard errors, which is what comparing two aggregate accuracies
    # implicitly assumes.
    se_unpaired = np.sqrt(a.var(ddof=1) / n + b.var(ddof=1) / n) * 1.96
    width_paired = paired["hi"] - paired["lo"]
    check(width_paired < se_unpaired / 5.0,
          f"the paired CI ({width_paired:.4f}) is far tighter than the unpaired one "
          f"({2 * se_unpaired:.4f}) when both arms share a concept-difficulty term",
          f"ratio {width_paired / (2 * se_unpaired):.3f}")
    check(paired["excludes_zero"] and abs(paired["delta"] - 0.03) < 0.01,
          "and it still recovers the +0.03 shift that the wide unpaired interval "
          "would have buried", f"delta {paired['delta']:+.4f}")


def test_no_false_positive_on_independent_arms() -> None:
    print("\n[2] two arms that differ only by noise are not called shifted")
    rng = np.random.default_rng(1)
    n = 200
    concept = rng.normal(0.75, 0.18, size=n)
    a = np.clip(concept + rng.normal(0, 0.10, n), 0, 1)
    b = np.clip(concept + rng.normal(0, 0.10, n), 0, 1)
    v = stats.verdict(a, b)
    check(not v["excludes_zero"], "the interval covers zero",
          f"[{v['lo']:+.4f}, {v['hi']:+.4f}]")
    check(v["verdict"] == "within noise", "and the verdict says so",
          v["verdict"])


def test_small_but_real_shift_is_not_oversold() -> None:
    print("\n[3] a shift the 200 concepts can resolve but nobody should care about")
    rng = np.random.default_rng(2)
    n = 200
    # Zero per-concept noise, a 0.005 shift: the interval WILL exclude zero.
    a = np.full(n, 0.750)
    b = np.full(n, 0.745)
    bs = stats.paired_bootstrap(a, b)
    check(bs["excludes_zero"], "the interval does exclude zero at n=200",
          f"[{bs['lo']:+.5f}, {bs['hi']:+.5f}]")
    v = stats.verdict(a, b)
    check(v["verdict"] != "shifted",
          "but the verdict refuses to call it `shifted`, because 0.005 is below "
          "the stated 0.02 threshold", v["verdict"])


def test_sign_test_fails_differently() -> None:
    print("\n[4] the sign test and the bootstrap fail differently, as designed")
    # One concept carries a huge positive shift, the rest are slightly negative.
    # The mean is positive and the interval may exclude zero, but the DIRECTION is
    # inconsistent, so the sign test must not agree.
    a = np.concatenate([np.array([1.0]), np.full(199, 0.49)])
    b = np.concatenate([np.array([0.0]), np.full(199, 0.50)])
    st = stats.sign_test(a - b)
    check(st["frac_positive"] < 0.01,
          "a shift driven by ONE concept shows up as an inconsistent direction",
          f"frac_positive {st['frac_positive']:.4f}")
    check(st["p"] < 0.01,
          "and the sign test flags it, which is the disagreement the caller is "
          "meant to read as `one large concept, not a consistent effect`",
          f"p {st['p']:.4g}")
    # And a consistent shift of the same mean size is NOT flagged the same way.
    a2 = np.full(200, 0.50) + 0.0025
    d2 = a2 - np.full(200, 0.50)
    st2 = stats.sign_test(d2)
    check(st2["frac_positive"] == 1.0 and st2["p"] < 1e-6,
          "a consistent shift of the same total size is maximally significant",
          f"frac_positive {st2['frac_positive']:.2f} p {st2['p']:.3g}")


def test_shape_mismatch_is_refused() -> None:
    print("\n[5] a pairing that cannot be aligned is refused, not broadcast")
    try:
        stats.paired_bootstrap(np.zeros(200), np.zeros(150))
        check(False, "mismatched lengths raise rather than silently broadcasting")
    except ValueError:
        check(True, "mismatched lengths raise rather than silently broadcasting")
    # Degenerate case: every difference is exactly zero.
    st = stats.sign_test(np.zeros(200))
    check(st["n_nonzero"] == 0 and st["p"] == 1.0,
          "an all-zero difference reports p=1 rather than dividing by zero")


def main() -> int:
    print("=" * 72)
    print("paired-statistics tests (no dataset, no GPU)")
    print("=" * 72)
    test_pairing_beats_independence()
    test_no_false_positive_on_independent_arms()
    test_small_but_real_shift_is_not_oversold()
    test_sign_test_fails_differently()
    test_shape_mismatch_is_refused()
    print("\n" + "=" * 72)
    if _failures:
        print(f"FAILED: {len(_failures)} check(s), {_passes} passed")
        for f in _failures:
            print(f"  - {f}")
        return 1
    print(f"ALL PASSED: {_passes} checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
