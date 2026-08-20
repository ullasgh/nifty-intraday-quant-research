"""Targeted coverage for Phase D branches left uncovered after the ratchet regression.

Every test here reaches a specific missing line/branch reported by the coverage gate and
asserts the BEHAVIOUR that branch is supposed to produce, not merely that the line executed
(CLAUDE.md rule 9 / the "no vacuous coverage" constraint in the task brief).

Scope, per the task brief:
  - persistence.py: `stitch_overnight_gaps` input-validation branches (363, 366, 376-377).
  - core.py: `breakout_strength` / `_ohlc_volatility` shape-mismatch guards (742, 813) and
    `_efficiency_ratio_no_day` degenerate-window branches (937-938, 950-951, and the
    933->936 / 954->956 branches).
  - research/ic.py: `_overlap_aware_se` zero-SE fallback (143) and `_fit_half_life`'s
    NaN / +inf branches (197, 205).

`core.py` lines 365, 368 and 405-436 (inside `ewma_volatility_ann`, a function with zero
production callers) are PRE-EXISTING debt that predates Phase D and are explicitly out of
scope per the task brief -- not touched here.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from nifty_quant.features import core, persistence
from nifty_quant.research import ic

# ---------------------------------------------------------------------------
# persistence.py: stitch_overnight_gaps input validation
# ---------------------------------------------------------------------------


def test_stitch_overnight_gaps_rejects_non_2d_array():
    """Line 363 / branch 362->363: a 1-D `close` array must raise, not silently reshape."""
    close_1d = np.array([100.0, 101.0, 102.0])
    day_offsets = np.array([0, 3], dtype=np.int64)

    with pytest.raises(ValueError, match="2-D array"):
        persistence.stitch_overnight_gaps(close_1d, day_offsets)


def test_stitch_overnight_gaps_rejects_empty_array():
    """Lines 365-366 / branch 365->366: zero rows and zero columns must both raise --
    neither degenerate panel has a defined stitched path."""
    day_offsets_for_zero_rows = np.array([0], dtype=np.int64)
    with pytest.raises(ValueError, match="must not be empty"):
        persistence.stitch_overnight_gaps(
            np.zeros((0, 2), dtype=np.float64), day_offsets_for_zero_rows
        )

    day_offsets_for_zero_cols = np.array([0, 3], dtype=np.int64)
    with pytest.raises(ValueError, match="must not be empty"):
        persistence.stitch_overnight_gaps(
            np.zeros((3, 0), dtype=np.float64), day_offsets_for_zero_cols
        )


def test_stitch_overnight_gaps_rejects_non_positive_finite_value():
    """Lines 376-377 / branch 375->376: a finite non-positive price is a contract violation
    (this is a price panel; the function requires strictly positive finite values) and the
    raised message must report the count of bad values found, not just that some exist."""
    close = np.full((4, 5), 100.0, dtype=np.float64)
    close[1, 2] = -5.0  # one finite non-positive value among otherwise-valid prices
    day_offsets = np.array([0, 4], dtype=np.int64)

    with pytest.raises(ValueError, match=r"found 1 non-positive value"):
        persistence.stitch_overnight_gaps(close, day_offsets)


# ---------------------------------------------------------------------------
# core.py: shape-mismatch guards
# ---------------------------------------------------------------------------


def test_breakout_strength_rejects_mismatched_shapes():
    """Line 742 / branch 741->742: close/high/low must share a shape; a caller that passes
    mismatched panels (e.g. a `low` column dropped) gets a clear error, not a broadcast."""
    close = np.full((5, 2), 100.0)
    high = np.full((5, 2), 101.0)
    low = np.full((5, 3), 99.0)  # wrong number of columns

    with pytest.raises(ValueError, match="close, high and low must have identical shapes"):
        core.breakout_strength(close, high, low, window=3)


def test_garman_klass_and_rogers_satchell_reject_mismatched_shapes():
    """Line 813 / branch 812->813: both OHLC-volatility estimators route through the shared
    `_ohlc_volatility` shape guard; a mismatched `open_` panel must raise for either one."""
    open_ = np.full((4, 2), 100.0)
    high = np.full((4, 2), 103.0)
    low = np.full((4, 2), 99.0)
    close = np.full((4, 3), 101.0)  # wrong number of columns

    with pytest.raises(
        ValueError, match="open, high, low and close must have identical shapes"
    ):
        core.garman_klass_volatility(open_, high, low, close, window=2)

    with pytest.raises(
        ValueError, match="open, high, low and close must have identical shapes"
    ):
        core.rogers_satchell_volatility(open_, high, low, close, window=2)


# ---------------------------------------------------------------------------
# core.py: efficiency_ratio degenerate-window branches
# ---------------------------------------------------------------------------


def test_efficiency_ratio_window_one_has_zero_denominator_everywhere():
    """Lines 937-938 / branch 936->937: `window=1` means `diff_window = window - 1 = 0`, so
    the denominator (a sum over `window - 1` first differences) is identically zero at every
    row by construction -- there is no "movement" to divide by. The function's own docstring
    says a zero denominator must be NaN, not 0/0; assert that holds at every row, not just
    that the zero-denominator branch was taken."""
    rng = np.random.default_rng(1)
    close = np.abs(rng.standard_normal((6, 1))) + 50.0

    result = core.efficiency_ratio(close, window=1, day_offsets=None)

    assert result.shape == close.shape
    assert np.all(np.isnan(result)), (
        "window=1 gives a zero denominator at every row; efficiency_ratio must be all-NaN, "
        f"got {result.ravel()}"
    )


def test_efficiency_ratio_session_shorter_than_window_is_nan_but_later_session_is_not():
    """Branch 933->936 (a 1-row session skips the diff_abs assignment), branch 944->950
    (lines 950-951: a session no longer than `diff_window` takes the cumulative-sum-as-is
    path rather than the sliding-difference path), and branch 954->956 (the same short
    session skips the `shifted` assignment, leaving it NaN) all fire together on a session
    that is too short to ever satisfy `window=2`.

    Two sessions via day_offsets=[0, 1, 4]: session 1 is a single bar (too short for
    window=2, by construction can never produce a finite ratio); session 2 is a 3-bar
    monotone ramp, long enough that its last two rows DO produce a defined ratio -- this
    is the behavioural contrast that proves the short session's NaN is the "too short"
    branch and not some other bug NaN-ing everything."""
    close = np.array([[100.0], [100.0], [101.0], [103.0]], dtype=np.float64)
    day_offsets = np.array([0, 1, 4], dtype=np.int64)

    result = core.efficiency_ratio(close, window=2, day_offsets=day_offsets)

    assert np.isnan(result[0, 0]), "the entire 1-row session must be NaN: window=2 can never fit"
    assert np.isnan(result[1, 0]), "warm-up row of the second session (window - 1 rows) is NaN"
    # Session 2 is a monotone ramp: the numerator (net move) equals the sum of absolute
    # first differences (no zig-zag), so efficiency_ratio is exactly 1.0 -- proving the
    # short-session NaN above is a real "too short" condition, not the function returning
    # NaN unconditionally.
    assert result[2, 0] == pytest.approx(1.0)
    assert result[3, 0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# research/ic.py: _overlap_aware_se zero-SE fallback
# ---------------------------------------------------------------------------


def test_information_coefficient_se_degenerates_to_zero_when_every_bar_fails_min_names():
    """Line 143 / branches 137->140 and 141->143: with only 3 symbols (< MIN_NAMES=5),
    `_per_bar_cross_sectional_ic` produces an all-NaN per-bar series at EVERY bar, so the
    block bootstrap resamples an all-NaN array -- no replicate ever has a finite mean.
    `_overlap_aware_se` must degenerate cleanly to 0.0 (a documented fallback), not NaN or
    an exception, and `ICResult.mean` must be NaN (there is no finite per-bar IC to
    average) -- these two facts together are the real behaviour under test, not just that
    line 143 executed."""
    rng = np.random.default_rng(2)
    n_rows, n_symbols = 10, 3  # deliberately below MIN_NAMES=5
    feature = rng.standard_normal((n_rows, n_symbols))
    fwd = rng.standard_normal((n_rows, n_symbols))
    day_offsets = np.array([0, n_rows], dtype=np.int64)

    result = ic.information_coefficient(feature, fwd, day_offsets, horizon=1)

    assert math.isnan(result.mean), "no bar ever reaches MIN_NAMES; mean must be NaN"
    assert result.se == 0.0, "the zero-SE fallback must be an exact 0.0, not NaN"
    assert result.n_bars == n_rows
    assert np.all(np.isnan(result.per_bar))


# ---------------------------------------------------------------------------
# research/ic.py: _fit_half_life NaN and +inf branches
# ---------------------------------------------------------------------------


def test_ic_decay_half_life_is_nan_with_fewer_than_two_valid_horizons():
    """Line 197 / branch 196->197: `_fit_half_life` needs >= 2 finite, strictly positive IC
    values to fit a decay slope. Session length 2 with horizons=[1, 5]: horizon=1 has a real
    (positive) cross-sectional IC at the only usable bar; horizon=5 can never fit inside a
    2-row session, so `forward_returns` is all-NaN and that horizon's IC is NaN. Exactly one
    valid horizon -> half_life must come back NaN, not a spurious finite value from a
    single-point fit."""
    n_symbols = 5  # exactly MIN_NAMES
    close = np.array(
        [
            [100.0, 100.0, 100.0, 100.0, 100.0],
            [100.0, 100.1, 100.2, 100.3, 100.4],
        ],
        dtype=np.float64,
    )
    feature = np.tile(np.arange(n_symbols, dtype=np.float64), (2, 1))
    day_offsets = np.array([0, 2], dtype=np.int64)

    result = ic.ic_decay(feature, close, day_offsets, horizons=[1, 5])

    assert np.isfinite(result.ic[0]) and result.ic[0] > 0.0, (
        "horizon=1 must produce a real positive IC (close is monotone-increasing in "
        "feature rank) -- otherwise this is not testing the '<2 valid horizons' case"
    )
    assert np.isnan(result.ic[1]), "horizon=5 cannot fit inside a 2-row session"
    assert math.isnan(result.half_life)


def test_ic_decay_half_life_is_infinite_when_ic_is_flat_not_decaying():
    """Line 205 / branch 203->205: a non-negative slope (flat or increasing IC across
    horizons) means the decay half-life is undefined/infinite. Built with the same exact
    orthonormal-noise construction as the existing obligation-12 test but with rho=1.0:
    `ret_h = k * (1**h * s + sqrt(1 - 1**(2h)) * noise) = k * s` for every horizon (the
    noise term vanishes exactly, independent of h), so the per-horizon forward-return
    vector is bit-for-bit IDENTICAL across h and every horizon's IC comes out identical
    (~1.0, all positive and finite) -- an exactly flat curve, slope == 0.0 >= 0.0."""
    rng = np.random.default_rng(42)
    n_symbols = 6
    raw = rng.standard_normal((n_symbols, 2))
    centered = raw - raw.mean(axis=0, keepdims=True)
    q, _ = np.linalg.qr(centered)
    scaled = q * np.sqrt(n_symbols)
    s = scaled[:, 0]

    rho = 1.0
    k = 1e-4
    horizons = [1, 2, 3]
    n_rows = max(horizons) + 1
    day_offsets = np.array([0, n_rows], dtype=np.int64)

    feature = np.full((n_rows, n_symbols), np.nan)
    feature[0, :] = s
    close = np.full((n_rows, n_symbols), np.nan)
    close[0, :] = 100.0
    for h in horizons:
        ret_h = k * (rho**h * s)
        close[h, :] = 100.0 * (1.0 + ret_h)

    result = ic.ic_decay(feature, close, day_offsets, horizons=horizons)

    assert np.all(np.isfinite(result.ic))
    assert np.all(result.ic > 0.0)
    # rho=1.0 makes every horizon's forward-return vector identical; the IC curve must
    # be exactly flat, which is the precondition for slope >= 0.0.
    np.testing.assert_allclose(result.ic, result.ic[0], rtol=1e-9)
    assert result.half_life == float("inf")
